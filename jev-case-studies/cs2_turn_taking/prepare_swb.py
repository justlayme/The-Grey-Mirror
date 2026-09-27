"""Extract turn-taking decision points from Switchboard telephone conversations.

A voice agent has to decide, every time the caller goes quiet, whether they have finished speaking
(respond now) or are pausing mid-thought (keep listening). Switchboard is ~260 hours of two-person
telephone calls with manually corrected forced alignments (Mississippi State ISIP "MS98" release),
so every real pause can be labelled by what actually happened next:

  HOLD   the same speaker resumed: an agent that responded here would have interrupted
  SHIFT  the other speaker took the floor: the turn was over

Rules (the standard IPU framing used in turn-taking research)
  * inter-pausal units (IPUs) split at >= 200 ms of silence within a channel
  * only "clean" pauses: the other channel is silent when the IPU ends
  * if the other speaker's next IPU is only a backchannel ("uh-huh", "yeah", laughter), the event is
    excluded as ambiguous
  * pauses longer than 10 s are excluded
  * text is lowercase with no punctuation (the MS98 transcripts have none), i.e. ASR-style

Why not AMI: the AMI manual word times are contiguous inside each transcription segment, so short
mid-turn pauses are invisible and only segment boundaries look like pauses. That biases the
hold-pause distribution towards long pauses, so AMI was not used for the latency simulation.

Splits are deterministic by conversation id (80/10/10). Evaluation samples at most
EVENTS_PER_CALL events per call so the eval set spans many different callers.
Output: data/prepared/swb_events_{train,dev,eval}.jsonl  (+ _sample files used for Jev)
"""
from __future__ import annotations

import json
import random
import re
import sys
import zlib
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevlab.paths import PREPARED, RAW  # noqa: E402

IPU_GAP = 0.2
MAX_PAUSE = 10.0
CONTEXT_SECONDS = 45.0
CONTEXT_MAX_WORDS = 140
BACKCHANNEL = {"uh-huh", "um-hum", "uh-hum", "mhm", "hm", "hmm", "yeah", "yes", "yep", "right", "okay", "oh", "sure",
               "really", "wow", "huh", "uh", "um", "mm", "no", "true", "exactly", "gosh", "ah", "alright", "yeh"}
EVENTS_PER_CALL = 10
N_EVAL_SAMPLE = 2400
N_DEV_SAMPLE = 1200

SRC = RAW / "swb_ms98_transcriptions"


def norm(tok: str) -> tuple[str | None, bool]:
    """-> (text or None, is_voice_activity)."""
    if tok in ("[silence]", "[noise]"):
        return None, False
    m = re.match(r"^\[laughter-(.+)\]$", tok)
    if m:
        tok = m.group(1)
    elif re.match(r"^\[.*\]$", tok):  # [laughter], [vocalized-noise]: voiced, but no words
        return None, True
    tok = re.sub(r"\[[^\]]*\]", "", tok)       # partial words: th[e]- -> th-
    tok = re.sub(r"_\d+$", "", tok).strip("{}")
    return (tok or None), True


def load_channel(path: Path):
    """Voice-activity segments with words: list of (start, end, word_or_None)."""
    out = []
    for line in path.open(errors="ignore"):
        parts = line.split()
        if len(parts) < 4:
            continue
        st, en, tok = float(parts[1]), float(parts[2]), parts[3]
        text, voiced = norm(tok)
        if voiced:
            out.append((st, en, text))
    return out


def ipus(segs, spk):
    units, cur = [], []
    for s in segs:
        if cur and s[0] - cur[-1][1] >= IPU_GAP:
            units.append((cur[0][0], cur[-1][1], spk, [x[2] for x in cur if x[2]]))
            cur = []
        cur.append(s)
    if cur:
        units.append((cur[0][0], cur[-1][1], spk, [x[2] for x in cur if x[2]]))
    return units


def is_backchannel(u) -> bool:
    return len(u[3]) <= 2 and all(t in BACKCHANNEL for t in u[3]) and (u[1] - u[0]) <= 1.0


def context_turns(units, t_end):
    window = [u for u in units if u[1] <= t_end + 1e-6 and u[1] >= t_end - CONTEXT_SECONDS and u[3]]
    turns = []
    for u in window:
        if turns and turns[-1][0] == u[2]:
            turns[-1][1].extend(u[3])
        else:
            turns.append([u[2], list(u[3])])
    total = sum(len(t[1]) for t in turns)
    while len(turns) > 1 and total > CONTEXT_MAX_WORDS:
        total -= len(turns[0][1])
        turns.pop(0)
    return [{"speaker": f"Speaker {s}", "text": " ".join(ws)} for s, ws in turns]


def events_for_call(cid: str, a_path: Path, b_path: Path) -> list[dict]:
    chans = {"A": load_channel(a_path), "B": load_channel(b_path)}
    units = sorted(ipus(chans["A"], "A") + ipus(chans["B"], "B"))
    by_spk = {"A": [u for u in units if u[2] == "A"], "B": [u for u in units if u[2] == "B"]}
    out = []
    for u in units:
        st, en, spk, toks = u
        if not toks:
            continue
        other = "B" if spk == "A" else "A"
        # clean ending: the other channel is not active when this IPU ends
        if any(s[0] < en < s[1] for s in chans[other] if s[0] < en):
            continue
        nxt_self = next((v for v in by_spk[spk] if v[0] > en), None)
        nxt_other = next((v for v in by_spk[other] if v[0] >= en), None)
        if nxt_self is None and nxt_other is None:
            continue
        if nxt_other is not None and (nxt_self is None or nxt_other[0] < nxt_self[0]):
            if is_backchannel(nxt_other) or not nxt_other[3]:
                continue
            label, silence = 1, nxt_other[0] - en
        else:
            label, silence = 0, nxt_self[0] - en
        if silence > MAX_PAUSE:
            continue
        ctx = context_turns(units, en)
        if not ctx or ctx[-1]["speaker"] != f"Speaker {spk}":
            continue
        out.append({
            "event_id": f"sw{cid}_{en:.2f}_{spk}", "call": cid, "speaker": f"Speaker {spk}", "t_end": round(en, 3),
            "label_shift": label, "silence_s": round(silence, 3), "ipu_words": len(toks),
            "turn_words": len(ctx[-1]["text"].split()), "last_words": toks[-3:], "context": ctx,
        })
    return out


def split_of(cid: str) -> str:
    h = zlib.crc32(cid.encode()) % 10
    return "dev" if h == 0 else "eval" if h == 1 else "train"


def main():
    buckets = defaultdict(list)
    for a in sorted(SRC.glob("*/*/sw*A-ms98-a-word.text")):
        cid = a.name[2:6]
        b = a.with_name(a.name.replace("A-ms98", "B-ms98"))
        if b.exists():
            buckets[split_of(cid)].extend(events_for_call(cid, a, b))
    rng = random.Random(20260927)
    for split in ("train", "dev", "eval"):
        evs = buckets[split]
        with (PREPARED / f"swb_events_{split}.jsonl").open("w") as fh:
            for e in evs:
                fh.write(json.dumps(e) + "\n")
        n, sh = len(evs), sum(e["label_shift"] for e in evs)
        print(f"{split:5s} calls={len({e['call'] for e in evs}):5d} events={n:7d} shift={sh / max(n, 1):.1%}")
        if split in ("dev", "eval"):
            per_call = defaultdict(list)
            for e in evs:
                per_call[e["call"]].append(e)
            pool = []
            for cid in sorted(per_call):
                es = per_call[cid]
                pool.extend(rng.sample(es, min(EVENTS_PER_CALL, len(es))))
            k = N_EVAL_SAMPLE if split == "eval" else N_DEV_SAMPLE
            sample = sorted(rng.sample(pool, min(k, len(pool))), key=lambda e: e["event_id"])
            with (PREPARED / f"swb_events_{split}_sample.jsonl").open("w") as fh:
                for e in sample:
                    fh.write(json.dumps(e) + "\n")
            ns = sum(e["label_shift"] for e in sample)
            print(f"      sample={len(sample)} from {len(per_call)} calls, shift={ns / len(sample):.1%}")


if __name__ == "__main__":
    main()
