"""Prepare Conversations Gone Awry (Wikipedia talk pages, and Reddit ChangeMyView) for forecasting.

Protocol follows CRAFT (Chang & Danescu-Niculescu-Mizil, EMNLP 2019): section headers removed,
the final comment (the personal attack, or the matched civil comment) is never shown, and a forecast
is made after every comment before it. Speakers are relabelled "Speaker 1..n" by first appearance.

Source: ConvoKit corpora (Cornell), downloaded by run_all.sh into data/raw/.
Output: data/prepared/cga_{wiki,cmv}_{train,val,test}.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevlab.paths import PREPARED, RAW  # noqa: E402

CORPORA = {
    "wiki": ("conversations-gone-awry-corpus", "conversation_has_personal_attack"),
    "cmv": ("conversations-gone-awry-cmv-corpus", "has_removed_comment"),
}
MAX_WORDS = 250  # per comment; keeps state well inside Jev's context budget


def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split(" ")
    if len(words) > MAX_WORDS:
        text = " ".join(words[:MAX_WORDS]) + " [...]"
    return text


def prepare(name: str) -> dict:
    folder, label_key = CORPORA[name]
    d = RAW / folder
    conv = json.loads((d / "conversations.json").read_text())
    utts = defaultdict(list)
    with (d / "utterances.jsonl").open() as fh:
        for i, line in enumerate(fh):
            u = json.loads(line)
            meta = u.get("meta") or {}
            if meta.get("is_section_header"):
                continue
            utts[u["conversation_id"]].append((u.get("timestamp") or 0, i, u.get("speaker"), u.get("text") or ""))
    counts = {}
    for split in ("train", "val", "test"):
        rows = []
        for cid, c in conv.items():
            meta = c.get("meta", c)
            if meta.get("split") != split:
                continue
            us = sorted(utts.get(cid, []), key=lambda x: (x[0], x[1]))
            if len(us) < 2:
                continue
            spk = {}
            seq = []
            for _, _, s, text in us:
                spk.setdefault(s, f"Speaker {len(spk) + 1}")
                seq.append({"speaker": spk[s], "text": clean(text)})
            rows.append({"conv_id": cid, "pair_id": meta.get("pair_id"), "label": int(bool(meta.get(label_key))),
                         # the last comment is withheld: it is the event being forecast
                         "context": seq[:-1]})
        with (PREPARED / f"cga_{name}_{split}.jsonl").open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[split] = {"conversations": len(rows), "positives": sum(r["label"] for r in rows),
                         "forecast_points": sum(len(r["context"]) for r in rows)}
    return counts


def main():
    names = sys.argv[1:] or list(CORPORA)
    for n in names:
        print(n, json.dumps(prepare(n)))


if __name__ == "__main__":
    main()
