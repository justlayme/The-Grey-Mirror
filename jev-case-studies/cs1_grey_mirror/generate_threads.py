"""Generate the synthetic turning-point benchmark (seeded, fully reproducible).

Three arms of 40 threads each, 90 days per thread, mirroring the design Grey Mirror publishes for
its own turning-point benchmark (40 steady threads, 40 threads with one planted change of known date
and direction), plus one harder arm:

  steady   no change. Includes decoys: isolated fights that get repaired, and a "busy week" with
           lower volume but unchanged warmth. Any detection here is a false positive.
  overt    one planted change (half cooling, half warming). The cold register is explicitly
           negative, replies slow down and volume drops, so lexical and behavioral signals both move.
  subtle   one planted change in *meaning only*. The cold register is polite, deflecting or
           passive-aggressive ("oh great, love that for me 🙂"), and reply timing and exchange volume
           are held constant. A sentiment lexicon should see little change.

Outputs
  data/prepared/cs1_threads.jsonl        one JSON thread per line (messages + ground truth)
  data/prepared/gm_uploads/*.txt          WhatsApp-style exports, uploadable to Grey Mirror
  data/prepared/gm_uploads/*.csv          timestamp,sender,message (Mirror Veil output format)
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs1_grey_mirror import banks as B  # noqa: E402
from jevlab.paths import PREPARED  # noqa: E402

DAYS = 90
N_PER_ARM = 40
START = datetime(2026, 1, 5)
ARMS = ("steady", "overt", "subtle")


def fill(rng: random.Random, template: str) -> str:
    out = template
    for slot, options in B.SLOTS.items():
        token = "{" + slot + "}"
        while token in out:
            out = out.replace(token, rng.choice(options), 1)
    return out


class DayWriter:
    def __init__(self, rng, day_start: datetime, a: str, b: str, b_delay_mode: str):
        self.rng, self.a, self.b = rng, a, b
        self.t = day_start
        self.msgs: list[dict] = []
        self.b_delay_mode = b_delay_mode

    def _delay(self, who: str, mode: str) -> timedelta:
        if who == self.a:
            minutes = self.rng.lognormvariate(math.log(2.0), 0.7)
        elif self.b_delay_mode == "slow" and mode == "cold":
            minutes = self.rng.lognormvariate(math.log(40.0), 0.9)
        else:
            minutes = self.rng.lognormvariate(math.log(3.0), 0.8)
        return timedelta(minutes=min(minutes, 240))

    def say(self, who: str, template: str, mode: str = "warm", first: bool = False):
        if not first:
            self.t += self._delay(who, mode)
        self.msgs.append({"ts": self.t.strftime("%Y-%m-%dT%H:%M"), "from": who, "text": fill(self.rng, template)})

    def jump_to(self, t: datetime):
        self.t = max(self.t + timedelta(minutes=5), t)


def make_thread(arm: str, idx: int) -> dict:
    rng = random.Random(f"{arm}-{idx}-v1")
    a, b = B.NAME_PAIRS[idx % len(B.NAME_PAIRS)]
    if arm == "steady":
        cold_style = "subtle" if idx % 2 == 0 else "overt"
        p_before = p_after = rng.uniform(0.3, 0.9)
        planted, direction = None, None
    else:
        cold_style = arm
        hi, lo = rng.uniform(0.8, 0.92), rng.uniform(0.08, 0.25)
        direction = -1 if idx % 2 == 0 else +1  # -1 cooling, +1 warming
        p_before, p_after = (hi, lo) if direction == -1 else (lo, hi)
        planted = rng.randint(25, 65)

    conflict_days = sorted(rng.sample(range(5, DAYS - 5), rng.randint(2, 3)))
    busy = None
    if rng.random() < (0.5 if arm == "steady" else 0.25):
        s = rng.randint(10, DAYS - 16)
        busy = (s, s + 5)

    cold_bank = {
        "checkin": B.CHECKIN_COLD_SUBTLE if cold_style == "subtle" else B.CHECKIN_COLD_OVERT,
        "plan": B.PLAN_COLD_SUBTLE if cold_style == "subtle" else B.PLAN_COLD_OVERT,
        "affection": B.AFFECTION_COLD_SUBTLE if cold_style == "subtle" else B.AFFECTION_COLD_OVERT,
        "share": B.SHARE_COLD_SUBTLE if cold_style == "subtle" else B.SHARE_COLD_OVERT,
        "goodnight": B.GOODNIGHT_COLD_SUBTLE if cold_style == "subtle" else B.GOODNIGHT_COLD_OVERT,
        "late": B.LATE_COLD_SUBTLE if cold_style == "subtle" else B.LATE_COLD_OVERT,
    }
    warm_bank = {"checkin": B.CHECKIN_WARM, "plan": B.PLAN_WARM, "affection": B.AFFECTION_WARM,
                 "share": B.SHARE_WARM, "goodnight": B.GOODNIGHT_WARM, "late": B.LATE_WARM}
    a_after_cold = B.A_AFTER_NEUTRAL if cold_style == "subtle" else B.A_AFTER_HURT
    # Overt cooling also slows replies and cuts volume; subtle holds both constant (meaning-only change).
    b_delay_mode = "slow" if cold_style == "overt" else "same"

    messages = []
    for d in range(DAYS):
        p_warm = p_before if planted is None or d < planted else p_after
        is_busy = busy is not None and busy[0] <= d < busy[1]
        day0 = START + timedelta(days=d)
        w = DayWriter(rng, day0 + timedelta(hours=7, minutes=rng.randint(30, 150)), a, b, b_delay_mode)
        mode = lambda: "warm" if rng.random() < p_warm else "cold"  # noqa: E731
        vol = 1.0 if cold_style == "subtle" else 0.45 + 0.55 * p_warm  # overt cold => fewer exchanges
        if is_busy:
            vol *= 0.5

        # morning check-in
        w.say(a, rng.choice(B.CHECKIN_Q), first=True)
        if is_busy:
            w.say(b, rng.choice(B.BUSY_WARM))
        else:
            m = mode()
            w.say(b, rng.choice(warm_bank["checkin"] if m == "warm" else cold_bank["checkin"]), m)
            w.say(a, rng.choice(B.A_AFTER_WARM if m == "warm" else a_after_cold))

        slots = []
        for _ in range(rng.randint(1, 2) if not is_busy else 1):
            slots.append("logistics")
        for kind, prob in (("plan", 0.5), ("affection", 0.6), ("share", 0.6), ("late", 0.25)):
            if rng.random() < prob * vol:
                slots.append(kind)
        if d in conflict_days:
            slots.append("conflict_repaired")
        if cold_style == "overt" and rng.random() < 0.35 * (1 - p_warm):
            slots.append("conflict_unrepaired")
        rng.shuffle(slots)

        hour = 10.0
        for kind in slots:
            hour += rng.uniform(0.8, 2.2)
            w.jump_to(day0 + timedelta(hours=min(hour, 21.5)))
            if kind == "logistics":
                q, answers = rng.choice(B.LOGISTICS)
                asker, answerer = (a, b) if rng.random() < 0.5 else (b, a)
                w.say(asker, q)
                w.say(answerer, rng.choice(answers))
            elif kind in ("plan", "affection", "late"):
                opener = {"plan": B.PLAN_PROPOSE, "affection": B.AFFECTION_A, "late": B.LATE_A}[kind]
                w.say(a, rng.choice(opener))
                m = mode()
                w.say(b, rng.choice(warm_bank[kind] if m == "warm" else cold_bank[kind]), m)
                if kind == "plan":
                    w.say(a, rng.choice(B.A_AFTER_WARM if m == "warm" else a_after_cold))
            elif kind == "share":
                m = mode()
                w.say(b, rng.choice(warm_bank["share"] if m == "warm" else cold_bank["share"]), m)
                w.say(a, rng.choice(B.A_AFTER_SHARE_WARM if m == "warm" else a_after_cold))
            elif kind == "conflict_repaired":
                first, second = (a, b) if rng.random() < 0.5 else (b, a)
                w.say(first, rng.choice(B.CONFLICT_OPEN))
                w.say(second, rng.choice(B.CONFLICT_DEFEND))
                w.say(first, rng.choice(B.CONFLICT_ESCALATE))
                w.jump_to(w.t + timedelta(hours=rng.uniform(1, 3)))
                w.say(second, rng.choice(B.REPAIR))
                w.say(first, rng.choice(B.REPAIR_ACCEPT))
            elif kind == "conflict_unrepaired":
                w.say(a, rng.choice(B.CONFLICT_OPEN))
                w.say(b, rng.choice(B.CONFLICT_DEFEND), "cold")
                w.say(a, rng.choice(B.CONFLICT_ESCALATE))

        if rng.random() < 0.85:
            w.jump_to(day0 + timedelta(hours=22, minutes=rng.randint(0, 70)))
            w.say(a, rng.choice(B.GOODNIGHT_A))
            m = mode()
            w.say(b, rng.choice(warm_bank["goodnight"] if m == "warm" else cold_bank["goodnight"]), m)

        for msg in w.msgs:
            msg["day"] = d
        messages.extend(w.msgs)

    return {
        "thread_id": f"{arm}_{idx:03d}",
        "arm": arm,
        "A": a,
        "B": b,
        "days": DAYS,
        "start_date": START.strftime("%Y-%m-%d"),
        "cold_style": cold_style,
        "planted_day": planted,
        "direction": direction,
        "p_warm_before": round(p_before, 3),
        "p_warm_after": round(p_after, 3),
        "decoys": {"conflict_days": conflict_days, "busy_week": busy},
        "messages": messages,
    }


def export_for_grey_mirror(thread: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tid = thread["thread_id"]
    with (out_dir / f"{tid}.txt").open("w") as fh:
        for m in thread["messages"]:
            t = datetime.strptime(m["ts"], "%Y-%m-%dT%H:%M")
            fh.write(f"{t.month}/{t.day}/{t.strftime('%y')}, {t.strftime('%H:%M')} - {m['from']}: {m['text']}\n")
    with (out_dir / f"{tid}.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["timestamp", "sender", "message"])
        for m in thread["messages"]:
            wr.writerow([m["ts"], m["from"], m["text"]])


def main():
    out = PREPARED / "cs1_threads.jsonl"
    gm_dir = PREPARED / "gm_uploads"
    n_msgs = 0
    with out.open("w") as fh:
        for arm in ARMS:
            for i in range(N_PER_ARM):
                th = make_thread(arm, i)
                n_msgs += len(th["messages"])
                fh.write(json.dumps(th, ensure_ascii=False) + "\n")
                export_for_grey_mirror(th, gm_dir)
    print(f"wrote {out} ({len(ARMS) * N_PER_ARM} threads, {n_msgs:,} messages) and Grey Mirror uploads in {gm_dir}")


if __name__ == "__main__":
    main()
