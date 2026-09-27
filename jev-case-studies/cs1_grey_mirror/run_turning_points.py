"""Experiment 1A: Jev as the per-day signal layer for turning-point detection.

For every thread-day, one Jev request asks eight typed questions about that day's messages
(speculative fan-out: all questions share one state, so they cost one request's latency).
A pre-registered composite of those answers is fed to the shared change-point detector.
Lexical (VADER) and behavioral (volume, length, questions, reply latency) baselines go through the
identical detector, so every difference in the table is a difference in the signal layer.

  python cs1_grey_mirror/run_turning_points.py [--limit-threads N] [--no-jev]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import zlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevlab import changepoint as cp  # noqa: E402
from jevlab.client import JevClient, is_mock, noul, run_all, score, write_json  # noqa: E402
from jevlab.paths import PREPARED, RESULTS  # noqa: E402

JEV_COMPOSITE = {"warmth_B": +1, "engagement_B": +1, "distant_B": -1, "sarcasm_B": -1,
                 "affection_B": +1, "future_B": +1}
BEHAV_COMPOSITE = {"count_B": +1, "len_B": +1, "qrate_B": +1, "latency_B": -1}
LEXBEHAV_COMPOSITE = {"vader_B": +1, **BEHAV_COMPOSITE}


def questions_for(a: str, b: str) -> dict:
    return {
        "warmth_B": {
            "type": "score",
            "instructions": f"How warm and affectionate is {b} toward {a} in these messages? Judge what {b}'s "
                            f"messages mean and how they would feel to {a}, not only the words used.",
            "criteria": [f"Hostile or contemptuous toward {a}", f"Cold, dismissive, or distant toward {a}",
                         f"Neutral or merely polite toward {a}", f"Warm and friendly toward {a}",
                         f"Very warm and openly affectionate toward {a}"],
        },
        "engagement_B": {
            "type": "score",
            "instructions": f"How engaged and invested is {b} in the conversation with {a} "
                            f"(curiosity about {a}, enthusiasm, effort in replies)?",
            "criteria": ["Disengaged: dismissive or minimal replies", "Low: brief replies with little interest",
                         "Moderate: responsive but not curious or enthusiastic",
                         "High: interested, responsive and enthusiastic",
                         "Very high: curious, enthusiastic and clearly invested"],
        },
        "distant_B": {
            "type": "noul",
            "instructions": f"Is {b} being distant, deflecting, or emotionally withholding toward {a}, "
                            f"even if the wording is polite?",
            "criteria": {"true": f"{b} avoids closeness: deflects plans, withholds affection, or gives polite "
                                 f"but closed replies",
                         "false": f"{b} is open and engaged with {a}"},
        },
        "sarcasm_B": {
            "type": "noul",
            "instructions": f"Does {b} make a sarcastic or passive-aggressive remark toward {a}?",
            "criteria": {"true": "A remark whose words sound pleasant but whose meaning is critical or resentful, "
                                 "or an openly sarcastic jab",
                         "false": f"No sarcasm or passive aggression from {b}"},
        },
        "affection_B": {"type": "noul",
                        "instructions": f"Does {b} explicitly express affection or love toward {a}?"},
        "future_B": {"type": "noul",
                     "instructions": f"Does {b} respond with enthusiasm to plans or to talk about the future with {a}?"},
        "conflict": {"type": "noul",
                     "instructions": f"Is there an argument or open conflict between {a} and {b} in these messages?"},
        "repair": {"type": "noul",
                   "instructions": "Does either person make a sincere attempt to apologize, repair, or de-escalate "
                                   "after tension?"},
    }


def day_state(thread: dict, day_msgs: list[dict]) -> dict:
    d = datetime.strptime(day_msgs[0]["ts"], "%Y-%m-%dT%H:%M")
    return {"date": d.strftime("%A, %B %-d"),
            "messages": [{"time": m["ts"][11:], "from": m["from"], "text": m["text"]} for m in day_msgs]}


def baseline_series(thread: dict, analyzer) -> dict[str, np.ndarray]:
    days = thread["days"]
    by_day = defaultdict(list)
    for m in thread["messages"]:
        by_day[m["day"]].append(m)
    out = {k: np.full(days, np.nan) for k in ("vader_B", "count_B", "len_B", "qrate_B", "latency_B")}
    for d in range(days):
        msgs = by_day.get(d, [])
        bm = [m for m in msgs if m["from"] == thread["B"]]
        if not bm:
            out["count_B"][d] = 0
            continue
        out["vader_B"][d] = np.mean([analyzer.polarity_scores(m["text"])["compound"] for m in bm])
        out["count_B"][d] = len(bm)
        out["len_B"][d] = np.mean([len(m["text"]) for m in bm])
        out["qrate_B"][d] = np.mean(["?" in m["text"] for m in bm])
        lats = []
        for prev, cur in zip(msgs, msgs[1:]):
            if cur["from"] == thread["B"] and prev["from"] == thread["A"]:
                t0 = datetime.strptime(prev["ts"], "%Y-%m-%dT%H:%M")
                t1 = datetime.strptime(cur["ts"], "%Y-%m-%dT%H:%M")
                lats.append((t1 - t0).total_seconds() / 60)
        out["latency_B"][d] = np.mean(lats) if lats else np.nan
    return out


def evaluate(threads: list[dict], layer_series: dict[str, dict[str, np.ndarray]]) -> dict:
    """layer_series[layer][thread_id] -> composite series. Returns per-thread and per-arm results."""
    per_thread = {}
    summary = {}
    for layer, series_by_thread in layer_series.items():
        rows = []
        for th in threads:
            tid = th["thread_id"]
            det = cp.detect(series_by_thread[tid], seed=zlib.crc32(tid.encode()))
            loc_err = abs(det.day - th["planted_day"]) if det.detected and th["planted_day"] is not None else None
            dir_ok = (det.direction == th["direction"]) if det.detected and th["direction"] is not None else None
            row = {"thread_id": tid, "arm": th["arm"], "cold_style": th["cold_style"], **cp.as_dict(det),
                   "planted_day": th["planted_day"], "true_direction": th["direction"],
                   "loc_error": loc_err, "direction_correct": dir_ok}
            rows.append(row)
            per_thread.setdefault(tid, {})[layer] = row
        summary[layer] = summarize_layer(rows)
    return {"per_thread": per_thread, "summary": summary}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def summarize_layer(rows: list[dict]) -> dict:
    out = {}
    for arm in ("steady", "overt", "subtle"):
        r = [x for x in rows if x["arm"] == arm]
        n, k = len(r), sum(x["detected"] for x in r)
        if arm == "steady":
            out[arm] = {"n": n, "false_positives": k, "false_positive_rate": k / n if n else None,
                        "fpr_ci95": wilson(k, n),
                        "fp_by_cold_style": {s: sum(x["detected"] for x in r if x["cold_style"] == s)
                                             for s in ("subtle", "overt")}}
        else:
            det = [x for x in r if x["detected"]]
            errs = [x["loc_error"] for x in det]
            out[arm] = {
                "n": n, "detected": k, "detection_rate": k / n if n else None, "detection_ci95": wilson(k, n),
                "direction_accuracy": (sum(x["direction_correct"] for x in det) / len(det)) if det else None,
                "median_loc_error_days": float(np.median(errs)) if errs else None,
                "within_3_days": (sum(e <= 3 for e in errs) / len(det)) if det else None,
                "detection_by_direction": {
                    "cooling": sum(x["detected"] for x in r if x["true_direction"] == -1),
                    "warming": sum(x["detected"] for x in r if x["true_direction"] == 1)},
            }
    return out


async def run_jev(threads: list[dict]) -> tuple[dict, dict, list[dict]]:
    daily_rows = []
    async with JevClient("cs1_turning_points", concurrency=12) as jev:
        tasks, keys = [], []
        for th in threads:
            by_day = defaultdict(list)
            for m in th["messages"]:
                by_day[m["day"]].append(m)
            qs = questions_for(th["A"], th["B"])
            for d in range(th["days"]):
                if by_day.get(d):
                    tasks.append(jev.ask(day_state(th, by_day[d]), qs))
                    keys.append((th["thread_id"], d, len(by_day[d])))
        print(f"Jev: {len(tasks)} thread-day requests")
        answers = await run_all(tasks, "turning points", every=500)
        stats = jev.stats.as_dict()
        model = next((a.get("model") for a in answers if a), None)

    series = defaultdict(lambda: defaultdict(lambda: None))
    for (tid, d, n_msgs), ans in zip(keys, answers):
        vals = {
            "warmth_B": score(ans, "warmth_B"), "engagement_B": score(ans, "engagement_B"),
            **{q: noul(ans, q) for q in ("distant_B", "sarcasm_B", "affection_B", "future_B", "conflict", "repair")},
        }
        daily_rows.append({"thread_id": tid, "day": d, "n_messages": n_msgs, **vals,
                           "input_tokens": (ans.get("usage") or {}).get("input_tokens"),
                           "latency_ms": ans.get("latency_ms")})
    by_thread = defaultdict(dict)
    for row in daily_rows:
        by_thread[row["thread_id"]][row["day"]] = row
    for th in threads:
        tid = th["thread_id"]
        cols = {}
        for q in list(JEV_COMPOSITE) + ["conflict", "repair"]:
            cols[q] = np.array([by_thread[tid].get(d, {}).get(q, np.nan) for d in range(th["days"])], dtype=float)
        series[tid] = cols
    return series, {**stats, "model": model}, daily_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-threads", type=int, default=0, help="per arm (0 = all)")
    ap.add_argument("--no-jev", action="store_true", help="baselines only")
    args = ap.parse_args()

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    threads = [json.loads(line) for line in (PREPARED / "cs1_threads.jsonl").open()]
    if args.limit_threads:
        keep = defaultdict(int)
        sel = []
        for th in threads:
            if keep[th["arm"]] < args.limit_threads:
                keep[th["arm"]] += 1
                sel.append(th)
        threads = sel
    analyzer = SentimentIntensityAnalyzer()

    base = {th["thread_id"]: baseline_series(th, analyzer) for th in threads}
    layers = {
        "lexicon_vader": {t: cp.composite(s, {"vader_B": +1}) for t, s in base.items()},
        "behavioral": {t: cp.composite(s, BEHAV_COMPOSITE) for t, s in base.items()},
        "lexicon_plus_behavioral": {t: cp.composite(s, LEXBEHAV_COMPOSITE) for t, s in base.items()},
    }
    jev_stats, daily_rows = None, []
    if not args.no_jev:
        jev_series, jev_stats, daily_rows = asyncio.run(run_jev(threads))
        layers["jev_composite"] = {t: cp.composite(s, JEV_COMPOSITE) for t, s in jev_series.items()}
        layers["jev_warmth_only"] = {t: cp.composite(s, {"warmth_B": +1}) for t, s in jev_series.items()}

    res = evaluate(threads, layers)
    n_msgs = sum(len(th["messages"]) for th in threads)
    surface = surface_stats(threads, base)
    out = {
        "experiment": "cs1a_turning_points",
        "mock": is_mock(),
        "detector": {"alpha": cp.ALPHA, "min_effect": cp.MIN_EFFECT, "min_segment": cp.MIN_SEGMENT,
                     "block": cp.BLOCK, "n_perm": cp.N_PERM},
        "composites": {"jev_composite": JEV_COMPOSITE, "behavioral": BEHAV_COMPOSITE,
                       "lexicon_plus_behavioral": LEXBEHAV_COMPOSITE},
        "n_threads": len(threads), "n_messages": n_msgs,
        "jev_usage": jev_stats,
        "surface_stats_subtle_vs_overt": surface,
        "summary": res["summary"],
        "per_thread": res["per_thread"],
        "series_examples": example_series(threads, layers),
    }
    write_json(RESULTS / "cs1a_turning_points.json", out)
    if daily_rows:
        with (RESULTS / "cs1a_turning_points_daily.jsonl").open("w") as fh:
            for r in daily_rows:
                fh.write(json.dumps(r) + "\n")
    print_summary(out)


def surface_stats(threads, base):
    """Mean of the partner's per-day surface statistics in the warm vs cold state, per arm."""
    out = {}
    for arm in ("overt", "subtle"):
        acc = defaultdict(lambda: {"warm": [], "cold": []})
        for th in threads:
            if th["arm"] != arm:
                continue
            s = base[th["thread_id"]]
            for d in range(th["days"]):
                warm = (d < th["planted_day"]) == (th["direction"] == -1)
                for k in ("vader_B", "count_B", "len_B", "qrate_B", "latency_B"):
                    if not np.isnan(s[k][d]):
                        acc[k]["warm" if warm else "cold"].append(s[k][d])
        out[arm] = {k: {"warm": float(np.mean(v["warm"])), "cold": float(np.mean(v["cold"]))} for k, v in acc.items()}
    return out


def example_series(threads, layers, per_arm: int = 3):
    ex = {}
    for arm in ("steady", "overt", "subtle"):
        for th in [t for t in threads if t["arm"] == arm][:per_arm]:
            ex[th["thread_id"]] = {"planted_day": th["planted_day"], "direction": th["direction"],
                                   **{layer: [None if np.isnan(v) else float(v) for v in s[th["thread_id"]]]
                                      for layer, s in layers.items()}}
    return ex


def print_summary(out):
    print(f"\nTurning points ({out['n_threads']} threads, {out['n_messages']:,} messages){'  [MOCK]' if out['mock'] else ''}")
    for layer, s in out["summary"].items():
        st, ov, sb = s["steady"], s["overt"], s["subtle"]
        print(f"  {layer:26s} FP {st['false_positives']}/{st['n']}  "
              f"overt det {ov['detected']}/{ov['n']} dir {fmt(ov['direction_accuracy'])} loc {ov['median_loc_error_days']}  "
              f"subtle det {sb['detected']}/{sb['n']} dir {fmt(sb['direction_accuracy'])} loc {sb['median_loc_error_days']}")
    if out["jev_usage"]:
        print("  Jev usage:", out["jev_usage"])


def fmt(x):
    return "-" if x is None else f"{x:.0%}"


if __name__ == "__main__":
    main()
