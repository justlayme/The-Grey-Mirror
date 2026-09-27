"""Experiment 1B: zero-shot derailment forecasting on Conversations Gone Awry, CRAFT protocol.

Jev sees each conversation one comment at a time and gives a calibrated probability that the NEXT
comment will be a personal attack. A conversation is flagged if any forecast before the attack
crosses a threshold learned on the validation split (exactly how CRAFT is evaluated), so the test
numbers are directly comparable with CRAFT's published Table 1.

Baselines
  published   CRAFT and its baselines, copied from Chang & Danescu-Niculescu-Mizil (EMNLP 2019), Table 1
  tfidf_lr    our re-implementation of "Cumulative BoW": TF-IDF + logistic regression trained on the
              train split (a supervised model; Jev never sees a training example)
  llm         optional System-Two comparator via Vercel AI Gateway (--llm MODEL), last prefix only

  python cs1_grey_mirror/run_cga.py [--corpus wiki|cmv|both] [--llm anthropic/claude-haiku-4.5] [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevlab import metrics as M  # noqa: E402
from jevlab.client import (JevClient, LLMClient, fetch_gateway_pricing, is_mock, noul, run_all,  # noqa: E402
                           score, write_json)
from jevlab.paths import PREPARED, RESULTS  # noqa: E402

PLATFORM = {"wiki": "Wikipedia article talk page", "cmv": "Reddit r/ChangeMyView discussion thread"}

# Chang & Danescu-Niculescu-Mizil (2019), Table 1, test splits. A=accuracy P=precision R=recall.
PUBLISHED = {
    "wiki": {"BoW": [56.5, 55.6, 65.5, 52.4, 60.1], "Awry": [58.9, 59.2, 57.6, 39.8, 58.4],
             "Cumulative BoW": [60.6, 57.7, 79.3, 58.1, 66.8], "Sliding Awry": [60.6, 60.2, 62.4, 41.2, 61.3],
             "CRAFT - CE": [64.9, 64.4, 66.7, 36.9, 65.5], "CRAFT": [66.5, 63.7, 77.1, 44.1, 69.8]},
    "cmv": {"BoW": [52.1, 51.8, 61.3, 57.0, 56.1], "Awry": [54.4, 55.0, 48.3, 39.5, 51.4],
            "Cumulative BoW": [59.9, 58.8, 65.9, 46.2, 62.1], "Sliding Awry": [56.8, 56.6, 58.2, 44.6, 57.4],
            "CRAFT - CE": [57.7, 56.1, 71.2, 55.7, 62.8], "CRAFT": [63.4, 60.4, 77.5, 50.7, 67.9]},
}
PUBLISHED_COLUMNS = ["accuracy", "precision", "recall", "fpr", "f1"]
PUBLISHED_EARLY_WARNING = {"wiki": "on average 3 comments before the attack", "cmv": "on average 4 comments"}

QUESTIONS = {
    "attack_next": {
        "type": "noul",
        "instructions": "Will the next comment in this conversation contain a personal attack, meaning an insult, "
                        "name-calling, or hostility aimed at a person rather than at their argument?",
        "criteria": {"true": "The next reply is likely to attack or insult one of the participants",
                     "false": "The next reply is likely to stay civil, even if people disagree"},
    },
    "derail": {"type": "noul",
               "instructions": "Is this conversation heading toward a hostile breakdown between the participants?"},
    "tension": {
        "type": "score",
        "instructions": "How tense or hostile has this conversation become so far?",
        "criteria": ["Friendly or collaborative", "Civil disagreement", "Frustrated or curt",
                     "Heated, with pointed, condescending or accusatory remarks", "Openly hostile or insulting"],
    },
}

LLM_SYSTEM = ("You forecast conversational derailment. Given an online conversation so far, estimate the probability "
              "that the NEXT comment will contain a personal attack (an insult, name-calling, or hostility aimed at a "
              'person rather than their argument). Reply with only JSON: {"probability": <number from 0 to 1>}')


def load(corpus: str, split: str, limit: int = 0) -> list[dict]:
    rows = [json.loads(line) for line in (PREPARED / f"cga_{corpus}_{split}.jsonl").open()]
    if limit:
        # keep whole pairs so pairwise accuracy stays defined
        pairs = defaultdict(list)
        for r in rows:
            pairs[tuple(sorted([r["conv_id"], r["pair_id"] or r["conv_id"]]))].append(r)
        rows = [r for grp in list(pairs.values())[: limit // 2] for r in grp]
    return rows


def state_for(corpus: str, context: list[dict]) -> dict:
    return {"platform": PLATFORM[corpus], "conversation": context}


async def jev_forecasts(corpus: str, splits: dict[str, list[dict]]) -> tuple[dict, dict]:
    """Returns {split: {conv_id: [per-prefix answers]}} and usage stats."""
    out = {s: {} for s in splits}
    async with JevClient(f"cs1_cga_{corpus}", concurrency=12) as jev:
        tasks, keys = [], []
        for split, rows in splits.items():
            for r in rows:
                for k in range(1, len(r["context"]) + 1):
                    tasks.append(jev.ask(state_for(corpus, r["context"][:k]), QUESTIONS))
                    keys.append((split, r["conv_id"], k))
        print(f"Jev [{corpus}]: {len(tasks)} forecast points")
        answers = await run_all(tasks, f"cga {corpus}", every=1000)
        stats = jev.stats.as_dict()
        stats["model"] = next((a.get("model") for a in answers if a), None)
    for (split, cid, k), ans in zip(keys, answers):
        out[split].setdefault(cid, []).append({
            "k": k, "attack_next": noul(ans, "attack_next"), "derail": noul(ans, "derail"),
            "tension": score(ans, "tension"), "input_tokens": (ans.get("usage") or {}).get("input_tokens"),
            "latency_ms": ans.get("latency_ms")})
    for split in out:
        for cid in out[split]:
            out[split][cid].sort(key=lambda x: x["k"])
    return out, stats


async def llm_forecasts(corpus: str, rows: list[dict], model: str) -> tuple[dict, dict]:
    pricing = {} if is_mock() else fetch_gateway_pricing(model)
    async with LLMClient(f"cs1_cga_{corpus}_llm_{model.replace('/', '_')}", model, pricing=pricing) as llm:
        tasks = [llm.probability(LLM_SYSTEM, json.dumps(state_for(corpus, r["context"]), ensure_ascii=False))
                 for r in rows]
        res = await run_all(tasks, f"llm {corpus}", every=200)
        stats = llm.stats.as_dict()
    return {r["conv_id"]: x for r, x in zip(rows, res)}, {**stats, "model": model, "pricing": pricing}


def tfidf_baseline(corpus: str, train: list[dict], evals: dict[str, list[dict]]) -> dict:
    """Cumulative bag-of-words: every prefix of a training conversation is an example."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    def prefixes(rows):
        for r in rows:
            for k in range(1, len(r["context"]) + 1):
                yield r["conv_id"], k, " ".join(u["text"] for u in r["context"][:k]), r["label"]

    tr = list(prefixes(train))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    X = vec.fit_transform([t for _, _, t, _ in tr])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X, [y for *_, y in tr])
    out = {}
    for split, rows in evals.items():
        pr = list(prefixes(rows))
        p = clf.predict_proba(vec.transform([t for _, _, t, _ in pr]))[:, 1]
        d = defaultdict(list)
        for (cid, k, _, _), pi in zip(pr, p):
            d[cid].append({"k": k, "p": float(pi)})
        out[split] = dict(d)
    return out


def conv_scores(rows, per_conv, field):
    """max over prefixes (CRAFT aggregation) and last-prefix score."""
    mx = np.array([max(x[field] for x in per_conv[r["conv_id"]]) for r in rows])
    last = np.array([per_conv[r["conv_id"]][-1][field] for r in rows])
    return mx, last


def pairwise_accuracy(rows, scores) -> float:
    by_id = {r["conv_id"]: (r, s) for r, s in zip(rows, scores)}
    seen, wins, n = set(), 0.0, 0
    for r in rows:
        pid = r["pair_id"]
        if pid not in by_id or r["conv_id"] in seen:
            continue
        seen |= {r["conv_id"], pid}
        a, sa = by_id[r["conv_id"]]
        b, sb = by_id[pid]
        if a["label"] == b["label"]:
            continue
        pos, neg = (sa, sb) if a["label"] == 1 else (sb, sa)
        wins += 1.0 if pos > neg else 0.5 if pos == neg else 0.0
        n += 1
    return wins / n if n else float("nan")


def early_warning(rows, per_conv, field, thr):
    """For correctly flagged derailments: comments between the first trigger and the attack."""
    h = []
    for r in rows:
        if r["label"] != 1:
            continue
        seq = per_conv[r["conv_id"]]
        first = next((x["k"] for x in seq if x[field] >= thr), None)
        if first is not None:
            h.append(len(r["context"]) - first + 1)  # 1 = flagged right before the attack
    return {"n": len(h), "mean_comments_ahead": float(np.mean(h)) if h else None,
            "median_comments_ahead": float(np.median(h)) if h else None,
            "hist": np.bincount(h).tolist() if h else []}


def evaluate_layer(val_rows, test_rows, val_pc, test_pc, field) -> dict:
    yv = np.array([r["label"] for r in val_rows])
    yt = np.array([r["label"] for r in test_rows])
    v_max, v_last = conv_scores(val_rows, val_pc, field)
    t_max, t_last = conv_scores(test_rows, test_pc, field)
    thr = M.best_threshold(yv, v_max, "accuracy")
    thr_last = M.best_threshold(yv, v_last, "accuracy")
    res = {
        "craft_protocol": {
            "val_threshold": thr,
            "test": M.binary_at(yt, t_max, thr),
            "test_at_0.5": M.binary_at(yt, t_max, 0.5),
            "auc": M.auc(yt, t_max), "auc_ci": M.bootstrap_ci(M.auc, yt, t_max), "aupr": M.aupr(yt, t_max),
            "pairwise_accuracy": pairwise_accuracy(test_rows, t_max),
            "early_warning": early_warning(test_rows, test_pc, field, thr),
            "acc_ci": M.bootstrap_ci(lambda y, p: M.binary_at(y, p, thr)["accuracy"], yt, t_max),
        },
        "last_prefix": {
            "val_threshold": thr_last,
            "test": M.binary_at(yt, t_last, thr_last),
            "auc": M.auc(yt, t_last), "pairwise_accuracy": pairwise_accuracy(test_rows, t_last),
            "calibration": M.summarize_binary(yt, t_last, with_ci=True),
        },
    }
    return res, t_max, t_last


def prefix_calibration(test_rows, test_pc, field) -> dict:
    """Per-forecast calibration: label = 1 only when the very next comment is the attack."""
    y, p = [], []
    for r in test_rows:
        n = len(r["context"])
        for x in test_pc[r["conv_id"]]:
            y.append(int(r["label"] == 1 and x["k"] == n))
            p.append(x[field])
    return M.summarize_binary(y, p)


def run_corpus(corpus: str, args) -> dict:
    val = load(corpus, "val", args.limit)
    test = load(corpus, "test", args.limit)
    train = load(corpus, "train")
    print(f"[{corpus}] train {len(train)}  val {len(val)}  test {len(test)}")

    out = {"corpus": corpus, "n_val": len(val), "n_test": len(test), "published": PUBLISHED[corpus],
           "published_columns": PUBLISHED_COLUMNS, "published_early_warning": PUBLISHED_EARLY_WARNING[corpus],
           "layers": {}}

    base = tfidf_baseline(corpus, train, {"val": val, "test": test})
    res, _, _ = evaluate_layer(val, test, base["val"], base["test"], "p")
    out["layers"]["tfidf_lr_supervised"] = res

    if not args.no_jev:
        jf, jstats = asyncio.run(jev_forecasts(corpus, {"val": val, "test": test}))
        out["jev_usage"] = jstats
        for field in ("attack_next", "derail", "tension"):
            vf = {c: [{**x, field: x[field] / 4 if field == "tension" else x[field]} for x in s] for c, s in jf["val"].items()}
            tf = {c: [{**x, field: x[field] / 4 if field == "tension" else x[field]} for x in s] for c, s in jf["test"].items()}
            res, t_max, t_last = evaluate_layer(val, test, vf, tf, field)
            out["layers"][f"jev_{field}"] = res
        out["jev_prefix_calibration"] = prefix_calibration(test, jf["test"], "attack_next")
        yt = np.array([r["label"] for r in test])
        jmax, _ = conv_scores(test, jf["test"], "attack_next")
        bmax, _ = conv_scores(test, base["test"], "p")
        out["jev_vs_tfidf_auc"] = M.paired_bootstrap_diff(M.auc, yt, jmax, bmax)
        out["tokens_per_forecast"] = jstats["input_tokens"] / max(1, jstats["calls"] + jstats["cached"])
        out["examples"] = pick_examples(test, jf["test"])

    if args.llm and not args.no_jev:
        rows = test if not args.llm_limit else test[: args.llm_limit]
        lf, lstats = asyncio.run(llm_forecasts(corpus, rows, args.llm))
        ok = [r for r in rows if lf[r["conv_id"]]["p"] is not None]
        y = np.array([r["label"] for r in ok])
        pl = np.array([lf[r["conv_id"]]["p"] for r in ok])
        pj = np.array([jf["test"][r["conv_id"]][-1]["attack_next"] for r in ok])
        out["llm_comparison_last_prefix"] = {
            "model": args.llm, "n": len(ok), "unparseable": len(rows) - len(ok), "usage": lstats,
            "llm": M.summarize_binary(y, pl), "jev_same_rows": M.summarize_binary(y, pj),
            "auc_diff_jev_minus_llm": M.paired_bootstrap_diff(M.auc, y, pj, pl),
            "llm_pairwise_accuracy": pairwise_accuracy(ok, pl), "jev_pairwise_accuracy": pairwise_accuracy(ok, pj),
        }
    return out


def pick_examples(test_rows, pc, n=3):
    """A few correctly-forecast derailments with their probability trajectory, for the report."""
    ex = []
    for r in test_rows:
        if r["label"] != 1 or len(r["context"]) < 3 or len(r["context"]) > 5:
            continue
        traj = [round(x["attack_next"], 3) for x in pc[r["conv_id"]]]
        if traj[-1] > 0.5 and traj[0] < 0.3:
            ex.append({"conv_id": r["conv_id"], "trajectory": traj,
                       "context": [{"speaker": u["speaker"], "text": u["text"][:280]} for u in r["context"]]})
        if len(ex) >= n:
            break
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="both", choices=["wiki", "cmv", "both"])
    ap.add_argument("--llm", default="", help="gateway model id for the System-Two comparator, e.g. anthropic/claude-haiku-4.5")
    ap.add_argument("--llm-limit", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="conversations per split (pairs kept whole)")
    ap.add_argument("--no-jev", action="store_true")
    args = ap.parse_args()
    corpora = ["wiki", "cmv"] if args.corpus == "both" else [args.corpus]
    out = {"experiment": "cs1b_conversations_gone_awry", "mock": is_mock(), "questions": QUESTIONS,
           "llm_system_prompt": LLM_SYSTEM, "corpora": {}}
    for c in corpora:
        out["corpora"][c] = run_corpus(c, args)
        print_summary(out["corpora"][c])
    write_json(RESULTS / "cs1b_cga.json", out)


def print_summary(o):
    print(f"\n[{o['corpus']}] test n={o['n_test']}   published CRAFT: A/P/R/FPR/F1 = {o['published']['CRAFT']}")
    for name, L in o["layers"].items():
        t = L["craft_protocol"]["test"]
        print(f"  {name:22s} A {t['accuracy']:.3f} P {t['precision']:.3f} R {t['recall']:.3f} FPR {t['fpr']:.3f} "
              f"F1 {t['f1']:.3f} | AUC {L['craft_protocol']['auc']:.3f} pairwise {L['craft_protocol']['pairwise_accuracy']:.3f}")
    if "jev_usage" in o:
        print("  Jev usage:", o["jev_usage"])
    if "llm_comparison_last_prefix" in o:
        c = o["llm_comparison_last_prefix"]
        print(f"  LLM {c['model']}: AUC {c['llm']['auc']:.3f} ECE {c['llm']['ece']:.3f} | Jev same rows AUC "
              f"{c['jev_same_rows']['auc']:.3f} ECE {c['jev_same_rows']['ece']:.3f} | llm usage {c['usage']}")


if __name__ == "__main__":
    main()
