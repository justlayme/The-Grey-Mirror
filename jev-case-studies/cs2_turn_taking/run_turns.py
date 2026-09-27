"""Experiment 2: Jev as the ears of a voice agent (zero-shot semantic endpointing on Switchboard).

At every clean pause in real telephone conversations, one Jev request (four questions, one state)
estimates whether the caller has finished their turn. Requests on the dev and eval samples are sent
one at a time, so every recorded latency is a clean round trip from this machine through Vercel AI
Gateway, and the policy simulation uses each event's own measured latency.

  python cs2_turn_taking/run_turns.py [--llm anthropic/claude-haiku-4.5 --llm-limit 600] [--no-jev]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs2_turn_taking import simulate as S  # noqa: E402
from jevlab import metrics as M  # noqa: E402
from jevlab.client import JevClient, LLMClient, fetch_gateway_pricing, is_mock, noul, score, write_json  # noqa: E402
from jevlab.paths import PREPARED, RESULTS  # noqa: E402

FILLERS = {"uh", "um", "uh-huh", "um-hum"}
FUNCTION = {"and", "but", "so", "because", "or", "the", "a", "an", "to", "of", "that", "with", "in", "for", "like",
            "if", "when", "my", "your", "their", "is", "was", "i", "we", "they", "it's", "i'm", "just", "then"}
QWORDS = {"what", "how", "do", "did", "are", "is", "can", "would", "where", "why", "who", "have", "does", "was"}

LLM_SYSTEM = ("You are the turn-taking detector inside a voice assistant. You receive a live phone-call transcript "
              "(speech recognition output, no punctuation). The last speaker listed has just gone silent. Estimate "
              "the probability that they have finished their turn and it is now the other person's turn to speak, as "
              "opposed to pausing mid-thought and about to continue. Reply with only JSON: "
              '{"probability": <number from 0 to 1>}')


def questions(spk: str) -> dict:
    return {
        "done": {
            "type": "noul",
            "instructions": f"{spk} has just gone quiet. Has {spk} finished what they were saying, so that it is now "
                            f"the other person's turn to talk?",
            "criteria": {"true": f"{spk}'s last words complete a thought and hand the conversation over",
                         "false": f"{spk} is mid-sentence, mid-list, searching for a word, or clearly has more to say"},
        },
        "unfinished": {"type": "noul",
                       "instructions": f"Do {spk}'s last words stop in the middle of a sentence or phrase?"},
        "question": {"type": "noul",
                     "instructions": f"Did {spk} just ask the other person a question or directly invite them to respond?"},
        "yield": {
            "type": "score",
            "instructions": f"How likely is it that {spk} is done talking for now?",
            "criteria": ["Clearly mid-sentence and will continue", "Probably continuing", "Could go either way",
                         "Probably finished", "Clearly finished and handing over the turn"],
        },
    }


def state(e: dict) -> dict:
    return {"setting": "Live two-person phone call, speech-recognition transcript without punctuation",
            "transcript": e["context"], "just_went_quiet": e["speaker"]}


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (PREPARED / name).open()]


# ---- lexical baseline (supervised, in-domain) -----------------------------------------------------

def lex_features(e: dict) -> dict:
    words = e["context"][-1]["text"].split()
    prev = e["context"][-2]["text"].split() if len(e["context"]) > 1 else []
    w = ["<s>", "<s>"] + words
    f = {f"l1={w[-1]}": 1, f"l2={w[-2]}_{w[-1]}": 1, f"l3={w[-3]}_{w[-2]}_{w[-1]}": 1 if len(w) > 2 else 0,
         f"f1={words[0] if words else ''}": 1, f"p1={prev[-1] if prev else '<none>'}": 1,
         "last_filler": int(w[-1] in FILLERS), "last_function": int(w[-1] in FUNCTION),
         "q_start": int(bool(words) and words[0] in QWORDS),
         f"turn_len={min(len(words), 25) // 3}": 1, f"ipu_len={min(e['ipu_words'], 25) // 3}": 1,
         "partial_word": int(w[-1].endswith("-"))}
    return f


def lexical_model(train: list[dict], seed: int = 0):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression

    rng = random.Random(seed)
    tr = rng.sample(train, min(80_000, len(train)))
    vec = DictVectorizer()
    X = vec.fit_transform([lex_features(e) for e in tr])
    clf = LogisticRegression(max_iter=3000, C=0.5)
    clf.fit(X, [e["label_shift"] for e in tr])
    return lambda evs: clf.predict_proba(vec.transform([lex_features(e) for e in evs]))[:, 1]


# ---- model calls ---------------------------------------------------------------------------------

async def jev_calls(evs: list[dict], namespace: str) -> tuple[list[dict], dict]:
    async with JevClient(namespace) as jev:
        out = []
        for i, e in enumerate(evs):
            a = await jev.ask(state(e), questions(e["speaker"]), sequential_latency=True)
            out.append({"done": noul(a, "done"), "unfinished": noul(a, "unfinished"), "question": noul(a, "question"),
                        "yield": score(a, "yield") / 4.0, "latency_ms": a["latency_ms"],
                        "input_tokens": (a.get("usage") or {}).get("input_tokens")})
            if (i + 1) % 300 == 0:
                print(f"  [jev {namespace}] {i + 1}/{len(evs)}", flush=True)
        stats = jev.stats.as_dict()
    return out, stats


async def llm_calls(evs: list[dict], model: str) -> tuple[list[dict], dict]:
    pricing = {} if is_mock() else fetch_gateway_pricing(model)
    async with LLMClient(f"cs2_turns_llm_{model.replace('/', '_')}", model, pricing=pricing) as llm:
        out = []
        for i, e in enumerate(evs):
            r = await llm.probability(LLM_SYSTEM, json.dumps(state(e)), sequential_latency=True)
            out.append({"p": r["p"], "latency_ms": r["latency_ms"]})
            if (i + 1) % 100 == 0:
                print(f"  [llm] {i + 1}/{len(evs)}", flush=True)
        stats = {**llm.stats.as_dict(), "model": model, "pricing": pricing}
    return out, stats


# ---- evaluation ------------------------------------------------------------------------------------

def sim_block(p_dev, lat_dev, dev, p_eval, lat_eval, ev) -> dict:
    """Operating points chosen on dev, reported on eval; full eval curve for the chart."""
    y_d = np.array([e["label_shift"] for e in dev], bool)
    s_d = np.array([e["silence_s"] for e in dev])
    y_e = np.array([e["label_shift"] for e in ev], bool)
    s_e = np.array([e["silence_s"] for e in ev])
    dev_pts = S.gated_points(p_dev, lat_dev, y_d, s_d)
    eval_pts = S.gated_points(p_eval, lat_eval, y_e, s_e)
    chosen = {}
    for target in (0.05, 0.10, 0.20):
        pt = S.best_under(dev_pts, target)
        chosen[f"interrupt_le_{int(target * 100)}pct"] = None if pt is None else {
            "policy": {k: pt[k] for k in ("family", "theta", "fallback", "a", "b") if k in pt}, "dev": pt,
            "eval": S.apply_policy(pt, p_eval, lat_eval, y_e, s_e)}
    return {"frontier_eval": S.frontier(eval_pts), "chosen_on_dev": chosen}


def silence_block(dev, ev) -> dict:
    y_d = np.array([e["label_shift"] for e in dev], bool)
    s_d = np.array([e["silence_s"] for e in dev])
    y_e = np.array([e["label_shift"] for e in ev], bool)
    s_e = np.array([e["silence_s"] for e in ev])
    dev_curve = S.silence_curve(y_d, s_d)
    chosen = {}
    for target in (0.05, 0.10, 0.20):
        pt = S.best_under(dev_curve, target)
        chosen[f"interrupt_le_{int(target * 100)}pct"] = {"policy": {"T": pt["T"]}, "dev": pt,
                                                          "eval": S.apply_policy(pt, None, None, y_e, s_e)}
    return {"curve_eval": S.silence_curve(y_e, s_e), "chosen_on_dev": chosen}


def combine_on_dev(dev_feats, eval_feats, y_dev, cols):
    """Logistic regression over Jev answers, fit on the dev sample only (the "feature extraction" pattern)."""
    from sklearn.linear_model import LogisticRegression

    Xd = np.array([[f[c] for c in cols] for f in dev_feats])
    Xe = np.array([[f[c] for c in cols] for f in eval_feats])
    clf = LogisticRegression(max_iter=2000).fit(Xd, y_dev)
    return clf.predict_proba(Xd)[:, 1], clf.predict_proba(Xe)[:, 1], dict(zip(cols, clf.coef_[0].round(3).tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="")
    ap.add_argument("--llm-limit", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0, help="events per sample (0 = all)")
    ap.add_argument("--no-jev", action="store_true")
    args = ap.parse_args()

    dev = load("swb_events_dev_sample.jsonl")
    ev = load("swb_events_eval_sample.jsonl")
    if args.limit:
        dev, ev = dev[: args.limit], ev[: args.limit]
    train = load("swb_events_train.jsonl")
    y_dev = np.array([e["label_shift"] for e in dev])
    y_ev = np.array([e["label_shift"] for e in ev])
    print(f"train {len(train)}  dev {len(dev)}  eval {len(ev)}  eval shift rate {y_ev.mean():.1%}")

    lex = lexical_model(train)
    p_lex_dev, p_lex_ev = lex(dev), lex(ev)
    lex_lat_dev, lex_lat_ev = np.full(len(dev), 0.002), np.full(len(ev), 0.002)  # local model: ~2 ms

    out = {"experiment": "cs2_turn_taking_switchboard", "mock": is_mock(), "n_dev": len(dev), "n_eval": len(ev),
           "eval_shift_rate": float(y_ev.mean()), "vad_s": S.VAD_S,
           "pause_stats_eval": {
               "hold_pause_pctl_50_75_90_95": np.percentile([e["silence_s"] for e in ev if not e["label_shift"]],
                                                           [50, 75, 90, 95]).round(3).tolist(),
               "shift_gap_pctl_50_75_90": np.percentile([e["silence_s"] for e in ev if e["label_shift"]],
                                                       [50, 75, 90]).round(3).tolist()},
           "questions_example": questions("Speaker A"), "llm_system_prompt": LLM_SYSTEM,
           "models": {}, "policies": {"silence_only": silence_block(dev, ev)}}
    out["models"]["lexical_lr_supervised"] = M.summarize_binary(y_ev, p_lex_ev)
    out["policies"]["lexical_lr_supervised"] = sim_block(p_lex_dev, lex_lat_dev, dev, p_lex_ev, lex_lat_ev, ev)

    if not args.no_jev:
        jd, jstats_d = asyncio.run(jev_calls(dev, "cs2_turns_dev"))
        je, jstats_e = asyncio.run(jev_calls(ev, "cs2_turns_eval"))
        out["jev_usage"] = {"dev": jstats_d, "eval": jstats_e}
        lat_d = np.array([x["latency_ms"] for x in jd]) / 1000
        lat_e = np.array([x["latency_ms"] for x in je]) / 1000
        out["jev_latency_eval_ms"] = {"p50": float(np.percentile(lat_e * 1000, 50)),
                                      "p90": float(np.percentile(lat_e * 1000, 90)),
                                      "p99": float(np.percentile(lat_e * 1000, 99)),
                                      "mean": float(np.mean(lat_e * 1000))}
        for q in ("done", "yield"):
            pd_ = np.array([x[q] for x in jd])
            pe = np.array([x[q] for x in je])
            out["models"][f"jev_{q}"] = M.summarize_binary(y_ev, pe)
            out["policies"][f"jev_{q}"] = sim_block(pd_, lat_d, dev, pe, lat_e, ev)
        for q in ("unfinished", "question"):
            out["models"][f"jev_{q}_raw"] = M.summarize_binary(y_ev, np.array([x[q] for x in je]), with_ci=False)
        cols = ["done", "unfinished", "question", "yield"]
        cd, ce, coefs = combine_on_dev(jd, je, y_dev, cols)
        out["models"]["jev_fanout_lr"] = {**M.summarize_binary(y_ev, ce), "coefficients": coefs}
        out["policies"]["jev_fanout_lr"] = sim_block(cd, lat_d, dev, ce, lat_e, ev)
        # Jev answers + the lexical model's probability, stacked on dev
        for f, pl in zip(jd, p_lex_dev):
            f["lexical"] = float(pl)
        for f, pl in zip(je, p_lex_ev):
            f["lexical"] = float(pl)
        sd, se, coefs2 = combine_on_dev(jd, je, y_dev, cols + ["lexical"])
        out["models"]["jev_plus_lexical_lr"] = {**M.summarize_binary(y_ev, se), "coefficients": coefs2}
        out["policies"]["jev_plus_lexical_lr"] = sim_block(sd, lat_d, dev, se, lat_e, ev)
        out["auc_diff_jev_done_minus_lexical"] = M.paired_bootstrap_diff(M.auc, y_ev, np.array([x["done"] for x in je]), p_lex_ev)
        out["auc_diff_jev_fanout_minus_lexical"] = M.paired_bootstrap_diff(M.auc, y_ev, ce, p_lex_ev)
        out["per_event_eval"] = [{"event_id": e["event_id"], "y": int(e["label_shift"]), "silence_s": e["silence_s"],
                                  "p_lex": round(float(pl), 4), **{k: round(float(v), 4) for k, v in f.items()}}
                                 for e, f, pl in zip(ev, je, p_lex_ev)]

        if args.llm:
            sub = ev[: args.llm_limit]
            lr, lstats = asyncio.run(llm_calls(sub, args.llm))
            ok = [i for i, r in enumerate(lr) if r["p"] is not None]
            y = y_ev[ok]
            pl = np.array([lr[i]["p"] for i in ok])
            ll = np.array([lr[i]["latency_ms"] for i in ok]) / 1000
            pj = np.array([je[i]["done"] for i in ok])
            jl = lat_e[ok]
            s_sub = np.array([ev[i]["silence_s"] for i in ok])
            ysub = y.astype(bool)
            out["llm_comparison"] = {
                "model": args.llm, "n": len(ok), "unparseable": len(sub) - len(ok), "usage": lstats,
                "llm": M.summarize_binary(y, pl), "jev_done_same_rows": M.summarize_binary(y, pj),
                "auc_diff_jev_minus_llm": M.paired_bootstrap_diff(M.auc, y, pj, pl),
                "llm_latency_ms": {"p50": float(np.percentile(ll * 1000, 50)), "p90": float(np.percentile(ll * 1000, 90))},
                "jev_latency_ms_same_rows": {"p50": float(np.percentile(jl * 1000, 50)), "p90": float(np.percentile(jl * 1000, 90))},
                "frontier_llm": S.frontier(S.gated_points(pl, ll, ysub, s_sub)),
                "frontier_jev_same_rows": S.frontier(S.gated_points(pj, jl, ysub, s_sub)),
                "silence_same_rows": S.silence_curve(ysub, s_sub),
            }

    write_json(RESULTS / "cs2_turn_taking.json", out)
    print_summary(out)


def print_summary(o):
    print(f"\nTurn-taking (eval n={o['n_eval']}, shift rate {o['eval_shift_rate']:.1%}){'  [MOCK]' if o['mock'] else ''}")
    for k, m in o["models"].items():
        print(f"  {k:24s} AUC {m['auc']:.3f}  ECE {m['ece']:.3f}  Brier {m['brier']:.3f}")
    for k, pol in o["policies"].items():
        for tgt, c in pol["chosen_on_dev"].items():
            if c:
                print(f"  {k:24s} {tgt}: eval interrupts {c['eval']['interruption_rate']:.1%}  "
                      f"latency mean {c['eval']['mean_latency_s'] * 1000:.0f} / median {c['eval']['median_latency_s'] * 1000:.0f} ms  {c['policy']}")
    if "jev_latency_eval_ms" in o:
        print("  Jev latency (sequential, ms):", {k: round(v) for k, v in o["jev_latency_eval_ms"].items()})
    if "llm_comparison" in o:
        c = o["llm_comparison"]
        print(f"  LLM {c['model']}: AUC {c['llm']['auc']:.3f} vs Jev {c['jev_done_same_rows']['auc']:.3f} | "
              f"latency p50 {c['llm_latency_ms']['p50']:.0f} ms vs {c['jev_latency_ms_same_rows']['p50']:.0f} ms")


if __name__ == "__main__":
    main()
