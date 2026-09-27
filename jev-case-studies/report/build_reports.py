"""Build the case-study reports from results/*.json.

Every number on the page is read from a results file. Sentences that interpret a Jev result are
generated from the numbers (comparisons are computed, not assumed), and any Jev figure that has not
been produced yet renders as "pending" rather than a placeholder value.

  python report/build_reports.py            -> reports/jev-case-studies.html, reports/*.md
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jevlab import svgcharts as C  # noqa: E402
from jevlab.paths import REPORTS, RESULTS  # noqa: E402
from jevlab.svgcharts import ENTITY, Raw, esc  # noqa: E402

GM = {  # Grey Mirror published figures (justlay.me/research/the-signal-is-rare, Aug-Sep 2026)
    "signal_cohort_messages": 4_600_611, "signal_histories": 29, "conflict_cohort_messages": 2_994_932,
    "operational_messages": 20_794_571, "no_signal_share": 0.974,
}
LIST_PRICES = {  # USD per 1M tokens, Vercel AI Gateway public model list, retrieved 2026-09-27
    "typesafe-ai/jev": (0.042, 0.0), "anthropic/claude-haiku-4.5": (1.0, 5.0), "anthropic/claude-sonnet-5": (2.0, 10.0),
}


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x, d=0):
    return "—" if x is None else f"{x * 100:.{d}f}%"


def ms(x):
    return "—" if x is None else f"{x * 1000:,.0f} ms"


def money(x):
    if x is None:
        return "—"
    return f"${x:,.2f}" if x >= 1 else f"${x:,.4f}" if x >= 0.001 else f"${x:.6f}"


def pending(text="pending"):
    return Raw(f'<span class="pending-tag">{esc(text)}</span>')


# =============================================================================================
# Case study 1A — turning points
# =============================================================================================

LAYER_NAMES = {"lexicon_vader": "Sentiment lexicon (VADER)", "behavioral": "Behavioral metrics",
               "lexicon_plus_behavioral": "Lexicon + behavioral", "jev_composite": "Jev (8-question composite)",
               "jev_warmth_only": "Jev (warmth score only)"}
LAYER_COLOR = {"lexicon_vader": ENTITY["lexical"], "behavioral": ENTITY["behavioral"],
               "lexicon_plus_behavioral": ENTITY["stack"], "jev_composite": ENTITY["jev"],
               "jev_warmth_only": ENTITY["llm"]}


def cs1a(r):
    if r is None:
        return {"html": "<p>Results not generated yet.</p>", "md": "", "has_jev": False, "hero": None}
    S = r["summary"]
    has_jev = "jev_composite" in S
    layers = [k for k in ("lexicon_vader", "behavioral", "lexicon_plus_behavioral", "jev_composite", "jev_warmth_only")
              if k in S]
    if not has_jev:
        layers += ["jev_composite"]

    def cell(layer, arm, key):
        if layer not in S:
            return pending()
        return S[layer][arm][key]

    rows = []
    for L in layers:
        if L not in S:
            rows.append([LAYER_NAMES[L], pending(), pending(), pending(), pending(), pending(), pending()])
            continue
        st, ov, sb = S[L]["steady"], S[L]["overt"], S[L]["subtle"]
        rows.append([LAYER_NAMES[L], f"{st['false_positives']}/{st['n']}",
                     f"{ov['detected']}/{ov['n']}", pct(ov["direction_accuracy"]),
                     "—" if ov["median_loc_error_days"] is None else f"{ov['median_loc_error_days']:.0f} d",
                     f"{sb['detected']}/{sb['n']}", pct(sb["direction_accuracy"])])
    tbl = C.table(["Signal layer", "False positives (40 steady)", "Overt: detected", "Overt: direction",
                   "Overt: median error", "Subtle: detected", "Subtle: direction"], rows)

    series = []
    for L in layers:
        if L == "jev_warmth_only":
            continue
        vals = [S[L]["steady"]["false_positive_rate"], S[L]["overt"]["detection_rate"], S[L]["subtle"]["detection_rate"]] \
            if L in S else [None, None, None]
        series.append({"name": LAYER_NAMES[L], "color": LAYER_COLOR[L], "values": [None if v is None else v * 100 for v in vals]})
    chart = C.bar_chart("Turning points flagged, by arm (same detector for every layer)",
                        ["Steady (false positives)", "Overt change (detected)", "Subtle change (detected)"],
                        series, "% of 40 threads", 100, yfmt=lambda v: f"{v:.0f}%", value_fmt=lambda v: f"{round(v * 0.4)}/40")

    # surface statistics table (proves the subtle arm is meaning-only)
    ss = r["surface_stats_subtle_vs_overt"]
    lab = {"vader_B": "VADER compound (partner)", "count_B": "Partner messages / day", "len_B": "Characters / message",
           "qrate_B": "Share of messages with a question", "latency_B": "Reply delay (min)"}
    srows = []
    for k, name in lab.items():
        srows.append([name, f"{ss['overt'][k]['warm']:.2f} → {ss['overt'][k]['cold']:.2f}",
                      f"{ss['subtle'][k]['warm']:.2f} → {ss['subtle'][k]['cold']:.2f}"])
    stbl = C.table(["Partner surface statistic", "Overt arm: warm → cold", "Subtle arm: warm → cold"], srows)

    # example timeline: first subtle thread
    ex_html = ""
    ex = {k: v for k, v in r["series_examples"].items() if k.startswith("subtle")}
    if ex:
        tid, e = next(iter(ex.items()))
        ser = []
        for L, name in (("lexicon_vader", "Sentiment lexicon"), ("behavioral", "Behavioral"), ("jev_composite", "Jev composite")):
            if L in e:
                ser.append({"name": name, "color": LAYER_COLOR[L],
                            "points": [(d, v, f"day {d}: {v:+.2f} SD (7-day mean)") for d, v in enumerate(rolling(e[L])) if v is not None]})
        ex_html = C.line_chart(
            f"One subtle thread ({tid}): signal z-scored within the thread, 7-day rolling mean",
            ser, "Day", "Standard deviations from the thread's own mean", (0, 89), (-2, 2),
            xfmt=lambda v: f"{v:.0f}", yfmt=lambda v: f"{v:+.0f}", markers=False,
            vlines=[(e["planted_day"], f"planted {'cooling' if e['direction'] == -1 else 'warming'} (day {e['planted_day']})")])

    hero = None
    if has_jev:
        j, v = S["jev_composite"], S["lexicon_vader"]
        hero = {"value": f"{j['subtle']['detected']}/40",
                "label": f"meaning-only turning points found by Jev (sentiment lexicon: {v['subtle']['detected']}/40), "
                         f"with {j['steady']['false_positives']} false alarms in 40 steady threads"}

    findings = []
    vs = S["lexicon_vader"]
    findings.append(
        f"A plain sentiment lexicon already matches Grey Mirror's published benchmark on overt changes: "
        f"{vs['overt']['detected']}/40 detected, {pct(vs['overt']['direction_accuracy'])} correct direction, "
        f"{vs['steady']['false_positives']}/40 false positives. The published benchmark does not separate a smart "
        f"signal layer from a dumb one; the subtle arm does.")
    findings.append(
        f"On meaning-only changes the lexicon finds {vs['subtle']['detected']}/40 and behavioral metrics "
        f"{S['behavioral']['subtle']['detected']}/40 (direction correct in {pct(S['behavioral']['subtle']['direction_accuracy'])} "
        f"of those): the surface statistics barely move, by construction.")
    if has_jev:
        j = S["jev_composite"]
        findings.append(
            f"Jev, reading one day at a time through eight typed questions, flags {j['subtle']['detected']}/40 subtle changes "
            f"({pct(j['subtle']['direction_accuracy'])} correct direction, median localisation error "
            f"{j['subtle']['median_loc_error_days']} days) and {j['overt']['detected']}/40 overt ones, with "
            f"{j['steady']['false_positives']}/40 false positives in threads that contain repaired fights and busy weeks.")
        w = S.get("jev_warmth_only")
        if w:
            findings.append(f"Ablation: the single warmth score alone flags {w['subtle']['detected']}/40 subtle changes "
                            f"({w['steady']['false_positives']}/40 false positives), so the fan-out questions "
                            f"{'add' if j['subtle']['detected'] > w['subtle']['detected'] else 'do not add'} detection power.")
    else:
        findings.append("Jev results for this experiment are pending (see the run instructions).")

    usage = r.get("jev_usage") or {}
    html_ = f"""
<h3 id="cs1a">1A · Turning points: can Jev see a relationship change that the words hide?</h3>
<p>Grey Mirror publishes a synthetic turning-point benchmark: 40 steady threads and 40 threads with one planted change of
known date and direction, reporting 0% false positives, 100% detection, 100% direction and a median localisation error of
0 days. We rebuilt that design ({r['n_threads']} threads, {r['n_messages']:,} messages, 90 days each, inside Grey Mirror's
500–2,000-message range) and added a third arm that it does not test.</p>
<ul>
<li><b>Steady (40):</b> no change, with decoys: two or three fights that are repaired the same day, and in half the threads a
"busy week" with lower volume but unchanged warmth. Any detection is a false positive.</li>
<li><b>Overt (40):</b> the partner cools or warms, and the cold register is explicitly negative ("stop asking. it was bad."),
replies slow down and volume drops. Every surface signal moves.</li>
<li><b>Subtle (40):</b> the partner cools or warms <i>in meaning only</i>. The cold register is polite, deflecting or
passive-aggressive ("oh great, love that for me 🙂", "sounds good in theory, let's play it by ear"), and message volume,
reply timing, length, question rate and sentiment vocabulary are held constant. This is the case Grey Mirror's own
benchmark report lists as a known weakness (sarcasm ~65%).</li>
</ul>
{stbl}
<p class="small">The subtle arm is meaning-only by construction: the partner's sentiment score, volume, question rate and reply
delay are essentially unchanged across the planted change, while the overt arm moves on every surface statistic.</p>
<p><b>Method.</b> For each thread-day, one Jev request carries that day's messages as <code>state</code> and eight typed
questions (two Scores: warmth and engagement; six Nouls: distant-but-polite, sarcasm or passive aggression, affection,
enthusiasm for plans, open conflict, repair). A pre-registered composite (warmth + engagement + affection + plans − distance − sarcasm,
each z-scored within the thread) feeds one change-point detector shared by every layer: maximum two-sample t-statistic over
candidate days, null tested by circular block permutation (999 permutations, 7-day blocks, p &lt; 0.01, effect size ≥ 0.8 SD).
The settings were fixed before any Jev output existed. Baselines go through the identical detector: VADER sentiment of the
partner's messages, and four behavioral metrics (volume, length, question rate, reply delay).</p>
{chart}
{tbl}
{ex_html}
<div class="findings"><h4>What the data says</h4><ul>{''.join(f'<li>{esc(f)}</li>' for f in findings)}</ul></div>
<p class="small">Usage: {usage.get('calls', 0) + usage.get('cached', 0):,} Jev requests,
{usage.get('input_tokens', 0):,} input tokens, {money(usage.get('cost_usd'))} at list price. The 120 threads are also
exported as WhatsApp-style .txt and CSV files (<code>data/prepared/gm_uploads/</code>) so Grey Mirror's own detector can be run on
exactly the same threads for a head-to-head.</p>"""
    md = "\n".join(["### 1A · Turning points", "", md_table(["Signal layer", "FP (40 steady)", "Overt det.", "Overt dir.",
                                                             "Overt median err.", "Subtle det.", "Subtle dir."], rows), "",
                    *[f"- {f}" for f in findings], ""])
    return {"html": html_, "md": md, "has_jev": has_jev, "hero": hero, "usage": usage,
            "tokens_per_message": (usage.get("input_tokens", 0) / r["n_messages"]) if usage else None}


# =============================================================================================
# Case study 1B — Conversations Gone Awry
# =============================================================================================

def cs1b(r):
    if r is None:
        return {"html": "<p>Results not generated yet.</p>", "md": "", "has_jev": False, "hero": None}
    blocks, md_blocks, heroes = [], [], {}
    has_jev = False
    for corpus, o in r["corpora"].items():
        name = "Wikipedia talk pages" if corpus == "wiki" else "Reddit r/ChangeMyView"
        L = o["layers"]
        jev = L.get("jev_attack_next")
        has_jev = has_jev or jev is not None
        rows = []
        for model, vals in o["published"].items():
            rows.append([f"{model} (published, supervised)", *[f"{v:.1f}" for v in vals], "—", "—"])
        t = L["tfidf_lr_supervised"]["craft_protocol"]
        rows.append(["TF-IDF + LR (our re-run, supervised)", *[f"{t['test'][k] * 100:.1f}" for k in
                                                               ("accuracy", "precision", "recall", "fpr", "f1")],
                     f"{t['auc']:.3f}", pct(t["pairwise_accuracy"], 1)])
        for key, label in (("jev_attack_next", "Jev: 'next comment is an attack?' (zero-shot)"),
                           ("jev_derail", "Jev: 'heading for a breakdown?' (zero-shot)"),
                           ("jev_tension", "Jev: tension score (zero-shot)")):
            if key in L:
                c = L[key]["craft_protocol"]
                rows.append([label, *[f"{c['test'][k] * 100:.1f}" for k in ("accuracy", "precision", "recall", "fpr", "f1")],
                             f"{c['auc']:.3f}", pct(c["pairwise_accuracy"], 1)])
            else:
                rows.append([label, *[pending()] * 7])
        tbl = C.table(["Model", "Accuracy", "Precision", "Recall", "FPR", "F1", "AUC", "Pairwise"], rows,
                      caption=f"{name}: test split (n = {o['n_test']}), CRAFT protocol")

        hrows = []
        for model, vals in o["published"].items():
            hrows.append((f"{model} (published)", vals[0], ENTITY["published"] if model == "CRAFT" else ENTITY["neutral"],
                          f"accuracy {vals[0]:.1f}%"))
        hrows.append(("TF-IDF + LR (our re-run)", t["test"]["accuracy"] * 100, ENTITY["lexical"], None))
        hrows.append(("Jev, zero-shot", jev["craft_protocol"]["test"]["accuracy"] * 100 if jev else None, ENTITY["jev"], None))
        chart = C.hbar_chart(f"{name}: forecasting accuracy on the test split", hrows, "Accuracy (%)", 80,
                             value_fmt=lambda v: f"{v:.1f}", ref=(50, "chance"), tick_fmt=lambda v: f"{v:.0f}")

        extra, findings = "", []
        craft = o["published"]["CRAFT"]
        if jev:
            c = jev["craft_protocol"]
            a = c["test"]["accuracy"] * 100
            findings.append(
                f"Zero-shot, with one threshold tuned on the validation split, Jev reaches {a:.1f}% accuracy "
                f"(95% CI {c['acc_ci'][0] * 100:.1f}–{c['acc_ci'][1] * 100:.1f}) and F1 {c['test']['f1'] * 100:.1f}, against "
                f"{craft[0]:.1f}% / {craft[4]:.1f} for CRAFT, a model pre-trained on this platform's conversations and fine-tuned "
                f"on {'2,508' if corpus == 'wiki' else '4,106'} labelled ones ({'above' if a > craft[0] else 'below'} CRAFT by "
                f"{abs(a - craft[0]):.1f} points).")
            findings.append(
                f"Threshold-free: AUC {c['auc']:.3f} (95% CI {c['auc_ci'][0]:.3f}–{c['auc_ci'][1]:.3f}); given a matched pair "
                f"from the same page, Jev ranks the conversation that derails higher {pct(c['pairwise_accuracy'], 1)} of the time.")
            ew = c["early_warning"]
            if ew["mean_comments_ahead"] is not None:
                findings.append(f"Early warning: on correctly flagged derailments Jev first fires {ew['mean_comments_ahead']:.1f} "
                                f"comments before the attack on average (CRAFT: {o['published_early_warning']}).")
            d = o.get("jev_vs_tfidf_auc")
            if d:
                findings.append(f"Against our supervised TF-IDF model on the same test conversations, Jev's AUC is "
                                f"{d['diff']:+.3f} (95% CI {d['ci'][0]:+.3f} to {d['ci'][1]:+.3f}).")
            cal = o.get("jev_prefix_calibration")
            if cal:
                extra += reliability_chart(f"{name}: calibration of every forecast ('is the NEXT comment an attack?')",
                                           cal["reliability"], "Jev", ENTITY["jev"])
                findings.append(f"Calibration across all {cal['n']:,} individual forecasts: ECE {cal['ece']:.3f}, "
                                f"Brier {cal['brier']:.3f}, base rate {pct(cal['base_rate'], 1)}.")
            heroes[corpus] = {"value": f"{a:.1f}%", "label": f"zero-shot derailment accuracy on {name} (CRAFT, trained: {craft[0]:.1f}%)"}
        else:
            findings.append("Jev results for this corpus are pending.")
        llm = o.get("llm_comparison_last_prefix")
        if llm:
            u = llm["usage"]
            n = max(1, u["calls"] + u["cached"])
            jl, ll = llm["jev_same_rows"], llm["llm"]
            ju = o.get("jev_usage", {})
            jn = max(1, ju.get("calls", 0) + ju.get("cached", 0))
            extra += C.table(["", "AUC", "ECE", "Brier", "Pairwise", "Median latency", "Cost / 1,000 forecasts"], [
                ["Jev (same conversations)", f"{jl['auc']:.3f}", f"{jl['ece']:.3f}", f"{jl['brier']:.3f}",
                 pct(llm["jev_pairwise_accuracy"], 1), ms((ju.get("latency_ms_p50") or 0) / 1000) + " *",
                 money(ju.get("cost_usd", 0) / jn * 1000)],
                [f"{llm['model']} (LLM)", f"{ll['auc']:.3f}", f"{ll['ece']:.3f}", f"{ll['brier']:.3f}",
                 pct(llm["llm_pairwise_accuracy"], 1), ms((u.get("latency_ms_p50") or 0) / 1000) + " *",
                 money(u["cost_usd"] / n * 1000)],
            ], caption=f"System One vs an LLM on the final prefix of {llm['n']} test conversations "
                       f"({llm['unparseable']} unparseable LLM replies excluded)")
            extra += '<p class="small">* latency under concurrent load from this run; see Case Study 2 for clean sequential latency.</p>'
            dd = llm["auc_diff_jev_minus_llm"]
            findings.append(f"Against {llm['model']} on the same {llm['n']} conversations: AUC {jl['auc']:.3f} vs {ll['auc']:.3f} "
                            f"(difference {dd['diff']:+.3f}, 95% CI {dd['ci'][0]:+.3f} to {dd['ci'][1]:+.3f}); ECE "
                            f"{jl['ece']:.3f} vs {ll['ece']:.3f}.")
        blocks.append(f"<h4>{esc(name)}</h4>{chart}{tbl}{extra}"
                      f"<div class='findings'><ul>{''.join(f'<li>{esc(f)}</li>' for f in findings)}</ul></div>")
        md_blocks.append("\n".join([f"#### {name}", "", md_table(["Model", "Acc", "P", "R", "FPR", "F1", "AUC", "Pairwise"], rows), "",
                                    *[f"- {f}" for f in findings], ""]))
    html_ = f"""
<h3 id="cs1b">1B · Before the fight: forecasting derailment in real conversations</h3>
<p>Grey Mirror's conflict study found escalation outrunning repair by about 36 to 1 in the median history. Detecting a fight after it happens is
easy; the product question is whether a model can see one coming. Conversations Gone Awry (Cornell, ConvoKit) is the standard public benchmark:
pairs of real conversations from the same page, one of which ends in a personal attack while the other stays civil, with every prefix before the
attack verified as civil. We followed the CRAFT evaluation protocol exactly (Chang &amp; Danescu-Niculescu-Mizil, EMNLP 2019): the final comment
is never shown, a forecast is made after every comment, a conversation is flagged if any forecast crosses a threshold learned on the validation
split, and a flag only counts if it fires before the attack. Jev receives the conversation so far as <code>state</code> and three questions
(will the next comment be a personal attack, is the conversation heading for a breakdown, and a five-level tension score). It never sees a
labelled example; the only fitted parameter is the decision threshold.</p>
{''.join(blocks)}"""
    return {"html": html_, "md": "### 1B · Conversations Gone Awry\n\n" + "\n".join(md_blocks), "has_jev": has_jev,
            "hero": heroes.get("wiki"), "raw": r}


def reliability_chart(title, rel_rows, name, color, extra_series=()):
    ser = [{"name": name, "color": color,
            "points": [(mp, fp, f"predicted {mp:.2f}, observed {fp:.2f} (n={n})") for _, _, mp, fp, n in rel_rows]}]
    for nm, col, rows in extra_series:
        ser.append({"name": nm, "color": col,
                    "points": [(mp, fp, f"predicted {mp:.2f}, observed {fp:.2f} (n={n})") for _, _, mp, fp, n in rows]})
    return C.line_chart(title, ser, "Predicted probability", "Observed frequency", (0, 1), (0, 1),
                        xfmt=lambda v: f"{v:.1f}", yfmt=lambda v: f"{v:.1f}", diagonal=True)


# =============================================================================================
# Case study 1C — economics at Grey Mirror scale
# =============================================================================================

def cs1c(a, b):
    tpm = a.get("tokens_per_message") if a else None
    rows = []
    if tpm:
        price_in = LIST_PRICES["typesafe-ai/jev"][0] / 1e6
        for label, n in (("Signal cohort: 29 histories", GM["signal_cohort_messages"]),
                         ("Conflict cohort: 44 histories", GM["conflict_cohort_messages"]),
                         ("Operational rollup: 347 conversations", GM["operational_messages"])):
            jev_cost = n * tpm * price_in
            reqs = n / 11  # thread-day windows average ~11 messages in this benchmark
            haiku = n * tpm * LIST_PRICES["anthropic/claude-haiku-4.5"][0] / 1e6 + reqs * 80 * LIST_PRICES["anthropic/claude-haiku-4.5"][1] / 1e6
            sonnet = n * tpm * LIST_PRICES["anthropic/claude-sonnet-5"][0] / 1e6 + reqs * 80 * LIST_PRICES["anthropic/claude-sonnet-5"][1] / 1e6
            rows.append([label, f"{n:,}", money(jev_cost), money(haiku), money(sonnet)])
        tbl = C.table(["Grey Mirror corpus (published size)", "Messages", "Jev, 8 questions per day-window",
                       "Claude Haiku 4.5 (est.)", "Claude Sonnet 5 (est.)"], rows,
                      caption=f"Cost to read every message with the 1A question set ({tpm:.0f} input tokens per message measured)")
        note = ("LLM estimates assume the same input tokens plus 80 output tokens per window at public gateway list prices "
                "(retrieved 2026-09-27); they exclude the longer prompt an LLM needs to return eight structured answers.")
        body = f"{tbl}<p class='small'>{esc(note)}</p>"
    else:
        body = f"<p>{pending('Pending Jev run: needs measured tokens per message from 1A')}</p>"
    return f"""
<h3 id="cs1c">1C · What it costs to read everything</h3>
<p>Grey Mirror's headline research finding is that the signal is rare: {pct(GM['no_signal_share'], 1)} of 4.6 million messages carried no
detectable emotional signal. A rare signal means the expensive part of the job is reading everything to find the few messages that matter.
That is the workload Jev is priced for: input at $0.042 per million tokens, output free.</p>{body}"""


# =============================================================================================
# Case study 2 — voice turn-taking
# =============================================================================================

MODEL_NAMES = {"silence_only": "Silence timeout only", "lexical_lr_supervised": "Lexical model (supervised, 80k examples)",
               "jev_done": "Jev 'done?' (zero-shot)", "jev_yield": "Jev yield score (zero-shot)",
               "jev_fanout_lr": "Jev 4-question fan-out + 5-weight LR", "jev_plus_lexical_lr": "Jev + lexical stacked"}
MODEL_COLOR = {"silence_only": ENTITY["silence"], "lexical_lr_supervised": ENTITY["lexical"], "jev_done": ENTITY["jev"],
               "jev_fanout_lr": ENTITY["jev"], "jev_plus_lexical_lr": ENTITY["stack"], "jev_yield": ENTITY["jev"]}


def cs2(r):
    if r is None:
        return {"html": "<p>Results not generated yet.</p>", "md": "", "has_jev": False, "hero": None}
    has_jev = "jev_done" in r["models"]
    P = r["policies"]
    # model quality table
    mrows = []
    for k in ("lexical_lr_supervised", "jev_done", "jev_yield", "jev_fanout_lr", "jev_plus_lexical_lr"):
        m = r["models"].get(k)
        if m is None:
            mrows.append([MODEL_NAMES[k], pending(), pending(), pending(), pending()])
        else:
            ci = m.get("auc_ci")
            mrows.append([MODEL_NAMES[k], f"{m['auc']:.3f}" + (f" ({ci[0]:.3f}–{ci[1]:.3f})" if ci else ""),
                          f"{m['ece']:.3f}", f"{m['brier']:.3f}", f"{m['aupr']:.3f}"])
    mtbl = C.table(["Model", "AUC (95% CI)", "ECE", "Brier", "AUPR"], mrows,
                   caption=f"Is the caller done? {r['n_eval']:,} held-out pauses, {pct(r['eval_shift_rate'], 1)} are real turn ends")

    # operating points table
    orows = []
    for k in ("silence_only", "lexical_lr_supervised", "jev_done", "jev_fanout_lr", "jev_plus_lexical_lr"):
        pol = P.get(k)
        cells = [MODEL_NAMES[k]]
        for tgt in ("interrupt_le_5pct", "interrupt_le_10pct", "interrupt_le_20pct"):
            c = pol["chosen_on_dev"].get(tgt) if pol else None
            cells.append(pending() if pol is None else "—" if c is None else
                         Raw(f"{c['eval']['mean_latency_s'] * 1000:,.0f} ms <span class='small'>({pct(c['eval']['interruption_rate'], 1)} interrupts)</span>"))
        orows.append(cells)
    otbl = C.table(["Endpointing policy", "Ceiling 5% interruptions", "Ceiling 10%", "Ceiling 20%"], orows,
                   caption="Mean response latency at turn ends, operating point chosen on dev, reported on eval")

    # frontier chart
    ser = []
    sil = P["silence_only"]["curve_eval"]
    ser.append({"name": "Silence timeout", "color": ENTITY["silence"],
                "points": [(p["interruption_rate"] * 100, p["mean_latency_s"] * 1000, f"T={p['T']:.2f}s: {pct(p['interruption_rate'], 1)} interrupts, {p['mean_latency_s'] * 1000:.0f} ms")
                           for p in sil]})
    for k, name in (("lexical_lr_supervised", "Lexical (supervised)"), ("jev_done", "Jev (zero-shot)"),
                    ("jev_plus_lexical_lr", "Jev + lexical")):
        if k in P:
            ser.append({"name": name, "color": MODEL_COLOR[k],
                        "points": [(p["interruption_rate"] * 100, p["mean_latency_s"] * 1000,
                                    f"{pct(p['interruption_rate'], 1)} interrupts, {p['mean_latency_s'] * 1000:.0f} ms") for p in thin(P[k]["frontier_eval"])]})
    frontier = C.line_chart("The endpointing trade-off: interruptions vs response latency (lower-left is better)", ser,
                            "Interruptions (% of mid-turn pauses the agent talks over)", "Mean response latency at turn ends (ms)",
                            (0, 30), (0, 1600), xfmt=lambda v: f"{v:.0f}%", yfmt=lambda v: f"{v:,.0f}", markers=True)

    extra, findings, hero = "", [], None
    s5 = P["silence_only"]["chosen_on_dev"]["interrupt_le_5pct"]["eval"]
    s10 = P["silence_only"]["chosen_on_dev"]["interrupt_le_10pct"]["eval"]
    findings.append(f"The problem, measured: a silence-only agent needs a {P['silence_only']['chosen_on_dev']['interrupt_le_5pct']['policy']['T']:.2f} s "
                    f"timeout to keep interruptions near 5% ({pct(s5['interruption_rate'], 1)} on eval), so every turn end costs the caller "
                    f"{s5['mean_latency_s'] * 1000:,.0f} ms of dead air. Mid-turn pauses in this data have a median of "
                    f"{r['pause_stats_eval']['hold_pause_pctl_50_75_90_95'][0] * 1000:.0f} ms and a 95th percentile of "
                    f"{r['pause_stats_eval']['hold_pause_pctl_50_75_90_95'][3] * 1000:,.0f} ms.")
    lx = r["models"]["lexical_lr_supervised"]
    findings.append(f"A supervised lexical model trained on 80,000 in-domain pauses reaches AUC {lx['auc']:.3f}: the words before a pause "
                    f"carry most of the signal, which is why a text-only model is worth putting in the loop at all.")
    if has_jev:
        jd = r["models"]["jev_done"]
        lat = r["jev_latency_eval_ms"]
        findings.append(f"Jev, zero-shot with one question, reaches AUC {jd['auc']:.3f} and ECE {jd['ece']:.3f}; the four-question fan-out "
                        f"with five fitted weights reaches {r['models']['jev_fanout_lr']['auc']:.3f}, and stacking Jev with the lexical model "
                        f"{r['models']['jev_plus_lexical_lr']['auc']:.3f}.")
        dd = r["auc_diff_jev_done_minus_lexical"]
        findings.append(f"Jev 'done?' vs the supervised lexical model: AUC difference {dd['diff']:+.3f} (95% CI {dd['ci'][0]:+.3f} to {dd['ci'][1]:+.3f}).")
        findings.append(f"Measured round-trip latency from a cloud container through Vercel AI Gateway, one request at a time: median "
                        f"{lat['p50']:,.0f} ms, p90 {lat['p90']:,.0f} ms, p99 {lat['p99']:,.0f} ms. Each simulated decision uses its own "
                        f"request's measured latency.")
        best = None
        for k in ("jev_done", "jev_fanout_lr", "jev_plus_lexical_lr"):
            c = P[k]["chosen_on_dev"]["interrupt_le_5pct"]
            if c and (best is None or c["eval"]["mean_latency_s"] < best[1]["eval"]["mean_latency_s"]):
                best = (k, c)
        if best:
            k, c = best
            saved = s5["mean_latency_s"] - c["eval"]["mean_latency_s"]
            findings.append(f"At a 5% interruption ceiling (chosen on dev), {MODEL_NAMES[k]} answers turn ends in "
                            f"{c['eval']['mean_latency_s'] * 1000:,.0f} ms on average ({pct(c['eval']['interruption_rate'], 1)} interruptions on eval), "
                            f"{'saving' if saved > 0 else 'adding'} {abs(saved) * 1000:,.0f} ms per turn against the silence timeout.")
            hero = {"value": f"{saved * 1000:,.0f} ms", "label": f"less dead air per turn vs a silence timeout ({MODEL_NAMES[k]}; tuned for 5% interruptions, "
                                                                     f"{pct(c['eval']['interruption_rate'], 1)} on held-out calls vs 5.0%)"}
        rel = [("Lexical (supervised)", ENTITY["lexical"], lx["reliability"])]
        extra += reliability_chart("Calibration: predicted vs observed turn ends (eval)", jd["reliability"], "Jev 'done?'",
                                   ENTITY["jev"], rel)
    else:
        findings.append("Jev results for this experiment are pending.")
    llm = r.get("llm_comparison")
    if llm:
        ser2 = [{"name": "Silence timeout", "color": ENTITY["silence"],
                 "points": [(p["interruption_rate"] * 100, p["mean_latency_s"] * 1000, f"T={p['T']:.2f}s") for p in llm["silence_same_rows"]]},
                {"name": "Jev (measured latency)", "color": ENTITY["jev"],
                 "points": [(p["interruption_rate"] * 100, p["mean_latency_s"] * 1000, f"{p['mean_latency_s'] * 1000:.0f} ms") for p in thin(llm["frontier_jev_same_rows"])]},
                {"name": f"{llm['model']} (measured latency)", "color": ENTITY["llm"],
                 "points": [(p["interruption_rate"] * 100, p["mean_latency_s"] * 1000, f"{p['mean_latency_s'] * 1000:.0f} ms") for p in thin(llm["frontier_llm"])]}]
        extra += C.line_chart(f"System One vs System Two in the loop ({llm['n']} eval pauses, each with its own measured latency)", ser2,
                              "Interruptions (%)", "Mean response latency at turn ends (ms)", (0, 30), (0, 2000),
                              xfmt=lambda v: f"{v:.0f}%", yfmt=lambda v: f"{v:,.0f}")
        dd = llm["auc_diff_jev_minus_llm"]
        findings.append(f"{llm['model']} on the same {llm['n']} pauses: AUC {llm['llm']['auc']:.3f} vs Jev {llm['jev_done_same_rows']['auc']:.3f} "
                        f"(difference {dd['diff']:+.3f}, 95% CI {dd['ci'][0]:+.3f} to {dd['ci'][1]:+.3f}), median latency "
                        f"{llm['llm_latency_ms']['p50']:,.0f} ms vs {llm['jev_latency_ms_same_rows']['p50']:,.0f} ms.")
    ps = r["pause_stats_eval"]
    html_ = f"""
<p>Every voice agent, including the phone-call companion JustLayMe runs, faces the same decision many times a minute: the caller has gone quiet;
are they done, or thinking? Answer too early and the agent talks over them. Answer too late and every turn ends in dead air. Most production
agents still decide with a silence timeout. This is a System One task in the literal sense: a fast, intuitive judgement made inside a real-time
loop, where a calibrated probability is exactly what the controller needs and a long answer is useless. Jev has never been trained for it.</p>
<p><b>Data.</b> Switchboard: two-person telephone calls with manually corrected word-level forced alignments (Mississippi State ISIP release).
Every clean pause (no overlap, the other side silent) is labelled by what actually happened next: the same speaker resumed (a <i>hold</i>, where
responding would have been an interruption) or the other person took the floor (a <i>shift</i>). Pauses followed only by a backchannel
("uh-huh") are excluded. Text is lowercase with no punctuation, like streaming speech recognition, so no annotator punctuation can leak the
answer. The held-out evaluation sample is {r['n_eval']:,} pauses from 252 calls never used for training or tuning; mid-turn pauses have
a median of {ps['hold_pause_pctl_50_75_90_95'][0] * 1000:.0f} ms (90th percentile {ps['hold_pause_pctl_50_75_90_95'][2] * 1000:,.0f} ms).
We first tried the AMI meeting corpus and rejected it for this purpose: its word times are contiguous within transcription segments, so short
mid-turn pauses are invisible.</p>
<p><b>Method.</b> At each pause Jev receives the last ~45 seconds of transcript and four questions in one request: a Noul "has the speaker
finished and handed over the turn?", Nouls for "stopped mid-phrase" and "just asked a question", and a five-level yield Score. Requests are sent
one at a time so every latency is a clean round trip, and the simulation charges each decision its own measured latency. The agent confirms
silence at 200 ms, asks the model, and then either answers as soon as the model returns or waits for a fallback timeout. Two policy families are
searched: a gate (answer now if p ≥ θ, else wait T) and an adaptive timeout (wait max(model latency, a + b·(1−p))). Operating points are chosen on a
separate dev sample of 1,200 pauses and reported on eval. The baselines are a silence timeout and a supervised logistic regression over
lexical features, trained on 80,000 in-domain pauses.</p>
{mtbl}{frontier}{otbl}{extra}
<div class="findings"><h4>What the data says</h4><ul>{''.join(f'<li>{esc(f)}</li>' for f in findings)}</ul></div>"""
    md = "\n".join(["## Case study 2 · Voice turn-taking", "", md_table(["Model", "AUC", "ECE", "Brier", "AUPR"], mrows), "",
                    md_table(["Policy", "≤5%", "≤10%", "≤20%"], orows), "", *[f"- {f}" for f in findings], ""])
    return {"html": html_, "md": md, "has_jev": has_jev, "hero": hero}


# =============================================================================================
# Page
# =============================================================================================

def md_table(headers, rows):
    def c(x):
        s = str(x)
        if isinstance(x, Raw):
            s = s.replace('<span class="pending-tag">', "").replace("<span class='small'>", "").replace("</span>", "")
        return s.replace("|", "/")
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(c(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


CSS = """
:root{color-scheme:light;--surface-0:#f6f5f2;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#8a8984;
--grid:#e4e2dc;--rule:#d6d4cc;--accent:#2a78d6;--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--series-4:#eda100;--series-5:#e87ba4;
--warn-bg:#fff4e0;--warn-ink:#6b4400;--bad-bg:#fde8e8;--bad-ink:#8a1c1c}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--surface-0:#121211;--surface-1:#1a1a19;--text-primary:#fff;
--text-secondary:#c3c2b7;--text-muted:#8f8e86;--grid:#2c2c2a;--rule:#3a3a37;--accent:#3987e5;--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;
--series-4:#c98500;--series-5:#d55181;--warn-bg:#3a2c10;--warn-ink:#f5d9a0;--bad-bg:#3d1616;--bad-ink:#f7c1c1}}
:root[data-theme="dark"]{color-scheme:dark;--surface-0:#121211;--surface-1:#1a1a19;--text-primary:#fff;--text-secondary:#c3c2b7;--text-muted:#8f8e86;
--grid:#2c2c2a;--rule:#3a3a37;--accent:#3987e5;--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#c98500;--series-5:#d55181;
--warn-bg:#3a2c10;--warn-ink:#f5d9a0;--bad-bg:#3d1616;--bad-ink:#f7c1c1}
*{box-sizing:border-box}html,body{margin:0;background:var(--surface-0);color:var(--text-primary)}
body{font:16px/1.6 "Inter",system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:960px;margin:0 auto;padding:32px 16px 80px}
h1{font-size:2.1rem;line-height:1.15;margin:.2em 0 .1em;letter-spacing:-.01em}h2{font-size:1.5rem;margin:2.2em 0 .4em;padding-top:.6em;border-top:1px solid var(--rule)}
h3{font-size:1.2rem;margin:1.8em 0 .4em}h4{margin:1.2em 0 .3em}
p,li{color:var(--text-primary)}.lede{font-size:1.1rem;color:var(--text-secondary)}.small{font-size:.85rem;color:var(--text-secondary)}
code{font:.9em ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--surface-1);border:1px solid var(--rule);border-radius:4px;padding:0 4px}
pre{background:var(--surface-1);border:1px solid var(--rule);border-radius:8px;padding:12px;overflow-x:auto}
.banner{border-radius:8px;padding:12px 16px;margin:16px 0;background:var(--warn-bg);color:var(--warn-ink)}.banner.bad{background:var(--bad-bg);color:var(--bad-ink)}
.heroes{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:20px 0}
.hero{background:var(--surface-1);border:1px solid var(--rule);border-radius:10px;padding:14px 16px}
.hero .v{font-size:1.9rem;font-weight:700;letter-spacing:-.02em;color:var(--text-primary)}.hero .l{font-size:.88rem;color:var(--text-secondary)}
.chart{margin:18px 0;background:var(--surface-1);border:1px solid var(--rule);border-radius:10px;padding:10px 8px 4px}
.chart figcaption{font-weight:600;font-size:.95rem;padding:2px 8px 6px}.chart svg{width:100%;height:auto;display:block}
.chart text{fill:var(--text-secondary);font-size:12px;font-family:inherit}.chart .legend{fill:var(--text-primary)}
.chart .value{fill:var(--text-primary);font-size:11px}.chart .axis-label{fill:var(--text-secondary);font-size:12px}
.chart .grid{stroke:var(--grid);stroke-width:1}.chart .axis{stroke:var(--rule);stroke-width:1}
.chart .ref{stroke:var(--text-muted);stroke-dasharray:4 4;stroke-width:1}.chart .note{fill:var(--text-muted);font-size:11px}
.chart .pending{fill:none;stroke:var(--text-muted);stroke-dasharray:3 3}
.chart .pt:hover circle:first-child{r:6}
.legend-row{display:flex;flex-wrap:wrap;gap:4px 16px;padding:0 8px 4px;font-size:.85rem;color:var(--text-primary)}
.lg{display:inline-flex;align-items:center;gap:6px}.key{display:inline-block;width:18px;height:0;border-top:3px solid var(--c)}.key.dashed{border-top-style:dashed}
.table-wrap{overflow-x:auto;margin:14px 0}table{border-collapse:collapse;width:100%;font-size:.9rem;background:var(--surface-1);border-radius:8px}
caption{text-align:left;font-weight:600;padding:6px 2px;color:var(--text-primary)}th,td{padding:7px 10px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}
th{color:var(--text-secondary);font-weight:600}.pending-tag{display:inline-block;font-size:.78rem;padding:1px 8px;border-radius:999px;border:1px dashed var(--text-muted);color:var(--text-muted)}
.findings{background:var(--surface-1);border-left:3px solid var(--accent);border-radius:6px;padding:4px 16px;margin:16px 0}
nav.toc{background:var(--surface-1);border:1px solid var(--rule);border-radius:10px;padding:8px 16px;margin:16px 0}nav.toc a{color:var(--accent)}
a{color:var(--accent)}details.data summary{cursor:pointer;color:var(--text-secondary);font-size:.9rem}
@media (max-width:600px){h1{font-size:1.6rem}.hero .v{font-size:1.5rem}}
"""


def request_examples() -> str:
    """The exact request shapes each experiment sends (synthetic content only)."""
    from cs1_grey_mirror.run_turning_points import questions_for
    from cs1_grey_mirror.run_cga import QUESTIONS as CGA_Q
    from cs2_turn_taking.run_turns import questions as turn_q

    ex1 = {"model": "typesafe-ai/jev",
           "state": {"date": "Tuesday, March 3", "messages": [
               {"time": "08:12", "from": "Sam", "text": "how'd it go with Priya?"},
               {"time": "08:15", "from": "Alex", "text": "fine thanks. i have a lot going on today. what time is your appointment?"},
               {"time": "18:40", "from": "Sam", "text": "sorry, running like 20 min late 😬"},
               {"time": "18:42", "from": "Alex", "text": "oh great, love that for me 🙂"}]},
           "questions": questions_for("Sam", "Alex")}
    ex2 = {"model": "typesafe-ai/jev",
           "state": {"platform": "Wikipedia article talk page", "conversation": [
               {"speaker": "Speaker 1", "text": "I reverted the move because the sources use the full name."},
               {"speaker": "Speaker 2", "text": "Did you actually read the policy page, or just skim it?"}]},
           "questions": CGA_Q}
    ex3 = {"model": "typesafe-ai/jev",
           "state": {"setting": "Live two-person phone call, speech-recognition transcript without punctuation",
                     "transcript": [{"speaker": "Speaker B", "text": "so what did you end up doing about the car"},
                                    {"speaker": "Speaker A", "text": "well we took it back to the dealer and they said it was the uh"}],
                     "just_went_quiet": "Speaker A"},
           "questions": turn_q("Speaker A")}
    blocks = []
    for title, ex in (("1A · one thread-day, eight questions", ex1), ("1B · one conversation prefix, three questions", ex2),
                      ("2 · one pause, four questions", ex3)):
        blocks.append(f"<details class='data'><summary>{esc(title)}</summary><pre><code>"
                      f"{esc(json.dumps(ex, indent=2, ensure_ascii=False))}</code></pre></details>")
    return "".join(blocks)


LESSONS = [
    ("Arithmetic stays in code.", "Jev never counts messages, compares dates or computes a trend. It answers one semantic question per "
     "day-window; the change-point statistics, thresholds and policy timing are ordinary code (TypeSafe's jaggedness guide, items 2–3)."),
    ("Small states.", "One day of messages, one conversation prefix, or the last 45 seconds of a call per request, never a whole history, "
     "to avoid the context-rot failure mode the docs warn about."),
    ("Literal questions with criteria.", "Every Noul spells out what yes and no mean (for example, 'polite but closed replies' counts "
     "as distant) because Jev answers the question as written."),
    ("Speculative fan-out.", "All questions about one state go in one request: eight in 1A, three in 1B, four in 2. They share one "
     "latency and one state ingestion."),
    ("Calibration used as a control signal.", "Probabilities feed thresholds and timeout policies directly; the only fitted "
     "parameters are thresholds or a five-weight logistic layer, always fit on a dev split and reported on held-out data."),
    ("Pre-registration.", "Detector settings, composites and question wording were fixed before any Jev output was seen, and "
     "baselines run through the identical detector."),
]


def thin(points, key="interruption_rate", step=0.005):
    """One frontier point per `step` of the x value, for readable markers."""
    out, seen = [], set()
    for p in points:
        b = round(p[key] / step)
        if b not in seen:
            seen.add(b)
            out.append(p)
    return out


def rolling(vals, k=7):
    out = []
    for i in range(len(vals)):
        w = [v for v in vals[max(0, i - k + 1): i + 1] if v is not None]
        out.append(sum(w) / len(w) if w else None)
    return out


def build():
    a_raw, b_raw, c_raw = load("cs1a_turning_points.json"), load("cs1b_cga.json"), load("cs2_turn_taking.json")
    mock = any(x and x.get("mock") for x in (a_raw, b_raw, c_raw))
    A, B, C2 = cs1a(a_raw), cs1b(b_raw), cs2(c_raw)
    econ = cs1c(A, B)
    all_jev = A["has_jev"] and B["has_jev"] and C2["has_jev"]

    banner = ""
    if mock:
        banner = ('<div class="banner bad"><b>Pipeline test: these numbers are NOT Jev results.</b> This page was built from '
                  'JEV_MOCK=1 output (random answers with the documented response shapes). Do not share.</div>')
    elif not all_jev:
        banner = ('<div class="banner"><b>Status: Jev runs pending.</b> Harness, datasets and every baseline are complete and shown below. '
                  'Jev figures fill in automatically when <code>./run_all.sh</code> runs with an <code>AI_GATEWAY_API_KEY</code> '
                  '(about $2–5 at list price for all three experiments).</div>')

    heroes = [h for h in (A["hero"], B["hero"], C2["hero"]) if h]
    tpm = A.get("tokens_per_message")
    if tpm:
        heroes.append({"value": money(GM["signal_cohort_messages"] * tpm * 0.042 / 1e6),
                       "label": "to read all 4.6M messages of Grey Mirror's signal cohort with eight typed questions per day"})
    hero_html = "".join(f'<div class="hero"><div class="v">{esc(h["value"])}</div><div class="l">{esc(h["label"])}</div></div>'
                        for h in heroes) or '<div class="hero"><div class="v">…</div><div class="l">Headline numbers appear after the Jev run.</div></div>'

    total_cost = sum(((x or {}).get("jev_usage") or {}).get("cost_usd", 0) for x in (a_raw,) if x)
    total_cost += sum((o.get("jev_usage") or {}).get("cost_usd", 0) for o in ((b_raw or {}).get("corpora") or {}).values())
    total_cost += sum(v.get("cost_usd", 0) for v in (((c_raw or {}).get("jev_usage")) or {}).values())

    html_ = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jev Case Studies</title><style>{CSS}</style></head><body><main>
<p class="small">Wentropy Labs · Grey Mirror by JustLayMe · {date.today():%B %-d, %Y}</p>
<h1>Same model, opposite ends of the clock</h1>
<p class="lede">Two case studies of TypeSafe's Jev. One reads months of relationship history offline to find the day something changed. The
other decides, inside a live phone call, whether a caller has finished speaking. Neither required a training example: only the typed questions
changed.</p>
{banner}
<div class="heroes">{hero_html}</div>
<nav class="toc"><b>Contents</b><ol>
<li><a href="#cs1">Case study 1 · Grey Mirror × Jev: relationship intelligence at System One speed</a>
 (<a href="#cs1a">turning points</a>, <a href="#cs1b">derailment forecasting</a>, <a href="#cs1c">economics</a>)</li>
<li><a href="#cs2">Case study 2 · Jev in the loop: when has the caller finished speaking?</a></li>
<li><a href="#method">Reproducibility, data and limits</a></li></ol></nav>

<h2 id="cs1">Case study 1 · Grey Mirror × Jev: relationship intelligence at System One speed</h2>
<p>Grey Mirror analyses complete exported message histories (iMessage, WhatsApp, Instagram and more) and measures initiation, reciprocity,
timing, repair, escalation and affection across the whole timeline. Its published research makes the design constraints unusually explicit:
signal is rare, claims must be withheld when the data does not support them, and every finding has to stand out from a thread's own rhythm.
Jev's properties line up with each one. It is cheap enough to read every message, its answers are typed so they drop straight into a
detector, and its probabilities are calibrated so the product can decide when to stay silent. This case study tests all three claims on data
anyone can reproduce. No customer conversation was used or sent to any model.</p>
{A['html']}{B['html']}{econ}

<h2 id="cs2">Case study 2 · Jev in the loop: when has the caller finished speaking?</h2>
{C2['html']}

<h2 id="method">Reproducibility, data and limits</h2>
<h4>How we built with Jev</h4>
<ul>{''.join(f"<li><b>{esc(a)}</b> {esc(b)}</li>" for a, b in LESSONS)}</ul>
<h4>The requests</h4>
{request_examples()}
<h4>Run it</h4>
<pre><code>cd jev-case-studies
export AI_GATEWAY_API_KEY=...        # Vercel AI Gateway key; Jev is typesafe-ai/jev
./run_all.sh                         # downloads data, runs all experiments, rebuilds this page</code></pre>
<p>Every Jev and LLM response is cached on disk keyed by a hash of the exact request, so re-running never re-spends and every number here can be
replayed. Jev is called through Vercel AI Gateway's TypeSafe-compatible endpoint (<code>POST https://ai-gateway.vercel.sh/typesafe/v1/systemone</code>,
model <code>typesafe-ai/jev</code>); setting <code>JEV_PROVIDER=typesafe</code> calls TypeSafe directly instead. Jev spend recorded for this build:
{money(total_cost)}.</p>
<h4>Data</h4>
<ul>
<li><b>Synthetic relationship threads</b> (1A): generated by <code>cs1_grey_mirror/generate_threads.py</code> from hand-written message banks, seeded
and fully reproducible; exported in Grey Mirror's upload formats.</li>
<li><b>Conversations Gone Awry</b> (1B): Zhang et al. 2018 and Chang &amp; Danescu-Niculescu-Mizil 2019, distributed with Cornell ConvoKit.
Published CRAFT numbers are copied from that paper's Table 1.</li>
<li><b>Switchboard</b> (2): Godfrey et al. 1992; word alignments and transcripts from the Mississippi State ISIP "MS98" release.
Raw transcripts are not redistributed in this repository.</li>
<li><b>Grey Mirror figures</b>: the public research package at justlay.me/research/the-signal-is-rare (aggregate data only).</li>
</ul>
<h4>Limits</h4>
<ul>
<li>1A uses synthetic threads with planted ground truth. It isolates one capability (reading meaning that surface statistics miss) and says
nothing directly about accuracy on real relationships. Real histories are also far sparser in signal than these threads.</li>
<li>1B's datasets are balanced by construction (half the conversations derail), so absolute precision and calibration differ from the wild,
where attacks are rare.</li>
<li>2 is text-only. Production endpointing also uses prosody and acoustics, which Jev cannot see; the fair reading is that Jev supplies the
semantic half of the decision at a latency the loop can afford. Latency was measured from one cloud region through a gateway and will vary by
deployment.</li>
<li>Each experiment is a single run of a deterministic pipeline; confidence intervals come from bootstrap resampling of the evaluation items, not
repeated model sampling.</li>
</ul>
<p class="small">Built by <code>report/build_reports.py</code> from <code>results/*.json</code>. Jev model version as reported by the API:
{esc(str(((a_raw or {}).get('jev_usage') or {}).get('model') or 'pending'))}.</p>
</main></body></html>"""
    out = REPORTS / "jev-case-studies.html"
    out.write_text(html_)
    md = "\n".join([f"# Jev case studies{' (PIPELINE TEST, NOT REAL RESULTS)' if mock else ''}", "",
                    "Rendered report: `reports/jev-case-studies.html`.", "",
                    "## Case study 1 · Grey Mirror × Jev", "", A["md"], B["md"], C2["md"]])
    (REPORTS / "jev-case-studies.md").write_text(md)
    print(f"wrote {out}  (mock={mock}, jev complete={all_jev})")
    return out


if __name__ == "__main__":
    build()
