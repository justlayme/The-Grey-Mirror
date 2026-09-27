# Jev case studies

Two case studies of [TypeSafe's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), a System One
model that returns typed, calibrated answers instead of text, called through
[Vercel AI Gateway](https://vercel.com/docs/ai-gateway/sdks-and-apis/typesafe).

| | Case study 1 · Grey Mirror × Jev | Case study 2 · Jev in the loop |
|---|---|---|
| Question | Can Jev be the reading layer of a relationship analyzer? | Can Jev decide when a caller has finished speaking? |
| Clock | Offline: 90-day histories, millions of messages | Real time: one decision inside a ~200 ms pause |
| Data | Synthetic threads modelled on Grey Mirror's benchmark; Conversations Gone Awry (Wikipedia, Reddit) | Switchboard telephone calls with forced-aligned pauses |
| Jev primitives | 8-question fan-out per day (2 Scores, 6 Nouls); 3 per comment | 4-question fan-out per pause (3 Nouls, 1 Score) |
| Baselines | Sentiment lexicon, behavioral metrics, TF-IDF LR, published CRAFT, Claude Haiku 4.5 | Silence timeout, supervised lexical LR (80k examples), Claude Haiku 4.5 |

The rendered report is `reports/jev-case-studies.html` (Markdown summary: `reports/jev-case-studies.md`).

## Status

Harness, datasets and every non-Jev baseline are built and their results are in `results/`. **Jev and LLM
numbers need one run with an `AI_GATEWAY_API_KEY`**; until then the report shows those cells as *pending*.

## Run

```bash
export AI_GATEWAY_API_KEY=...     # Vercel AI Gateway key
./run_all.sh                      # ~60-80 min, a few dollars at list price
```

`run_all.sh` fetches the public data, rebuilds every prepared file deterministically, runs the three
experiments and rebuilds the report. Every API response is cached in `data/cache/` keyed by the exact
request, so re-running is free and reproduces the same numbers.

* `LLM_COMPARATOR=...` picks the System-Two comparator (default `anthropic/claude-haiku-4.5`, `""` to skip).
* `JEV_PROVIDER=typesafe` calls `api.typesafe.ai` directly with `TYPESAFE_API_KEY`.
* `JEV_MOCK=1 ./run_all.sh` tests the whole pipeline without an API key. It uses fake answers with the
  documented response shapes and writes only to `results/mock/` and `reports/mock/`. The mock report
  carries a "not real results" banner.

## Layout

```
jevlab/                 shared harness
  client.py             async Jev + LLM clients (gateway), disk cache, retries, RPM limiter, latency/cost logging
  metrics.py            AUC, ECE, Brier, reliability, selective accuracy, bootstrap CIs
  changepoint.py        pre-registered change-point detector (circular block permutation null)
  svgcharts.py          dependency-free SVG charts for the report
cs1_grey_mirror/
  banks.py              message banks (warm / subtle-cold / overt-cold registers)
  generate_threads.py   120 synthetic threads (steady / overt / subtle), plus Grey Mirror upload exports
  run_turning_points.py 1A: Jev per day-window vs lexicon/behavioral, same detector
  prepare_cga.py        Conversations Gone Awry, CRAFT protocol
  run_cga.py            1B: zero-shot derailment forecasting vs published CRAFT, TF-IDF, LLM
cs2_turn_taking/
  prepare_swb.py        hold/shift pause events from Switchboard alignments
  run_turns.py          Jev at every pause (sequential, clean latency), lexical baseline, LLM
  simulate.py           latency-aware endpointing policy simulation
report/build_reports.py builds reports/ from results/
```

## Data and privacy

No Grey Mirror customer upload was used or sent to any model. Case study 1 uses synthetic threads plus
Grey Mirror's published aggregate research figures. The 120 synthetic threads are also exported in Grey
Mirror's upload formats (`data/prepared/gm_uploads/`) so Grey Mirror's own detector can be run on the
same threads for a head-to-head comparison.

Third-party corpora are downloaded at run time and are not committed: Conversations Gone Awry (Cornell
ConvoKit) and Switchboard (Mississippi State ISIP MS98 alignments).
