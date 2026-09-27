# Jev case studies

Rendered report: `reports/jev-case-studies.html`.

## Case study 1 · Grey Mirror × Jev

### 1A · Turning points

| Signal layer | FP (40 steady) | Overt det. | Overt dir. | Overt median err. | Subtle det. | Subtle dir. |
|---|---|---|---|---|---|---|
| Sentiment lexicon (VADER) | 0/40 | 40/40 | 100% | 0 d | 0/40 | — |
| Behavioral metrics | 0/40 | 38/40 | 100% | 0 d | 0/40 | — |
| Lexicon + behavioral | 0/40 | 40/40 | 100% | 0 d | 1/40 | 0% |
| Jev (8-question composite) | 0/40 | 40/40 | 100% | 0 d | 40/40 | 100% |
| Jev (warmth score only) | 0/40 | 40/40 | 100% | 0 d | 40/40 | 100% |

- A plain sentiment lexicon already matches Grey Mirror's published benchmark on overt changes: 40/40 detected, 100% correct direction, 0/40 false positives. The published benchmark does not separate a smart signal layer from a dumb one; the subtle arm does.
- On meaning-only changes the lexicon finds 0/40 and behavioral metrics 0/40 (direction correct in — of those): the surface statistics barely move, by construction.
- Jev, reading one day at a time through eight typed questions, flags 40/40 subtle changes (100% correct direction, median localisation error 0.0 days) and 40/40 overt ones, with 0/40 false positives in threads that contain repaired fights and busy weeks.
- Ablation: the single warmth score alone flags 40/40 subtle changes (0/40 false positives), so the fan-out questions do not add detection power.

### 1B · Conversations Gone Awry

#### Wikipedia talk pages

| Model | Acc | P | R | FPR | F1 | AUC | Pairwise |
|---|---|---|---|---|---|---|---|
| BoW (published, supervised) | 56.5 | 55.6 | 65.5 | 52.4 | 60.1 | — | — |
| Awry (published, supervised) | 58.9 | 59.2 | 57.6 | 39.8 | 58.4 | — | — |
| Cumulative BoW (published, supervised) | 60.6 | 57.7 | 79.3 | 58.1 | 66.8 | — | — |
| Sliding Awry (published, supervised) | 60.6 | 60.2 | 62.4 | 41.2 | 61.3 | — | — |
| CRAFT - CE (published, supervised) | 64.9 | 64.4 | 66.7 | 36.9 | 65.5 | — | — |
| CRAFT (published, supervised) | 66.5 | 63.7 | 77.1 | 44.1 | 69.8 | — | — |
| TF-IDF + LR (our re-run, supervised) | 58.7 | 59.4 | 54.8 | 37.4 | 57.0 | 0.621 | 62.4% |
| Jev: 'next comment is an attack?' (zero-shot) | 67.0 | 64.1 | 77.4 | 43.3 | 70.1 | 0.732 | 74.2% |
| Jev: 'heading for a breakdown?' (zero-shot) | 67.6 | 65.7 | 73.8 | 38.6 | 69.5 | 0.747 | 76.0% |
| Jev: tension score (zero-shot) | 67.1 | 65.1 | 73.8 | 39.5 | 69.2 | 0.726 | 72.7% |

- Zero-shot, with one threshold tuned on the validation split, Jev reaches 67.0% accuracy (95% CI 63.8–70.2) and F1 70.1, against 66.5% / 69.8 for CRAFT, a model pre-trained on this platform's conversations and fine-tuned on 2,508 labelled ones (above CRAFT by 0.5 points).
- Threshold-free: AUC 0.732 (95% CI 0.696–0.764); given a matched pair from the same page, Jev ranks the conversation that derails higher 74.2% of the time.
- Early warning: on correctly flagged derailments Jev first fires 3.8 comments before the attack on average (CRAFT: on average 3 comments before the attack).
- Against our supervised TF-IDF model on the same test conversations, Jev's AUC is +0.110 (95% CI +0.072 to +0.147).
- Calibration across all 4,365 individual forecasts: ECE 0.144, Brier 0.106, base rate 9.6%.

#### Reddit r/ChangeMyView

| Model | Acc | P | R | FPR | F1 | AUC | Pairwise |
|---|---|---|---|---|---|---|---|
| BoW (published, supervised) | 52.1 | 51.8 | 61.3 | 57.0 | 56.1 | — | — |
| Awry (published, supervised) | 54.4 | 55.0 | 48.3 | 39.5 | 51.4 | — | — |
| Cumulative BoW (published, supervised) | 59.9 | 58.8 | 65.9 | 46.2 | 62.1 | — | — |
| Sliding Awry (published, supervised) | 56.8 | 56.6 | 58.2 | 44.6 | 57.4 | — | — |
| CRAFT - CE (published, supervised) | 57.7 | 56.1 | 71.2 | 55.7 | 62.8 | — | — |
| CRAFT (published, supervised) | 63.4 | 60.4 | 77.5 | 50.7 | 67.9 | — | — |
| TF-IDF + LR (our re-run, supervised) | 55.8 | 55.2 | 61.1 | 49.6 | 58.0 | 0.575 | 57.9% |
| Jev: 'next comment is an attack?' (zero-shot) | 65.6 | 64.3 | 70.2 | 38.9 | 67.1 | 0.712 | 72.2% |
| Jev: 'heading for a breakdown?' (zero-shot) | 65.6 | 64.9 | 67.7 | 36.5 | 66.3 | 0.711 | 73.5% |
| Jev: tension score (zero-shot) | 66.0 | 65.0 | 69.4 | 37.4 | 67.1 | 0.719 | 73.3% |

- Zero-shot, with one threshold tuned on the validation split, Jev reaches 65.6% accuracy (95% CI 63.1–68.1) and F1 67.1, against 63.4% / 67.9 for CRAFT, a model pre-trained on this platform's conversations and fine-tuned on 4,106 labelled ones (above CRAFT by 2.2 points).
- Threshold-free: AUC 0.712 (95% CI 0.687–0.739); given a matched pair from the same page, Jev ranks the conversation that derails higher 72.2% of the time.
- Early warning: on correctly flagged derailments Jev first fires 3.7 comments before the attack on average (CRAFT: on average 4 comments).
- Against our supervised TF-IDF model on the same test conversations, Jev's AUC is +0.138 (95% CI +0.103 to +0.173).
- Calibration across all 7,098 individual forecasts: ECE 0.166, Brier 0.112, base rate 9.6%.

## Case study 2 · Voice turn-taking

| Model | AUC | ECE | Brier | AUPR |
|---|---|---|---|---|
| Lexical model (supervised, 80k examples) | 0.832 (0.814–0.848) | 0.018 | 0.142 | 0.689 |
| Jev 'done?' (zero-shot) | 0.848 (0.832–0.864) | 0.065 | 0.144 | 0.648 |
| Jev yield score (zero-shot) | 0.864 (0.848–0.878) | 0.076 | 0.138 | 0.686 |
| Jev 4-question fan-out + 5-weight LR | 0.864 (0.848–0.879) | 0.038 | 0.134 | 0.692 |
| Jev + lexical stacked | 0.875 (0.860–0.889) | 0.037 | 0.127 | 0.718 |

| Policy | ≤5% | ≤10% | ≤20% |
|---|---|---|---|
| Silence timeout only | 1,200 ms (5.0% interrupts) | 1,000 ms (9.7% interrupts) | 800 ms (17.9% interrupts) |
| Lexical model (supervised, 80k examples) | 993 ms (3.9% interrupts) | 795 ms (9.5% interrupts) | 522 ms (19.3% interrupts) |
| Jev 'done?' (zero-shot) | 952 ms (5.7% interrupts) | 675 ms (13.4% interrupts) | 518 ms (25.4% interrupts) |
| Jev 4-question fan-out + 5-weight LR | 925 ms (6.6% interrupts) | 671 ms (12.3% interrupts) | 524 ms (23.1% interrupts) |
| Jev + lexical stacked | 867 ms (6.5% interrupts) | 616 ms (13.9% interrupts) | 514 ms (23.8% interrupts) |

- The problem, measured: a silence-only agent needs a 1.20 s timeout to keep interruptions near 5% (5.0% on eval), so every turn end costs the caller 1,200 ms of dead air. Mid-turn pauses in this data have a median of 456 ms and a 95th percentile of 1,196 ms.
- A supervised lexical model trained on 80,000 in-domain pauses reaches AUC 0.832: the words before a pause carry most of the signal, which is why a text-only model is worth putting in the loop at all.
- Jev, zero-shot with one question, reaches AUC 0.848 and ECE 0.065; the four-question fan-out with five fitted weights reaches 0.864, and stacking Jev with the lexical model 0.875.
- Jev 'done?' vs the supervised lexical model: AUC difference +0.016 (95% CI -0.000 to +0.033).
- Measured round-trip latency from a cloud container through Vercel AI Gateway, one request at a time: median 258 ms, p90 314 ms, p99 430 ms. Each simulated decision uses its own request's measured latency.
- At a 5% interruption ceiling (chosen on dev), Jev + lexical stacked answers turn ends in 867 ms on average (6.5% interruptions on eval), saving 333 ms per turn against the silence timeout.
