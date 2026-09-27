# Jev case studies

Rendered report: `reports/jev-case-studies.html`.

## Case study 1 · Grey Mirror × Jev

### 1A · Turning points

| Signal layer | FP (40 steady) | Overt det. | Overt dir. | Overt median err. | Subtle det. | Subtle dir. |
|---|---|---|---|---|---|---|
| Sentiment lexicon (VADER) | 0/40 | 40/40 | 100% | 0 d | 0/40 | — |
| Behavioral metrics | 0/40 | 38/40 | 100% | 0 d | 0/40 | — |
| Lexicon + behavioral | 0/40 | 40/40 | 100% | 0 d | 1/40 | 0% |
| Jev (8-question composite) | pending | pending | pending | pending | pending | pending |

- A plain sentiment lexicon already matches Grey Mirror's published benchmark on overt changes: 40/40 detected, 100% correct direction, 0/40 false positives. The published benchmark does not separate a smart signal layer from a dumb one; the subtle arm does.
- On meaning-only changes the lexicon finds 0/40 and behavioral metrics 0/40 (direction correct in — of those): the surface statistics barely move, by construction.
- Jev results for this experiment are pending (see the run instructions).

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
| Jev: 'next comment is an attack?' (zero-shot) | pending | pending | pending | pending | pending | pending | pending |
| Jev: 'heading for a breakdown?' (zero-shot) | pending | pending | pending | pending | pending | pending | pending |
| Jev: tension score (zero-shot) | pending | pending | pending | pending | pending | pending | pending |

- Jev results for this corpus are pending.

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
| Jev: 'next comment is an attack?' (zero-shot) | pending | pending | pending | pending | pending | pending | pending |
| Jev: 'heading for a breakdown?' (zero-shot) | pending | pending | pending | pending | pending | pending | pending |
| Jev: tension score (zero-shot) | pending | pending | pending | pending | pending | pending | pending |

- Jev results for this corpus are pending.

## Case study 2 · Voice turn-taking

| Model | AUC | ECE | Brier | AUPR |
|---|---|---|---|---|
| Lexical model (supervised, 80k examples) | 0.832 (0.814–0.848) | 0.018 | 0.142 | 0.689 |
| Jev 'done?' (zero-shot) | pending | pending | pending | pending |
| Jev yield score (zero-shot) | pending | pending | pending | pending |
| Jev 4-question fan-out + 5-weight LR | pending | pending | pending | pending |
| Jev + lexical stacked | pending | pending | pending | pending |

| Policy | ≤5% | ≤10% | ≤20% |
|---|---|---|---|
| Silence timeout only | 1,200 ms (5.0% interrupts) | 1,000 ms (9.7% interrupts) | 800 ms (17.9% interrupts) |
| Lexical model (supervised, 80k examples) | 993 ms (3.9% interrupts) | 795 ms (9.5% interrupts) | 522 ms (19.3% interrupts) |
| Jev 'done?' (zero-shot) | pending | pending | pending |
| Jev 4-question fan-out + 5-weight LR | pending | pending | pending |
| Jev + lexical stacked | pending | pending | pending |

- The problem, measured: a silence-only agent needs a 1.20 s timeout to keep interruptions near 5% (5.0% on eval), so every turn end costs the caller 1,200 ms of dead air. Mid-turn pauses in this data have a median of 456 ms and a 95th percentile of 1,196 ms.
- A supervised lexical model trained on 80,000 in-domain pauses reaches AUC 0.832: the words before a pause carry most of the signal, which is why a text-only model is worth putting in the loop at all.
- Jev results for this experiment are pending.
