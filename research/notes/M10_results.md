# M10 — Fill Probability and Queue Hazard

## Status

**Research Complete — Date1**

Dataset:
- Nasdaq TotalView ITCH 5.0
- 2019-07-30
- AAPL, AMZN, GOOG, INTC, MSFT

Limitation:
- Single-date exploratory evidence.
- Independent-date validation remains deferred.

---

## Research objective

M10 models the discrete-time hazard of a hypothetical quote filling after placement.

The objective is:

> estimate interval-specific fill risk conditional on the candidate surviving to that interval.

The analysis uses the existing M9 candidate dataset and does not require another ITCH replay.

---

## Data integrity

The corrected M9 state-transition rebuild was completed:

- 282,229,684 ITCH messages
- 134,832 execution observations
- 134,832 completed state-transition rows
- 0 duplicate full rows
- 0 timestamp-order violations
- 0 unmatched hazard/state candidates
- corrected execution-flow semantics

Flow definitions:

- `executions_*`: execution events only
- `buy_flow_*`: execution events with BUY side only
- `sell_flow_*`: execution events with SELL side only
- `X` cancellation events remain cancellations and do not contribute to execution flow

One execution observation maps to two hypothetical quote candidates; both inherit the same pre-event state and flow features.

---

## Frozen methodology

Common candidate-level chronological 70/30 split was retained from M9.

Person-period hazard intervals:

- 0–10ms
- 10–100ms
- 100–500ms
- 500–1000ms

Queue position is measured at quote placement only.

Two models were compared:

### Model 1
State + queue

Features:
- imbalance
- relative spread
- log-transformed queue ahead

### Model 2
State + queue + recent execution flow

Additional features:
- events_10ms
- executions_10ms
- buy_flow_10ms
- sell_flow_10ms

No new feature search was performed.

---

## Test results

### Raw cumulative fill discrimination

| Horizon | State + Queue AUC | State + Queue + Flow AUC |
|---|---:|---:|
| 0–10ms | 0.780626 | 0.809704 |
| 10–100ms | 0.740131 | 0.771379 |
| 100–500ms | 0.684307 | 0.712147 |
| 500–1000ms | 0.655476 | 0.677919 |

The flow model improves AUC at every horizon.

### Calibrated cumulative discrimination

| Horizon | State + Queue AUC | State + Queue + Flow AUC |
|---|---:|---:|
| 0–10ms | 0.780621 | 0.809699 |
| 10–100ms | 0.757563 | 0.785184 |
| 100–500ms | 0.709921 | 0.734583 |
| 500–1000ms | 0.675068 | 0.695257 |

### Incremental flow effect

Block bootstrap of Model 2 minus Model 1:

| Horizon | AUC Δ | 95% CI |
|---|---:|---:|
| 0–10ms | +0.029078 | [+0.015003, +0.047809] |
| 10–100ms | +0.027621 | [+0.015231, +0.045544] |
| 100–500ms | +0.024662 | [+0.011044, +0.046055] |
| 500–1000ms | +0.020189 | [+0.005804, +0.034403] |

All intervals exclude zero.

### Brier difference

Model 2 minus Model 1:

| Horizon | Δ Brier | 95% CI |
|---|---:|---:|
| 0–10ms | +0.000011 | [−0.000134, +0.000145] |
| 10–100ms | +0.000019 | [−0.000127, +0.000176] |
| 100–500ms | +0.000004 | [−0.000160, +0.000200] |
| 500–1000ms | +0.000016 | [−0.000167, +0.000203] |

Because lower Brier is better, the flow model does not demonstrate an improvement in probability accuracy.

---

## Calibration

Held-out cumulative probabilities remained below observed cumulative fill rates.

For the calibrated models:

| Horizon | State + Queue predicted | Flow predicted | Observed |
|---|---:|---:|---:|
| 0–10ms | 1.448% | 1.801% | 3.011% |
| 10–100ms | 1.693% | 2.107% | 3.843% |
| 100–500ms | 2.002% | 2.452% | 5.446% |
| 500–1000ms | 2.346% | 2.810% | 7.028% |

The earlier train/test base-rate diagnostic showed substantial chronological and symbol-level variation in fill rates.

This suggests that the calibration problem cannot be treated as a simple universal offset.

No additional Date1 calibration tuning is performed.

Formal calibration slope/intercept were not retained in this run; cumulative predicted-vs-observed rates and Brier scores are retained as the calibration evidence.

---

## Interpretation

M10 establishes two distinct findings:

1. Recent execution flow contains significant incremental information for **ranking fill hazard**.
2. That additional discrimination does **not** translate into a demonstrated improvement in **probability accuracy/calibration**.

Therefore:

> flow is retained as a useful discriminative variable for the M10 hazard representation, but not as evidence of improved calibrated fill probabilities.

The persistent under-calibration is treated as a limitation associated with the chronological/symbol regime structure of the single-date sample.

---

## Stopping rule

No further Date1 feature tuning or calibration optimization is performed.

M10 is considered complete for this research date.

Independent-date validation remains deferred.