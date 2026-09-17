# M9 — State, Fill Hazard, and Post-Fill Markout

## Status

**Research Complete — Date1**

Dataset:
- Nasdaq TotalView ITCH 5.0
- 2019-07-30
- Five symbols: AAPL, AMZN, GOOG, INTC, MSFT

Important limitation:
- Results are single-date and exploratory.
- Independent-date validation is deferred because no second raw ITCH date is currently available.

---

## Research objective

M9 investigates whether:

> current market state + state transitions + recent order flow

provide information about:

1. quote fill probability,
2. post-fill price movement,
3. joint quote value,
4. basic exposure control.

The objective is measurement and reproducibility, not forcing a profitable signal.

---

## M9.4 — Fill Hazard

The final fill-hazard reconstruction processed:

- 282,229,684 ITCH messages
- 134,832 execution observations
- 269,664 hypothetical quote candidates

Candidate outcomes:

- FILL: 13,805
- ADVERSE: 121,203
- TIMEOUT: 134,656

A chronological state/queue model produced useful fill discrimination.

AUC:

- State + queue: ~0.7376
- + state transitions: ~0.7378
- + recent flow: ~0.7435

Interpretation:

Recent order flow provided measurable incremental information about fill probability.

For adverse-selection hazard, adding transitions and recent flow did not provide meaningful improvement over the state/queue specification.

---

## M9.5 — Post-Fill Markout

13,805 completed fills were evaluated at:

- 10ms
- 100ms
- 500ms
- 1s

Overall mean signed markout:

| Horizon | Mean |
|---|---:|
| 10ms | -0.0257 bps |
| 100ms | -0.0959 bps |
| 500ms | -0.0837 bps |
| 1s | -0.0900 bps |

Aggregate bootstrap intervals included zero.

There was a clear side asymmetry in this sample:

- BUY fills had negative signed markout.
- SELL fills had positive signed markout.

A chronological predictive model showed that pre-fill state had modest out-of-sample predictive value for post-fill markout.

R²:

| Horizon | State | State + transitions |
|---|---:|---:|
| 10ms | 0.05698 | 0.05832 |
| 100ms | 0.07560 | 0.07613 |
| 500ms | 0.02983 | 0.03020 |
| 1s | 0.04569 | 0.04632 |

Interpretation:

Post-fill markout contains predictable structure, but state-transition variables add only a small incremental contribution.

---

## M9.6 — Joint Quote Value

A common chronological 70/30 candidate-level split was frozen across the evaluation.

Fill probability model:

- Test AUC: 0.726823

The state-aware joint quote-value model was compared with a side-specific baseline.

Final common-split results:

| Horizon | Baseline EV | State-aware EV | Gap | 95% CI |
|---|---:|---:|---:|---:|
| 10ms | -0.01294 | -0.02445 | -0.01151 | [-0.02003, -0.00299] |
| 100ms | -0.01460 | -0.02602 | -0.01142 | [-0.02066, -0.00322] |
| 500ms | -0.01527 | -0.02376 | -0.00849 | [-0.01530, -0.00261] |
| 1s | -0.01672 | -0.02435 | -0.00763 | [-0.01415, -0.00207] |

The state-aware model was worse than the baseline at all four horizons, and the block-bootstrap confidence intervals for the EV gap excluded zero.

Interpretation:

> Predictive information about fill probability does not automatically translate into improved joint quote economic value.

The current state-aware formulation therefore does not demonstrate an economic improvement over the baseline on this date.

This result is recorded rather than optimized away.

---

## M9.7 — Exposure Control

A lightweight inventory-cap diagnostic was performed using the existing candidate/fill dataset.

The experiment confirmed deterministic enforcement of hard inventory limits.

This is treated only as an exposure-control diagnostic, not as a P&L backtest or trading-strategy result.

---

## Convention Audit

The ITCH reconstruction underwent a side-convention audit.

The canonical passive-side convention is:

- Resting SELL -> passive BUY
- Resting BUY -> passive SELL

The ITCH message semantics were also corrected so that:

- `E` / `C` are execution messages
- `X` is an order cancellation

The corrected convention is used by the final M8/M9 pipeline.

Therefore the M9.5 BUY/SELL asymmetry is not attributable to the previously identified M8 side-convention error.

---

## Validation status

M9 currently has only one empirical date.

Independent-date validation is therefore **deferred**, not claimed.

The current findings should be interpreted as:

> single-date exploratory evidence from the 2019-07-30 Nasdaq ITCH sample.

A second independent date should be acquired before making claims about generalization across dates or market regimes.

---

## Final M9 conclusion

M9 establishes a coherent measurement chain:

```text
Market state
    ↓
Fill probability
    ↓
Actual hypothetical fill
    ↓
Post-fill markout
    ↓
Joint quote-value assessment