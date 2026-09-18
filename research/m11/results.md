# M11 — Conditional Adverse Selection After Fill

## Status

**Research Complete — Date1**

Dataset:
- Nasdaq TotalView ITCH 5.0
- 2019-07-30
- AAPL, AMZN, GOOG, INTC, MSFT

Limitation:
- Single-date exploratory evidence.
- Independent-date validation remains deferred.
- The M9.5 artifact records complete fills only, so M11 evaluates conditional markout after complete fill rather than a full partial-fill specification.
- 13,500 fills had an exact fill-to-state linkage; 305 used a prior-state fallback because the existing artifacts do not uniquely identify the triggering execution in every case.

## Research objective

M11 tests whether information available before and through a quote fill explains subsequent adverse selection.

The primary estimand is:

> Conditional expected signed midpoint markout at 1 second given fill.

Sign convention:

- Positive = adverse to the passive quoter.
- Negative = favorable to the passive quoter.

## Frozen methodology

The common candidate-level chronological 70/30 split from M9/M10 was retained.

Only candidates with complete `FILL` outcomes and a valid fill timestamp were used.

Predictors:

- pre-placement imbalance
- relative spread
- queue ahead
- execution flow through the fill interval
- quote side
- time to fill
- fill fraction
- side × imbalance interaction

The existing M9 state-transition observations were linked to fills at the same symbol and execution timestamp where possible. When an exact match was unavailable, the latest prior state observation was used.

No M10 predicted hazard, residual, or other downstream prediction was used.

A single linear regression was evaluated. No nonlinear model or additional feature search was performed.

## Sample

| Quantity | Value |
|---|---:|
| Candidate observations | 269,664 |
| Complete fills | 13,805 |
| Training fills | 8,119 |
| Test fills | 5,686 |
| Exact flow/state matches | 13,500 |
| Fallback matches | 305 |

`fill_fraction` was constant among the complete-fill observations and was therefore not used in the fitted model.

## Out-of-sample results

| Horizon | R² | 95% CI | MAE (bps) | Mean markout (bps) | P(adverse) |
|---|---:|---:|---:|---:|---:|
| 10ms | -0.074294 | [-0.227446, -0.002950] | 1.197957 | -0.273248 | 0.4949 |
| 100ms | -0.013903 | [-0.084834, +0.014028] | 1.304599 | -0.145827 | 0.5014 |
| 500ms | -0.024338 | [-0.106265, +0.017564] | 1.411535 | -0.193177 | 0.5121 |
| 1s | -0.006139 | [-0.072268, +0.031159] | 1.814029 | -0.215619 | 0.5317 |

At 1 second, the side-specific mean markouts were:

| Quote side | Mean markout |
|---|---:|
| BUY | +0.198778 bps |
| SELL | -0.623365 bps |

## Primary diagnostic

The earlier M9.5 1-second state-based reference was approximately:

- R² = 0.0463

M11 produced:

- OOS R² = -0.0061

The M11 result therefore does not demonstrate improved out-of-sample predictive performance relative to the earlier M9.5 markout model.

The side × imbalance interaction confidence interval included zero.

Three other model coefficients had bootstrap confidence intervals excluding zero, but this did not correspond to improved primary out-of-sample R² and therefore does not satisfy the predefined predictive-improvement stopping condition.

## Conclusion

M11 does not demonstrate incremental predictive value for conditional post-fill markout on the 2019-07-30 sample.

The expanded linear specification is therefore frozen without escalation to a nonlinear model or further Date1 feature tuning.

The result is retained as a negative research finding rather than optimized away.

## Interface to M12

M11 does not provide a demonstrated state-dependent predictive improvement in conditional markout.

For M12, the conditional markout component should therefore be treated conservatively. The M10 fill-hazard estimate and M11 conditional markout estimate remain separate components, with the M11 selection effect and single-date limitation carried forward.

## Validation status

Independent-date validation remains deferred because no second raw ITCH date is currently available.