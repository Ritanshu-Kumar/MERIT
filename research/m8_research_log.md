# M8 Research Log

## M8-E0 — Conditional fill-quality discovery

The initial M8 exploration investigated whether relative spread, imbalance, and their interaction were associated with post-fill midpoint movement.

Exploratory evidence suggested that relative spread and imbalance contain short-horizon information, but pooled effects were heterogeneous across symbols.

A continuous spread effect was observed across several symbols, with meaningful heterogeneity.

The interaction between spread and imbalance was retained as the candidate mechanism.

No ML strategy claim was made.

## M8-E1 — Frozen mechanism

The mechanism was frozen as:

interaction_score =
relative_spread_pct × signed_imbalance

with:

signed_imbalance =
+imbalance for BUY
-imbalance for SELL

Validation-selected threshold:

0.00957966

The final test set was kept untouched during threshold selection.

## M8-E2 — Frozen test result

Frozen test population:

5,736 observations

M8-selected:

1,654 observations

The selected-vs-unselected conditional post-fill difference was approximately:

+1.512 bps

This should be described as a conditional fill-quality edge, not as strategy P&L.

## M8-E3 — Temporal placebo

The contemporaneous imbalance signal decayed when increasingly stale imbalance was substituted.

Approximate selected-vs-unselected effects:

- current imbalance: +1.512 bps
- 1-event lag: +1.4 bps
- 5-event lag: +1.1 bps
- 20-event lag: +0.36 bps
- 100-event lag: approximately zero

Within-symbol randomization produced a near-zero null distribution.

Interpretation:

The conditional signal appears to depend on contemporaneous order-book state rather than only static symbol composition.

## M8-E4 — Tick-regime diagnostic

The frozen policy was evaluated in one-tick and multi-tick spread regimes.

The result did not support narrowing the mechanism to a specifically tick-constrained explanation.

Symbol-level behavior remained heterogeneous.

Conclusion:

Do not modify the frozen mechanism on the basis of this diagnostic.

## M8-E5 — Economic replay

An initial economic replay was found to use a mismatched test boundary.

That result was invalidated.

A subsequent corrected replay was developed using the frozen test interval and the LOBSTER execution framework.

## M8-E6 — Execution sensitivity

The corrected economic replay compared the frozen M8 policy with the baseline market maker under queue assumptions.

Point estimates showed substantial single-day variation.

The zero-queue point estimates were approximately:

Baseline P&L: +$1.26k
M8 P&L: +$0.43k

However, this difference is not statistically resolved on the single-day sample.

## M8-E7 — Hybrid residual-size experiment

Exploratory residual-size hybrids were tested.

Some variants improved validation P&L, but none produced a robust test improvement.

These variants are discarded and will not be iteratively tuned.

This experiment is retained as evidence of researcher-degrees-of-freedom risk.

## M8-E8 — Economic uncertainty

Time-block bootstrap showed that the single-day M8-vs-baseline P&L difference has a confidence interval that includes zero.

Likewise, the apparent per-share execution-quality improvement is not statistically resolved on the current day.

Interpretation:

The economic question cannot be answered reliably from this single date.

## M8-E9 — Implementation consistency finding

During economic validation, multiple replay implementations were found to differ in boundary or execution assumptions.

Those discrepancies were corrected rather than averaged together.

This is now treated as a standing methodological requirement:

all reported economic numbers must come from the same frozen dataset, causal convention, execution implementation, and evaluation boundary.

## Current research conclusion

M8 currently has evidence of a conditional fill-quality signal.

There is not yet sufficient evidence that the signal translates into economically superior market-making P&L.

The single-day economic sample is too noisy to support a strong win/loss conclusion.

The correct next step is additional data and a fully audited common evaluation pipeline.

Do not introduce a new sizing mechanism yet.

## Next phase

1. Implementation-consistency audit.
2. Per-symbol fill-count and outlier audit.
3. Bootstrap block-length sensitivity.
4. Second independent trading date.
5. Repeat the exact frozen evaluation.
6. Only after cross-date evidence is established consider a new mechanism.

## M8-E10 — Empirical research checkpoint

This checkpoint consolidates the main simulation findings.

The key result is not a confirmed P&L advantage.

The key result is that the frozen spread × contemporaneous signed-imbalance mechanism shows
short-horizon conditional fill-quality information that decays when the state becomes stale.

Economic conversion to a complete quoting strategy remains unresolved because the single-day
sample has high P&L variance.

The next research step is therefore validation and generalization, not additional strategy
optimization.

## Economic Replay Robustness — Date 1

The frozen Date 1 economic replay was audited against the canonical
LOBSTER event population. Baseline and M8 replay fills achieved 100%
event reconciliation with zero quantity mismatches.

Economic point estimates:

- Baseline total P&L: +$313.87
- M8 total P&L: +$161.78
- M8 minus baseline: -$152.09

The minute-level portfolio equity reconstruction reproduced the audited
P&L difference exactly, with reconstruction error of $0.000000.

Time-block bootstrap of the M8-minus-baseline P&L difference:

| Block | Observed | 95% interval |
|---|---:|---:|
| 1 minute | -$152.09 | [-$1,041.59, +$717.82] |
| 5 minutes | -$152.09 | [-$696.33, +$453.82] |
| 10 minutes | -$152.09 | [-$656.33, +$388.68] |

All intervals include zero. Therefore, the single Date 1 replay does
not establish a statistically robust economic P&L difference between
M8 and the baseline.

The Date 1 microstructure signal remains detectable, but its economic
value is not established from this single day.