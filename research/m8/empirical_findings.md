# M8 Empirical Findings

## Purpose

This document records the empirical findings obtained during the M8 research program through
frozen-policy analysis, placebo tests, execution simulations, queue sensitivity, and economic
replay.

These findings are distinct from software unit/integration tests.

The purpose is to preserve the research trail, including positive results, failed hypotheses,
invalidated runs, and unresolved questions.

---

# 1. Frozen M8 mechanism

The frozen mechanism is:

interaction_score =
relative_spread_pct × signed_imbalance

where:

signed_imbalance =
    +imbalance for BUY
    -imbalance for SELL

The frozen validation threshold is:

0.00957966

Direction:

high

The threshold was selected using the validation set only and was not re-optimized on the
test set.

---

# 2. Primary conditional fill-quality finding

The frozen M8 policy was evaluated on the untouched test population.

Test population:

- 5,736 observations
- 1,654 M8-selected observations

The selected observations had approximately:

+1.512 bps

higher 1-second post-fill return than the unselected observations.

The important interpretation is:

This is a conditional fill-quality effect.

It is NOT evidence that an executable market-making strategy earns +1.512 bps.

The effect measures post-fill behavior conditional on a historical passive execution opportunity.

---

# 3. Temporal locality / placebo finding

A temporal placebo test replaced contemporaneous imbalance with progressively older imbalance
while retaining the frozen interaction structure.

Approximate selected-vs-unselected effects:

| Imbalance state | Effect |
|---|---:|
| Current | +1.512 bps |
| 1-event lag | ~+1.4 bps |
| 5-event lag | ~+1.1 bps |
| 20-event lag | ~+0.36 bps |
| 100-event lag | approximately 0 |

A within-symbol randomization placebo produced an approximately null effect.

Interpretation:

The M8 effect appears to depend on contemporaneous order-book state.

This is stronger evidence than a result that could be explained entirely by static symbol
characteristics or spread differences.

The effect also appears short-lived rather than representing a persistent unconditional
predictor.

---

# 4. Tick-regime diagnostic

The frozen mechanism was evaluated separately in one-tick and multi-tick spread regimes.

The results did not establish that the M8 effect is uniquely caused by tick-constrained
spreads.

Some symbols showed positive effects in both regimes, while others were weak or negative.

Conclusion:

The research should NOT currently claim that M8 is specifically a tick-bound mechanism.

The frozen mechanism remains unchanged.

---

# 5. Economic replay: first important qualification

The first economic replay result was invalidated.

Reason:

The replay reconstructed its own test boundary rather than using the exact frozen research
test population.

The resulting test window differed from the frozen boundary.

Therefore those initial P&L numbers are not part of the valid empirical record.

This invalidation is intentional and should remain documented.

---

# 6. Corrected economic replay

A corrected economic replay was constructed using the frozen test interval and the LOBSTER
execution framework.

The comparison was:

- frozen M8 policy
- baseline market maker

Representative zero-queue / zero-latency point estimates:

Baseline:
approximately +$1.26k

M8:
approximately +$0.43k

Point estimate difference:

M8 - baseline
approximately -$0.83k

However, this difference is NOT statistically resolved on the available single-day sample.

---

# 7. Economic uncertainty finding

A time-block bootstrap was used to account for intraday dependence.

Using moving time blocks, the 95% confidence interval for M8 minus baseline P&L was
approximately:

[-$6.1k, +$3.7k]

The interval includes zero.

Therefore:

We CANNOT conclude that M8 economically loses to baseline.

The correct conclusion is:

The single-day sample does not distinguish the economic performance of M8 and the baseline
with sufficient statistical precision.

---

# 8. Per-share quality finding

Point estimates for P&L per traded share were approximately:

Baseline:
$0.00557/share

M8:
$0.00600/share

Difference:
approximately +$0.00043/share

M8 therefore showed better point-estimate execution quality per traded share.

However, the corresponding time-block bootstrap confidence interval also included zero.

Therefore the per-share improvement is suggestive but not statistically resolved on this
single date.

---

# 9. Symbol decomposition

At zero queue, the approximate M8-minus-baseline P&L differences were:

| Symbol | M8 - Baseline |
|---|---:|
| AAPL | -$1,003.75 |
| AMZN | +$76.06 |
| GOOG | -$158.24 |
| INTC | +$138.71 |
| MSFT | +$117.15 |

This decomposition is heterogeneous.

It does not justify a claim that the economic result is purely an INTC/MSFT effect.

However, per-symbol sample sizes and outlier sensitivity still require explicit auditing before
drawing stronger cross-symbol conclusions.

---

# 10. Queue sensitivity

Using one consistent replay implementation, representative point estimates were:

| Queue ahead | Baseline | M8 |
|---:|---:|---:|
| 0% | +$1.26k | +$0.43k |
| 25% | +$0.76k | +$0.17k |
| 50% | -$1.87k | -$0.80k |

The earlier interpretation that M8 uniquely collapses under queue assumptions was rejected.

At 50% queue, M8 actually had a better point estimate than the baseline.

This result should also NOT be over-interpreted because uncertainty increases as executable fills
become scarcer.

Queue sensitivity is therefore currently an unresolved execution interaction rather than a
confirmed mechanism.

---

# 11. Hybrid residual-size experiment

Exploratory hybrid policies were tested by retaining some residual quoting on observations
not selected by M8.

Several residual-size fractions were evaluated.

Some variants improved validation performance but failed on the untouched test set.

Example:

M8 + 25% residual size:

Validation:
approximately +$1.50k

Test:
approximately -$2.57k

Other residual fractions also exhibited large validation-to-test reversals.

Conclusion:

The residual-size experiment is NOT retained as a candidate strategy.

It is recorded as evidence of researcher-degrees-of-freedom risk and single-day economic
variance.

No further residual-size sweep should be performed without a new preregistered hypothesis.

---

# 12. Core empirical conclusion

The strongest finding from M8 is currently:

> The interaction between relative spread and contemporaneous signed imbalance appears to
> identify a subset of passive executions with better short-horizon post-fill behavior.

The evidence supporting this statement includes:

1. A positive frozen test-set conditional effect.
2. Temporal decay when imbalance is made stale.
3. Near-null within-symbol randomization.
4. Replication across multiple symbols, although heterogeneous.

The following stronger claim is NOT supported:

> M8 is a profitable market-making strategy.

Economic P&L remains unresolved on the single available trading day.

---

# 13. Research interpretation

M8 should currently be treated as:

A potentially useful fill-quality signal.

M8 should NOT currently be treated as:

A validated profitable market-making policy.

The central unresolved bridge is:

microstructure state
→ probability of receiving a fill
→ quality of the fill
→ executable quote policy
→ realized P&L

The first two stages show evidence of structure.

The final economic conversion remains unresolved.

---

# 14. Important methodological lessons

The M8 experiments exposed several methodological requirements.

### Frozen boundaries matter

Economic replay must use the exact frozen test population and boundary.

### One replay implementation must be authoritative

Different replay implementations produced materially different intermediate conclusions.

Economic figures should therefore only be reported after implementation-consistency auditing.

### Single-day P&L is highly noisy

Large validation-to-test sign reversals were observed even for simple hybrid policies.

Point estimates must therefore always be accompanied by time-block uncertainty estimates.

### Conditional fill quality is not strategy P&L

A positive post-fill edge does not imply that a strategy can capture the edge at realistic
fill probabilities, queue positions, latency, and costs.

---

# 15. Current research status

Status:

M8 conditional mechanism:
PROMISING

M8 profitable market-making strategy:
UNRESOLVED

Generalization across dates:
UNTESTED

Generalization across regimes:
UNTESTED

Live-market validity:
UNTESTED

Sizing mechanism:
NOT YET JUSTIFIED

---

# 16. Next empirical checkpoint

Before introducing a new strategy mechanism:

1. Audit all existing reported results for implementation consistency.
2. Audit per-symbol M8 fill counts.
3. Check economic outlier sensitivity.
4. Repeat the economic bootstrap using multiple block lengths.
5. Obtain an independent trading date.
6. Run the frozen M8 policy without modification on that date.
7. Compare the same metrics and uncertainty intervals.

Only if the underlying phenomenon survives cross-date validation should a new sizing or
quote-intensity mechanism be considered.