# M8 Preregistration

## Research question

Does the interaction between relative spread and contemporaneous signed order-book imbalance identify passive fills with lower adverse selection?

## Primary target

Primary horizon: 1 second.

For a passive BUY fill:

post-fill move = markout - initial edge

For a passive SELL fill:

post-fill move = markout - initial edge

Primary reported metric:

post-fill return (bps) =
post-fill move / fill price * 10000

Secondary horizons:

- 100 ms
- 5 seconds

## Frozen mechanism

The frozen score is:

interaction_score =
relative_spread_pct × signed_imbalance

where:

signed_imbalance =
    +imbalance for BUY
    -imbalance for SELL

The policy selects a candidate when:

interaction_score >= 0.00957966

Direction:

high

This threshold was selected on the validation period only.

It must not be re-optimized on the final test set.

## Data and causal convention

Initial research dataset:

LOBSTER sample date:
2012-06-21

Symbols:

- AAPL
- AMZN
- GOOG
- INTC
- MSFT

The feature state used for an execution event is the immediately preceding order-book state.

Execution is therefore evaluated causally using state information available before the observed execution event.

## Dataset split

Chronological split:

- Train: 60%
- Validation: 20%
- Test: 20%

A 5-second purge/embargo is applied around split boundaries.

Frozen combined dataset:

28,734 usable observations

Frozen test set:

5,736 observations

Frozen M8 selections on test:

1,654

## Economic evaluation

The economic evaluation compares:

1. frozen M8 policy
2. baseline market maker

The economic simulator must use the same:

- test interval
- causal book state
- order quantity
- starting capital
- position limits
- execution-event convention
- portfolio accounting
- mark-to-market convention

Execution sensitivity is evaluated through:

- queue assumptions
- decision-to-arrival latency
- fees

Raw total P&L is not sufficient.

Report:

- total P&L
- P&L per fill
- P&L per traded unit
- fill count
- traded quantity
- average absolute inventory
- inventory/exposure time where available
- maximum inventory
- per-symbol results
- uncertainty intervals

## Statistical evaluation

Economic P&L must not be treated as resolved from a single point estimate.

Use time-block bootstrap rather than independent tick/fill resampling.

Primary economic comparisons should report confidence intervals.

Per-symbol decomposition should also be reported before making cross-symbol claims.

Bootstrap block-length sensitivity should be checked.

## Evidence standard

A positive conditional fill-quality result is not equivalent to a profitable quoting strategy.

The economic claim is considered unresolved when uncertainty intervals include zero or results are materially unstable across reasonable execution assumptions.

No new mechanism should be introduced solely because it improves one validation sweep.

Any new strategy mechanism must be:

1. explicitly hypothesized,
2. frozen before validation,
3. validated once,
4. evaluated once on the untouched test set.

## Scope limitation

The current evidence is from one trading date.

A single-day result is not sufficient to claim generalization across dates, regimes, symbols, venues, or live markets.

A second independent trading date is required before treating the economic relationship as a generalizable phenomenon.

## Current status

The conditional M8 fill-quality effect has been frozen.

The economic P&L question remains unresolved on the current single-day sample.

The next research phase is audit and cross-date validation, not additional threshold or sizing optimization.

## Date 2 Replication Protocol

The second trading date will be evaluated using the same frozen M8
mechanism, threshold, symbols, order quantity, causal execution
convention, and economic evaluation framework used for Date 1.

No threshold, mechanism, quantity, or execution assumption may be
changed after inspecting Date 2 results.

### Pre-specified Date 2 analyses

The following analyses will be run without modification:

1. Pooled 1-second conditional effect.
2. Temporal placebo at 0, 1, 5, 20, and 100 event lags.
3. Within-symbol permutation placebo.
4. Symbol decomposition.
5. Per-symbol bootstrap intervals.
6. Economic replay against the baseline market maker.
7. Economic P&L block bootstrap using 1-, 5-, and 10-minute blocks.

No additional analysis may be introduced solely because of the observed
Date 2 results.

### Replication criteria

Date 2 will be classified as a signal replication only if:

1. The pooled 1-second conditional effect is positive.
2. The contemporary effect is materially larger than the
   100-event placebo.
3. The temporal relationship shows decay away from the contemporaneous
   state rather than an arbitrary lagged pattern.
4. The within-symbol permutation placebo remains centered near zero.
5. No Date 1 symbol with a robust positive effect develops a robust
   negative effect on Date 2.

A symbol is considered to have a robust sign when its 95% bootstrap
interval excludes zero.

The economic result is evaluated separately from signal replication.

### Failure criteria

Date 2 will be classified as a signal failure if:

- the pooled 1-second effect is non-positive,
- the temporal decay pattern is absent, or
- the permutation placebo indicates a comparable non-zero effect.

### Inconclusive criteria

Date 2 will be classified as inconclusive when the evidence is mixed,
including cases where the signal characteristics replicate but the
economic P&L comparison remains statistically unresolved.

A signal replication does not imply economic profitability.

### Cross-date analysis

After Date 2 is evaluated, Date 1 and Date 2 will be analyzed jointly.

The cross-date analysis will report:

- date-level conditional effects,
- symbol-level effects,
- pooled effect,
- between-date heterogeneity,
- random-effects meta-analysis where appropriate,
- economic P&L comparison for each date.

GOOG will be reported explicitly rather than absorbed into a single
heterogeneity statistic because its Date 1 effect is robustly negative.

The two-date analysis is still considered limited evidence and is not
sufficient by itself to establish generalization across broader market
regimes, venues, or live trading.