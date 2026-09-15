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