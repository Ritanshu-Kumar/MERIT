# M8 Research Protocol

## Research question

Can short-horizon microstructure state identify passive quote opportunities
with different expected post-fill adverse selection, and can exploiting that
information improve economic performance over the frozen M5 baseline market
maker?

## Primary economic target

For every simulated passive fill, calculate signed post-fill mid-price
markout at fixed horizons:

- 100 ms
- 1 s
- 5 s

For a buy fill:

M_h = Mid(t+h) - P_fill

For a sell fill:

M_h = P_fill - Mid(t+h)

Positive values indicate favorable movement after the fill.
Negative values indicate adverse selection.

The primary horizon is 1 second.

## Feature set

The initial frozen feature set is:

- mid price
- spread
- relative spread
- bid size
- ask size
- L1 imbalance
- microprice
- L5 imbalance
- L10 imbalance
- L5 depth-weighted imbalance
- L10 depth-weighted imbalance

## Quote opportunities

Quote opportunities are generated from observed market states for both
passive buy and passive sell quotes.

Fill/no-fill outcomes must use the frozen M7 execution assumptions.

The target is conditional on a simulated passive fill for the primary
economic analysis.

## Baselines

### Predictive baseline

Zero predicted markout.

### Economic baseline

The frozen M5 baseline market maker.

Predictive significance alone is not sufficient evidence for the primary
research claim.

The primary economic comparison is research strategy versus the frozen M5
baseline under identical execution assumptions.

## Data splitting

Evaluation is chronological.

Training, validation, and test periods must be specified before the final
test is run.

Samples whose target intervals overlap the evaluation boundary are purged.

A fixed embargo is applied around each boundary.

The maximum target horizon is 5 seconds, so the protocol currently specifies
5 seconds of purge and 5 seconds of embargo.

## Statistical evaluation

High-frequency observations are treated as dependent observations.

Inference will use block bootstrap procedures rather than an IID assumption.

The multiple-testing family includes the specified horizons and symbols.
Bonferroni correction is applied to the predefined family.

## Reproducibility

The final experiment must record:

- data provenance
- symbol universe
- time periods
- execution parameters
- feature definitions
- target definitions
- random seed
- software version
- configuration version

The final test set must not be used to select features, thresholds,
horizons, models, or strategy parameters.

## Development-data restriction

The current AAPL LOBSTER sample is development/reference data only.

Performance conclusions from that sample must not be presented as
universe-level evidence.