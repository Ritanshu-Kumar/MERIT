# MERIT M9 v1 — Research Results

## Objective

Evaluate whether a machine-learning fill-outcome ranking improves the
estimated economics of passive quotes compared with the frozen M8 heuristic.

## Data and validation

- Sample: July 30, 2019, across AAPL, AMZN, GOOG, INTC, and MSFT.
- Candidate rows: 269,664.
- Chronological split: 183,788 training rows and 85,762 test rows.
- Embargo: 2 seconds around the split boundary.
- Timestamp integrity tests passed for all four post-fill markout horizons.
- Full test suite: 259 passed.

This is a single-day experiment. The held-out portion of this day has already
been inspected during research and is not an untouched confirmatory sample.

## Primary economics

Primary P&L uses fully filled candidates only. Partial-fill estimates using
resolution midpoints are reported separately as sensitivity analysis.

In the central scenario, before the overlap proxy:

| Metric | M8 | Prospective ML |
|---|---:|---:|
| Selected candidates | 14,405 | 13,231 |
| Full-fill candidates | 744 | 1,929 |
| Net P&L per selected candidate | $0.218790 | $0.134879 |

The estimated ML-minus-M8 difference was -$0.083911 per selected candidate,
with a 95% 60-second block-bootstrap interval of
[-$0.154626, +$0.032662]. This interval includes zero.

## Activity-matched diagnostic

At 14,405 selected candidates per policy, activity-matched ML produced
2,123 full fills and central-scenario net P&L of approximately $0.117506
per selected candidate.

This ranking uses scores on the test period to match M8's activity budget.
It is a retrospective diagnostic, not a frozen-threshold deployable policy.

## Markout timing

Midpoint observation timestamps were recorded and checked against the
required post-fill horizons. Observations occur at the first recorded event
at or after the target, so they can overshoot the nominal horizon.

Coverage and P&L sensitivity were evaluated at overshoot limits from 25 ms
to 1,000 ms. Filtering can affect policies differently; filtered results are
sensitivity analyses, not replacements for the unfiltered primary comparison.

## Overlap proxy

A greedy one-active-quote-per-symbol/side proxy was used as a lightweight
execution constraint. It does not model inventory, cancellations, queue
priority, capital constraints, or liquidation.

In the central scenario:

| Metric | M8 | Prospective ML |
|---|---:|---:|
| Accepted candidates after proxy | 4,918 | 2,089 |
| Full fills after proxy | 69 | 164 |
| Net P&L per accepted candidate | $0.081862 | -$0.038131 |

The estimated prospective ML-minus-M8 difference was -$0.119994 per accepted
candidate, with a 95% block-bootstrap interval of
[-$0.180259, -$0.058736].

The interval was below zero in the conservative, central, and optimistic
overlap-proxy scenarios. These results are conditional on the proxy and
the one-day sample, and should not be generalized to other dates.

## Conclusion

ML identified substantially more fills but did not improve estimated net
economics over M8 in this experiment. The result is consistent with a gap
between fill probability and fill profitability, but does not establish the
sole cause of the underperformance.

Independent-date validation is required before making a general performance
claim. The overlap proxy is not a full portfolio or execution simulator.

## Reproducibility

Run tests with:

    python -m pytest -q

The timestamped dataset and generated candidate-level and diagnostic CSVs
are research outputs and should not be committed as source code.