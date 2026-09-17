# M8 Empirical Research Checkpoint

This checkpoint contains the reproducible analysis used to validate the main M8 empirical
finding.

It is separate from the software test suite.

The software tests verify that MERIT components work correctly.

This research checkpoint verifies the empirical behavior discovered through the M8 experiments.

## Frozen mechanism

The M8 score is:

`relative_spread_pct × signed_imbalance`

with:

`+imbalance` for BUY

`-imbalance` for SELL

Frozen threshold:

`0.00957966`

The threshold is not re-optimized by this script.

## Required dataset

The script expects:

`data/sample/M8_replication_research_dataset.csv`

Generate it using the existing repository pipeline:

```cmd
python scripts/build_m8_replication.py