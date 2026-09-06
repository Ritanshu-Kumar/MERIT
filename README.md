# MERIT

**Market Execution, Risk & Intelligence Testbed**

MERIT is a first-principles research framework for studying **quantitative equity market making**.

The project is designed to answer a difficult question:

> Can market-microstructure information be used to make better liquidity-provision and quoting decisions after realistic execution costs, inventory effects, and risk constraints?

MERIT is not intended to reproduce the proprietary strategies of firms such as Jane Street or Citadel Securities. Instead, the goal is to build the research infrastructure ourselves, establish rigorous baselines, formulate testable hypotheses, and develop our own market-making mechanisms through reproducible experimentation.

MERIT is a **research laboratory**, not a production trading bot. No real-money trading is part of the core project, and no market-making performance claims are made until they are backed by frozen, out-of-sample results.

---

## Table of Contents

- [Research Philosophy](#research-philosophy)
- [What MERIT Is Studying](#what-merit-is-studying)
- [System Architecture](#system-architecture)
- [Universe Expansion](#universe-expansion)
- [Development Roadmap](#development-roadmap)
- [Trade Accounting & Reporting](#trade-accounting--reporting)
- [Execution Realism](#execution-realism)
- [Data](#data)
- [Research Protocol](#research-protocol)
- [What Counts as a Successful Result?](#what-counts-as-a-successful-result)
- [Reproducibility](#reproducibility)
- [Repository Structure](#repository-structure)
- [Current Status](#current-status)
- [Long-Term Goal](#long-term-goal)

---

## Research Philosophy

MERIT follows a research-first approach:

1. Build the infrastructure from first principles.
2. Reconstruct the market from Level-2 order-book data.
3. Create a deterministic event-driven simulator.
4. Establish a deliberately simple market-making baseline.
5. Model execution, latency, fees, partial fills, and queue effects realistically.
6. Separate strategy decisions from risk controls.
7. Freeze hypotheses, metrics, and test sets before final evaluation.
8. Test proposed mechanisms out of sample across multiple stocks and market regimes.
9. Report failures as well as successful results.
10. Avoid novelty claims without serious prior-art research.

## What MERIT Is Studying

The research is centered on market microstructure and liquidity provision. Potential signals and mechanisms include:

- Order-book imbalance
- Trade flow and signed volume
- Short-horizon volatility
- Liquidity and spread regimes
- Adverse selection
- Inventory pressure
- Latency and quote staleness
- Short-horizon price dynamics
- State-dependent quote placement and sizing

These are research directions, not predetermined conclusions. The eventual strategy will be developed only after the simulator, baseline, exploratory analysis, and literature review establish a defensible research question.

## System Architecture

```text
                    EQUITY MARKET DATA
                            │
                            ▼
                    EVENT NORMALIZER
                            │
                            ▼
                  PER-SYMBOL LOCAL L2 BOOK
                            │
                            ▼
                MICROSTRUCTURE FEATURES
                            │
                            ▼
                    MARKET-MAKING STRATEGY
                            │
                            ▼
                       RISK ENGINE
                            │
                            ▼
                    EXECUTION SIMULATOR
                            │
                            ▼
                          FILLS
                            │
                            ▼
                    PORTFOLIO / P&L
                            │
                            ▼
                       TRADE LEDGER
                            │
                            ▼
                    REPORTING / ANALYSIS
```

Every stateful component is designed to be symbol-aware and symbol-agnostic at the same time: the code does not hardcode a specific ticker, while each symbol maintains its own book, features, position, risk state, and ledger.

This allows the same codebase to scale from one stock to a larger research universe.

## Universe Expansion

MERIT is architected for a growing equity universe, but the research rollout is deliberately staged.

```text
T0 → 1 symbol
T1 → 5–10 symbols
T2 → 20–40 symbols
T3 → Full liquid universe
```

**T0 — Single Name**
Prove that the complete system works end-to-end:
- Local order book
- Feature generation
- Baseline market maker
- Execution
- Risk
- Portfolio accounting
- Trade ledger
- Reporting

**T1 — Small Basket**
Demonstrate that the architecture is genuinely symbol-agnostic and that:
- Multiple independent books work correctly
- Positions remain isolated
- Trade ledgers remain isolated
- Portfolio-level P&L aggregates correctly
- No cross-symbol state leakage occurs

**T2 — Research Basket**
Use a sector- and market-condition-diverse basket large enough to investigate whether a research mechanism generalizes beyond a single stock. The current target is approximately 20–40 symbols.

**T3 — Full Liquid Universe**
Optionally expand to a clearly defined liquid-equity universe using an explicit liquidity criterion rather than claiming to model every listed ticker. This is a later-stage extension, not a prerequisite for the core research result.

## Development Roadmap

| Milestone | Description |
|---|---|
| **M0 — Research Infrastructure** | Repository, environment, configuration, logging, testing, documentation, and experiment metadata. |
| **M1 — Portfolio & Wallet** | Cash, positions, fees, order accounting, trade ledger, and realized/unrealized P&L. |
| **M2 — Deterministic Simulator** | Event-driven market replay with reproducible results. |
| **M3 — Local L2 Order Book** | Incremental reconstruction and validation of per-symbol order books. |
| **M4 — Market-Microstructure Features** | Order-book, trade-flow, volatility, liquidity, and inventory features. |
| **M5 — Baseline Market Maker** | A deliberately simple frozen baseline used as the primary controlled comparison. |
| **M6 — Risk Engine** | Per-symbol and portfolio-level exposure limits, loss controls, stale-data protection, and kill switches. |
| **M7 — Realistic Execution** | Latency, fees, spread costs, slippage, partial fills, and an explicit L2 queue-position approximation. |
| **M8 — Original Research** | Formulate, preregister, and test an original state-dependent market-making mechanism. |
| **M9 — ML Extension** *(optional)* | Machine-learning extensions after the non-ML mechanism is understood. |
| **M10 — Robustness & Portfolio Research** *(optional)* | Broader-universe validation, cross-symbol research, correlation analysis, and related extensions. |

## Trade Accounting & Reporting

Trade accounting is a core part of MERIT. Every simulated fill is recorded with enough information to reconstruct the complete trading history.

**Example ledger entry:**

| Timestamp | Symbol | Side | Qty | Price | Fee | Position | Realized P&L |
|---|---|---|---|---|---|---|---|
| 09:30:01.245 | AAPL | BUY | 100 | 50.10 | ... | +100 | — |
| 09:30:03.882 | AAPL | SELL | 100 | 50.18 | ... | 0 | +8.00 |

At the end of a simulation, MERIT reports both per-symbol and portfolio-level results, including:

- Starting capital
- Total buys
- Total sells
- Total fees
- Realized P&L
- Unrealized P&L
- Final equity
- Maximum position
- Maximum exposure
- Maximum drawdown
- Turnover
- Fill rate

The trade ledger is the source of truth for these calculations.

## Execution Realism

A market-making simulation cannot assume that every submitted quote is filled. MERIT therefore models:

- Decision-to-arrival latency
- Limit and market orders where supported
- Partial fills
- Cancellation
- Fees
- Spread costs
- Slippage
- Order lifecycle
- Queue-position effects

Level-2 data provides aggregated quantity at each price rather than individual order identities. Therefore, exact queue position is not observable from L2 alone. MERIT explicitly models estimated queue position using documented assumptions rather than presenting it as ground truth.

## Data

The initial asset class is **US equities**.

The data layer is intentionally separated from the rest of the architecture so that a dataset or vendor can be changed through a documented protocol change rather than by rewriting the simulator or strategy.

The selected feed will be documented with:

- Vendor
- Venue
- Feed type
- Schema
- Historical availability
- Timestamp semantics
- Licensing
- Known limitations
- Exact dataset/version used

## Research Protocol

Before final test-period evaluation, MERIT will freeze:

- Research hypothesis
- Primary metric
- Secondary metrics
- Baseline definition
- Training/validation/test split
- Exact symbol set
- Minimum improvement criterion
- Robustness checks
- Failure criteria

```text
HYPOTHESIS + METRIC + THRESHOLD + TEST WINDOWS + SYMBOL SET
                            │
                            ▼
                           FREEZE
                            │
                            ▼
                        RUN TEST
                            │
                            ▼
                          REPORT
```

Test-period results must not influence feature selection, hyperparameters, architecture, thresholds, or symbol selection.

## What Counts as a Successful Result?

A strategy does not "beat the baseline" simply because it has higher raw P&L. MERIT evaluates performance using multiple dimensions, including:

- Net realized spread
- Sharpe ratio
- Maximum drawdown
- Fill rate
- Inventory variance
- Turnover
- Adverse-selection / markout metrics
- Fee burden
- Cross-sectional consistency
- Robustness across independent test periods

A claimed improvement must survive predefined risk constraints and should not depend on a small number of cherry-picked symbols.

## Reproducibility

Every experiment records enough information to reproduce the result, including:

- Dataset
- Dataset version / hash
- Universe configuration
- Code revision
- Configuration
- Random seed
- Simulator version
- Experiment configuration

Identical input data and configuration produce reproducible simulation results.

## Repository Structure

```text
MERIT/
├── data/                # datasets, manifests, and data metadata
├── schemas/             # normalized market-event contracts
├── book/                # local L2 order-book reconstruction
├── simulator/            # deterministic event-driven replay
├── portfolio/             # wallet, positions, P&L, trade ledger
├── risk/                  # risk limits and kill switches
├── strategy/              # baseline and research strategies
├── execution/             # latency, fees, fills, queue model
├── experiments/            # experiment configurations and evaluation
├── research/               # literature review and hypotheses
├── reports/                # generated experiment reports
├── tests/                  # unit, integration, and determinism tests
├── configs/                # T0/T1/T2/T3 configurations
├── scripts/                # utility and experiment scripts
├── docs/                   # project documentation
├── src/
│   └── merit/              # Python package
├── README.md
├── pyproject.toml
├── LICENSE
└── .gitignore
```

## Current Status

**M0 — Research Infrastructure**

The project is currently establishing the software architecture, configuration system, testing framework, and reproducibility foundation.

- No market-making performance claims have been made.
- No real-money trading is part of the core project.

## Long-Term Goal

The objective is not to build the largest trading platform or to imitate an existing proprietary market maker.

The objective is to understand market making deeply enough to:

1. Build a trustworthy experimental environment.
2. Develop a clearly defined market-making mechanism.
3. Evaluate it against meaningful baselines.
4. Determine where it works and where it fails.
5. Establish whether its effect generalizes across a diverse equity universe.
6. Produce a reproducible research artifact that may eventually support a technical report, preprint, or peer-reviewed publication.