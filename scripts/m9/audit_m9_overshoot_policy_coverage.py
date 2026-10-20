from pathlib import Path

import pandas as pd

DATA = Path("research/m9_economic_decision_test_candidates.csv")

SCENARIOS = {
    "optimistic": "10ms",
    "central": "100ms",
    "conservative": "1s",
}

HORIZONS_NS = {
    "10ms": 10_000_000,
    "100ms": 100_000_000,
    "500ms": 500_000_000,
    "1s": 1_000_000_000,
}

LIMITS_MS = [25, 50, 100, 250, 500, 1000]

POLICIES = [
    "m8_selected",
    "ml_selected",
    "ml_stratified_selected",
]


def parse_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
        .fillna(False)
    )


def main():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Missing {DATA}. Run evaluate_m8_ml_economics.py first."
        )

    df = pd.read_csv(DATA)

    required = {
        "scenario",
        "outcome",
        "filled_qty",
        "fill_timestamp_ns",
        *POLICIES,
    }

    for horizon in HORIZONS_NS:
        required.add(f"future_mid_{horizon}_timestamp_ns")
        required.add(f"future_mid_{horizon}")

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Candidate export lacks required columns: {missing}\n"
            "Confirm that the timestamp-aware evaluator generated the file."
        )

    for policy in POLICIES:
        df[policy] = parse_bool(df[policy])

    df["outcome"] = df["outcome"].astype(str).str.upper()
    df["filled_qty"] = pd.to_numeric(
        df["filled_qty"], errors="coerce"
    ).fillna(0)

    df["fill_timestamp_ns"] = pd.to_numeric(
        df["fill_timestamp_ns"], errors="coerce"
    )

    print(f"Candidate-export rows: {len(df):,}")
    print(
        "Note: policies are evaluated only in their corresponding "
        "scenario and use full-fill candidates for markout coverage."
    )

    for scenario, horizon in SCENARIOS.items():
        timestamp_col = f"future_mid_{horizon}_timestamp_ns"
        price_col = f"future_mid_{horizon}"
        target_ns = HORIZONS_NS[horizon]

        section = df[df["scenario"].eq(scenario)].copy()

        section[timestamp_col] = pd.to_numeric(
            section[timestamp_col], errors="coerce"
        )
        section[price_col] = pd.to_numeric(
            section[price_col], errors="coerce"
        )

        section["overshoot_ms"] = (
            section[timestamp_col]
            - section["fill_timestamp_ns"]
            - target_ns
        ) / 1_000_000

        print(f"\n{'=' * 72}")
        print(f"Scenario: {scenario} | Horizon: {horizon}")
        print(f"{'=' * 72}")

        for policy in POLICIES:
            selected = section[section[policy]]
            full_fills = selected[
                selected["outcome"].eq("FILL")
                & selected["filled_qty"].gt(0)
            ]

            valid = (
                full_fills["fill_timestamp_ns"].notna()
                & full_fills[timestamp_col].notna()
                & full_fills[price_col].notna()
            )

            valid_fills = full_fills[valid]
            overshoot = valid_fills["overshoot_ms"]

            print(f"\nPolicy: {policy}")
            print(f"Selected candidates: {len(selected):,}")
            print(f"Full-fill candidates: {len(full_fills):,}")
            print(f"Valid timestamped markouts: {len(valid_fills):,}")
            print(
                "Missing timestamp or midpoint: "
                f"{len(full_fills) - len(valid_fills):,}"
            )

            if overshoot.empty:
                print("No valid markout observations.")
                continue

            print(
                f"Overshoot p50/p95/max: "
                f"{overshoot.quantile(0.50):.3f} / "
                f"{overshoot.quantile(0.95):.3f} / "
                f"{overshoot.max():.3f} ms"
            )

            for limit in LIMITS_MS:
                count = int((overshoot <= limit).sum())
                pct = 100 * count / len(overshoot)

                print(
                    f"  <= {limit:4d} ms: "
                    f"{count:5,}/{len(overshoot):,} ({pct:6.2f}%)"
                )


if __name__ == "__main__":
    main()