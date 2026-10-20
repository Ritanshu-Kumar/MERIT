
from pathlib import Path

import pandas as pd


INPUT = Path("research/m9_m8_vs_ml_selection.csv")
OUTPUT = Path("research/m9_m8_ml_stratified.csv")
REPORT = Path("research/m9_m8_ml_stratified.txt")

STRATA = ["symbol", "quote_side"]


def parse_boolean(series):
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
    )


def summarize(name, subset, total_test):
    counts = subset["outcome"].value_counts()
    n = len(subset)

    return {
        "policy": name,
        "selected": n,
        "selection_rate_pct": 100 * n / max(total_test, 1),
        "full_fill_rate_pct": 100 * counts.get("FILL", 0) / max(n, 1),
        "adverse_rate_pct": 100 * counts.get("ADVERSE", 0) / max(n, 1),
        "timeout_rate_pct": 100 * counts.get("TIMEOUT", 0) / max(n, 1),
    }


def outcome_row(symbol, side, policy, subset):
    counts = subset["outcome"].value_counts()
    n = len(subset)

    return {
        "symbol": symbol,
        "quote_side": side,
        "policy": policy,
        "selected": n,
        "full_fill_rate_pct": 100 * counts.get("FILL", 0) / max(n, 1),
        "adverse_rate_pct": 100 * counts.get("ADVERSE", 0) / max(n, 1),
        "timeout_rate_pct": 100 * counts.get("TIMEOUT", 0) / max(n, 1),
    }


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"{INPUT} not found. Run compare_m8_ml_selection.py first."
        )

    df = pd.read_csv(INPUT)

    required = {
        "timestamp_ns",
        "symbol",
        "quote_side",
        "outcome",
        "m8_selected",
        "ml_score",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["m8_selected"] = parse_boolean(df["m8_selected"])
    df["ml_score"] = pd.to_numeric(df["ml_score"], errors="coerce")

    if df["m8_selected"].isna().any():
        raise ValueError("Could not parse m8_selected.")
    if df["ml_score"].isna().any():
        raise ValueError("ML scores contain missing or invalid values.")

    total_test = len(df)
    m8 = df[df["m8_selected"]].copy()

    selected_groups = []
    stratum_audit = []

    # Match M8's selected count within every symbol × quote-side group.
    # The ML ranking uses scores only, never outcome labels.
    for (symbol, side), group in df.groupby(STRATA, sort=True):
        target_count = int(group["m8_selected"].sum())

        ranked = group.sort_values(
            ["ml_score", "timestamp_ns"],
            ascending=[False, True],
            kind="mergesort",
        )

        chosen = ranked.head(target_count).copy()
        selected_groups.append(chosen)

        stratum_audit.append({
            "symbol": symbol,
            "quote_side": side,
            "test_candidates": len(group),
            "m8_selected": target_count,
            "ml_selected": len(chosen),
        })

    ml_matched = pd.concat(selected_groups, ignore_index=True)

    # Verify exact count matching within every stratum.
    audit = pd.DataFrame(stratum_audit)
    if not (audit["m8_selected"] == audit["ml_selected"]).all():
        raise AssertionError("ML and M8 counts do not match within all strata.")

    if len(ml_matched) != len(m8):
        raise AssertionError("Overall ML selection count does not match M8.")

    overall = pd.DataFrame([
        summarize("All test candidates", df, total_test),
        summarize("Frozen M8", m8, total_test),
        summarize("ML matched by symbol + side", ml_matched, total_test),
    ])

    per_stratum = []
    for (symbol, side), group in df.groupby(STRATA, sort=True):
        per_stratum.append(
            outcome_row(
                symbol, side, "Frozen M8",
                group[group["m8_selected"]],
            )
        )
        # The chosen ML candidates for this group are identifiable
        # by their group keys in ml_matched.
        chosen = ml_matched[
            (ml_matched["symbol"] == symbol)
            & (ml_matched["quote_side"] == side)
        ]
        per_stratum.append(
            outcome_row(
                symbol, side, "ML matched",
                chosen,
            )
        )

    per_stratum = pd.DataFrame(per_stratum)

    selected_rows = pd.concat([
        m8.assign(policy="Frozen M8"),
        ml_matched.assign(policy="ML matched by symbol + side"),
    ], ignore_index=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    selected_rows.to_csv(OUTPUT, index=False)

    report = (
        "MERIT: Stratified M8 vs ML comparison\n"
        + "=" * 58 + "\n\nOverall outcomes\n"
        + overall.to_string(index=False, float_format=lambda x: f"{x:.4f}")
        + "\n\nSelection-count audit by symbol and side\n"
        + audit.to_string(index=False)
        + "\n\nOutcome breakdown by symbol and side\n"
        + per_stratum.to_string(index=False, float_format=lambda x: f"{x:.4f}")
        + "\n\nLimitations:\n"
        + "- Single trading date; exploratory result only.\n"
        + "- ML ranks test candidates retrospectively using scores, not outcomes.\n"
        + "- Matching group counts controls symbol and side mix, not every market condition.\n"
        + "- Candidate outcomes are not net P&L; partial fills still require valuation.\n"
        + "- Candidates overlap in time and may not be independent.\n"
        + f"\nSelected candidates saved to: {OUTPUT}\n"
    )

    REPORT.write_text(report, encoding="utf-8")

    print("MERIT: Stratified M8 vs ML comparison")
    print("=" * 58)
    print("\nOverall outcomes")
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSelection-count audit")
    print(audit.to_string(index=False))
    print("\nOutcome breakdown by symbol and side")
    print(per_stratum.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {OUTPUT}")
    print(f"Saved: {REPORT}")


if __name__ == "__main__":
    main()
