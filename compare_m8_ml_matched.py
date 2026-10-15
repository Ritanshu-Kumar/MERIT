
from pathlib import Path

import pandas as pd


INPUT = Path("research/m9_m8_vs_ml_selection.csv")
OUTPUT = Path("research/m9_m8_ml_activity_matched.csv")
REPORT = Path("research/m9_m8_ml_activity_matched.txt")


def summarize(name, subset, total_test):
    counts = subset["outcome"].value_counts()
    n = len(subset)

    return {
        "policy": name,
        "selected_candidates": n,
        "test_selection_rate_pct": 100 * n / total_test,
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

    df["m8_selected"] = (
        df["m8_selected"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )

    if df["m8_selected"].isna().any():
        raise ValueError("Could not parse all m8_selected values.")

    df["ml_score"] = pd.to_numeric(df["ml_score"], errors="coerce")

    if df["ml_score"].isna().any():
        raise ValueError("ML scores contain missing or invalid values.")

    total_test = len(df)
    m8 = df[df["m8_selected"]].copy()
    target_count = len(m8)

    if target_count == 0:
        raise ValueError("M8 selected zero candidates.")

    # Retrospective ranking diagnostic:
    # match M8's test selection count using ML scores only.
    # Outcome labels are not used to rank or select candidates.
    ml_matched = (
        df.sort_values(
            ["ml_score", "timestamp_ns"],
            ascending=[False, True],
            kind="mergesort",
        )
        .head(target_count)
        .copy()
    )

    all_candidates = df.copy()

    summary = pd.DataFrame([
        summarize("All test candidates", all_candidates, total_test),
        summarize("Frozen M8", m8, total_test),
        summarize("ML, test-activity-matched", ml_matched, total_test),
    ])

    print("MERIT: M8 vs ML activity-matched diagnostic")
    print("=" * 58)
    print(f"Test candidates: {total_test:,}")
    print(f"M8-selected candidates: {target_count:,}")
    print()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Show whether the result is concentrated in particular symbols.
    detail_rows = []

    for symbol in sorted(df["symbol"].unique()):
        for policy, selected in [
            ("Frozen M8", m8),
            ("ML matched", ml_matched),
        ]:
            group = selected[selected["symbol"] == symbol]
            counts = group["outcome"].value_counts()
            n = len(group)

            detail_rows.append({
                "symbol": symbol,
                "policy": policy,
                "selected": n,
                "fill_rate_pct": 100 * counts.get("FILL", 0) / max(n, 1),
                "adverse_rate_pct": 100 * counts.get("ADVERSE", 0) / max(n, 1),
                "timeout_rate_pct": 100 * counts.get("TIMEOUT", 0) / max(n, 1),
            })

    by_symbol = pd.DataFrame(detail_rows)

    # Save selected candidate rows for later economic analysis.
    selected_rows = pd.concat([
        m8.assign(policy="Frozen M8"),
        ml_matched.assign(policy="ML activity-matched"),
    ], ignore_index=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    selected_rows.to_csv(OUTPUT, index=False)

    report = (
        "MERIT: M8 vs ML activity-matched diagnostic\n"
        + "=" * 58 + "\n"
        + f"Test candidates: {total_test:,}\n"
        + f"M8-selected candidates: {target_count:,}\n\n"
        + summary.to_string(index=False, float_format=lambda x: f"{x:.4f}")
        + "\n\nPer-symbol breakdown\n"
        + by_symbol.to_string(index=False, float_format=lambda x: f"{x:.4f}")
        + "\n\nLimitations:\n"
        + "- One trading date; results are exploratory.\n"
        + "- ML selection count is matched retrospectively on test scores.\n"
        + "- This is not a prospective threshold evaluation.\n"
        + "- Outcome rates are not net P&L; partial fills are not fully valued.\n"
        + "- Candidates may overlap in time and are not independent observations.\n"
        + f"\nSelected candidates saved to: {OUTPUT}\n"
    )

    REPORT.write_text(report, encoding="utf-8")
    print("\nPer-symbol breakdown")
    print(by_symbol.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {OUTPUT}")
    print(f"Saved: {REPORT}")


if __name__ == "__main__":
    main()
