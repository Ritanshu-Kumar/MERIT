
from pathlib import Path
import pandas as pd

ROOT = Path("data")

print("MERIT DATA AUDIT")
print("=" * 80)

for path in sorted(ROOT.rglob("*")):
    if not path.is_file():
        continue

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"\nFILE: {path}")
    print(f"SIZE: {size_mb:.2f} MB")

    if path.suffix.lower() != ".csv":
        print("Inspection: non-CSV file; format needs separate identification.")
        continue

    try:
        df = pd.read_csv(path, nrows=3, low_memory=False)

        print(f"COLUMNS ({len(df.columns)}):")
        print(list(df.columns))
        print("\nSAMPLE ROWS:")
        print(df.to_string(index=False, max_cols=12))

    except Exception as exc:
        print(f"Could not inspect CSV: {exc}")

print("\nAudit complete. No data files were changed.")
