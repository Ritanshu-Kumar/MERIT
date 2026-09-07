import csv
from datetime import date
from pathlib import Path

from merit.execution.queue_model import QueueModel
from merit.research.lobster_m8 import build_lobster_research_dataset


BASE = Path("data/sample")

MESSAGE_FILE = (
    BASE / "AAPL_2012-06-21_34200000_57600000_message_10.csv"
)

ORDERBOOK_FILE = (
    BASE / "AAPL_2012-06-21_34200000_57600000_orderbook_10.csv"
)

OUTPUT_FILE = BASE / "AAPL_m8_research_dataset.csv"

SYMBOL = "AAPL"
TRADING_DATE = date(2012, 6, 21)

QUEUE_AHEAD_FRACTION = 0.0
ORDER_QUANTITY = 100
LEVELS = 10


def write_dataset(path: Path, observations) -> None:
    if not observations:
        raise ValueError("No research observations were generated")

    fieldnames = list(observations[0].__dataclass_fields__.keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for observation in observations:
            writer.writerow(
                {
                    field: getattr(observation, field)
                    for field in fieldnames
                }
            )


def main() -> None:
    if not MESSAGE_FILE.exists():
        raise FileNotFoundError(MESSAGE_FILE)

    if not ORDERBOOK_FILE.exists():
        raise FileNotFoundError(ORDERBOOK_FILE)

    observations = build_lobster_research_dataset(
        
        message_path=MESSAGE_FILE,
        orderbook_path=ORDERBOOK_FILE,
        symbol=SYMBOL,
        trading_date=TRADING_DATE,
        queue_model=QueueModel(ahead_fraction=QUEUE_AHEAD_FRACTION),
        order_quantity=100,
        levels=10,
        max_observations=None,
    )

    print(
        f"Generated {len(observations):,} "
        "research observations."
    )

    write_dataset(
        OUTPUT_FILE,
        observations,
    )

    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()