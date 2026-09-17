import pandas as pd
from datetime import date

from merit.execution.queue_model import QueueModel
from merit.research.lobster_m8 import build_lobster_research_dataset


BASE = "data/sample"

MESSAGE_FILE = (
    f"{BASE}/AAPL_2012-06-21_34200000_57600000_message_10.csv"
)

ORDERBOOK_FILE = (
    f"{BASE}/AAPL_2012-06-21_34200000_57600000_orderbook_10.csv"
)

TRADING_DATE = date(2012, 6, 21)

QUEUE_FRACTIONS = [
    0.0,
    0.25,
    0.50,
    0.75,
    1.0,
]

ORDER_QUANTITY = 100
LEVELS = 10


def summarize(
    observations,
    queue_fraction: float,
) -> dict[str, float]:
    if not observations:
        return {
            "queue_fraction": queue_fraction,
            "fills": 0,
            "mean_100ms": float("nan"),
            "mean_1s": float("nan"),
            "mean_5s": float("nan"),
            "negative_1s_pct": float("nan"),
        }

    df = pd.DataFrame(
        {
            "markout_100ms": [
                observation.markout_100ms
                for observation in observations
            ],
            "markout_1s": [
                observation.markout_1s
                for observation in observations
            ],
            "markout_5s": [
                observation.markout_5s
                for observation in observations
            ],
        }
    )

    return {
        "queue_fraction": queue_fraction,
        "fills": len(df),
        "mean_100ms": df["markout_100ms"].mean(),
        "mean_1s": df["markout_1s"].mean(),
        "mean_5s": df["markout_5s"].mean(),
        "negative_1s_pct": (
            (df["markout_1s"] < 0).mean() * 100
        ),
    }


def main() -> None:
    results = []

    for queue_fraction in QUEUE_FRACTIONS:
        observations = build_lobster_research_dataset(
            message_path=MESSAGE_FILE,
            orderbook_path=ORDERBOOK_FILE,
            symbol="AAPL",
            trading_date=TRADING_DATE,
            queue_model=QueueModel(
                ahead_fraction=queue_fraction,
            ),
            order_quantity=ORDER_QUANTITY,
            levels=LEVELS,
            max_observations=None,
        )

        results.append(
            summarize(
                observations,
                queue_fraction,
            )
        )

    result = pd.DataFrame(results)

    print(
        result.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )


if __name__ == "__main__":
    main()