
import pandas as pd
import numpy as np

d = pd.read_csv(
    "data/itch/M8_ITCH_2019-07-30_events_corrected.csv"
)
d = d.dropna(
    subset=["return_bps_1s", "selected", "symbol", "timestamp_ns"]
).copy()
d["selected"] = d["selected"].astype(bool)
d = d.sort_values(["symbol", "timestamp_ns"])
d["centered"] = (
    d["return_bps_1s"]
    - d.groupby("symbol")["return_bps_1s"].transform("mean")
)

rng = np.random.default_rng(20261012)
B = 1000
block = 500


def effect(parts):
    selected_returns = np.concatenate(
        [x[z] for x, z in parts]
    )
    unselected_returns = np.concatenate(
        [x[~z] for x, z in parts]
    )
    return selected_returns.mean() - unselected_returns.mean()


prepared = []
for _, g in d.groupby("symbol"):
    prepared.append((
        g["centered"].to_numpy(),
        g["selected"].to_numpy()
    ))

observed = effect(prepared)
simulations = []

for _ in range(B):
    parts = []
    for x, z in prepared:
        n = len(x)
        starts = rng.integers(
            0, n, size=int(np.ceil(n / block))
        )
        indices = np.concatenate([
            np.arange(start, start + block) % n
            for start in starts
        ])[:n]
        parts.append((x[indices], z[indices]))
    simulations.append(effect(parts))

simulations = np.array(simulations)

print("Rows:", len(d))
print("Block length:", block, "events")
print("Observed centered difference:", round(observed, 6), "bps")
print(
    "Block-bootstrap 95% interval:",
    np.round(np.quantile(simulations, [0.025, 0.975]), 6),
    "bps"
)
print(
    "Bootstrap share above zero:",
    round(np.mean(simulations > 0), 4)
)
print("Replications:", B)
