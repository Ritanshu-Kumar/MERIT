from __future__ import annotations

import argparse
import csv
import heapq
import os
import tempfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from . import build_state_transitions as m9
else:
    import build_state_transitions as m9

TARGET_SYMBOLS = {
    "AAPL",
    "AMZN",
    "GOOG",
    "INTC",
    "MSFT",
}

QUOTE_QTY = 100
TIMEOUT_NS = 1_000_000_000
DEFAULT_SAMPLING_INTERVAL_NS = 1_000_000_000
TICK_SIZE_UNITS = 100

MARKOUT_HORIZONS_NS = {
    "10ms": 10_000_000,
    "100ms": 100_000_000,
    "500ms": 500_000_000,
    "1s": 1_000_000_000,
    "5s": 5_000_000_000,
}

BOOK_UPDATE_TYPES = frozenset({"A", "F", "D", "X", "E", "C", "U"})


def advance_sample_grid(
    next_due_ns: int,
    current_timestamp_ns: int,
    interval_ns: int,
) -> tuple[list[int], int]:
    """Return scheduled times strictly before the current event."""
    if interval_ns <= 0:
        raise ValueError("interval_ns must be positive")
    due_times = []
    while next_due_ns < current_timestamp_ns:
        due_times.append(next_due_ns)
        next_due_ns += interval_ns
    return due_times, next_due_ns


OUTPUT_COLUMNS = [
    "candidate_id",
    "event_id",
    "cluster_id",
    "cluster_size",
    "timestamp_ns",
    "snapshot_timestamp_ns",
    "snapshot_age_ns",
    "symbol",
    "stock_locate",
    "quote_side",
    "quote_price",
    "quote_qty",
    "mid",
    "spread",
    "relative_spread",
    "imbalance",
    "delta_imbalance",
    "delta_microprice",
    "delta_spread",
    "queue_ahead_initial",
    "queue_ahead_at_activation",
    "filled_qty",
    "fill_fraction",
    "time_to_resolution_ns",
    "resolution_mid",
    "resolution_mid_timestamp_ns",
    "resolution_mid_age_ns",
    "adverse_threshold",
    "level_resets",
    "outcome",
    "fill_before_adverse",
    "fill_timestamp_ns",
    "fill_resolution_message_index",
    "fill_resolution_type",
    "future_mid_10ms",
    "future_mid_100ms",
    "future_mid_500ms",
    "future_mid_1s",
    "future_mid_5s",
    "future_mid_10ms_timestamp_ns",
    "future_mid_100ms_timestamp_ns",
    "future_mid_500ms_timestamp_ns",
    "future_mid_1s_timestamp_ns",
    "future_mid_5s_timestamp_ns",
]


@dataclass
class LevelState:
    epoch: int = 0
    execution_volume: int = 0


@dataclass
class Candidate:
    candidate_id: int
    timestamp_ns: int
    symbol: str
    stock_locate: int

    quote_side: str
    quote_price: int
    quote_qty: int

    mid: float
    spread: float
    relative_spread: float
    imbalance: float
    delta_imbalance: float
    delta_microprice: float
    delta_spread: float

    queue_ahead_initial: int

    level_epoch: int
    start_execution_volume: int
    queue_ahead_current: int

    filled_qty: int
    expiry_ns: int
    adverse_threshold: float
    source_event_id: int | None = None
    queue_ahead_at_activation: int | None = None
    snapshot_timestamp_ns: int | None = None

    level_resets: int = 0
    status: str = "ACTIVE"
    resolved_timestamp_ns: int | None = None
    resolution_mid: float | None = None
    resolution_mid_timestamp_ns: int | None = None

    required_execution_volume: int = 0
    fill_timestamp_ns: int | None = None
    fill_resolution_message_index: int | None = None
    fill_resolution_type: str | None = None
    future_mids: dict[str, float | None] | None = None
    future_mid_timestamps: dict[str, int | None] | None = None
    markout_resolved: dict[str, bool] | None = None

    def __post_init__(self) -> None:
        if self.future_mids is None:
            self.future_mids = {
                horizon: None
                for horizon in MARKOUT_HORIZONS_NS
            }
        if self.future_mid_timestamps is None:
            self.future_mid_timestamps = {
                horizon: None
                for horizon in MARKOUT_HORIZONS_NS
            }
        if self.markout_resolved is None:
            self.markout_resolved = {
                horizon: False
                for horizon in MARKOUT_HORIZONS_NS
            }


class HazardEngine:
    def __init__(self) -> None:
        self.next_id = 0
        # Latest post-event midpoint per stock, with its exchange timestamp.
        # Timeout valuation must never borrow another symbol's midpoint or
        # use an observation from after the candidate's expiry.
        self.latest_mid_by_locate: dict[int, tuple[int, float]] = {}

        self.candidates: dict[
            int,
            Candidate,
        ] = {}

        self.levels: dict[
            tuple[int, str, int],
            LevelState,
        ] = defaultdict(LevelState)

        self.active_by_level: dict[
            tuple[int, str, int],
            set[int],
        ] = defaultdict(set)

        self.fill_heaps: dict[
            tuple[int, str, int],
            list[tuple[int, int]],
        ] = defaultdict(list)

        self.buy_adverse_heap: dict[
            int,
            list[tuple[float, int]],
        ] = defaultdict(list)

        self.sell_adverse_heap: dict[
            int,
            list[tuple[float, int]],
        ] = defaultdict(list)

        self.expiry_heap: list[
            tuple[int, int]
        ] = []

        self.completed: deque[Candidate] = deque()
        self.pending_fill_marks: dict[int, deque[Candidate]] = defaultdict(deque)

    def level_key(
        self,
        stock_locate: int,
        resting_side: str,
        price: int,
    ) -> tuple[int, str, int]:
        return (
            stock_locate,
            resting_side,
            price,
        )

    def level(
        self,
        key: tuple[int, str, int],
    ) -> LevelState:
        return self.levels[key]

    def _remove_active(
        self,
        candidate: Candidate,
    ) -> None:
        key = self.level_key(
            candidate.stock_locate,
            "B"
            if candidate.quote_side == "BUY"
            else "S",
            candidate.quote_price,
        )

        self.active_by_level[key].discard(
            candidate.candidate_id
        )

    def _resolve_fill(
        self,
        candidate: Candidate,
        timestamp_ns: int,
        resolution_mid: float | None,
        fill_resolution_message_index: int | None = None,
        fill_resolution_type: str = "unknown",
    ) -> None:
        if candidate.status != "ACTIVE":
            return

        candidate.filled_qty = candidate.quote_qty
        candidate.fill_resolution_message_index = (
            fill_resolution_message_index
        )
        candidate.fill_resolution_type = fill_resolution_type
        candidate.status = "FILL"
        candidate.resolved_timestamp_ns = (
            timestamp_ns
        )
        candidate.resolution_mid = resolution_mid
        candidate.resolution_mid_timestamp_ns = (
            timestamp_ns if resolution_mid is not None else None
        )

        self._remove_active(candidate)
        candidate.fill_timestamp_ns = timestamp_ns
        if resolution_mid is not None:
            for horizon_name in MARKOUT_HORIZONS_NS:
                candidate.future_mids[horizon_name] = resolution_mid
                candidate.future_mid_timestamps[horizon_name] = timestamp_ns
        self.pending_fill_marks[candidate.stock_locate].append(candidate)

    def _partial_fill_at_current_epoch(
        self,
        candidate: Candidate,
        level: LevelState,
    ) -> int:
        consumed = (
            level.execution_volume
            - candidate.start_execution_volume
            - candidate.queue_ahead_current
        )

        if consumed <= 0:
            return 0

        remaining = (
            candidate.quote_qty
            - candidate.filled_qty
        )

        return max(
            0,
            min(
                remaining,
                consumed,
            ),
        )

    def create_candidate(
        self,
        timestamp_ns: int,
        symbol: str,
        stock_locate: int,
        quote_side: str,
        quote_price: int,
        quote_qty: int,
        state: m9.BookState,
        delta_imbalance: float,
        delta_microprice: float,
        delta_spread: float,
        pre_level_epoch: int,
        source_event_id: int | None = None,
        activation_queue_ahead: int | None = None,
        snapshot_timestamp_ns: int | None = None,
    ) -> None:
        resting_side = (
            "B"
            if quote_side == "BUY"
            else "S"
        )

        key = self.level_key(
            stock_locate,
            resting_side,
            quote_price,
        )

        level = self.level(key)

        if quote_side == "BUY":
            pre_queue_ahead = state.bid_size
            adverse_threshold = (
                state.mid
                - TICK_SIZE_UNITS / 10000.0
            )
        else:
            pre_queue_ahead = state.ask_size
            adverse_threshold = (
                state.mid
                + TICK_SIZE_UNITS / 10000.0
            )

        # If the price level disappeared during the triggering
        # message, the old queue has already been consumed/cancelled.
        # The hypothetical order is therefore re-anchored at the
        # front of the newly established epoch.
        same_epoch = (
            pre_level_epoch
            == level.epoch
        )

        queue_ahead = (
            pre_queue_ahead
            if same_epoch
            else 0
        )

        if activation_queue_ahead is not None:
            if activation_queue_ahead < 0:
                raise ValueError(
                    "activation_queue_ahead must be non-negative"
                )
            queue_ahead = activation_queue_ahead

        start_execution_volume = (
            level.execution_volume
        )

        required_execution_volume = (
            start_execution_volume
            + queue_ahead
            + quote_qty
        )

        candidate = Candidate(
            candidate_id=self.next_id,
            timestamp_ns=timestamp_ns,
            symbol=symbol,
            stock_locate=stock_locate,
            quote_side=quote_side,
            quote_price=quote_price,
            quote_qty=quote_qty,
            mid=state.mid,
            spread=state.spread,
            relative_spread=state.relative_spread,
            imbalance=state.imbalance,
            delta_imbalance=delta_imbalance,
            delta_microprice=delta_microprice,
            delta_spread=delta_spread,
            queue_ahead_initial=pre_queue_ahead,
            level_epoch=level.epoch,
            start_execution_volume=(
                start_execution_volume
            ),
            queue_ahead_current=queue_ahead,
            filled_qty=0,
            expiry_ns=(
                timestamp_ns
                + TIMEOUT_NS
            ),
            adverse_threshold=adverse_threshold,
            source_event_id=source_event_id,
            required_execution_volume=(
                required_execution_volume
            ),
            queue_ahead_at_activation=queue_ahead,
            snapshot_timestamp_ns=snapshot_timestamp_ns,
        )

        candidate_id = candidate.candidate_id

        self.candidates[candidate_id] = candidate
        self.active_by_level[key].add(
            candidate_id
        )

        heapq.heappush(
            self.fill_heaps[key],
            (
                required_execution_volume,
                candidate_id,
            ),
        )

        if quote_side == "BUY":
            heapq.heappush(
                self.buy_adverse_heap[
                    stock_locate
                ],
                (
                    -adverse_threshold,
                    candidate_id,
                ),
            )
        else:
            heapq.heappush(
                self.sell_adverse_heap[
                    stock_locate
                ],
                (
                    adverse_threshold,
                    candidate_id,
                ),
            )

        heapq.heappush(
            self.expiry_heap,
            (
                candidate.expiry_ns,
                candidate_id,
            ),
        )

        self.next_id += 1

    def register_execution(
        self,
        stock_locate: int,
        resting_side: str,
        price: int,
        shares: int,
        timestamp_ns: int,
        current_mid: float | None,
        message_index: int | None = None,
    ) -> None:
        key = self.level_key(
            stock_locate,
            resting_side,
            price,
        )

        level = self.level(key)
        level.execution_volume += shares

        cumulative = level.execution_volume

        heap = self.fill_heaps[key]

        while heap:
            required, candidate_id = heap[0]

            if required > cumulative:
                break

            heapq.heappop(heap)

            candidate = self.candidates.get(
                candidate_id
            )

            if (
                candidate is None
                or candidate.status != "ACTIVE"
            ):
                continue

            if (
                candidate.level_epoch
                != level.epoch
            ):
                continue

            self._resolve_fill(
                candidate,
                timestamp_ns,
                current_mid,
                fill_resolution_message_index=message_index,
                fill_resolution_type="execution",
            )

    def register_cancellation(
        self,
        stock_locate: int,
        resting_side: str,
        price: int,
        cancelled_shares: int,
        order_added_message_index: int | None,
    ) -> None:
        """
        Reduce the simulated queue threshold when shares are
        cancelled from an order that was ahead at activation.

        Shares cancelled from orders added after the candidate's
        activation do not reduce its queue-ahead estimate.
        """
        if cancelled_shares <= 0:
            return

        if order_added_message_index is None:
            return

        key = self.level_key(
            stock_locate,
            resting_side,
            price,
        )

        level = self.level(key)
        heap = self.fill_heaps[key]

        for candidate_id in list(self.active_by_level[key]):
            candidate = self.candidates.get(candidate_id)

            if candidate is None or candidate.status != "ACTIVE":
                continue

            if candidate.level_epoch != level.epoch:
                continue

            # Only orders already present when the hypothetical
            # candidate was activated can have been ahead of it.
            if candidate.source_event_id is None:
                continue

            if order_added_message_index >= candidate.source_event_id:
                continue

            executed_since_start = max(
                0,
                level.execution_volume
                - candidate.start_execution_volume,
            )

            remaining_queue_ahead = max(
                0,
                candidate.queue_ahead_current
                - executed_since_start,
            )

            # Do not count cancellations of historical shares
            # that would already have been behind our simulated order.
            queue_reduction = min(
                cancelled_shares,
                remaining_queue_ahead,
            )

            if queue_reduction <= 0:
                continue

            candidate.queue_ahead_current -= queue_reduction

            candidate.required_execution_volume = (
                candidate.start_execution_volume
                + candidate.queue_ahead_current
                + candidate.quote_qty
                - candidate.filled_qty
            )

            # Push the updated threshold. The old heap entry may
            # remain, but status checks make stale entries harmless.
            heapq.heappush(
                heap,
                (
                    candidate.required_execution_volume,
                    candidate.candidate_id,
                ),
            )

    def handle_level_reset(
        self,
        stock_locate: int,
        resting_side: str,
        price: int,
        timestamp_ns: int,
        current_mid: float | None,
        message_index: int | None = None,
    ) -> None:
        key = self.level_key(
            stock_locate,
            resting_side,
            price,
        )

        level = self.level(key)

        old_execution_volume = (
            level.execution_volume
        )

        level.epoch += 1
        level.execution_volume = 0

        active_ids = list(
            self.active_by_level[key]
        )

        new_heap: list[
            tuple[int, int]
        ] = []

        for candidate_id in active_ids:
            candidate = self.candidates.get(
                candidate_id
            )

            if (
                candidate is None
                or candidate.status != "ACTIVE"
            ):
                continue

            prior_consumed = (
                old_execution_volume
                - candidate.start_execution_volume
                - candidate.queue_ahead_current
            )

            if prior_consumed > 0:
                remaining = (
                    candidate.quote_qty
                    - candidate.filled_qty
                )

                partial = min(
                    remaining,
                    max(0, prior_consumed),
                )

                candidate.filled_qty += (
                    partial
                )

            if (
                candidate.filled_qty
                >= candidate.quote_qty
            ):
                self._resolve_fill(
                    candidate,
                    timestamp_ns,
                    current_mid,
                    fill_resolution_message_index=message_index,
                    fill_resolution_type="level_reset",
                )
                continue

            candidate.level_epoch = level.epoch
            candidate.start_execution_volume = 0
            candidate.queue_ahead_current = 0
            candidate.level_resets += 1

            remaining = (
                candidate.quote_qty
                - candidate.filled_qty
            )

            candidate.required_execution_volume = (
                remaining
            )

            heapq.heappush(
                new_heap,
                (
                    remaining,
                    candidate_id,
                ),
            )

        self.fill_heaps[key] = new_heap

    def resolve_adverse(
        self,
        stock_locate: int,
        current_mid: float | None,
        timestamp_ns: int,
    ) -> None:
        if current_mid is None:
            return

        buy_heap = self.buy_adverse_heap[
            stock_locate
        ]

        while buy_heap:
            negative_threshold, candidate_id = (
                buy_heap[0]
            )

            threshold = -negative_threshold

            if current_mid > threshold:
                break

            heapq.heappop(
                buy_heap
            )

            candidate = self.candidates.get(
                candidate_id
            )

            if (
                candidate is None
                or candidate.status != "ACTIVE"
            ):
                continue

            key = self.level_key(
                stock_locate,
                "B",
                candidate.quote_price,
            )

            level = self.level(key)

            partial = (
                self._partial_fill_at_current_epoch(
                    candidate,
                    level,
                )
            )

            candidate.filled_qty += partial

            candidate.status = "ADVERSE"
            candidate.resolved_timestamp_ns = (
                timestamp_ns
            )
            candidate.resolution_mid = current_mid
            candidate.resolution_mid_timestamp_ns = (
                timestamp_ns if current_mid is not None else None
            )

            self._remove_active(candidate)
            self.completed.append(candidate)

        sell_heap = self.sell_adverse_heap[
            stock_locate
        ]

        while sell_heap:
            threshold, candidate_id = (
                sell_heap[0]
            )

            if current_mid < threshold:
                break

            heapq.heappop(
                sell_heap
            )

            candidate = self.candidates.get(
                candidate_id
            )

            if (
                candidate is None
                or candidate.status != "ACTIVE"
            ):
                continue

            key = self.level_key(
                stock_locate,
                "S",
                candidate.quote_price,
            )

            level = self.level(key)

            partial = (
                self._partial_fill_at_current_epoch(
                    candidate,
                    level,
                )
            )

            candidate.filled_qty += partial

            candidate.status = "ADVERSE"
            candidate.resolved_timestamp_ns = (
                timestamp_ns
            )
            candidate.resolution_mid = current_mid
            candidate.resolution_mid_timestamp_ns = (
                timestamp_ns if current_mid is not None else None
            )

            self._remove_active(candidate)
            self.completed.append(candidate)

    def resolve_timeouts_before(
        self,
        timestamp_ns: int,
        current_mid: float | None = None,
    ) -> None:
        while self.expiry_heap:
            expiry, candidate_id = (
                self.expiry_heap[0]
            )

            if expiry >= timestamp_ns:
                break

            heapq.heappop(
                self.expiry_heap
            )

            candidate = self.candidates.get(
                candidate_id
            )

            if (
                candidate is None
                or candidate.status != "ACTIVE"
            ):
                continue

            key = self.level_key(
                candidate.stock_locate,
                "B"
                if candidate.quote_side == "BUY"
                else "S",
                candidate.quote_price,
            )

            level = self.level(key)

            partial = (
                self._partial_fill_at_current_epoch(
                    candidate,
                    level,
                )
            )

            candidate.filled_qty += partial
            candidate.status = "TIMEOUT"
            candidate.resolved_timestamp_ns = (
                candidate.expiry_ns
            )
            cached_mid = self.latest_mid_by_locate.get(
                candidate.stock_locate
            )
            if (
                cached_mid is not None
                and cached_mid[0] <= candidate.expiry_ns
            ):
                (
                    candidate.resolution_mid_timestamp_ns,
                    candidate.resolution_mid,
                ) = cached_mid
            else:
                candidate.resolution_mid_timestamp_ns = None
                candidate.resolution_mid = None

            self._remove_active(candidate)
            self.completed.append(candidate)

    def resolve_timeouts_at_or_before(
        self,
        timestamp_ns: int,
        current_mid: float | None = None,
    ) -> None:
        while self.expiry_heap:
            expiry, candidate_id = (
                self.expiry_heap[0]
            )

            if expiry > timestamp_ns:
                break

            heapq.heappop(
                self.expiry_heap
            )

            candidate = self.candidates.get(
                candidate_id
            )

            if (
                candidate is None
                or candidate.status != "ACTIVE"
            ):
                continue

            key = self.level_key(
                candidate.stock_locate,
                "B"
                if candidate.quote_side == "BUY"
                else "S",
                candidate.quote_price,
            )

            level = self.level(key)

            partial = (
                self._partial_fill_at_current_epoch(
                    candidate,
                    level,
                )
            )

            candidate.filled_qty += partial
            candidate.status = "TIMEOUT"
            candidate.resolved_timestamp_ns = (
                candidate.expiry_ns
            )
            cached_mid = self.latest_mid_by_locate.get(
                candidate.stock_locate
            )
            if (
                cached_mid is not None
                and cached_mid[0] <= candidate.expiry_ns
            ):
                (
                    candidate.resolution_mid_timestamp_ns,
                    candidate.resolution_mid,
                ) = cached_mid
            else:
                candidate.resolution_mid_timestamp_ns = None
                candidate.resolution_mid = None

            self._remove_active(candidate)
            self.completed.append(candidate)

    def resolve_post_fill_markouts(
        self,
        stock_locate: int,
        timestamp_ns: int,
        current_mid: float | None,
    ) -> None:
        """
        Resolve post-fill midpoint markouts using as-of semantics.

        For each horizon T = fill time + horizon:
          - At or before T, update the provisional midpoint.
          - After T, freeze the last midpoint observed at or before T.
          - Never use a midpoint from after T for that horizon.

        The caller must process events in exchange-timestamp order.
        Events sharing the target timestamp may update the provisional mark
        until the stream advances beyond that timestamp.
        """
        pending = self.pending_fill_marks[stock_locate]
        remaining: deque[Candidate] = deque()

        while pending:
            candidate = pending.popleft()

            if candidate.fill_timestamp_ns is None:
                remaining.append(candidate)
                continue

            for horizon_name, horizon_ns in MARKOUT_HORIZONS_NS.items():
                if candidate.markout_resolved[horizon_name]:
                    continue
                target_ns = candidate.fill_timestamp_ns + horizon_ns
                if timestamp_ns <= target_ns:
                    if current_mid is not None:
                        candidate.future_mids[horizon_name] = current_mid
                        candidate.future_mid_timestamps[horizon_name] = (
                            timestamp_ns
                        )
                else:
                    candidate.markout_resolved[horizon_name] = True

            if all(candidate.markout_resolved.values()):
                self.completed.append(candidate)
            else:
                remaining.append(candidate)

        self.pending_fill_marks[stock_locate] = remaining

    def finalize_pending_markouts(self, last_timestamp_ns: int) -> None:
        """
        Finalize horizons reached by the available feed.

        A horizon beyond the end of the feed remains unresolved and must not
        be assigned a fabricated future midpoint.
        """
        for stock_locate, pending in list(self.pending_fill_marks.items()):
            remaining: deque[Candidate] = deque()
            while pending:
                candidate = pending.popleft()
                if candidate.fill_timestamp_ns is None:
                    remaining.append(candidate)
                    continue
                for horizon_name, horizon_ns in MARKOUT_HORIZONS_NS.items():
                    target_ns = candidate.fill_timestamp_ns + horizon_ns
                    if (
                        not candidate.markout_resolved[horizon_name]
                        and last_timestamp_ns >= target_ns
                    ):
                        candidate.markout_resolved[horizon_name] = True
                if all(candidate.markout_resolved.values()):
                    self.completed.append(candidate)
                else:
                    remaining.append(candidate)
            self.pending_fill_marks[stock_locate] = remaining

    def pop_completed(
        self,
    ) -> list[Candidate]:
        completed = list(
            self.completed
        )

        self.completed.clear()

        for candidate in completed:
            self.candidates.pop(
                candidate.candidate_id,
                None,
            )

        return completed


def cancel_order(
    book: m9.OrderBook,
    order_ref: int,
    shares: int,
) -> m9.Order | None:
    order = book.orders.get(order_ref)

    if order is None:
        return None

    cancelled = min(
        shares,
        order.shares,
    )

    levels = (
        book.bids
        if order.side == "B"
        else book.asks
    )

    levels[order.price] -= cancelled

    if levels[order.price] <= 0:
        del levels[order.price]

    order.shares -= cancelled

    if order.shares <= 0:
        del book.orders[order_ref]

    return m9.Order(
        order_ref=order_ref,
        side=order.side,
        shares=cancelled,
        price=order.price,
    )


def candidate_to_row(
    candidate: Candidate,
) -> dict[str, object]:
    assert (
        candidate.resolved_timestamp_ns
        is not None
    )

    return {
        "candidate_id": candidate.candidate_id,
        "event_id": candidate.source_event_id,
        "cluster_id": f"{candidate.symbol}:{candidate.timestamp_ns}",
        "cluster_size": None,
        "timestamp_ns": candidate.timestamp_ns,
        "snapshot_timestamp_ns": candidate.snapshot_timestamp_ns,
        "snapshot_age_ns": (
            candidate.timestamp_ns - candidate.snapshot_timestamp_ns
            if candidate.snapshot_timestamp_ns is not None
            else None
        ),
        "symbol": candidate.symbol,
        "stock_locate": candidate.stock_locate,
        "quote_side": candidate.quote_side,
        "quote_price": (
            candidate.quote_price
            / 10000.0
        ),
        "quote_qty": candidate.quote_qty,
        "mid": candidate.mid,
        "spread": candidate.spread,
        "relative_spread": (
            candidate.relative_spread
        ),
        "imbalance": candidate.imbalance,
        "delta_imbalance": (
            candidate.delta_imbalance
        ),
        "delta_microprice": (
            candidate.delta_microprice
        ),
        "delta_spread": candidate.delta_spread,
        "queue_ahead_initial": (
            candidate.queue_ahead_initial
        ),
        "queue_ahead_at_activation": (
            candidate.queue_ahead_at_activation
        ),
        "filled_qty": candidate.filled_qty,
        "fill_fraction": (
            candidate.filled_qty
            / candidate.quote_qty
        ),
        "time_to_resolution_ns": (
            candidate.resolved_timestamp_ns
            - candidate.timestamp_ns
        ),
        "resolution_mid": candidate.resolution_mid,
        "resolution_mid_timestamp_ns": (
            candidate.resolution_mid_timestamp_ns
        ),
        "resolution_mid_age_ns": (
            candidate.resolved_timestamp_ns
            - candidate.resolution_mid_timestamp_ns
            if candidate.resolved_timestamp_ns is not None
            and candidate.resolution_mid_timestamp_ns is not None
            else None
        ),
        "adverse_threshold": (
            candidate.adverse_threshold
        ),
        "level_resets": candidate.level_resets,
        "outcome": candidate.status,
        "fill_before_adverse": (
            candidate.status == "FILL"
        ),
        "fill_timestamp_ns": candidate.fill_timestamp_ns,
        "fill_resolution_message_index": (
            candidate.fill_resolution_message_index
        ),
        "fill_resolution_type": candidate.fill_resolution_type,
        "future_mid_10ms": candidate.future_mids["10ms"],
        "future_mid_100ms": candidate.future_mids["100ms"],
        "future_mid_500ms": candidate.future_mids["500ms"],
        "future_mid_1s": candidate.future_mids["1s"],
        "future_mid_5s": candidate.future_mids["5s"],
        "future_mid_10ms_timestamp_ns": candidate.future_mid_timestamps["10ms"],
        "future_mid_100ms_timestamp_ns": candidate.future_mid_timestamps["100ms"],
        "future_mid_500ms_timestamp_ns": candidate.future_mid_timestamps["500ms"],
        "future_mid_1s_timestamp_ns": candidate.future_mid_timestamps["1s"],
        "future_mid_5s_timestamp_ns": candidate.future_mid_timestamps["5s"],
    }


def write_completed(
    writer: csv.DictWriter,
    engine: HazardEngine,
) -> int:
    completed = engine.pop_completed()

    for candidate in completed:
        writer.writerow(
            candidate_to_row(candidate)
        )

    return len(completed)


def finalize_cluster_metadata(output_path: Path) -> None:
    """
    Populate cluster_size using rows actually present in the final CSV.

    Clusters are defined by the existing cluster_id field: symbol plus
    decision timestamp. A temporary file is written first so the original
    CSV is not replaced unless the rewrite completes successfully.
    """
    counts: Counter[str] = Counter()
    with output_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError("Dataset has no CSV header.")
        required = {
            "candidate_id",
            "event_id",
            "cluster_id",
            "cluster_size",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Dataset is missing required columns: {sorted(missing)}"
            )
        for row in reader:
            cluster_id = row["cluster_id"]
            if not cluster_id:
                raise ValueError("Dataset contains a missing cluster_id.")
            counts[cluster_id] += 1
        fieldnames = reader.fieldnames

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temp_path = Path(temporary.name)
            writer = csv.DictWriter(
                temporary,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            with output_path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as source:
                reader = csv.DictReader(source)
                for row in reader:
                    row["cluster_size"] = str(counts[row["cluster_id"]])
                    writer.writerow(row)
        os.replace(temp_path, output_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def run(
    input_path: Path,
    output_path: Path,
    max_messages: int | None,
    sampling_mode: str = "execution",
    sampling_interval_ns: int = DEFAULT_SAMPLING_INTERVAL_NS,
) -> None:
    if sampling_mode not in {"execution", "time"}:
        raise ValueError(
            "sampling_mode must be 'execution' or 'time'"
        )
    if sampling_interval_ns <= 0:
        raise ValueError("sampling_interval_ns must be positive")

    locate_to_symbol = (
        m9.discover_stock_locates(
            input_path,
            TARGET_SYMBOLS,
        )
    )

    print("Target stock locates:")

    for locate, symbol in sorted(
        locate_to_symbol.items()
    ):
        print(
            f"  {symbol}: {locate}"
        )

    missing = (
        TARGET_SYMBOLS
        - set(locate_to_symbol.values())
    )

    if missing:
        raise RuntimeError(
            f"Missing target locates: "
            f"{sorted(missing)}"
        )

    books: dict[
        int,
        m9.OrderBook,
    ] = defaultdict(
        m9.OrderBook
    )

    previous_execution_state: dict[
        int,
        m9.BookState,
    ] = {}
    previous_sample_state: dict[
        int,
        m9.BookState,
    ] = {}
    next_sample_ns: dict[int, int] = {}
    latest_book_timestamp_ns: dict[int, int] = {}
    # Track the original ITCH add-message index for each resting order.
    order_add_message_index: dict[
        tuple[int, int],
        int,
    ] = {}

    engine = HazardEngine()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    message_count = 0
    execution_count = 0
    candidate_count = 0
    completed_count = 0
    last_timestamp_ns: int | None = None

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=OUTPUT_COLUMNS,
        )

        writer.writeheader()

        for msg in m9.iter_itch_messages(
            input_path,
            max_messages=max_messages,
        ):
            message_count += 1

            if not msg:
                continue

            message_type = chr(msg[0])

            if message_type == "R":
                continue

            stock_locate = (
                m9.parse_stock_locate(msg)
            )

            if stock_locate not in locate_to_symbol:
                continue

            timestamp_ns = (
                m9.parse_timestamp_ns(msg)
            )

            symbol = locate_to_symbol[
                stock_locate
            ]

            book = books[stock_locate]

            last_timestamp_ns = timestamp_ns

            pre_state = book.snapshot()
            pre_snapshot_timestamp_ns = latest_book_timestamp_ns.get(
                stock_locate
            )

            # Save the exact pre-event best quote and level epochs.
            pre_quotes: dict[
                str,
                tuple[int, int],
            ] = {}

            if pre_state is not None:
                best_bid = book.best_bid()
                best_ask = book.best_ask()

                if best_bid is not None:
                    key = engine.level_key(
                        stock_locate,
                        "B",
                        best_bid,
                    )

                    pre_quotes["BUY"] = (
                        best_bid,
                        engine.level(key).epoch,
                    )

                if best_ask is not None:
                    key = engine.level_key(
                        stock_locate,
                        "S",
                        best_ask,
                    )

                    pre_quotes["SELL"] = (
                        best_ask,
                        engine.level(key).epoch,
                    )

            # Time-mode candidates use the prior available book state.
            # Their timestamp records the logical grid decision time.
            if (
                sampling_mode == "time"
                and pre_state is not None
                and pre_snapshot_timestamp_ns is not None
            ):
                if stock_locate not in next_sample_ns:
                    next_sample_ns[stock_locate] = (
                        pre_snapshot_timestamp_ns
                        + sampling_interval_ns
                    )
                due_times, next_due = advance_sample_grid(
                    next_sample_ns[stock_locate],
                    timestamp_ns,
                    sampling_interval_ns,
                )
                next_sample_ns[stock_locate] = next_due
                for decision_ns in due_times:
                    previous = previous_sample_state.get(stock_locate)
                    if previous is None:
                        delta_imbalance = 0.0
                        delta_microprice = 0.0
                        delta_spread = 0.0
                    else:
                        delta_imbalance = (
                            pre_state.imbalance - previous.imbalance
                        )
                        delta_microprice = (
                            pre_state.microprice - previous.microprice
                        )
                        delta_spread = pre_state.spread - previous.spread

                    if "BUY" in pre_quotes:
                        bid_price, bid_epoch = pre_quotes["BUY"]
                        engine.create_candidate(
                            timestamp_ns=decision_ns,
                            symbol=symbol,
                            stock_locate=stock_locate,
                            quote_side="BUY",
                            quote_price=bid_price,
                            quote_qty=QUOTE_QTY,
                            state=pre_state,
                            delta_imbalance=delta_imbalance,
                            delta_microprice=delta_microprice,
                            delta_spread=delta_spread,
                            pre_level_epoch=bid_epoch,
                            source_event_id=message_count,
                            activation_queue_ahead=book.bids.get(
                                bid_price,
                                0,
                            ),
                            snapshot_timestamp_ns=pre_snapshot_timestamp_ns,
                        )
                        candidate_count += 1

                    if "SELL" in pre_quotes:
                        ask_price, ask_epoch = pre_quotes["SELL"]
                        engine.create_candidate(
                            timestamp_ns=decision_ns,
                            symbol=symbol,
                            stock_locate=stock_locate,
                            quote_side="SELL",
                            quote_price=ask_price,
                            quote_qty=QUOTE_QTY,
                            state=pre_state,
                            delta_imbalance=delta_imbalance,
                            delta_microprice=delta_microprice,
                            delta_spread=delta_spread,
                            pre_level_epoch=ask_epoch,
                            source_event_id=message_count,
                            activation_queue_ahead=book.asks.get(
                                ask_price,
                                0,
                            ),
                            snapshot_timestamp_ns=pre_snapshot_timestamp_ns,
                        )
                        candidate_count += 1

                    previous_sample_state[stock_locate] = pre_state

            # Candidates whose exact one-second horizon has
            # already passed cannot interact with this event.
            engine.resolve_timeouts_before(
                timestamp_ns,
                current_mid=(
                    pre_state.mid
                    if pre_state is not None
                    else None
                ),
            )

            completed_count += write_completed(
                writer,
                engine,
            )

            # Capture the order involved in the current message
            # before mutating the book.
            resting_order: m9.Order | None = None
            execution_shares = 0

            if message_type in {"E", "C"}:
                (
                    order_ref,
                    execution_shares,
                    _,
                ) = m9.parse_execution(msg)

                resting_order = book.orders.get(
                    order_ref
                )

            # Prices whose displayed depth may change.
            affected_levels: set[
                tuple[str, int]
            ] = set()

            if message_type in {"A", "F"}:
                (
                    _,
                    side,
                    _,
                    price,
                ) = m9.parse_add(msg)

                if side in {"B", "S"}:
                    affected_levels.add(
                        (side, price)
                    )

            elif message_type == "D" or message_type == "X":
                order_ref = (
                    m9.parse_order_ref(msg)
                )

                existing = book.orders.get(
                    order_ref
                )

                if existing is not None:
                    affected_levels.add(
                        (
                            existing.side,
                            existing.price,
                        )
                    )

            elif message_type in {"E", "C"}:
                if resting_order is not None:
                    affected_levels.add(
                        (
                            resting_order.side,
                            resting_order.price,
                        )
                    )

            elif message_type == "U":
                (
                    original_ref,
                    new_ref,
                    shares,
                    price,
                ) = m9.parse_replace(msg)

                existing = book.orders.get(
                    original_ref
                )

                if existing is not None:
                    affected_levels.add(
                        (
                            existing.side,
                            existing.price,
                        )
                    )

                    affected_levels.add(
                        (
                            existing.side,
                            price,
                        )
                    )

            before_depth: dict[
                tuple[str, int],
                int,
            ] = {}

            for side, price in affected_levels:
                levels = (
                    book.bids
                    if side == "B"
                    else book.asks
                )

                before_depth[
                    (side, price)
                ] = levels.get(price, 0)

            # -------------------------------------------------------
            # Apply the actual ITCH message.
            # -------------------------------------------------------
            if message_type in {"A", "F"}:
                (
                    order_ref,
                    side,
                    shares,
                    price,
                ) = m9.parse_add(msg)

                if (
                    side in {"B", "S"}
                    and shares > 0
                ):
                    book.add_order(
                        order_ref=order_ref,
                        side=side,
                        shares=shares,
                        price=price,
                    )
                    order_add_message_index[
                        (stock_locate, order_ref)
                    ] = message_count

            elif message_type == "D":
                order_ref = m9.parse_order_ref(msg)
                order_key = (stock_locate, order_ref)
                existing = book.orders.get(order_ref)
                if existing is not None:
                    engine.register_cancellation(
                        stock_locate=stock_locate,
                        resting_side=existing.side,
                        price=existing.price,
                        cancelled_shares=existing.shares,
                        order_added_message_index=(
                            order_add_message_index.get(order_key)
                        ),
                    )
                removed = book.remove_order(order_ref)
                if removed is not None:
                    order_add_message_index.pop(order_key, None)

            elif message_type == "X":
                order_ref = m9.parse_order_ref(msg)
                order_key = (stock_locate, order_ref)
                cancelled_shares = int.from_bytes(
                    msg[19:23],
                    "big",
                )

                existing = book.orders.get(order_ref)
                if existing is not None:
                    actual_cancelled = min(
                        cancelled_shares,
                        existing.shares,
                    )
                    engine.register_cancellation(
                        stock_locate=stock_locate,
                        resting_side=existing.side,
                        price=existing.price,
                        cancelled_shares=actual_cancelled,
                        order_added_message_index=(
                            order_add_message_index.get(order_key)
                        ),
                    )

                cancel_order(
                    book,
                    order_ref,
                    cancelled_shares,
                )
                if order_ref not in book.orders:
                    order_add_message_index.pop(order_key, None)

            elif message_type in {"E", "C"}:
                order_ref = int.from_bytes(
                    msg[11:19],
                    "big",
                )

                executed_shares = int.from_bytes(
                    msg[19:23],
                    "big",
                )

                book.execute_order(
                    order_ref,
                    executed_shares,
                )

            elif message_type == "U":
                (
                    original_ref,
                    new_ref,
                    shares,
                    price,
                ) = m9.parse_replace(msg)

                original_key = (stock_locate, original_ref)
                new_key = (stock_locate, new_ref)
                existing = book.orders.get(original_ref)
                if existing is not None:
                    engine.register_cancellation(
                        stock_locate=stock_locate,
                        resting_side=existing.side,
                        price=existing.price,
                        cancelled_shares=existing.shares,
                        order_added_message_index=(
                            order_add_message_index.get(original_key)
                        ),
                    )
                replaced = book.replace_order(
                    original_ref,
                    new_ref,
                    shares,
                    price,
                )
                if replaced is not None:
                    order_add_message_index.pop(original_key, None)
                    order_add_message_index[new_key] = message_count

            post_state = book.snapshot()

            if post_state is not None:
                engine.latest_mid_by_locate[stock_locate] = (
                    timestamp_ns,
                    post_state.mid,
                )
                if message_type in BOOK_UPDATE_TYPES:
                    latest_book_timestamp_ns[stock_locate] = timestamp_ns
                    if (
                        sampling_mode == "time"
                        and stock_locate not in next_sample_ns
                    ):
                        next_sample_ns[stock_locate] = (
                            timestamp_ns + sampling_interval_ns
                        )

            engine.resolve_post_fill_markouts(
                stock_locate=stock_locate,
                timestamp_ns=timestamp_ns,
                current_mid=(
                    post_state.mid
                    if post_state is not None
                    else None
                ),
            )

            # -------------------------------------------------------
            # Current execution belongs to EXISTING candidates.
            # Newly created candidates below do not see it.
            # Queue consumption is tied to the resting/display price,
            # not C's potentially price-improved execution price.
            # -------------------------------------------------------
            if (
                message_type in {"E", "C"}
                and resting_order is not None
                and execution_shares > 0
            ):
                engine.register_execution(
                    stock_locate=stock_locate,
                    resting_side=(
                        resting_order.side
                    ),
                    price=resting_order.price,
                    shares=execution_shares,
                    timestamp_ns=timestamp_ns,
                    current_mid=(
                        post_state.mid
                        if post_state is not None
                        else None
                    ),
                    message_index=message_count,
                )

                execution_count += 1

            # -------------------------------------------------------
            # Detect complete disappearance of a price level.
            # -------------------------------------------------------
            for side, price in affected_levels:
                levels = (
                    book.bids
                    if side == "B"
                    else book.asks
                )

                before = before_depth.get(
                    (side, price),
                    0,
                )

                after = levels.get(
                    price,
                    0,
                )

                if (
                    before > 0
                    and after == 0
                ):
                    engine.handle_level_reset(
                        stock_locate=stock_locate,
                        resting_side=side,
                        price=price,
                        timestamp_ns=timestamp_ns,
                        current_mid=(
                            post_state.mid
                            if post_state is not None
                            else None
                        ),
                        message_index=message_count,
                    )

            # Existing candidates see this post-event midpoint.
            engine.resolve_adverse(
                stock_locate=stock_locate,
                current_mid=(
                    post_state.mid
                    if post_state is not None
                    else None
                ),
                timestamp_ns=timestamp_ns,
            )

            # At the exact timeout timestamp, a fill/adverse event
            # on this message wins; unresolved candidates timeout.
            engine.resolve_timeouts_at_or_before(
                timestamp_ns,
                current_mid=(
                    post_state.mid
                    if post_state is not None
                    else None
                ),
            )

            completed_count += write_completed(
                writer,
                engine,
            )

            # -------------------------------------------------------
            # Create execution-triggered candidates from pre-event
            # state; time-mode candidates were created before the event.
            # -------------------------------------------------------
            should_sample = False
            sample_state: m9.BookState | None = None
            sample_quotes: dict[str, tuple[int, int]] = {}
            previous: m9.BookState | None = None

            if sampling_mode == "execution":
                if (
                    message_type in {"E", "C"}
                    and pre_state is not None
                    and post_state is not None
                ):
                    should_sample = True
                    sample_state = pre_state
                    sample_quotes = pre_quotes
                    previous = previous_execution_state.get(
                        stock_locate
                    )
                    previous_execution_state[stock_locate] = pre_state

            if should_sample and sample_state is not None:
                if previous is None:
                    delta_imbalance = 0.0
                    delta_microprice = 0.0
                    delta_spread = 0.0
                else:
                    delta_imbalance = (
                        sample_state.imbalance
                        - previous.imbalance
                    )
                    delta_microprice = (
                        sample_state.microprice
                        - previous.microprice
                    )
                    delta_spread = (
                        sample_state.spread
                        - previous.spread
                    )

                if "BUY" in sample_quotes:
                    bid_price, bid_epoch = sample_quotes["BUY"]
                    engine.create_candidate(
                        timestamp_ns=timestamp_ns,
                        symbol=symbol,
                        stock_locate=stock_locate,
                        quote_side="BUY",
                        quote_price=bid_price,
                        quote_qty=QUOTE_QTY,
                        state=sample_state,
                        delta_imbalance=delta_imbalance,
                        delta_microprice=delta_microprice,
                        delta_spread=delta_spread,
                        pre_level_epoch=bid_epoch,
                        source_event_id=message_count,
                        activation_queue_ahead=book.bids.get(
                            bid_price,
                            0,
                        ),
                        snapshot_timestamp_ns=pre_snapshot_timestamp_ns,
                    )
                    candidate_count += 1

                if "SELL" in sample_quotes:
                    ask_price, ask_epoch = sample_quotes["SELL"]

                    engine.create_candidate(
                        timestamp_ns=timestamp_ns,
                        symbol=symbol,
                        stock_locate=stock_locate,
                        quote_side="SELL",
                        quote_price=ask_price,
                        quote_qty=QUOTE_QTY,
                        state=sample_state,
                        delta_imbalance=delta_imbalance,
                        delta_microprice=delta_microprice,
                        delta_spread=delta_spread,
                        pre_level_epoch=ask_epoch,
                        source_event_id=message_count,
                        activation_queue_ahead=book.asks.get(
                            ask_price,
                            0,
                        ),
                        snapshot_timestamp_ns=pre_snapshot_timestamp_ns,
                    )

                    candidate_count += 1

            if (
                message_count % 5_000_000
                == 0
            ):
                active_count = len(
                    engine.candidates
                )

                print(
                    f"messages={message_count:,} "
                    f"executions={execution_count:,} "
                    f"candidates={candidate_count:,} "
                    f"completed={completed_count:,} "
                    f"active={active_count:,}"
                )

        if last_timestamp_ns is not None:
            engine.finalize_pending_markouts(last_timestamp_ns)
            completed_count += write_completed(writer, engine)

    finalize_cluster_metadata(output_path)

    print()
    print("Completed")
    print(
        f"Messages processed: "
        f"{message_count:,}"
    )
    print(
        f"Execution observations: "
        f"{execution_count:,}"
    )
    print(
        f"Candidates created: "
        f"{candidate_count:,}"
    )
    print(
        f"Completed candidates: "
        f"{completed_count:,}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the M9 passive quote "
            "fill-hazard dataset."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/itch/"
            "07302019.NASDAQ_ITCH50.gz"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "research/"
            "m9_fill_hazard_markouts_2019-07-30.csv"
        ),
    )

    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--sampling-mode",
        choices=("execution", "time"),
        default="execution",
        help=(
            "Candidate sampling mode: execution-triggered "
            "or scheduled time sampling."
        ),
    )
    parser.add_argument(
        "--sampling-interval-ms",
        type=int,
        default=1000,
        help="Scheduled sampling interval for time mode, in milliseconds.",
    )

    args = parser.parse_args()

    if args.sampling_interval_ms <= 0:
        parser.error("--sampling-interval-ms must be positive")

    run(
        input_path=args.input,
        output_path=args.output,
        max_messages=args.max_messages,
        sampling_mode=args.sampling_mode,
        sampling_interval_ns=(
            args.sampling_interval_ms * 1_000_000
        ),
    )


if __name__ == "__main__":
    main()