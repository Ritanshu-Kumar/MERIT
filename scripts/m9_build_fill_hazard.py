from __future__ import annotations

import argparse
import csv
import heapq
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import m9_build_state_transitions as m9


TARGET_SYMBOLS = {
    "AAPL",
    "AMZN",
    "GOOG",
    "INTC",
    "MSFT",
}

QUOTE_QTY = 100
TIMEOUT_NS = 1_000_000_000
TICK_SIZE_UNITS = 100

OUTPUT_COLUMNS = [
    "timestamp_ns",
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
    "filled_qty",
    "fill_fraction",
    "time_to_resolution_ns",
    "resolution_mid",
    "adverse_threshold",
    "level_resets",
    "outcome",
    "fill_before_adverse",
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

    level_resets: int = 0
    status: str = "ACTIVE"
    resolved_timestamp_ns: Optional[int] = None
    resolution_mid: Optional[float] = None

    required_execution_volume: int = 0


class HazardEngine:
    def __init__(self) -> None:
        self.next_id = 0

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
        resolution_mid: Optional[float],
    ) -> None:
        if candidate.status != "ACTIVE":
            return

        candidate.filled_qty = candidate.quote_qty
        candidate.status = "FILL"
        candidate.resolved_timestamp_ns = (
            timestamp_ns
        )
        candidate.resolution_mid = resolution_mid

        self._remove_active(candidate)

        self.completed.append(candidate)

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
            required_execution_volume=(
                required_execution_volume
            ),
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
        current_mid: Optional[float],
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
            )

    def handle_level_reset(
        self,
        stock_locate: int,
        resting_side: str,
        price: int,
        timestamp_ns: int,
        current_mid: Optional[float],
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
        current_mid: Optional[float],
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
            candidate.resolution_mid = (
                current_mid
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
            candidate.resolution_mid = (
                current_mid
            )

            self._remove_active(candidate)
            self.completed.append(candidate)

    def resolve_timeouts_before(
        self,
        timestamp_ns: int,
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
            candidate.resolution_mid = None

            self._remove_active(candidate)
            self.completed.append(candidate)

    def resolve_timeouts_at_or_before(
        self,
        timestamp_ns: int,
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
            candidate.resolution_mid = None

            self._remove_active(candidate)
            self.completed.append(candidate)

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
) -> Optional[m9.Order]:
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
        "timestamp_ns": candidate.timestamp_ns,
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
        "filled_qty": candidate.filled_qty,
        "fill_fraction": (
            candidate.filled_qty
            / candidate.quote_qty
        ),
        "time_to_resolution_ns": (
            candidate.resolved_timestamp_ns
            - candidate.timestamp_ns
        ),
        "resolution_mid": (
            candidate.resolution_mid
        ),
        "adverse_threshold": (
            candidate.adverse_threshold
        ),
        "level_resets": candidate.level_resets,
        "outcome": candidate.status,
        "fill_before_adverse": (
            candidate.status == "FILL"
        ),
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


def run(
    input_path: Path,
    output_path: Path,
    max_messages: Optional[int],
) -> None:
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

    engine = HazardEngine()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    message_count = 0
    execution_count = 0
    candidate_count = 0
    completed_count = 0

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

            # Candidates whose exact one-second horizon has
            # already passed cannot interact with this event.
            engine.resolve_timeouts_before(
                timestamp_ns
            )

            completed_count += write_completed(
                writer,
                engine,
            )

            pre_state = book.snapshot()

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

            # Capture the order involved in the current message
            # before mutating the book.
            resting_order: Optional[m9.Order] = None
            execution_shares = 0
            actual_execution_price: Optional[int] = None

            if message_type in {"E", "C"}:
                (
                    order_ref,
                    execution_shares,
                    printable_price,
                ) = m9.parse_execution(msg)

                resting_order = book.orders.get(
                    order_ref
                )

                if resting_order is not None:
                    actual_execution_price = (
                        printable_price
                        if printable_price is not None
                        else resting_order.price
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

            elif message_type == "D":
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

            elif message_type == "X":
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

            elif message_type == "D":
                order_ref = (
                    m9.parse_order_ref(msg)
                )

                book.remove_order(
                    order_ref
                )

            elif message_type == "X":
                order_ref = (
                    m9.parse_order_ref(msg)
                )

                cancelled_shares = int.from_bytes(
                    msg[19:23],
                    "big",
                )

                cancel_order(
                    book,
                    order_ref,
                    cancelled_shares,
                )

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

                book.replace_order(
                    original_ref,
                    new_ref,
                    shares,
                    price,
                )

            post_state = book.snapshot()

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
                timestamp_ns
            )

            completed_count += write_completed(
                writer,
                engine,
            )

            # -------------------------------------------------------
            # Create NEW candidates using the PRE-event quote.
            # The triggering execution is therefore excluded.
            # -------------------------------------------------------
            if (
                message_type in {"E", "C"}
                and pre_state is not None
                and post_state is not None
            ):
                previous = (
                    previous_execution_state.get(
                        stock_locate
                    )
                )

                if previous is None:
                    delta_imbalance = 0.0
                    delta_microprice = 0.0
                    delta_spread = 0.0
                else:
                    delta_imbalance = (
                        pre_state.imbalance
                        - previous.imbalance
                    )

                    delta_microprice = (
                        pre_state.microprice
                        - previous.microprice
                    )

                    delta_spread = (
                        pre_state.spread
                        - previous.spread
                    )

                if "BUY" in pre_quotes:
                    (
                        bid_price,
                        bid_epoch,
                    ) = pre_quotes["BUY"]

                    engine.create_candidate(
                        timestamp_ns=timestamp_ns,
                        symbol=symbol,
                        stock_locate=stock_locate,
                        quote_side="BUY",
                        quote_price=bid_price,
                        quote_qty=QUOTE_QTY,
                        state=pre_state,
                        delta_imbalance=(
                            delta_imbalance
                        ),
                        delta_microprice=(
                            delta_microprice
                        ),
                        delta_spread=delta_spread,
                        pre_level_epoch=bid_epoch,
                    )

                    candidate_count += 1

                if "SELL" in pre_quotes:
                    (
                        ask_price,
                        ask_epoch,
                    ) = pre_quotes["SELL"]

                    engine.create_candidate(
                        timestamp_ns=timestamp_ns,
                        symbol=symbol,
                        stock_locate=stock_locate,
                        quote_side="SELL",
                        quote_price=ask_price,
                        quote_qty=QUOTE_QTY,
                        state=pre_state,
                        delta_imbalance=(
                            delta_imbalance
                        ),
                        delta_microprice=(
                            delta_microprice
                        ),
                        delta_spread=delta_spread,
                        pre_level_epoch=ask_epoch,
                    )

                    candidate_count += 1

                previous_execution_state[
                    stock_locate
                ] = pre_state

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

        # Remaining candidates have not reached their exact
        # one-second horizon because the stream ended.
        # They are deliberately excluded rather than fabricating
        # a future state after the sample boundary.

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
            "m9_fill_hazard_2019-07-30.csv"
        ),
    )

    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        max_messages=args.max_messages,
    )


if __name__ == "__main__":
    main()