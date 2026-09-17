from __future__ import annotations

import argparse
import csv
import gzip
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, Optional, Tuple


TARGET_SYMBOLS = {"AAPL", "AMZN", "GOOG", "INTC", "MSFT"}

HORIZONS_NS = {
    "10ms": 10_000_000,
    "50ms": 50_000_000,
    "100ms": 100_000_000,
    "500ms": 500_000_000,
    "1s": 1_000_000_000,
}

FLOW_WINDOWS_NS = {
    "1ms": 1_000_000,
    "10ms": 10_000_000,
    "100ms": 100_000_000,
}

OUTPUT_COLUMNS = [
    "timestamp_ns",
    "symbol",
    "stock_locate",
    "execution_side",
    "execution_price",
    "execution_shares",
    "mid",
    "spread",
    "relative_spread",
    "bid_size",
    "ask_size",
    "imbalance",
    "imbalance_l5",
    "imbalance_l10",
    "microprice",
    "delta_mid",
    "delta_spread",
    "delta_imbalance",
    "delta_imbalance_l5",
    "delta_imbalance_l10",
    "delta_microprice",
    "events_1ms",
    "events_10ms",
    "events_100ms",
    "adds_1ms",
    "adds_10ms",
    "adds_100ms",
    "cancels_1ms",
    "cancels_10ms",
    "cancels_100ms",
    "executions_1ms",
    "executions_10ms",
    "executions_100ms",
    "buy_flow_1ms",
    "buy_flow_10ms",
    "buy_flow_100ms",
    "sell_flow_1ms",
    "sell_flow_10ms",
    "sell_flow_100ms",
    "mid_return_bps_10ms",
    "mid_return_bps_50ms",
    "mid_return_bps_100ms",
    "mid_return_bps_500ms",
    "mid_return_bps_1s",
    "spread_change_bps_10ms",
    "spread_change_bps_100ms",
    "spread_change_bps_1s",
    "imbalance_change_10ms",
    "imbalance_change_100ms",
    "imbalance_change_1s",
    "microprice_return_bps_10ms",
    "microprice_return_bps_100ms",
    "microprice_return_bps_1s",
    "l5_imbalance_change_100ms",
    "l10_imbalance_change_100ms",
    "future_mid_10ms",
    "future_mid_100ms",
]


@dataclass
class Order:
    order_ref: int
    side: str
    shares: int
    price: int


@dataclass
class FlowEvent:
    timestamp_ns: int
    kind: str
    side: str
    shares: int


@dataclass
class BookState:
    mid: float
    spread: float
    relative_spread: float
    bid_size: int
    ask_size: int
    imbalance: float
    imbalance_l5: float
    imbalance_l10: float
    microprice: float


@dataclass
class PendingObservation:
    timestamp_ns: int
    symbol: str
    stock_locate: int
    execution_side: str
    execution_price: float
    execution_shares: int

    state: BookState

    delta_mid: float
    delta_spread: float
    delta_imbalance: float
    delta_imbalance_l5: float
    delta_imbalance_l10: float
    delta_microprice: float

    events_1ms: int
    events_10ms: int
    events_100ms: int

    adds_1ms: int
    adds_10ms: int
    adds_100ms: int

    cancels_1ms: int
    cancels_10ms: int
    cancels_100ms: int

    executions_1ms: int
    executions_10ms: int
    executions_100ms: int

    buy_flow_1ms: int
    buy_flow_10ms: int
    buy_flow_100ms: int

    sell_flow_1ms: int
    sell_flow_10ms: int
    sell_flow_100ms: int

    future_states: Dict[str, Optional[BookState]]
    resolved: Dict[str, bool]


class OrderBook:
    def __init__(self) -> None:
        self.bids: Dict[int, int] = defaultdict(int)
        self.asks: Dict[int, int] = defaultdict(int)
        self.orders: Dict[int, Order] = {}

    def add_order(
        self,
        order_ref: int,
        side: str,
        shares: int,
        price: int,
    ) -> None:
        if order_ref in self.orders:
            raise ValueError(
                f"duplicate order reference {order_ref}"
            )

        order = Order(
            order_ref=order_ref,
            side=side,
            shares=shares,
            price=price,
        )

        self.orders[order_ref] = order

        levels = self.bids if side == "B" else self.asks
        levels[price] += shares

    def remove_order(
        self,
        order_ref: int,
    ) -> Optional[Order]:
        order = self.orders.pop(order_ref, None)

        if order is None:
            return None

        levels = self.bids if order.side == "B" else self.asks
        levels[order.price] -= order.shares

        if levels[order.price] <= 0:
            del levels[order.price]

        return order

    def execute_order(
        self,
        order_ref: int,
        shares: int,
    ) -> Optional[Order]:
        order = self.orders.get(order_ref)

        if order is None:
            return None

        executed = min(shares, order.shares)

        levels = self.bids if order.side == "B" else self.asks
        levels[order.price] -= executed

        if levels[order.price] <= 0:
            del levels[order.price]

        order.shares -= executed

        if order.shares <= 0:
            del self.orders[order_ref]

        return Order(
            order_ref=order_ref,
            side=order.side,
            shares=executed,
            price=order.price,
        )

    def replace_order(
        self,
        original_ref: int,
        new_ref: int,
        shares: int,
        price: int,
    ) -> Optional[Order]:
        old = self.remove_order(original_ref)

        if old is None:
            return None

        self.add_order(
            order_ref=new_ref,
            side=old.side,
            shares=shares,
            price=price,
        )

        return old

    def best_bid(self) -> Optional[int]:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> Optional[int]:
        return min(self.asks) if self.asks else None

    def level_imbalance(self, depth: int) -> float:
        bid_prices = sorted(
            self.bids.keys(),
            reverse=True,
        )[:depth]

        ask_prices = sorted(
            self.asks.keys(),
        )[:depth]

        bid_size = sum(
            self.bids[p]
            for p in bid_prices
        )

        ask_size = sum(
            self.asks[p]
            for p in ask_prices
        )

        total = bid_size + ask_size

        if total == 0:
            return 0.0

        return (bid_size - ask_size) / total

    def snapshot(self) -> Optional[BookState]:
        bid = self.best_bid()
        ask = self.best_ask()

        if bid is None or ask is None:
            return None

        bid_size = self.bids[bid]
        ask_size = self.asks[ask]

        mid = (bid + ask) / 20000.0
        spread = (ask - bid) / 10000.0

        if mid <= 0:
            return None

        relative_spread = spread / mid

        total = bid_size + ask_size

        if total > 0:
            imbalance = (
                bid_size - ask_size
            ) / total

            microprice = (
                (
                    ask * bid_size
                    + bid * ask_size
                )
                / total
                / 10000.0
            )
        else:
            imbalance = 0.0
            microprice = mid

        return BookState(
            mid=mid,
            spread=spread,
            relative_spread=relative_spread,
            bid_size=bid_size,
            ask_size=ask_size,
            imbalance=imbalance,
            imbalance_l5=self.level_imbalance(5),
            imbalance_l10=self.level_imbalance(10),
            microprice=microprice,
        )


# ---------------------------------------------------------------------------
# ITCH 5.0 decoding.
# Offsets are relative to the message body, i.e. after the
# two-byte ITCH message-length prefix.
# ---------------------------------------------------------------------------

def parse_timestamp_ns(msg: bytes) -> int:
    return int.from_bytes(
        msg[5:11],
        "big",
    )


def parse_stock_locate(msg: bytes) -> int:
    return int.from_bytes(
        msg[1:3],
        "big",
    )


def parse_order_ref(msg: bytes) -> int:
    return int.from_bytes(
        msg[11:19],
        "big",
    )


def parse_stock_directory(
    msg: bytes,
) -> Tuple[int, str]:
    stock_locate = int.from_bytes(
        msg[1:3],
        "big",
    )

    symbol = (
        msg[11:19]
        .decode("ascii", errors="replace")
        .strip()
    )

    return stock_locate, symbol


def parse_add(
    msg: bytes,
) -> Tuple[int, str, int, int]:
    order_ref = int.from_bytes(
        msg[11:19],
        "big",
    )

    side = chr(msg[19])

    shares = int.from_bytes(
        msg[20:24],
        "big",
    )

    # A/F:
    # 24:32 stock
    # 32:36 price
    price = int.from_bytes(
        msg[32:36],
        "big",
    )

    return (
        order_ref,
        side,
        shares,
        price,
    )


def parse_execution(
    msg: bytes,
) -> Tuple[int, int, Optional[int]]:
    order_ref = int.from_bytes(
        msg[11:19],
        "big",
    )

    shares = int.from_bytes(
        msg[19:23],
        "big",
    )

    if msg[0] == ord("C"):
        execution_price = int.from_bytes(
            msg[32:36],
            "big",
        )

        return (
            order_ref,
            shares,
            execution_price,
        )

    return (
        order_ref,
        shares,
        None,
    )


def parse_replace(
    msg: bytes,
) -> Tuple[int, int, int, int]:
    original_ref = int.from_bytes(
        msg[11:19],
        "big",
    )

    new_ref = int.from_bytes(
        msg[19:27],
        "big",
    )

    shares = int.from_bytes(
        msg[27:31],
        "big",
    )

    price = int.from_bytes(
        msg[31:35],
        "big",
    )

    return (
        original_ref,
        new_ref,
        shares,
        price,
    )


def price_to_float(price_units: int) -> float:
    return price_units / 10000.0


def passive_to_aggressive(
    resting_side: str,
) -> str:
    if resting_side == "S":
        return "BUY"

    if resting_side == "B":
        return "SELL"

    return "UNKNOWN"


def flow_side(
    resting_side: str,
) -> str:
    if resting_side == "B":
        return "BUY"

    if resting_side == "S":
        return "SELL"

    return "UNKNOWN"


def get_window_events(
    flow: Deque[FlowEvent],
    timestamp_ns: int,
    window_ns: int,
) -> Iterable[FlowEvent]:
    cutoff = timestamp_ns - window_ns

    for event in reversed(flow):
        if event.timestamp_ns < cutoff:
            break

        yield event


def trailing_features(
    flow: Deque[FlowEvent],
    timestamp_ns: int,
) -> Dict[str, int]:
    result: Dict[str, int] = {}

    for label, window_ns in FLOW_WINDOWS_NS.items():
        events = list(
            get_window_events(
                flow,
                timestamp_ns,
                window_ns,
            )
        )

        result[f"events_{label}"] = len(events)

        result[f"adds_{label}"] = sum(
            1
            for event in events
            if event.kind == "ADD"
        )

        result[f"cancels_{label}"] = sum(
            1
            for event in events
            if event.kind == "CANCEL"
        )

        result[f"executions_{label}"] = sum(
            1
            for event in events
            if event.kind == "EXECUTION"
        )

        result[f"buy_flow_{label}"] = sum(
            event.shares
            for event in events
            if event.kind == "EXECUTION"
            and event.side == "BUY"
        )

        result[f"sell_flow_{label}"] = sum(
            event.shares
            for event in events
            if event.kind == "EXECUTION"
            and event.side == "SELL"
        )

    return result


def mid_return_bps(
    initial_mid: float,
    future_mid: Optional[float],
) -> Optional[float]:
    if future_mid is None or initial_mid <= 0:
        return None

    return (
        (future_mid - initial_mid)
        / initial_mid
        * 10000.0
    )


def relative_change_bps(
    initial_value: float,
    future_value: Optional[float],
) -> Optional[float]:
    if future_value is None or initial_value == 0:
        return None

    return (
        (future_value - initial_value)
        / initial_value
        * 10000.0
    )


def absolute_change(
    initial_value: float,
    future_value: Optional[float],
) -> Optional[float]:
    if future_value is None:
        return None

    return future_value - initial_value


def resolve_pending(
    pending: Deque[PendingObservation],
    current_timestamp_ns: int,
    current_state: Optional[BookState],
) -> list[PendingObservation]:
    if current_state is None:
        return []

    completed: list[PendingObservation] = []
    remaining: Deque[PendingObservation] = deque()

    while pending:
        observation = pending.popleft()

        age = (
            current_timestamp_ns
            - observation.timestamp_ns
        )

        for horizon_name, horizon_ns in HORIZONS_NS.items():
            if (
                not observation.resolved[horizon_name]
                and age >= horizon_ns
            ):
                observation.future_states[horizon_name] = (
                    current_state
                )

                observation.resolved[horizon_name] = True

        if all(observation.resolved.values()):
            completed.append(observation)
        else:
            remaining.append(observation)

    pending.extend(remaining)

    return completed


def observation_to_row(
    observation: PendingObservation,
) -> Optional[Dict[str, object]]:
    required = (
        "10ms",
        "50ms",
        "100ms",
        "500ms",
        "1s",
    )

    if not all(
        observation.resolved[h]
        for h in required
    ):
        return None

    state = observation.state

    s10 = observation.future_states["10ms"]
    s50 = observation.future_states["50ms"]
    s100 = observation.future_states["100ms"]
    s500 = observation.future_states["500ms"]
    s1s = observation.future_states["1s"]

    assert s10 is not None
    assert s50 is not None
    assert s100 is not None
    assert s500 is not None
    assert s1s is not None

    return {
        "timestamp_ns": observation.timestamp_ns,
        "symbol": observation.symbol,
        "stock_locate": observation.stock_locate,
        "execution_side": observation.execution_side,
        "execution_price": observation.execution_price,
        "execution_shares": observation.execution_shares,
        "mid": state.mid,
        "spread": state.spread,
        "relative_spread": state.relative_spread,
        "bid_size": state.bid_size,
        "ask_size": state.ask_size,
        "imbalance": state.imbalance,
        "imbalance_l5": state.imbalance_l5,
        "imbalance_l10": state.imbalance_l10,
        "microprice": state.microprice,
        "delta_mid": observation.delta_mid,
        "delta_spread": observation.delta_spread,
        "delta_imbalance": observation.delta_imbalance,
        "delta_imbalance_l5": observation.delta_imbalance_l5,
        "delta_imbalance_l10": observation.delta_imbalance_l10,
        "delta_microprice": observation.delta_microprice,
        "events_1ms": observation.events_1ms,
        "events_10ms": observation.events_10ms,
        "events_100ms": observation.events_100ms,
        "adds_1ms": observation.adds_1ms,
        "adds_10ms": observation.adds_10ms,
        "adds_100ms": observation.adds_100ms,
        "cancels_1ms": observation.cancels_1ms,
        "cancels_10ms": observation.cancels_10ms,
        "cancels_100ms": observation.cancels_100ms,
        "executions_1ms": observation.executions_1ms,
        "executions_10ms": observation.executions_10ms,
        "executions_100ms": observation.executions_100ms,
        "buy_flow_1ms": observation.buy_flow_1ms,
        "buy_flow_10ms": observation.buy_flow_10ms,
        "buy_flow_100ms": observation.buy_flow_100ms,
        "sell_flow_1ms": observation.sell_flow_1ms,
        "sell_flow_10ms": observation.sell_flow_10ms,
        "sell_flow_100ms": observation.sell_flow_100ms,
        "mid_return_bps_10ms": mid_return_bps(
            state.mid,
            s10.mid,
        ),
        "mid_return_bps_50ms": mid_return_bps(
            state.mid,
            s50.mid,
        ),
        "mid_return_bps_100ms": mid_return_bps(
            state.mid,
            s100.mid,
        ),
        "mid_return_bps_500ms": mid_return_bps(
            state.mid,
            s500.mid,
        ),
        "mid_return_bps_1s": mid_return_bps(
            state.mid,
            s1s.mid,
        ),
        "spread_change_bps_10ms": relative_change_bps(
            state.spread,
            s10.spread,
        ),
        "spread_change_bps_100ms": relative_change_bps(
            state.spread,
            s100.spread,
        ),
        "spread_change_bps_1s": relative_change_bps(
            state.spread,
            s1s.spread,
        ),
        "imbalance_change_10ms": absolute_change(
            state.imbalance,
            s10.imbalance,
        ),
        "imbalance_change_100ms": absolute_change(
            state.imbalance,
            s100.imbalance,
        ),
        "imbalance_change_1s": absolute_change(
            state.imbalance,
            s1s.imbalance,
        ),
        "microprice_return_bps_10ms": mid_return_bps(
            state.microprice,
            s10.microprice,
        ),
        "microprice_return_bps_100ms": mid_return_bps(
            state.microprice,
            s100.microprice,
        ),
        "microprice_return_bps_1s": mid_return_bps(
            state.microprice,
            s1s.microprice,
        ),
        "l5_imbalance_change_100ms": absolute_change(
            state.imbalance_l5,
            s100.imbalance_l5,
        ),
        "l10_imbalance_change_100ms": absolute_change(
            state.imbalance_l10,
            s100.imbalance_l10,
        ),
        "future_mid_10ms": s10.mid,
        "future_mid_100ms": s100.mid,
    }


def process_message(
    msg: bytes,
    locate_to_symbol: Dict[int, str],
    books: Dict[int, OrderBook],
    previous_execution_state: Dict[int, BookState],
    flow: Dict[int, Deque[FlowEvent]],
    pending: Dict[int, Deque[PendingObservation]],
) -> list[PendingObservation]:
    message_type = chr(msg[0])
    stock_locate = parse_stock_locate(msg)

    if stock_locate not in locate_to_symbol:
        return []

    timestamp_ns = parse_timestamp_ns(msg)

    book = books[stock_locate]
    symbol = locate_to_symbol[stock_locate]

    symbol_flow = flow[stock_locate]
    symbol_pending = pending[stock_locate]

    pre_state = book.snapshot()

    completed_observations: list[PendingObservation] = []

    # ---------------------------------------------------------------
    # Capture execution observation using the PRE-event book.
    # ---------------------------------------------------------------
    if (
        message_type in {"E", "C"}
        and pre_state is not None
    ):
        (
            order_ref,
            execution_shares,
            execution_price_units,
        ) = parse_execution(msg)

        resting_order = book.orders.get(
            order_ref
        )

        if resting_order is not None:
            previous = previous_execution_state.get(
                stock_locate
            )

            if previous is None:
                (
                    delta_mid,
                    delta_spread,
                    delta_imbalance,
                    delta_imbalance_l5,
                    delta_imbalance_l10,
                    delta_microprice,
                ) = (0.0,) * 6
            else:
                # M9 transition:
                # current execution PRE-state
                # minus previous execution PRE-state.
                delta_mid = (
                    pre_state.mid
                    - previous.mid
                )

                delta_spread = (
                    pre_state.spread
                    - previous.spread
                )

                delta_imbalance = (
                    pre_state.imbalance
                    - previous.imbalance
                )

                delta_imbalance_l5 = (
                    pre_state.imbalance_l5
                    - previous.imbalance_l5
                )

                delta_imbalance_l10 = (
                    pre_state.imbalance_l10
                    - previous.imbalance_l10
                )

                delta_microprice = (
                    pre_state.microprice
                    - previous.microprice
                )

            # Current execution is NOT included.
            flow_stats = trailing_features(
                symbol_flow,
                timestamp_ns,
            )

            execution_price = (
                price_to_float(
                    execution_price_units
                )
                if execution_price_units is not None
                else price_to_float(
                    resting_order.price
                )
            )

            observation = PendingObservation(
                timestamp_ns=timestamp_ns,
                symbol=symbol,
                stock_locate=stock_locate,
                execution_side=passive_to_aggressive(
                    resting_order.side
                ),
                execution_price=execution_price,
                execution_shares=execution_shares,
                state=pre_state,
                delta_mid=delta_mid,
                delta_spread=delta_spread,
                delta_imbalance=delta_imbalance,
                delta_imbalance_l5=delta_imbalance_l5,
                delta_imbalance_l10=delta_imbalance_l10,
                delta_microprice=delta_microprice,
                events_1ms=flow_stats["events_1ms"],
                events_10ms=flow_stats["events_10ms"],
                events_100ms=flow_stats["events_100ms"],
                adds_1ms=flow_stats["adds_1ms"],
                adds_10ms=flow_stats["adds_10ms"],
                adds_100ms=flow_stats["adds_100ms"],
                cancels_1ms=flow_stats["cancels_1ms"],
                cancels_10ms=flow_stats["cancels_10ms"],
                cancels_100ms=flow_stats["cancels_100ms"],
                executions_1ms=flow_stats["executions_1ms"],
                executions_10ms=flow_stats["executions_10ms"],
                executions_100ms=flow_stats["executions_100ms"],
                buy_flow_1ms=flow_stats["buy_flow_1ms"],
                buy_flow_10ms=flow_stats["buy_flow_10ms"],
                buy_flow_100ms=flow_stats["buy_flow_100ms"],
                sell_flow_1ms=flow_stats["sell_flow_1ms"],
                sell_flow_10ms=flow_stats["sell_flow_10ms"],
                sell_flow_100ms=flow_stats["sell_flow_100ms"],
                future_states={
                    "10ms": None,
                    "50ms": None,
                    "100ms": None,
                    "500ms": None,
                    "1s": None,
                },
                resolved={
                    "10ms": False,
                    "50ms": False,
                    "100ms": False,
                    "500ms": False,
                    "1s": False,
                },
            )

            symbol_pending.append(observation)

            # Store the PRE-event state for the NEXT execution.
            previous_execution_state[stock_locate] = (
                pre_state
            )

    # ---------------------------------------------------------------
    # Apply current ITCH message.
    # ---------------------------------------------------------------
    if message_type in {"A", "F"}:
        (
            order_ref,
            side,
            shares,
            price,
        ) = parse_add(msg)

        if side in {"B", "S"} and shares > 0:
            book.add_order(
                order_ref=order_ref,
                side=side,
                shares=shares,
                price=price,
            )

            symbol_flow.append(
                FlowEvent(
                    timestamp_ns=timestamp_ns,
                    kind="ADD",
                    side=flow_side(side),
                    shares=shares,
                )
            )

    elif message_type == "D":
        order_ref = parse_order_ref(msg)

        removed = book.remove_order(
            order_ref
        )

        if removed is not None:
            symbol_flow.append(
                FlowEvent(
                    timestamp_ns=timestamp_ns,
                    kind="CANCEL",
                    side=flow_side(
                        removed.side
                    ),
                    shares=removed.shares,
                )
            )

    elif message_type == "X":
        (
            order_ref,
            executed_shares,
            _,
        ) = parse_execution(msg)

        existing = book.orders.get(
            order_ref
        )

        if existing is not None:
            executed = min(
                executed_shares,
                existing.shares,
            )

            book.execute_order(
                order_ref=order_ref,
                shares=executed,
            )

            symbol_flow.append(
                FlowEvent(
                    timestamp_ns=timestamp_ns,
                    kind="CANCEL",
                    side=flow_side(
                        existing.side
                    ),
                    shares=executed,
                )
            )

    elif message_type in {"E", "C"}:
        (
            order_ref,
            executed_shares,
            _,
        ) = parse_execution(msg)

        existing = book.orders.get(
            order_ref
        )

        if existing is not None:
            executed = min(
                executed_shares,
                existing.shares,
            )

            book.execute_order(
                order_ref=order_ref,
                shares=executed,
            )

            symbol_flow.append(
                FlowEvent(
                    timestamp_ns=timestamp_ns,
                    kind="EXECUTION",
                    side=flow_side(
                        existing.side
                    ),
                    shares=executed,
                )
            )

    elif message_type == "U":
        (
            original_ref,
            new_ref,
            shares,
            price,
        ) = parse_replace(msg)

        old = book.orders.get(
            original_ref
        )

        if old is not None:
            old_side = old.side
            old_shares = old.shares

            book.replace_order(
                original_ref=original_ref,
                new_ref=new_ref,
                shares=shares,
                price=price,
            )

            symbol_flow.append(
                FlowEvent(
                    timestamp_ns=timestamp_ns,
                    kind="CANCEL",
                    side=flow_side(
                        old_side
                    ),
                    shares=old_shares,
                )
            )

            symbol_flow.append(
                FlowEvent(
                    timestamp_ns=timestamp_ns,
                    kind="ADD",
                    side=flow_side(
                        old_side
                    ),
                    shares=shares,
                )
            )

    post_state = book.snapshot()

    if post_state is not None:
        if not (
            0.0
            < post_state.mid
            < 10000.0
        ):
            raise RuntimeError(
                "Invalid reconstructed midpoint: "
                f"symbol={symbol} "
                f"timestamp={timestamp_ns} "
                f"mid={post_state.mid}"
            )

    completed_observations.extend(
        resolve_pending(
            symbol_pending,
            timestamp_ns,
            post_state,
        )
    )

    cutoff = (
        timestamp_ns
        - FLOW_WINDOWS_NS["100ms"]
    )

    while (
        symbol_flow
        and symbol_flow[0].timestamp_ns < cutoff
    ):
        symbol_flow.popleft()

    return completed_observations


def iter_itch_messages(
    path: Path,
    max_messages: Optional[int] = None,
) -> Iterable[bytes]:
    message_count = 0
    buffer = bytearray()
    position = 0

    with gzip.open(path, "rb") as fh:
        while True:
            chunk = fh.read(
                4 * 1024 * 1024
            )

            if not chunk:
                break

            buffer.extend(chunk)

            while (
                len(buffer) - position >= 2
            ):
                message_length = int.from_bytes(
                    buffer[
                        position:
                        position + 2
                    ],
                    "big",
                )

                frame_length = (
                    2 + message_length
                )

                if (
                    len(buffer) - position
                    < frame_length
                ):
                    break

                start = position + 2
                end = start + message_length

                yield bytes(
                    buffer[start:end]
                )

                position += frame_length
                message_count += 1

                if (
                    max_messages is not None
                    and message_count >= max_messages
                ):
                    return

            if position > 16 * 1024 * 1024:
                del buffer[:position]
                position = 0


def discover_stock_locates(
    path: Path,
    target_symbols: set[str],
) -> Dict[int, str]:
    locate_to_symbol: Dict[int, str] = {}

    for msg in iter_itch_messages(path):
        if not msg:
            continue

        if msg[0] != ord("R"):
            continue

        stock_locate, symbol = (
            parse_stock_directory(msg)
        )

        if symbol in target_symbols:
            locate_to_symbol[stock_locate] = symbol

            if (
                len(locate_to_symbol)
                == len(target_symbols)
            ):
                break

    return locate_to_symbol


def run(
    input_path: Path,
    output_path: Path,
    max_messages: Optional[int],
) -> None:
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    locate_to_symbol = discover_stock_locates(
        input_path,
        TARGET_SYMBOLS,
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
            "Missing target stock directory "
            f"entries: {sorted(missing)}"
        )

    books: Dict[
        int,
        OrderBook,
    ] = defaultdict(OrderBook)

    # Previous EXECUTION observation state,
    # not previous ITCH-event state.
    previous_execution_state: Dict[
        int,
        BookState,
    ] = {}

    flow: Dict[
        int,
        Deque[FlowEvent],
    ] = defaultdict(deque)

    pending: Dict[
        int,
        Deque[PendingObservation],
    ] = defaultdict(deque)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    message_count = 0
    execution_count = 0
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

        for msg in iter_itch_messages(
            input_path,
            max_messages=max_messages,
        ):
            message_count += 1

            if not msg:
                continue

            message_type = chr(msg[0])

            if message_type == "R":
                continue

            stock_locate = parse_stock_locate(
                msg
            )

            if stock_locate not in locate_to_symbol:
                continue

            if message_type in {"E", "C"}:
                execution_count += 1

            completed_observations = process_message(
                msg=msg,
                locate_to_symbol=locate_to_symbol,
                books=books,
                previous_execution_state=(
                    previous_execution_state
                ),
                flow=flow,
                pending=pending,
            )

            for observation in completed_observations:
                row = observation_to_row(
                    observation
                )

                if row is not None:
                    writer.writerow(row)
                    completed_count += 1

            if (
                message_count % 500_000
                == 0
            ):
                pending_count = sum(
                    len(q)
                    for q in pending.values()
                )

                print(
                    f"messages={message_count:,} "
                    f"executions={execution_count:,} "
                    f"completed={completed_count:,} "
                    f"pending={pending_count:,}"
                )

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
        f"Completed rows: "
        f"{completed_count:,}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the M9 ITCH "
            "state-transition dataset."
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
            "m9_state_transitions_2019-07-30.csv"
        ),
    )

    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        max_messages=args.max_messages,
    )