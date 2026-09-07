from dataclasses import dataclass


@dataclass(frozen=True)
class FillAllocation:
    quantity_ahead_before: int
    execution_quantity: int
    hypothetical_fill: int
    quantity_ahead_after: int


def allocate_execution(
    quantity_ahead: int,
    execution_quantity: int,
) -> FillAllocation:
    if quantity_ahead < 0:
        raise ValueError("quantity_ahead cannot be negative")

    if execution_quantity <= 0:
        raise ValueError("execution_quantity must be positive")

    consumed_ahead = min(quantity_ahead, execution_quantity)
    hypothetical_fill = execution_quantity - consumed_ahead

    return FillAllocation(
        quantity_ahead_before=quantity_ahead,
        execution_quantity=execution_quantity,
        hypothetical_fill=hypothetical_fill,
        quantity_ahead_after=quantity_ahead - consumed_ahead,
    )