"""Lead-time rule: the outcome of origin r is available at origin o if r + horizon - 1 < o."""

from __future__ import annotations

from typing import Iterable, TypeVar


T = TypeVar("T")


def outcome_end_day(origin_day: int, horizon_days: int) -> int:
    return int(origin_day) + int(horizon_days) - 1


def is_outcome_available(update_origin: int, current_origin: int, horizon_days: int) -> bool:
    return outcome_end_day(update_origin, horizon_days) < int(current_origin)


def split_available_updates(
    pending: Iterable[tuple[int, T]],
    current_origin: int,
    horizon_days: int,
) -> tuple[list[tuple[int, T]], list[tuple[int, T]]]:
    available: list[tuple[int, T]] = []
    blocked: list[tuple[int, T]] = []
    for update_origin, payload in pending:
        if is_outcome_available(update_origin, current_origin, horizon_days):
            available.append((update_origin, payload))
        else:
            blocked.append((update_origin, payload))
    for update_origin, _ in available:
        assert is_outcome_available(update_origin, current_origin, horizon_days), (
            f"Future outcome used: update_origin={update_origin}, "
            f"current_origin={current_origin}, horizon_days={horizon_days}"
        )
    return available, blocked
