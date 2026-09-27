"""Shared ingestion validation helpers (API + import script)."""

from datetime import UTC, datetime, timedelta


def utc_now() -> datetime:
    return datetime.now(UTC)


def is_future_offset(
    occurred_at: datetime, *, max_skew_minutes: int, now: datetime | None = None
) -> bool:
    """True when the event time exceeds allowed future skew (naive treated as UTC)."""
    reference = now or utc_now()
    occurred = occurred_at if occurred_at.tzinfo else occurred_at.replace(tzinfo=UTC)
    return occurred > reference + timedelta(minutes=max_skew_minutes)


class ImportPartialFailure(RuntimeError):
    """Raised when a batch import fails mid-way; carries the partial summary."""

    def __init__(self, summary: dict, message: str):
        super().__init__(message)
        self.summary = summary
