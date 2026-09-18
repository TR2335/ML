"""Event bus adapters.

HTTP remains the default local mode. Redis Streams is available for production
and preserves the validated event envelope for retryable asynchronous scoring.
"""

import json
from typing import Protocol

from .domain import ClinicalEvent


class EventPublisher(Protocol):
    def publish(self, event: ClinicalEvent) -> str: ...


class RedisStreamPublisher:
    def __init__(self, url: str, stream: str):
        try:
            import redis
        except ModuleNotFoundError as exc:
            raise RuntimeError("Redis Streams requires the 'prod' dependency extra") from exc
        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.stream = stream

    def publish(self, event: ClinicalEvent) -> str:
        payload = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
        return str(
            self.client.xadd(self.stream, {"event": payload}, maxlen=100000, approximate=True)
        )

    def health(self) -> bool:
        return bool(self.client.ping())


class RedisStreamConsumer:
    def __init__(
        self,
        url: str,
        stream: str,
        group: str = "medintel-workers",
        consumer: str = "worker-1",
        dead_letter_stream: str | None = None,
    ):
        try:
            import redis
        except ModuleNotFoundError as exc:
            raise RuntimeError("Redis Streams requires the 'prod' dependency extra") from exc
        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.stream = stream
        self.group = group
        self.consumer = consumer
        self.dead_letter_stream = dead_letter_stream or f"{stream}:dead-letter"
        try:
            self.client.xgroup_create(stream, group, id="0", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def read(
        self, count: int = 50, block_ms: int = 1000, claim_idle_ms: int = 60000
    ) -> list[tuple[str, ClinicalEvent]]:
        records: list[tuple[str, list[tuple[str, dict]]]] = []
        try:
            _next_id, claimed, *_ = self.client.xautoclaim(
                self.stream,
                self.group,
                self.consumer,
                min_idle_time=claim_idle_ms,
                start_id="0-0",
                count=count,
            )
            if claimed:
                records.append((self.stream, claimed))
        except (AttributeError, TypeError):
            pass
        remaining = max(count - sum(len(messages) for _, messages in records), 0)
        if remaining:
            records.extend(
                self.client.xreadgroup(
                    self.group, self.consumer, {self.stream: ">"}, count=remaining, block=block_ms
                )
            )
        events = []
        for _, messages in records:
            for message_id, values in messages:
                try:
                    event = ClinicalEvent.model_validate(json.loads(values["event"]))
                except Exception as exc:
                    # A malformed envelope can never succeed on retry. Move it
                    # to the dead-letter stream and ACK it so the worker keeps
                    # processing healthy events.
                    self.client.xadd(
                        self.dead_letter_stream,
                        {"event": values.get("event", ""), "error": f"invalid_event:{exc}"},
                        maxlen=100000,
                        approximate=True,
                    )
                    self.ack(str(message_id))
                    continue
                events.append((str(message_id), event))
        return events

    def ack(self, message_id: str) -> None:
        self.client.xack(self.stream, self.group, message_id)

    def reject(self, message_id: str, error: str, max_delivery_attempts: int = 5) -> bool:
        """Move poison messages to a dead-letter stream after repeated failures."""
        pending = self.client.xpending_range(
            self.stream, self.group, min=message_id, max=message_id, count=1
        )
        deliveries = int(pending[0].get("times_delivered", 0)) if pending else 0
        if deliveries < max_delivery_attempts:
            return False
        entries = self.client.xrange(self.stream, min=message_id, max=message_id, count=1)
        payload = entries[0][1].get("event", "") if entries else ""
        self.client.xadd(
            self.dead_letter_stream,
            {"event": payload, "error": error},
            maxlen=100000,
            approximate=True,
        )
        self.ack(message_id)
        return True


class SyncPublisher:
    def publish(self, event: ClinicalEvent) -> str:
        return event.event_id

    def health(self) -> bool:
        return True
