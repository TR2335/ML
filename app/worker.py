import logging
import time

from .config import get_settings
from .domain import ClinicalEvent, TargetId
from .messaging import RedisStreamConsumer
from .observability import metrics
from .risk import RiskEngine
from .storage import create_store
from .store import Store

LOGGER = logging.getLogger(__name__)


def rescore_once(store: Store | None = None, engine: RiskEngine | None = None) -> int:
    settings = get_settings()
    owned_store = store is None
    store = store or create_store(settings)
    engine = engine or RiskEngine(settings)
    engines: dict[TargetId, RiskEngine] = {
        "deterioration_24h": engine,
        "readmission_30d": RiskEngine(settings, target_id="readmission_30d"),
    }
    try:
        count = 0
        for patient in store.list_all_patients():
            events = store.events_for(
                patient["encounter_id"], patient["tenant_id"], patient["facility_id"]
            )
            if events:
                snapshot = store.feature_snapshot(
                    patient["encounter_id"],
                    patient["tenant_id"],
                    patient["facility_id"],
                    as_of=events[-1].occurred_at,
                )
                scoring_event = events[-1].model_copy(update={"payload": snapshot})
                for target_engine in engines.values():
                    store.upsert_score(target_engine.score(scoring_event))
                count += 1
        return count
    finally:
        if owned_store:
            store.close()


def score_event(store: Store, engines: dict[TargetId, RiskEngine], event: ClinicalEvent) -> None:
    """Score one durable event at its event time and persist every target."""
    snapshot = store.feature_snapshot(
        event.encounter_id,
        event.tenant_id,
        event.facility_id,
        as_of=event.occurred_at,
    )
    scoring_event = event.model_copy(update={"payload": snapshot})
    for target_engine in engines.values():
        store.upsert_score(target_engine.score(scoring_event))
    metrics.inc("medintel_events_scored_total")


def consume_once(
    store: Store,
    engines: dict[TargetId, RiskEngine],
    consumer: RedisStreamConsumer,
    claim_idle_ms: int = 60000,
    max_delivery_attempts: int = 5,
) -> int:
    processed = 0
    for message_id, event in consumer.read(claim_idle_ms=claim_idle_ms):
        try:
            score_event(store, engines, event)
            consumer.ack(message_id)
            metrics.inc("medintel_queue_acked_total")
            processed += 1
        except Exception:
            metrics.inc("medintel_queue_processing_errors_total")
            LOGGER.exception("failed to process Redis event %s; leaving it pending", message_id)
            try:
                if consumer.reject(message_id, "processing_error", max_delivery_attempts):
                    metrics.inc("medintel_queue_dead_letter_total")
            except Exception:
                LOGGER.exception("failed to move Redis event %s to dead letter", message_id)
    return processed


def run_redis_worker() -> None:
    settings = get_settings()
    store = create_store(settings)
    engines: dict[TargetId, RiskEngine] = {
        "deterioration_24h": RiskEngine(settings),
        "readmission_30d": RiskEngine(settings, target_id="readmission_30d"),
    }
    consumer = RedisStreamConsumer(
        settings.redis_url,
        settings.redis_stream,
        group=settings.redis_consumer_group,
        consumer=settings.redis_consumer_name,
        dead_letter_stream=settings.redis_dead_letter_stream,
    )
    try:
        while True:
            try:
                consume_once(
                    store,
                    engines,
                    consumer,
                    settings.redis_claim_idle_ms,
                    settings.redis_max_delivery_attempts,
                )
            except Exception:
                # Redis outages should trigger reconnect/backoff rather than
                # silently stopping the worker process.
                metrics.inc("medintel_queue_read_errors_total")
                LOGGER.exception("Redis stream read failed; retrying")
                time.sleep(min(max(settings.score_interval_seconds, 2), 30))
    finally:
        store.close()


if __name__ == "__main__":
    settings = get_settings()
    if settings.event_backend == "redis":
        run_redis_worker()
        raise SystemExit(0)
    while True:
        rescore_once()
        time.sleep(settings.score_interval_seconds)
