import logging
import os
import tempfile
import time
from pathlib import Path

from .config import get_settings
from .domain import ClinicalEvent, TargetId
from .messaging import RedisStreamConsumer
from .observability import metrics
from .risk import RiskEngine
from .storage import create_store
from .store import Store

LOGGER = logging.getLogger(__name__)
HEARTBEAT_PATH = Path(
    os.environ.get("WORKER_HEARTBEAT_PATH")
    or Path(tempfile.gettempdir()) / "medintel-worker-heartbeat"
)


def _require_known_backend(settings) -> None:
    if settings.event_backend not in {"sync", "redis"}:
        raise ValueError(f"unsupported event backend: {settings.event_backend}")


def _touch_heartbeat() -> None:
    """Healthcheck signal: written every live cycle so a hang stops the updates."""
    try:
        HEARTBEAT_PATH.touch()
    except OSError:
        LOGGER.warning("failed to update worker heartbeat at %s", HEARTBEAT_PATH)


def sync_backoff_seconds(base_interval: int, consecutive_failures: int) -> float:
    """Fixed cadence after success; capped exponential backoff after failures."""
    if consecutive_failures <= 0:
        return float(max(base_interval, 1))
    return float(min(max(base_interval, 1) * 2**consecutive_failures, 3600))


def run_sync_worker() -> None:
    settings = get_settings()
    failures = 0
    while True:
        try:
            rescore_once()
            failures = 0
        except Exception:
            failures += 1
            LOGGER.exception("sync rescore cycle failed (%s consecutive)", failures)
        _touch_heartbeat()
        time.sleep(sync_backoff_seconds(settings.score_interval_seconds, failures))


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


def reconcile(store: Store, engines: dict[TargetId, RiskEngine]) -> int:
    """Backfill missing or stale scores from durable events without regressing newer ones."""
    count = 0
    for tenant_id, facility_id, encounter_id, latest in store.encounter_latest_event_times():
        scores = store.get_scores(encounter_id, tenant_id, facility_id)
        stale = False
        for target_id in engines:
            score = scores.get(target_id)
            if score is None:
                stale = True
                break
            effective = score.as_of or score.scored_at
            if effective < latest:
                stale = True
                break
        if not stale:
            continue
        events = store.events_for(encounter_id, tenant_id, facility_id)
        if not events:
            continue
        snapshot = store.feature_snapshot(encounter_id, tenant_id, facility_id, as_of=latest)
        scoring_event = max(events, key=lambda item: item.occurred_at).model_copy(
            update={"payload": snapshot}
        )
        for target_engine in engines.values():
            store.upsert_score(target_engine.score(scoring_event))
        count += 1
    return count


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
    cycle = 0
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
                cycle += 1
                if cycle % max(settings.reconcile_interval_cycles, 1) == 0:
                    reconcile(store, engines)
            except Exception:
                # Redis outages should trigger reconnect/backoff rather than
                # silently stopping the worker process.
                metrics.inc("medintel_queue_read_errors_total")
                LOGGER.exception("Redis stream read failed; retrying")
                time.sleep(min(max(settings.score_interval_seconds, 2), 30))
            _touch_heartbeat()
    finally:
        store.close()


if __name__ == "__main__":
    settings = get_settings()
    _require_known_backend(settings)
    if settings.event_backend == "redis":
        run_redis_worker()
        raise SystemExit(0)
    run_sync_worker()
