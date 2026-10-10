import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

from .config import get_settings
from .domain import ClinicalEvent, TargetId, targets_for_event
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


def _first_discharge(events: list[ClinicalEvent]) -> ClinicalEvent | None:
    discharges = [event for event in events if event.event_type == "discharge"]
    if not discharges:
        return None
    return min(discharges, key=lambda event: (event.occurred_at, event.event_id))


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
                store.upsert_score(engines["deterioration_24h"].score(scoring_event))
                discharge = _first_discharge(events)
                if discharge is not None:
                    discharge_snapshot = store.feature_snapshot(
                        patient["encounter_id"],
                        patient["tenant_id"],
                        patient["facility_id"],
                        as_of=discharge.occurred_at,
                    )
                    discharge_event = discharge.model_copy(update={"payload": discharge_snapshot})
                    # upsert_score 的 as_of 防倒退：只在缺失或落后于出院时刻时写入。
                    store.upsert_score(engines["readmission_30d"].score(discharge_event))
                count += 1
        return count
    finally:
        if owned_store:
            store.close()


def reconcile(store: Store, engines: dict[TargetId, RiskEngine]) -> int:
    """Backfill missing or stale scores from durable events without regressing newer ones.

    readmission_30d 以首个出院事件为基准（WBS T8.2）：无出院事件则不要求该分数。
    """
    count = 0
    for tenant_id, facility_id, encounter_id, latest in store.encounter_latest_event_times():
        scores = store.get_scores(encounter_id, tenant_id, facility_id)
        events = store.events_for(encounter_id, tenant_id, facility_id)
        if not events:
            continue
        discharge = _first_discharge(events)
        plans: list[tuple[TargetId, RiskEngine, datetime, ClinicalEvent | None]] = []
        if "deterioration_24h" in engines:
            current = scores.get("deterioration_24h")
            if current is None or (current.as_of or current.scored_at) < latest:
                plans.append(("deterioration_24h", engines["deterioration_24h"], latest, None))
        if "readmission_30d" in engines and discharge is not None:
            current = scores.get("readmission_30d")
            if current is None or (current.as_of or current.scored_at) < discharge.occurred_at:
                plans.append(
                    (
                        "readmission_30d",
                        engines["readmission_30d"],
                        discharge.occurred_at,
                        discharge,
                    )
                )
        if not plans:
            continue
        latest_event = max(events, key=lambda item: item.occurred_at)
        for _target_id, target_engine, as_of, trigger in plans:
            snapshot = store.feature_snapshot(encounter_id, tenant_id, facility_id, as_of=as_of)
            source = trigger or latest_event
            scoring_event = source.model_copy(update={"payload": snapshot})
            store.upsert_score(target_engine.score(scoring_event))
        count += 1
    return count


def score_event(store: Store, engines: dict[TargetId, RiskEngine], event: ClinicalEvent) -> None:
    """Score one durable event at its event time and persist policy targets."""
    snapshot = store.feature_snapshot(
        event.encounter_id,
        event.tenant_id,
        event.facility_id,
        as_of=event.occurred_at,
    )
    scoring_event = event.model_copy(update={"payload": snapshot})
    # 按触发策略过滤：readmission 仅出院事件（WBS T8.2）。
    for target_id in targets_for_event(event):
        target_engine = engines.get(target_id)
        if target_engine is not None:
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
