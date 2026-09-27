from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse

from .config import Settings
from .domain import ClinicalEvent, ReviewRequest, RiskScore, TargetId
from .messaging import RedisStreamPublisher, SyncPublisher
from .observability import metrics
from .risk import RiskEngine
from .security import Principal, audit, require_permission
from .store import Store
from .validation import is_future_offset


def score_all_targets(engines: dict[TargetId, RiskEngine], event: ClinicalEvent) -> list[RiskScore]:
    return [target_engine.score(event) for target_engine in engines.values()]


def build_router(store: Store, engine: RiskEngine, settings: Settings) -> APIRouter:
    router = APIRouter(prefix="/api")
    engines: dict[TargetId, RiskEngine] = {
        "deterioration_24h": engine,
        "readmission_30d": RiskEngine(settings, target_id="readmission_30d"),
    }
    if settings.event_backend == "sync":
        publisher = SyncPublisher()
    elif settings.event_backend == "redis":
        publisher = RedisStreamPublisher(settings.redis_url, settings.redis_stream)
    else:
        raise ValueError(f"unsupported event backend: {settings.event_backend}")
    model_reader = Depends(require_permission("model:read", settings))
    risk_reader = Depends(require_permission("risk:read", settings))
    risk_reviewer = Depends(require_permission("risk:review", settings))
    audit_reader = Depends(require_permission("audit:read", settings))

    @router.get("/health")
    def health() -> dict:
        degraded = [target for target, target_engine in engines.items() if target_engine.degraded]
        event_bus_ok = True
        if settings.event_backend == "redis":
            event_bus_ok = publisher.health()
        return {
            "status": "degraded" if degraded or not event_bus_ok else "ok",
            "environment": settings.app_env,
            "event_backend": settings.event_backend,
            "degraded_models": degraded,
            "degraded_reasons": {
                target: target_engine.degraded_reason
                for target, target_engine in engines.items()
                if target_engine.degraded_reason
            },
            "event_bus": "ok" if event_bus_ok else "unavailable",
            "model_version": engine.version,
            "models": {target: target_engine.version for target, target_engine in engines.items()},
        }

    @router.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
    def metrics_endpoint() -> str:
        if not settings.metrics_enabled:
            raise HTTPException(status_code=404, detail="metrics_disabled")
        return metrics.render()

    @router.get("/model-card")
    def model_card(principal: Principal = model_reader) -> dict:
        return {
            "model_version": engine.version,
            "purpose": "成人普通住院患者风险分层；两个临床目标独立建模",
            "label_definition": engine.label_definition,
            "algorithm": engine.algorithm,
            "models": {
                target: {
                    "model_version": target_engine.version,
                    "label_definition": target_engine.label_definition,
                    "algorithm": target_engine.algorithm,
                    "thresholds": target_engine.thresholds,
                }
                for target, target_engine in engines.items()
            },
            "decision_role": "仅供人工复核与资源分配，不单独改变诊疗方案",
            "features": list(engine.weights),
            "validation_status": "demo-baseline-not-clinically-validated",
        }

    @router.get("/patients")
    def patients(
        band: str | None = Query(default=None, pattern="^(low|medium|high|critical)$"),
        target_id: TargetId = "deterioration_24h",
        principal: Principal = risk_reader,
    ) -> list[dict]:
        return store.list_patients(principal.tenant_id, principal.facility_id, band, target_id)

    @router.get("/patients/{encounter_id}")
    def patient(
        encounter_id: str,
        principal: Principal = risk_reader,
    ) -> dict:
        row = store.get_patient(encounter_id, principal.tenant_id, principal.facility_id)
        if not row:
            raise HTTPException(status_code=404, detail="encounter_not_found")
        return {
            **row,
            "risks": {
                target: score.model_dump(mode="json")
                for target, score in store.get_scores(
                    encounter_id, principal.tenant_id, principal.facility_id
                ).items()
            },
            "events": [
                event.model_dump(mode="json")
                for event in store.events_for(
                    encounter_id, principal.tenant_id, principal.facility_id
                )
            ],
        }

    @router.post("/events", status_code=202)
    def ingest_event(
        event: ClinicalEvent,
        principal: Principal = risk_reviewer,
    ) -> dict:
        metrics.inc("medintel_events_received_total")
        if is_future_offset(
            event.occurred_at, max_skew_minutes=settings.max_event_future_skew_minutes
        ):
            metrics.inc("medintel_events_rejected_total")
            raise HTTPException(status_code=422, detail="event_time_in_future")
        if event.tenant_id != principal.tenant_id or (
            principal.facility_id and event.facility_id != principal.facility_id
        ):
            metrics.inc("medintel_events_rejected_total")
            raise HTTPException(status_code=403, detail="event_scope_forbidden")
        if not store.has_encounter(
            event.encounter_id, event.patient_id, principal.tenant_id, event.facility_id
        ):
            metrics.inc("medintel_events_rejected_total")
            raise HTTPException(status_code=404, detail="encounter_not_found")
        inserted = store.add_event(event)
        if not inserted:
            metrics.inc("medintel_events_duplicate_total")
            existing_event = store.get_event(event.event_id, event.tenant_id)
            if existing_event is None:
                # The insert may have raced with a transaction that has not
                # become visible yet. Let the retry path below rebuild from
                # the current snapshot once the row is visible.
                existing_event = event
            elif existing_event.model_dump(
                mode="json", exclude={"occurred_at"}
            ) != event.model_dump(mode="json", exclude={"occurred_at"}):
                raise HTTPException(status_code=409, detail="event_id_conflict")
            existing_scores = store.get_scores(
                event.encounter_id, principal.tenant_id, principal.facility_id
            )
            scores_missing = not existing_scores
            # A concurrent first delivery may have committed the event before
            # its score. Rebuild it from the complete snapshot so retries are
            # still useful and never return an empty score for a valid event.
            if settings.event_backend == "redis" and scores_missing:
                try:
                    message_id = publisher.publish(existing_event)
                except Exception as exc:
                    metrics.inc("medintel_event_publish_errors_total")
                    raise HTTPException(status_code=503, detail="event_queue_unavailable") from exc
                metrics.inc("medintel_events_queued_total")
                return {
                    "accepted": False,
                    "duplicate": True,
                    "queued": True,
                    "message_id": message_id,
                }
            if not existing_scores:
                snapshot = store.feature_snapshot(
                    event.encounter_id,
                    principal.tenant_id,
                    principal.facility_id,
                    as_of=store.latest_event_time(
                        event.encounter_id, principal.tenant_id, principal.facility_id
                    ),
                )
                scoring_event = event.model_copy(update={"payload": snapshot})
                existing_scores = {
                    score.target_id: score for score in score_all_targets(engines, scoring_event)
                }
                for score in existing_scores.values():
                    store.upsert_score(score)
            primary = existing_scores.get("deterioration_24h") or next(
                iter(existing_scores.values())
            )
            return {
                "accepted": False,
                "duplicate": True,
                "score": primary,
                "scores": existing_scores,
            }
        if settings.event_backend == "redis":
            try:
                message_id = publisher.publish(event)
            except Exception as exc:
                metrics.inc("medintel_event_publish_errors_total")
                # Keep the durable event; callers can retry publication safely
                # because event_id is idempotent.
                audit(
                    store,
                    principal,
                    "event_publish_failed",
                    "event",
                    event.event_id,
                    {"error": str(exc)},
                )
                raise HTTPException(status_code=503, detail="event_queue_unavailable") from exc
            metrics.inc("medintel_events_queued_total")
            audit(
                store,
                principal,
                "event_queued",
                "event",
                event.event_id,
                {"message_id": message_id},
            )
            return {"accepted": True, "duplicate": False, "queued": True, "message_id": message_id}
        snapshot = store.feature_snapshot(
            event.encounter_id,
            principal.tenant_id,
            principal.facility_id,
            as_of=store.latest_event_time(
                event.encounter_id, principal.tenant_id, principal.facility_id
            ),
        )
        scoring_event = event.model_copy(update={"payload": snapshot})
        scored = score_all_targets(engines, scoring_event)
        for score in scored:
            store.upsert_score(score)
        metrics.inc("medintel_events_scored_total")
        score = next(item for item in scored if item.target_id == "deterioration_24h")
        audit(
            store,
            principal,
            "event_scored",
            "encounter",
            event.encounter_id,
            {
                "event_type": event.event_type,
                "models": {item.target_id: item.model_version for item in scored},
                "missing_features": {item.target_id: item.missing_features for item in scored},
            },
        )
        return {
            "accepted": True,
            "duplicate": False,
            "score": score,
            "scores": {item.target_id: item for item in scored},
        }

    @router.post("/patients/{encounter_id}/review")
    def review(
        encounter_id: str,
        request: ReviewRequest,
        principal: Principal = risk_reviewer,
    ) -> dict:
        score = store.get_score(
            encounter_id,
            principal.tenant_id,
            principal.facility_id,
            target_id=request.target_id,
        )
        if not score:
            raise HTTPException(status_code=404, detail="score_not_found")
        score.status = request.status
        score.human_review = request.note
        store.upsert_score(score, preserve_review=False)
        metrics.inc("medintel_reviews_total")
        audit(
            store,
            principal,
            "human_review",
            "risk_score",
            score.score_id,
            {"status": request.status, "note": request.note, "target_id": request.target_id},
        )
        return score.model_dump(mode="json")

    @router.get("/audit")
    def audit_log(
        limit: int = Query(default=50, ge=1, le=500),
        principal: Principal = audit_reader,
    ) -> list[dict]:
        return [
            item.model_dump(mode="json")
            for item in store.audits(principal.tenant_id, principal.facility_id, limit)
        ]

    return router
