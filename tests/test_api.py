import re
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.security import principal_from_claims


def test_health_and_patient_queue(tmp_path):
    application = create_app(
        Settings(store_path=str(tmp_path / "store.db"), model_path=str(tmp_path / "missing.json"))
    )
    with TestClient(application) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        patients = client.get(
            "/api/patients", headers={"Authorization": "Bearer demo-clinician-token"}
        )
        assert patients.status_code == 200
        assert len(patients.json()) == 18
        assert "risk" in patients.json()[0]
        detail = client.get(
            "/api/patients/enc-001", headers={"Authorization": "Bearer demo-clinician-token"}
        )
        # readmission 仅出院触发（WBS T8.2）：演示种子不再预置，故只有 deterioration。
        assert set(detail.json()["risks"]) == {"deterioration_24h"}


def test_audit_requires_quality_role(tmp_path):
    application = create_app(
        Settings(store_path=str(tmp_path / "store.db"), model_path=str(tmp_path / "missing.json"))
    )
    with TestClient(application) as client:
        response = client.get(
            "/api/audit", headers={"Authorization": "Bearer demo-clinician-token"}
        )
        assert response.status_code == 403


def test_auth_scope_and_event_idempotency(tmp_path):
    application = create_app(
        Settings(store_path=str(tmp_path / "store.db"), model_path=str(tmp_path / "missing.json"))
    )
    event = {
        "event_id": "evt-test-1",
        "event_type": "lab_result",
        "patient_id": "pat-001",
        "encounter_id": "enc-001",
        "tenant_id": "group-demo",
        "facility_id": "hospital-01",
        "payload": {"lactate": 3.0},
    }
    with TestClient(application) as client:
        assert client.get("/api/patients").status_code == 401
        assert (
            client.post(
                "/api/events",
                json={**event, "encounter_id": "unknown"},
                headers={"Authorization": "Bearer demo-clinician-token"},
            ).status_code
            == 404
        )
        first = client.post(
            "/api/events", json=event, headers={"Authorization": "Bearer demo-clinician-token"}
        )
        second = client.post(
            "/api/events", json=event, headers={"Authorization": "Bearer demo-clinician-token"}
        )
        assert first.status_code == 202
        assert first.json()["accepted"] is True
        assert second.json()["duplicate"] is True


def test_invalid_clinical_value_is_rejected(tmp_path):
    application = create_app(
        Settings(store_path=str(tmp_path / "store.db"), model_path=str(tmp_path / "missing.json"))
    )
    with TestClient(application) as client:
        response = client.post(
            "/api/events",
            json={
                "event_type": "vital_signs",
                "patient_id": "pat-001",
                "encounter_id": "enc-001",
                "payload": {"heart_rate": "bad"},
            },
            headers={"Authorization": "Bearer demo-clinician-token"},
        )
        assert response.status_code == 422


def test_boolean_payload_value_returns_422(tmp_path):
    application = create_app(
        Settings(store_path=str(tmp_path / "store.db"), model_path=str(tmp_path / "missing.json"))
    )
    with TestClient(application) as client:
        response = client.post(
            "/api/events",
            json={
                "event_type": "vital_signs",
                "patient_id": "pat-001",
                "encounter_id": "enc-001",
                "payload": {"age": True},
            },
            headers={"Authorization": "Bearer demo-clinician-token"},
        )
        assert response.status_code == 422


def test_partial_event_uses_existing_feature_snapshot(tmp_path):
    application = create_app(
        Settings(store_path=str(tmp_path / "store.db"), model_path=str(tmp_path / "missing.json"))
    )
    with TestClient(application) as client:
        patient = client.get(
            "/api/patients/enc-002", headers={"Authorization": "Bearer demo-clinician-token"}
        ).json()
        initial = patient["risk"]["score"]
        response = client.post(
            "/api/events",
            json={
                "event_type": "lab_result",
                "patient_id": "pat-002",
                "encounter_id": "enc-002",
                "payload": {"lactate": 3.0},
            },
            headers={"Authorization": "Bearer demo-clinician-token"},
        )
        assert response.status_code == 202
        assert response.json()["score"]["score"] >= initial * 0.5


def test_model_card_requires_authenticated_quality_access(tmp_path):
    application = create_app(
        Settings(store_path=str(tmp_path / "store.db"), model_path=str(tmp_path / "missing.json"))
    )
    with TestClient(application) as client:
        assert client.get("/api/model-card").status_code == 401
        assert (
            client.get(
                "/api/model-card", headers={"Authorization": "Bearer demo-clinician-token"}
            ).status_code
            == 403
        )
        card = client.get("/api/model-card", headers={"Authorization": "Bearer demo-quality-token"})
        assert card.status_code == 200
        assert set(card.json()["models"]) == {"deterioration_24h", "readmission_30d"}


def test_event_id_conflict_is_rejected_and_missingness_is_exposed(tmp_path):
    application = create_app(
        Settings(store_path=str(tmp_path / "store.db"), model_path=str(tmp_path / "missing.json"))
    )
    event = {
        "event_id": "evt-conflict",
        "event_type": "lab_result",
        "occurred_at": "2026-09-09T08:30:00Z",
        "patient_id": "pat-001",
        "encounter_id": "enc-001",
        "tenant_id": "group-demo",
        "facility_id": "hospital-01",
        "payload": {"lactate": 3.0},
    }
    headers = {"Authorization": "Bearer demo-clinician-token"}
    with TestClient(application) as client:
        first = client.post("/api/events", json=event, headers=headers)
        assert first.status_code == 202
        assert "missing_features" in first.json()["score"]
        assert set(first.json()["scores"]) == {"deterioration_24h"}
        conflict = client.post(
            "/api/events", json={**event, "payload": {"lactate": 4.0}}, headers=headers
        )
        assert conflict.status_code == 409


def test_future_event_time_is_rejected(tmp_path):
    application = create_app(
        Settings(store_path=str(tmp_path / "store.db"), model_path=str(tmp_path / "missing.json"))
    )
    with TestClient(application) as client:
        response = client.post(
            "/api/events",
            json={
                "event_type": "vital_signs",
                "patient_id": "pat-001",
                "encounter_id": "enc-001",
                "occurred_at": "2099-01-01T00:00:00Z",
                "payload": {"heart_rate": 80},
            },
            headers={"Authorization": "Bearer demo-clinician-token"},
        )
        assert response.status_code == 422
        assert response.json()["detail"] == "event_time_in_future"


def test_oidc_claims_map_to_scoped_principal():
    principal = principal_from_claims(
        {
            "sub": "user-123",
            "roles": ["clinician"],
            "tenant_id": "tenant-a",
            "facility_id": "facility-a",
        },
        Settings(),
    )
    assert principal.subject == "user-123"
    assert principal.role == "clinician"
    assert principal.tenant_id == "tenant-a"
    assert principal.facility_id == "facility-a"


def test_production_refuses_to_start_without_oidc(tmp_path):
    with pytest.raises(RuntimeError, match="require OIDC"):
        create_app(
            Settings(
                app_env="prod",
                oidc_enabled=False,
                demo_auth_enabled=False,
                store_path=str(tmp_path / "store.db"),
                model_path=str(tmp_path / "model.json"),
            )
        )


def test_metrics_and_degraded_health(tmp_path):
    application = create_app(
        Settings(
            store_path=str(tmp_path / "store.db"),
            model_path=str(tmp_path / "missing.json"),
            readmission_model_path=str(tmp_path / "missing-readmission.json"),
            model_fallback_enabled=True,
        )
    )
    with TestClient(application) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "degraded"
        assert set(health.json()["degraded_models"]) == {
            "deterioration_24h",
            "readmission_30d",
        }
        metrics_response = client.get("/api/metrics")
        assert metrics_response.status_code == 200
        assert (
            "medintel_events_received_total" in metrics_response.text or metrics_response.text == ""
        )


def test_metrics_can_be_disabled(tmp_path):
    application = create_app(
        Settings(
            store_path=str(tmp_path / "store.db"),
            model_path=str(tmp_path / "missing.json"),
            metrics_enabled=False,
        )
    )
    with TestClient(application) as client:
        assert client.get("/api/metrics").status_code == 404


def test_review_status_transitions(tmp_path):
    application = create_app(
        Settings(store_path=str(tmp_path / "store.db"), model_path=str(tmp_path / "missing.json"))
    )
    headers = {"Authorization": "Bearer demo-clinician-token"}
    with TestClient(application) as client:
        for status, note in [
            ("acknowledged", "已确认告警"),
            ("in_progress", "责任团队处置中"),
            ("resolved", "处置完成"),
        ]:
            response = client.post(
                "/api/patients/enc-001/review",
                json={"status": status, "note": note},
                headers=headers,
            )
            assert response.status_code == 200
            stored = client.get("/api/patients/enc-001", headers=headers).json()["risks"][
                "deterioration_24h"
            ]
            assert stored["status"] == status
            assert stored["human_review"] == note


def test_index_security_headers(tmp_path):
    application = create_app(
        Settings(store_path=str(tmp_path / "store.db"), model_path=str(tmp_path / "missing.json"))
    )
    with TestClient(application) as client:
        response = client.get("/")
        assert response.status_code == 200
        csp = response.headers.get("content-security-policy", "")
        assert "default-src 'self'" in csp
        assert "script-src 'self'" in csp
        assert response.headers.get("x-content-type-options") == "nosniff"
        html = response.text
        assert "<script src=" in html
        assert re.findall(r"<script(?![^>]*\bsrc=)[^>]*>", html) == []
        assert "onclick=" not in html


def test_static_app_js_is_served(tmp_path):
    application = create_app(
        Settings(store_path=str(tmp_path / "store.db"), model_path=str(tmp_path / "missing.json"))
    )
    with TestClient(application) as client:
        response = client.get("/static/app.js")
        assert response.status_code == 200
        assert "const safe =" in response.text


def test_auth_dependency_runs_in_threadpool(tmp_path, monkeypatch):
    import asyncio as _asyncio

    from app import security

    application = create_app(
        Settings(store_path=str(tmp_path / "store.db"), model_path=str(tmp_path / "missing.json"))
    )
    observed: dict[str, bool] = {}
    original = security.authenticate

    def spy(authorization, settings):
        try:
            _asyncio.get_running_loop()
            observed["in_event_loop"] = True
        except RuntimeError:
            observed["in_event_loop"] = False
        return original(authorization, settings)

    monkeypatch.setattr(security, "authenticate", spy)
    with TestClient(application) as client:
        response = client.get(
            "/api/patients", headers={"Authorization": "Bearer demo-clinician-token"}
        )
    assert response.status_code == 200
    assert observed["in_event_loop"] is False


def test_admin_action_visible_in_facility_audit(tmp_path):
    application = create_app(
        Settings(store_path=str(tmp_path / "store.db"), model_path=str(tmp_path / "missing.json"))
    )
    with TestClient(application) as client:
        # admin（facility None → "*"）复核 hospital-01 的评分
        review = client.post(
            "/api/patients/enc-001/review",
            json={
                "status": "acknowledged",
                "note": "跨院区管理员处理",
                "target_id": "deterioration_24h",
            },
            headers={"Authorization": "Bearer demo-admin-token"},
        )
        assert review.status_code == 200
        score_id = review.json()["score_id"]
        audit = client.get("/api/audit", headers={"Authorization": "Bearer demo-quality-token"})
        assert audit.status_code == 200
        entries = [
            item
            for item in audit.json()
            if item["action"] == "human_review" and item["resource_id"] == score_id
        ]
        assert entries, "hospital-01 过滤必须能看到对 hospital-01 资源的管理操作"
        assert entries[0]["resource_facility_id"] == "hospital-01"
        assert entries[0]["actor_facility_id"] == "*"


def test_health_does_not_leak_paths(tmp_path):
    application = create_app(
        Settings(store_path=str(tmp_path / "store.db"), model_path=str(tmp_path / "missing.json"))
    )
    with TestClient(application) as client:
        body = client.get("/api/health").json()
    reasons = body["degraded_reasons"]
    assert reasons, "缺失工件应记录 degraded_reason"
    allowed = {"model_artifact_missing", "model_signature_invalid", "model_artifact_invalid"}
    for reason in reasons.values():
        assert reason in allowed, f"health 泄漏原始原因/路径: {reason!r}"
        assert str(tmp_path) not in reason


def test_duplicate_event_with_different_time_rejected_409(tmp_path):
    application = create_app(
        Settings(store_path=str(tmp_path / "store.db"), model_path=str(tmp_path / "missing.json"))
    )
    event = {
        "event_id": "evt-time-1",
        "event_type": "lab_result",
        "patient_id": "pat-001",
        "encounter_id": "enc-001",
        "occurred_at": "2026-01-01T06:00:00+00:00",
        "payload": {"lactate": 3.0},
    }
    headers = {"Authorization": "Bearer demo-clinician-token"}
    with TestClient(application) as client:
        first = client.post("/api/events", json=event, headers=headers)
        assert first.status_code == 202
        # 同 id、显式不同的时间 → 冲突而非幂等
        changed = client.post(
            "/api/events",
            json={**event, "occurred_at": "2026-01-01T07:00:00+00:00"},
            headers=headers,
        )
        assert changed.status_code == 409
        # 同 id、同显式时间（其余一致）→ 仍是幂等重复
        again = client.post("/api/events", json=event, headers=headers)
        assert again.status_code == 202
        assert again.json()["duplicate"] is True


def test_patient_pagination(tmp_path):
    application = create_app(
        Settings(store_path=str(tmp_path / "store.db"), model_path=str(tmp_path / "missing.json"))
    )
    headers = {"Authorization": "Bearer demo-clinician-token"}
    with TestClient(application) as client:  # dev 种子共 18 名患者
        page = client.get("/api/patients?limit=5", headers=headers)
        assert page.status_code == 200
        assert len(page.json()) == 5
        assert page.headers["X-Total-Count"] == "18"

        tail = client.get("/api/patients?limit=5&offset=15", headers=headers)
        assert len(tail.json()) == 3
        assert tail.headers["X-Total-Count"] == "18"  # 总数与切片无关

        beyond = client.get("/api/patients?limit=5&offset=999", headers=headers)
        assert beyond.json() == []
        assert beyond.headers["X-Total-Count"] == "18"

        filtered = client.get("/api/patients?band=critical&limit=5", headers=headers)
        total = int(filtered.headers["X-Total-Count"])
        assert total == len(client.get("/api/patients?band=critical", headers=headers).json())

        too_big = client.get("/api/patients?limit=1001", headers=headers)
        assert too_big.status_code == 422


def test_oversized_payload_rejected_422(tmp_path):
    application = create_app(
        Settings(store_path=str(tmp_path / "store.db"), model_path=str(tmp_path / "missing.json"))
    )
    headers = {"Authorization": "Bearer demo-clinician-token"}
    base = {
        "event_type": "note",
        "patient_id": "pat-001",
        "encounter_id": "enc-001",
    }
    with TestClient(application) as client:
        # 单值 > 4KB → 422
        too_big_value = client.post(
            "/api/events",
            json={**base, "event_id": "evt-too-big-value", "payload": {"comment": "x" * 5000}},
            headers=headers,
        )
        assert too_big_value.status_code == 422
        # 每个值 ≤4096 但总 payload > 64KB → 422（17 × 4000 ≈ 68KB；键数 17 < 100 上限）
        too_big_total = client.post(
            "/api/events",
            json={
                **base,
                "event_id": "evt-too-big-total",
                "payload": {f"k{i}": "v" * 4000 for i in range(17)},
            },
            headers=headers,
        )
        assert too_big_total.status_code == 422
        # 通过校验的正常事件仍可入库（边界守卫）
        ok = client.post(
            "/api/events",
            json={**base, "event_id": "evt-ok", "payload": {"comment": "y" * 4000}},
            headers=headers,
        )
        assert ok.status_code == 202


def test_invalid_event_backend_rejected_consistently(tmp_path):
    import pytest

    from app import worker
    from app.api import build_router
    from app.risk import RiskEngine
    from app.store import Store

    settings = Settings(
        event_backend="kafka",
        store_path=str(tmp_path / "store.db"),
        model_path=str(tmp_path / "missing.json"),
    )
    store = Store(str(tmp_path / "store.db"))
    engine = RiskEngine(settings)
    with pytest.raises(ValueError, match="unsupported event backend: kafka"):
        build_router(store, engine, settings)
    with pytest.raises(ValueError, match="unsupported event backend: kafka"):
        worker._require_known_backend(settings)
    store.close()


def test_non_discharge_event_only_scores_deterioration(tmp_path):
    application = create_app(
        Settings(store_path=str(tmp_path / "store.db"), model_path=str(tmp_path / "missing.json"))
    )
    headers = {"Authorization": "Bearer demo-clinician-token"}
    with TestClient(application) as client:
        response = client.post(
            "/api/events",
            json={
                "event_id": "evt-vital-only",
                "event_type": "vital_signs",
                "patient_id": "pat-001",
                "encounter_id": "enc-001",
                "occurred_at": "2026-09-09T08:30:00Z",
                "payload": {"heart_rate": 120},
            },
            headers=headers,
        )
        assert response.status_code == 202
        assert set(response.json()["scores"]) == {"deterioration_24h"}
        assert response.json()["score"]["target_id"] == "deterioration_24h"


def test_discharge_event_scores_readmission_exactly_once(tmp_path):
    application = create_app(
        Settings(store_path=str(tmp_path / "store.db"), model_path=str(tmp_path / "missing.json"))
    )
    headers = {"Authorization": "Bearer demo-clinician-token"}
    discharge = {
        "event_id": "evt-discharge-1",
        "event_type": "discharge",
        "patient_id": "pat-001",
        "encounter_id": "enc-001",
        "occurred_at": "2026-09-09T10:00:00Z",
        "payload": {
            "discharge_destination": "home",
            "los_days": 4.0,
            "diagnosis_codes": ["I50.9"],
            "comorbidity_index": 1.0,
            "prior_utilization_12m": 0,
            "ed_visits_6m": 1,
        },
    }
    with TestClient(application) as client:
        first = client.post("/api/events", json=discharge, headers=headers)
        assert first.status_code == 202
        assert set(first.json()["scores"]) == {"deterioration_24h", "readmission_30d"}
        later = client.post(
            "/api/events",
            json={
                "event_id": "evt-vital-after",
                "event_type": "vital_signs",
                "patient_id": "pat-001",
                "encounter_id": "enc-001",
                "occurred_at": "2026-09-09T12:00:00Z",
                "payload": {"heart_rate": 130},
            },
            headers=headers,
        )
        assert later.status_code == 202
        assert set(later.json()["scores"]) == {"deterioration_24h"}
    from app.store import Store

    with Store(str(tmp_path / "store.db")) as store:
        readmission = store.get_score("enc-001", "group-demo", "hospital-01", "readmission_30d")
        assert readmission is not None
        assert readmission.as_of == datetime(2026, 9, 9, 10, 0, tzinfo=UTC)


def test_second_discharge_event_does_not_rescore_readmission_api(tmp_path):
    """P8 deferred 4：API live 路径与 worker/批路径一致——第二个出院事件不返回
    readmission、不改写 as_of（deterioration 照常更新）。"""
    application = create_app(
        Settings(store_path=str(tmp_path / "store.db"), model_path=str(tmp_path / "missing.json"))
    )
    headers = {"Authorization": "Bearer demo-clinician-token"}
    first_discharge = {
        "event_id": "evt-d1",
        "event_type": "discharge",
        "patient_id": "pat-001",
        "encounter_id": "enc-001",
        "occurred_at": "2026-09-09T10:00:00Z",
        "payload": {
            "discharge_destination": "home",
            "los_days": 4.0,
            "diagnosis_codes": ["I50.9"],
            "comorbidity_index": 1.0,
            "prior_utilization_12m": 0,
            "ed_visits_6m": 1,
        },
    }
    second_discharge = {
        **first_discharge,
        "event_id": "evt-d2",
        "occurred_at": "2026-09-11T10:00:00Z",
        "payload": {**first_discharge["payload"], "los_days": 6.0},
    }
    with TestClient(application) as client:
        first = client.post("/api/events", json=first_discharge, headers=headers)
        assert first.status_code == 202
        assert set(first.json()["scores"]) == {"deterioration_24h", "readmission_30d"}
        second = client.post("/api/events", json=second_discharge, headers=headers)
        assert second.status_code == 202
        # deterioration 逐事件重打（响应即为当次计算结果，as_of 钉 d2）；
        # readmission 单次评分不重打（响应不含该目标）
        assert set(second.json()["scores"]) == {"deterioration_24h"}
        assert second.json()["scores"]["deterioration_24h"]["as_of"].startswith("2026-09-11")
    from app.store import Store

    with Store(str(tmp_path / "store.db")) as store:
        readmission = store.get_score("enc-001", "group-demo", "hospital-01", "readmission_30d")
        assert readmission is not None
        assert readmission.as_of == datetime(2026, 9, 9, 10, 0, tzinfo=UTC)


def test_discharge_event_missing_snapshot_is_422(tmp_path):
    application = create_app(
        Settings(store_path=str(tmp_path / "store.db"), model_path=str(tmp_path / "missing.json"))
    )
    headers = {"Authorization": "Bearer demo-clinician-token"}
    with TestClient(application) as client:
        response = client.post(
            "/api/events",
            json={
                "event_id": "evt-discharge-bad",
                "event_type": "discharge",
                "patient_id": "pat-001",
                "encounter_id": "enc-001",
                "occurred_at": "2026-09-09T10:00:00Z",
                "payload": {"discharge_destination": "home"},
            },
            headers=headers,
        )
        assert response.status_code == 422
        assert "los_days" in str(response.json())
