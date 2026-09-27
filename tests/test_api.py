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
        assert set(detail.json()["risks"]) == {"deterioration_24h", "readmission_30d"}


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
        assert set(first.json()["scores"]) == {"deterioration_24h", "readmission_30d"}
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
