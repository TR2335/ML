"""Cross-backend consistency: same event sequence → identical scores (WBS Review Focus #3)."""

import os
import socket
import time

import pytest

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://medintel:medintel@127.0.0.1:5432/medintel"
)


def _reachable(port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def test_same_event_sequence_scores_identically_across_backends(tmp_path):
    if not _reachable(5432):
        pytest.skip("环境性守卫：需要 docker compose --profile future-infra 的 PostgreSQL")
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app

    # demo clinician 固定 group-demo 租户（跨租户会被 403）；用时间戳后缀隔离历次运行。
    suffix = f"{int(time.time())}"
    sqlite_app = create_app(
        Settings(store_path=str(tmp_path / "sqlite.db"), model_path=str(tmp_path / "missing.json"))
    )
    pg_app = create_app(
        Settings(
            storage_backend="postgres",
            database_url=DATABASE_URL,
            model_path=str(tmp_path / "missing.json"),
        )
    )
    headers = {"Authorization": "Bearer demo-clinician-token"}
    enc = f"enc-it-{suffix}"
    events = [
        {
            "event_id": f"evt-it1-{suffix}",
            "event_type": "lab_result",
            "patient_id": "pat-001",
            "encounter_id": enc,
            "facility_id": "hospital-01",
            "occurred_at": "2026-01-01T06:00:00+00:00",
            "payload": {"lactate": 4.5},
        },
        # 乱序投递：时间更早的事件后到
        {
            "event_id": f"evt-it2-{suffix}",
            "event_type": "vital_signs",
            "patient_id": "pat-001",
            "encounter_id": enc,
            "facility_id": "hospital-01",
            "occurred_at": "2026-01-01T05:00:00+00:00",
            "payload": {"heart_rate": 132, "systolic_bp": 88},
        },
        # 同 id 重投递（幂等）
        {
            "event_id": f"evt-it1-{suffix}",
            "event_type": "lab_result",
            "patient_id": "pat-001",
            "encounter_id": enc,
            "facility_id": "hospital-01",
            "occurred_at": "2026-01-01T06:00:00+00:00",
            "payload": {"lactate": 4.5},
        },
    ]
    with TestClient(sqlite_app) as sqlite_client, TestClient(pg_app) as pg_client:
        for body in events:
            first = sqlite_client.post("/api/events", json=body, headers=headers)
            second = pg_client.post("/api/events", json=body, headers=headers)
            assert first.status_code == second.status_code, body["event_id"]
        sqlite_patient = sqlite_client.get(f"/api/patients/{enc}", headers=headers).json()
        pg_patient = pg_client.get(f"/api/patients/{enc}", headers=headers).json()
    assert set(sqlite_patient["risks"]) == set(pg_patient["risks"])
    for target, sqlite_score in sqlite_patient["risks"].items():
        pg_score = pg_patient["risks"][target]
        assert sqlite_score["band"] == pg_score["band"], target
        assert abs(sqlite_score["score"] - pg_score["score"]) < 1e-9, target
        assert sqlite_score["as_of"] == pg_score["as_of"], target
