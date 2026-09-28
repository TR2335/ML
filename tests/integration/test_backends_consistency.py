"""Cross-backend consistency: same event sequence → identical scores (WBS Review Focus #3)."""

import os
import socket
import time

import pytest

pytestmark = pytest.mark.integration

# 凭据从环境派生：CI/本地 compose 用什么密码就连什么密码（审查 C2）。
_PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "ci")
DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    f"postgresql://medintel:{_PG_PASSWORD}@127.0.0.1:5432/medintel",
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
    from app.storage import create_store

    # 每次运行自建专属就诊（时间戳后缀）：既满足 has_encounter 门（审查 C1），
    # 又让 PG 侧跨次运行互不污染（sqlite 每次都是全新 tmp 库）。
    suffix = f"{int(time.time())}"
    enc = f"enc-it-{suffix}"
    patient_id = f"pat-it-{suffix}"
    sqlite_settings = Settings(
        store_path=str(tmp_path / "sqlite.db"), model_path=str(tmp_path / "missing.json")
    )
    pg_settings = Settings(
        storage_backend="postgres",
        database_url=DATABASE_URL,
        model_path=str(tmp_path / "missing.json"),
    )
    sqlite_store = create_store(sqlite_settings)
    pg_store = create_store(pg_settings)
    patient = {
        "patient_id": patient_id,
        "encounter_id": enc,
        "display_name": f"集成患者 {suffix}",
        "age": 60,
        "sex": "M",
        "department": "综合内科",
        "tenant_id": "group-demo",
        "facility_id": "hospital-01",
    }
    sqlite_app = create_app(sqlite_settings, store_override=sqlite_store)
    pg_app = create_app(pg_settings, store_override=pg_store)
    headers = {"Authorization": "Bearer demo-clinician-token"}
    events = [
        {
            "event_id": f"evt-it1-{suffix}",
            "event_type": "lab_result",
            "patient_id": patient_id,
            "encounter_id": enc,
            "facility_id": "hospital-01",
            "occurred_at": "2026-01-01T06:00:00+00:00",
            "payload": {"lactate": 4.5},
        },
        # 乱序投递：时间更早的事件后到
        {
            "event_id": f"evt-it2-{suffix}",
            "event_type": "vital_signs",
            "patient_id": patient_id,
            "encounter_id": enc,
            "facility_id": "hospital-01",
            "occurred_at": "2026-01-01T05:00:00+00:00",
            "payload": {"heart_rate": 132, "systolic_bp": 88},
        },
        # 同 id 重投递（幂等）
        {
            "event_id": f"evt-it1-{suffix}",
            "event_type": "lab_result",
            "patient_id": patient_id,
            "encounter_id": enc,
            "facility_id": "hospital-01",
            "occurred_at": "2026-01-01T06:00:00+00:00",
            "payload": {"lactate": 4.5},
        },
    ]
    try:
        with TestClient(sqlite_app) as sqlite_client, TestClient(pg_app) as pg_client:
            # 必须在 lifespan 种子（demo 18 患者）之后插入专属就诊：
            # 预插入会让 seed() 见非空表提前返回，lifespan 评分随即外键崩溃。
            sqlite_store.upsert_patient(patient)
            pg_store.upsert_patient(patient)
            for body in events:
                first = sqlite_client.post("/api/events", json=body, headers=headers)
                second = pg_client.post("/api/events", json=body, headers=headers)
                # 自建就诊后两侧都必须真正入库（修复前曾双双 404 静默通过）
                assert (first.status_code, second.status_code) == (202, 202), body["event_id"]
            sqlite_patient = sqlite_client.get(f"/api/patients/{enc}", headers=headers).json()
            pg_patient = pg_client.get(f"/api/patients/{enc}", headers=headers).json()
    finally:
        sqlite_store.close()
        pg_store.close()
    assert set(sqlite_patient["risks"]) == set(pg_patient["risks"])
    for target, sqlite_score in sqlite_patient["risks"].items():
        pg_score = pg_patient["risks"][target]
        assert sqlite_score["band"] == pg_score["band"], target
        assert abs(sqlite_score["score"] - pg_score["score"]) < 1e-9, target
        assert sqlite_score["as_of"] == pg_score["as_of"], target
