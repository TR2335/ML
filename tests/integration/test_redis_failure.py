"""Redis outage: publish fails → 503 + audit; retry after recovery is idempotent."""

import os
import socket
import subprocess
from datetime import UTC, datetime

import pytest

pytestmark = pytest.mark.integration


def _reachable(port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def _compose(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "compose", *args], capture_output=True, text=True, timeout=180, check=False
    )


@pytest.mark.skipif(
    os.environ.get("RUN_REDIS_FAILURE_TEST") != "1",
    reason="环境性守卫：会 stop/start redis 容器，仅 CI integration job（RUN_REDIS_FAILURE_TEST=1）或手动显式开启",
)
def test_publish_failure_returns_503_and_retry_recovers(tmp_path):
    if not _reachable(6379):
        pytest.skip("环境性守卫：Redis 未就绪")
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app

    event = {
        "event_id": "evt-redis-1",
        "event_type": "lab_result",
        "patient_id": "pat-001",
        "encounter_id": "enc-001",
        "facility_id": "hospital-01",
        "occurred_at": datetime.now(UTC).isoformat(),
        "payload": {"lactate": 3.3},
    }
    headers = {"Authorization": "Bearer demo-clinician-token"}
    application = create_app(
        Settings(
            event_backend="redis",
            redis_url="redis://127.0.0.1:6379/0",
            model_path=str(tmp_path / "missing.json"),
        )
    )
    with TestClient(application) as client:
        healthy = client.post("/api/events", json=event, headers=headers)
        assert healthy.status_code == 202
        assert _compose("stop", "redis").returncode == 0
        try:
            down = client.post(
                "/api/events", json={**event, "event_id": "evt-redis-2"}, headers=headers
            )
            assert down.status_code == 503
            audit = client.get("/api/audit", headers={"Authorization": "Bearer demo-quality-token"})
            assert any(item["action"] == "event_publish_failed" for item in audit.json())
        finally:
            assert _compose("start", "redis").returncode == 0
        # 恢复后同 id 重试 → 幂等重发路径（入流成功）
        retry = client.post("/api/events", json=event, headers=headers)
        assert retry.status_code == 202
        # in-process 消费一轮，断言事件最终评分成功（幂等重发闭环）
        from app.messaging import RedisStreamConsumer
        from app.risk import RiskEngine
        from app.storage import create_store
        from app.worker import consume_once

        settings = Settings(
            event_backend="redis",
            redis_url="redis://127.0.0.1:6379/0",
            model_path=str(tmp_path / "missing.json"),
        )
        store = create_store(settings)
        engines = {
            "deterioration_24h": RiskEngine(settings),
            "readmission_30d": RiskEngine(settings, target_id="readmission_30d"),
        }
        consumer = RedisStreamConsumer(
            settings.redis_url,
            settings.redis_stream,
            group=settings.redis_consumer_group,
            consumer="it-consumer",
            dead_letter_stream=settings.redis_dead_letter_stream,
        )
        consume_once(store, engines, consumer, max_delivery_attempts=1)
        scores = store.get_scores("enc-001", "group-demo", "hospital-01")
        assert scores, "恢复后重试的事件最终被评分"
        store.close()
