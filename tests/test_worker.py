from datetime import UTC, datetime

import pytest

from app.config import Settings
from app.domain import ClinicalEvent
from app.risk import RiskEngine
from app.store import Store
from app.worker import reconcile, rescore_once, score_event


def test_worker_rescores_aggregated_events_without_resetting_review(tmp_path):
    settings = Settings(model_path=str(tmp_path / "missing.json"))
    with Store(str(tmp_path / "store.db")) as store:
        event = ClinicalEvent(
            event_type="vital_signs",
            patient_id="pat-1",
            encounter_id="enc-1",
            payload={"heart_rate": 130, "respiratory_rate": 30, "systolic_bp": 90},
        )
        store.seed(
            [{"patient_id": "pat-1", "encounter_id": "enc-1", "display_name": "测试患者"}],
            [event],
        )
        engine = RiskEngine(settings)
        store.upsert_score(engine.score(event))
        score = store.get_score("enc-1", "group-demo", "hospital-01")
        score.status = "resolved"
        score.human_review = "已处置"
        store.upsert_score(score)
        assert rescore_once(store, engine) == 1
        updated = store.get_score("enc-1", "group-demo", "hospital-01")
        assert updated.status == "resolved"
        assert updated.human_review == "已处置"


def test_out_of_order_events_keep_latest_as_of(tmp_path):
    settings = Settings(model_path=str(tmp_path / "missing.json"))
    with Store(str(tmp_path / "store.db")) as store:
        e_early = ClinicalEvent(
            event_id="evt-early",
            event_type="vital_signs",
            patient_id="pat-1",
            encounter_id="enc-1",
            occurred_at=datetime(2026, 1, 1, 6, 0, tzinfo=UTC),
            payload={"heart_rate": 80},
        )
        e_late = ClinicalEvent(
            event_id="evt-late",
            event_type="vital_signs",
            patient_id="pat-1",
            encounter_id="enc-1",
            occurred_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            payload={"heart_rate": 140},
        )
        store.seed(
            [{"patient_id": "pat-1", "encounter_id": "enc-1", "display_name": "测试患者"}],
            [e_early, e_late],
        )
        engine = RiskEngine(settings)
        engines = {"deterioration_24h": engine}
        score_event(store, engines, e_late)
        after_late = store.get_score("enc-1", "group-demo", "hospital-01").score
        score_event(store, engines, e_early)
        final = store.get_score("enc-1", "group-demo", "hospital-01")
        assert final.as_of == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        assert final.score == after_late


def test_reconcile_scores_missing_events(tmp_path):
    settings = Settings(model_path=str(tmp_path / "missing.json"))
    with Store(str(tmp_path / "store.db")) as store:
        event = ClinicalEvent(
            event_id="evt-orphan",
            event_type="vital_signs",
            patient_id="pat-1",
            encounter_id="enc-1",
            occurred_at=datetime(2026, 1, 1, 8, 0, tzinfo=UTC),
            payload={"heart_rate": 135, "respiratory_rate": 28},
        )
        store.seed(
            [{"patient_id": "pat-1", "encounter_id": "enc-1", "display_name": "测试患者"}],
            [event],
        )
        # 事件在库但没有分数（模拟 worker 丢分场景）
        assert store.get_scores("enc-1", "group-demo", "hospital-01") == {}
        engine = RiskEngine(settings)
        engines = {"deterioration_24h": engine}
        assert reconcile(store, engines) == 1
        scores = store.get_scores("enc-1", "group-demo", "hospital-01")
        assert set(scores) == {"deterioration_24h"}
        assert scores["deterioration_24h"].as_of == datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
        # 已是最新的分数不再重复补算
        assert reconcile(store, engines) == 0


def test_sync_loop_survives_rescore_failure(tmp_path, monkeypatch, caplog):
    import logging

    from app import worker
    from app.config import get_settings

    calls = {"n": 0}

    def flaky_rescore():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        raise KeyboardInterrupt  # 第二轮停止循环（BaseException，不被 except Exception 吞掉）

    monkeypatch.setattr(worker, "rescore_once", flaky_rescore)
    monkeypatch.setattr(worker, "HEARTBEAT_PATH", tmp_path / "heartbeat")
    sleeps: list[float] = []
    monkeypatch.setattr(worker.time, "sleep", lambda seconds: sleeps.append(seconds))
    with caplog.at_level(logging.ERROR, logger="app.worker"), pytest.raises(KeyboardInterrupt):
        worker.run_sync_worker()
    assert calls["n"] == 2  # 失败后循环继续
    assert "sync rescore cycle failed" in caplog.text
    assert sleeps == [worker.sync_backoff_seconds(get_settings().score_interval_seconds, 1)]
    assert (tmp_path / "heartbeat").exists()


def test_sync_backoff_grows_and_resets():
    from app.worker import sync_backoff_seconds

    assert sync_backoff_seconds(300, 0) == 300
    assert sync_backoff_seconds(300, 1) == 600
    assert sync_backoff_seconds(300, 2) == 1200
    assert sync_backoff_seconds(300, 10) == 3600  # 上限
