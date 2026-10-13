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


def test_score_event_discharge_triggers_readmission_only(tmp_path):
    settings = Settings(model_path=str(tmp_path / "missing.json"))
    with Store(str(tmp_path / "store.db")) as store:
        vital = ClinicalEvent(
            event_id="evt-v",
            event_type="vital_signs",
            patient_id="pat-1",
            encounter_id="enc-1",
            occurred_at=datetime(2026, 1, 1, 8, 0, tzinfo=UTC),
            payload={"heart_rate": 120},
        )
        discharge = ClinicalEvent(
            event_id="evt-d",
            event_type="discharge",
            patient_id="pat-1",
            encounter_id="enc-1",
            occurred_at=datetime(2026, 1, 1, 20, 0, tzinfo=UTC),
            payload={
                "discharge_destination": "snf",
                "los_days": 3.5,
                "diagnosis_codes": ["N17.9"],
                "comorbidity_index": 4.0,
                "prior_utilization_12m": 2,
                "ed_visits_6m": 2,
            },
        )
        store.seed(
            [{"patient_id": "pat-1", "encounter_id": "enc-1", "display_name": "测试患者"}],
            [vital, discharge],
        )
        engines = {
            "deterioration_24h": RiskEngine(settings),
            "readmission_30d": RiskEngine(settings, target_id="readmission_30d"),
        }
        score_event(store, engines, vital)
        assert store.get_score("enc-1", "group-demo", "hospital-01", "readmission_30d") is None
        score_event(store, engines, discharge)
        readmission = store.get_score("enc-1", "group-demo", "hospital-01", "readmission_30d")
        assert readmission is not None
        assert readmission.as_of == datetime(2026, 1, 1, 20, 0, tzinfo=UTC)


def test_second_discharge_event_does_not_rescore_readmission(tmp_path):
    """P8 deferred 4：live 与批路径/契约一致——readmission 单次评分、as_of 钉首个
    出院事件；第二个出院事件不重打分（deterioration 照常随事件更新）。"""
    settings = Settings(model_path=str(tmp_path / "missing.json"))
    with Store(str(tmp_path / "store.db")) as store:
        d1 = ClinicalEvent(
            event_id="evt-d1",
            event_type="discharge",
            patient_id="pat-1",
            encounter_id="enc-1",
            occurred_at=datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
            payload={
                "discharge_destination": "home",
                "los_days": 2.0,
                "diagnosis_codes": [],
                "comorbidity_index": 0.0,
                "prior_utilization_12m": 0,
                "ed_visits_6m": 0,
            },
        )
        d2 = d1.model_copy(
            update={
                "event_id": "evt-d2",
                "occurred_at": datetime(2026, 1, 3, 10, 0, tzinfo=UTC),
                "payload": {**d1.payload, "los_days": 5.0},
            }
        )
        store.seed(
            [{"patient_id": "pat-1", "encounter_id": "enc-1", "display_name": "测试患者"}],
            [d1, d2],
        )
        engines = {
            "deterioration_24h": RiskEngine(settings),
            "readmission_30d": RiskEngine(settings, target_id="readmission_30d"),
        }
        score_event(store, engines, d1)
        first = store.get_score("enc-1", "group-demo", "hospital-01", "readmission_30d")
        assert first is not None and first.as_of == d1.occurred_at
        scored_at_before = first.scored_at

        score_event(store, engines, d2)
        after = store.get_score("enc-1", "group-demo", "hospital-01", "readmission_30d")
        assert after.as_of == d1.occurred_at
        assert after.scored_at == scored_at_before  # 未被重打分
        # deterioration 仍随第二个出院事件正常重打
        det = store.get_score("enc-1", "group-demo", "hospital-01", "deterioration_24h")
        assert det is not None and det.as_of == d2.occurred_at


def test_rescore_uses_discharge_time_for_readmission(tmp_path):
    with Store(str(tmp_path / "store.db")) as store:
        discharge = ClinicalEvent(
            event_id="evt-d1",
            event_type="discharge",
            patient_id="pat-1",
            encounter_id="enc-1",
            occurred_at=datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
            payload={
                "discharge_destination": "home",
                "los_days": 2.0,
                "diagnosis_codes": [],
                "comorbidity_index": 0.0,
                "prior_utilization_12m": 0,
                "ed_visits_6m": 0,
            },
        )
        later_vital = ClinicalEvent(
            event_id="evt-v1",
            event_type="vital_signs",
            patient_id="pat-1",
            encounter_id="enc-1",
            occurred_at=datetime(2026, 1, 1, 22, 0, tzinfo=UTC),
            payload={"heart_rate": 140},
        )
        store.seed(
            [{"patient_id": "pat-1", "encounter_id": "enc-1", "display_name": "测试患者"}],
            [discharge, later_vital],
        )
        assert rescore_once(store) == 1
        readmission = store.get_score("enc-1", "group-demo", "hospital-01", "readmission_30d")
        assert readmission is not None
        # 单次评分：as_of 钉在出院时刻，不被更晚的生命体征刷新
        assert readmission.as_of == datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
        deterioration = store.get_score("enc-1", "group-demo", "hospital-01")
        assert deterioration.as_of == datetime(2026, 1, 1, 22, 0, tzinfo=UTC)


def test_reconcile_backfills_readmission_at_discharge_and_is_idempotent(tmp_path):
    """Review I3：reconcile 的 readmission 分支——有出院 → 回填钉出院时刻；再跑幂等 0。"""
    settings = Settings(model_path=str(tmp_path / "missing.json"))
    with Store(str(tmp_path / "store.db")) as store:
        discharge = ClinicalEvent(
            event_id="evt-rc-d",
            event_type="discharge",
            patient_id="pat-1",
            encounter_id="enc-1",
            occurred_at=datetime(2026, 1, 1, 10, 0, tzinfo=UTC),
            payload={
                "discharge_destination": "home",
                "los_days": 3.0,
                "diagnosis_codes": [],
                "comorbidity_index": 1.0,
                "prior_utilization_12m": 0,
                "ed_visits_6m": 1,
            },
        )
        vital = ClinicalEvent(
            event_id="evt-rc-v",
            event_type="vital_signs",
            patient_id="pat-1",
            encounter_id="enc-1",
            occurred_at=datetime(2026, 1, 1, 18, 0, tzinfo=UTC),
            payload={"heart_rate": 125},
        )
        store.seed(
            [{"patient_id": "pat-1", "encounter_id": "enc-1", "display_name": "测试患者"}],
            [discharge, vital],
        )
        engines = {
            "deterioration_24h": RiskEngine(settings),
            "readmission_30d": RiskEngine(settings, target_id="readmission_30d"),
        }
        assert store.get_scores("enc-1", "group-demo", "hospital-01") == {}
        assert reconcile(store, engines) == 1
        scores = store.get_scores("enc-1", "group-demo", "hospital-01")
        readmission = scores.get("readmission_30d")
        assert readmission is not None
        # 回填钉在首出院时刻，而非较晚的生命体征时刻
        assert readmission.as_of == datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
        assert scores["deterioration_24h"].as_of == datetime(2026, 1, 1, 18, 0, tzinfo=UTC)
        # 幂等：再次 reconcile 不再补算
        assert reconcile(store, engines) == 0


def test_reconcile_requires_no_readmission_score_without_discharge(tmp_path):
    """Review I3：无出院事件 → reconcile 不要求/不产生 readmission 分数，且幂等 0。"""
    settings = Settings(model_path=str(tmp_path / "missing.json"))
    with Store(str(tmp_path / "store.db")) as store:
        vital = ClinicalEvent(
            event_id="evt-rr-v",
            event_type="vital_signs",
            patient_id="pat-1",
            encounter_id="enc-1",
            occurred_at=datetime(2026, 1, 1, 8, 0, tzinfo=UTC),
            payload={"heart_rate": 130},
        )
        store.seed(
            [{"patient_id": "pat-1", "encounter_id": "enc-1", "display_name": "测试患者"}],
            [vital],
        )
        engines = {
            "deterioration_24h": RiskEngine(settings),
            "readmission_30d": RiskEngine(settings, target_id="readmission_30d"),
        }
        assert reconcile(store, engines) == 1
        scores = store.get_scores("enc-1", "group-demo", "hospital-01")
        assert "deterioration_24h" in scores
        assert "readmission_30d" not in scores
        assert reconcile(store, engines) == 0


def test_sync_backoff_grows_and_resets():
    from app.worker import sync_backoff_seconds

    assert sync_backoff_seconds(300, 0) == 300
    assert sync_backoff_seconds(300, 1) == 600
    assert sync_backoff_seconds(300, 2) == 1200
    assert sync_backoff_seconds(300, 10) == 3600  # 上限
