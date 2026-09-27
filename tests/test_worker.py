from datetime import UTC, datetime

from app.config import Settings
from app.domain import ClinicalEvent
from app.risk import RiskEngine
from app.store import Store
from app.worker import rescore_once, score_event


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
