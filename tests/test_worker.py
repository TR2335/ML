from app.config import Settings
from app.domain import ClinicalEvent
from app.risk import RiskEngine
from app.store import Store
from app.worker import rescore_once


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
