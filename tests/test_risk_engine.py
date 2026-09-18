from app.config import Settings
from app.domain import ClinicalEvent
from app.risk import RiskEngine


def event(payload: dict) -> ClinicalEvent:
    return ClinicalEvent(
        event_type="vital_signs", patient_id="pat-1", encounter_id="enc-1", payload=payload
    )


def test_high_risk_patient_scores_above_low_risk_patient(tmp_path):
    engine = RiskEngine(Settings(model_path=str(tmp_path / "missing.json")))
    low = engine.score(
        event(
            {
                "age": 35,
                "heart_rate": 72,
                "respiratory_rate": 16,
                "systolic_bp": 128,
                "temperature": 36.8,
                "spo2": 98,
                "wbc": 7,
                "creatinine": 0.8,
                "lactate": 1.0,
                "prior_admissions_30d": 0,
            }
        )
    )
    high = engine.score(
        event(
            {
                "age": 78,
                "heart_rate": 132,
                "respiratory_rate": 32,
                "systolic_bp": 85,
                "temperature": 39.2,
                "spo2": 88,
                "wbc": 19,
                "creatinine": 2.8,
                "lactate": 4.1,
                "prior_admissions_30d": 2,
            }
        )
    )
    assert high.score > low.score
    assert high.band in {"high", "critical"}
    assert len(high.top_factors) == 4


def test_score_reports_missing_features(tmp_path):
    engine = RiskEngine(Settings(model_path=str(tmp_path / "missing.json")))
    score = engine.score(event({"heart_rate": 110}))
    assert "heart_rate" not in score.missing_features
    assert "lactate" in score.missing_features


def test_model_artifact_overrides_version_and_weights(tmp_path):
    path = tmp_path / "model.json"
    from app.synthetic import FEATURES

    weights = {feature: 0.0 for feature in FEATURES}
    weights["lactate"] = 1.2
    path.write_text(
        __import__("json").dumps(
            {
                "model_version": "approved-v2",
                "bias": -2,
                "feature_names": FEATURES,
                "weights": weights,
            }
        ),
        encoding="utf-8",
    )
    engine = RiskEngine(Settings(model_path=str(path)))
    assert engine.version == "approved-v2"
    assert engine.weights["lactate"] == 1.2


def test_production_requires_valid_model_artifact(tmp_path):
    settings = Settings(
        app_env="prod", model_path=str(tmp_path / "missing.json"), model_version="approved-v2"
    )
    try:
        RiskEngine(settings)
    except RuntimeError as error:
        assert "model artifact not found" in str(error)
    else:
        raise AssertionError("production must not fall back to default weights")
