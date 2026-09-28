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


def test_preprocess_from_artifact_overrides_hardcoded(tmp_path):
    import json
    import math

    from app.synthetic import FEATURES

    path = tmp_path / "model.json"
    weights = {feature: 0.0 for feature in FEATURES}
    weights["heart_rate"] = 1.0
    path.write_text(
        json.dumps(
            {
                "model_version": "preprocess-v1",
                "feature_names": FEATURES,
                "feature_order": FEATURES,
                "preprocess": {
                    feature: (
                        {"median": 0.0, "mean": 100.0, "std": 20.0}
                        if feature == "heart_rate"
                        else {"median": 0.0, "mean": 0.0, "std": 1.0}
                    )
                    for feature in FEATURES
                },
                "weights": weights,
                "bias": 0.0,
                "thresholds": {"medium": 0.2, "high": 0.45, "critical": 0.75},
            }
        ),
        encoding="utf-8",
    )
    engine = RiskEngine(Settings(model_path=str(path)))
    score = engine.score(event({"heart_rate": 120}))
    # z = (120-100)/20 = 1.0 → sigmoid(1.0) ≈ 0.7311；硬编码路径会得到 sigmoid(120) ≈ 1.0
    assert abs(score.score - 1 / (1 + math.exp(-1))) < 0.001
    assert score.model_version == "preprocess-v1"


def test_legacy_artifact_without_preprocess_still_scores(tmp_path):
    import json

    from app.synthetic import FEATURES

    path = tmp_path / "model.json"
    path.write_text(
        json.dumps(
            {
                "model_version": "legacy-v1",
                "feature_names": FEATURES,
                "weights": {feature: 0.01 for feature in FEATURES},
                "bias": -1.0,
                "thresholds": {"medium": 0.2, "high": 0.45, "critical": 0.75},
            }
        ),
        encoding="utf-8",
    )
    engine = RiskEngine(Settings(model_path=str(path)))
    score = engine.score(event({"heart_rate": 110}))
    assert 0.0 <= score.score <= 1.0
    assert score.model_version == "legacy-v1"
    assert engine.preprocess is None
    assert engine.feature_order == FEATURES


def test_preprocess_without_feature_order_is_rejected(tmp_path):
    import json

    import pytest

    from app.synthetic import FEATURES

    path = tmp_path / "model.json"
    path.write_text(
        json.dumps(
            {
                "model_version": "bad-v1",
                "target_id": "deterioration_24h",
                "feature_names": FEATURES,
                "preprocess": {f: {"median": 0, "mean": 0, "std": 1} for f in FEATURES},
                "weights": {feature: 0.01 for feature in FEATURES},
                "bias": -1.0,
                "thresholds": {"medium": 0.2, "high": 0.45, "critical": 0.75},
            }
        ),
        encoding="utf-8",
    )
    # prod 严格模式才抛 RuntimeError；显式关闭验签以确保失败原因确实是 preprocess 校验
    with pytest.raises(RuntimeError, match="invalid model artifact"):
        RiskEngine(
            Settings(
                model_path=str(path),
                app_env="prod",
                model_signature_required=False,
            )
        )


def test_preprocess_imputed_feature_still_reported_missing(tmp_path):
    """Plan Ruling R10：median 填补的特征参与计算，但仍须列入 missing_features。"""
    import json
    import math

    from app.synthetic import FEATURES

    path = tmp_path / "model.json"
    weights = {feature: 0.0 for feature in FEATURES}
    weights["heart_rate"] = 1.0
    path.write_text(
        json.dumps(
            {
                "model_version": "impute-v1",
                "feature_names": FEATURES,
                "feature_order": FEATURES,
                "preprocess": {
                    feature: (
                        {"median": 50.0, "mean": 100.0, "std": 20.0}
                        if feature == "heart_rate"
                        else {"median": 0.0, "mean": 0.0, "std": 1.0}
                    )
                    for feature in FEATURES
                },
                "weights": weights,
                "bias": 0.0,
                "thresholds": {"medium": 0.2, "high": 0.45, "critical": 0.75},
            }
        ),
        encoding="utf-8",
    )
    engine = RiskEngine(Settings(model_path=str(path)))
    score = engine.score(event({}))
    # 缺失 → median=50 填补 → z=(50-100)/20=-2.5 → sigmoid(-2.5)
    assert abs(score.score - 1 / (1 + math.exp(2.5))) < 0.001
    assert "heart_rate" in score.missing_features
    assert "lactate" in score.missing_features
