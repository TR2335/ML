import contextlib
import json
import sys
from pathlib import Path


def _train(tmp_path, extra: list[str] | None = None) -> str:
    from scripts.train_model import main as train_main

    output = str(tmp_path / "model.json")
    argv = [
        "train_model.py",
        "--output",
        output,
        "--rows",
        "480",
        "--version",
        "p6-test-v1",
        *(extra or []),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        with contextlib.redirect_stdout(open(str(tmp_path / "train.log"), "w", encoding="utf-8")):
            train_main()
    finally:
        sys.argv = old_argv
    return output


def test_sklearn_artifact_loads_and_scores(tmp_path):
    from app.config import Settings
    from app.domain import ClinicalEvent
    from app.risk import RiskEngine

    path = _train(tmp_path)
    engine = RiskEngine(Settings(model_path=path))
    score = engine.score(
        ClinicalEvent(
            event_type="vital_signs",
            patient_id="pat-1",
            encounter_id="enc-1",
            payload={"age": 70, "heart_rate": 128, "creatinine": 2.6},
        )
    )
    assert 0.0 <= score.score <= 1.0
    assert score.model_version == "p6-test-v1"


def test_artifact_contract_shape(tmp_path):
    from app.synthetic import FEATURES

    path = _train(tmp_path)
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = [
        *FEATURES,
        *[f"{feature}__missing" for feature in FEATURES],
        "news2_score",
    ]
    assert artifact["feature_names"] == expected
    assert artifact["feature_order"] == expected
    assert artifact["algorithm"] == "logistic_regression_sklearn"
    assert artifact["calibration"] is None  # T6.2 占位
    assert set(artifact["preprocess"]) == set(expected)
    for entry in artifact["preprocess"].values():
        assert {"median", "mean", "std"} <= set(entry)


def test_missing_indicator_features_are_computed_by_serving(tmp_path):
    """R2：serving 对 `__missing` 特征名即时计算，不走 median 填补、不算缺失。"""
    from app.config import Settings
    from app.domain import ClinicalEvent
    from app.risk import RiskEngine
    from app.synthetic import FEATURES

    path = _train(tmp_path)
    engine = RiskEngine(Settings(model_path=path))
    score = engine.score(
        ClinicalEvent(event_type="vital_signs", patient_id="p", encounter_id="e", payload={})
    )
    assert 0.0 <= score.score <= 1.0
    missing = set(score.missing_features)
    assert set(FEATURES) <= missing  # 基特征全部缺失
    assert not any(name.endswith("__missing") for name in missing)  # 指示列本身不算缺失
    assert engine.feature_order[-1] == "news2_score"
