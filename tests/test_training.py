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
    assert artifact["calibration"] is not None  # T6.2 已填充（占位断言随之更新）
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


def test_calibration_improves_or_ties_brier(tmp_path):
    import copy

    from app.config import Settings
    from app.domain import ClinicalEvent
    from app.risk import RiskEngine
    from app.synthetic import FEATURES
    from scripts.evaluate_model import binary_metrics
    from scripts.train_model import build_demo_dataset

    path = _train(tmp_path)
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    assert artifact["calibration"] is not None
    assert artifact["calibration"]["method"] == "platt"

    _times, matrix, labels = build_demo_dataset("deterioration_24h", 480)
    calibrated = RiskEngine(Settings(model_path=path))
    raw_artifact = copy.deepcopy(artifact)
    raw_artifact["calibration"] = None
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(raw_artifact), encoding="utf-8")
    uncalibrated = RiskEngine(Settings(model_path=str(raw_path)))

    cal_scores: list[float] = []
    raw_scores: list[float] = []
    for row_index, row in enumerate(matrix):
        payload = {
            feature: row[i]
            for i, feature in enumerate(FEATURES)
            if row[i] == row[i]  # 非 NaN
        }
        if row[-1] == row[-1]:  # news2 非 NaN
            payload["news2_score"] = row[-1]
        event = ClinicalEvent(
            event_type="vital_signs",
            patient_id=f"cal-p-{row_index}",
            encounter_id=f"cal-e-{row_index}",
            payload=payload,
        )
        cal_scores.append(calibrated.score(event).score)
        raw_scores.append(uncalibrated.score(event).score)
    brier_cal = binary_metrics(labels, cal_scores, calibrated.thresholds["high"])["brier"]
    brier_raw = binary_metrics(labels, raw_scores, uncalibrated.thresholds["high"])["brier"]
    assert brier_cal <= brier_raw + 1e-9


def test_calibration_fitted_on_train_segment_by_default(tmp_path):
    """Review I2：默认数据校准段单类 → 回退 train 段拟合，val 段标签保持未见。"""
    path = _train(tmp_path)
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    assert artifact["calibration"]["fitted_on"] == "train_segment"
