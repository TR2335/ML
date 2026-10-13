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
    # Ruling R-P7-5a：deterioration 的 in-sample isotonic 结构性退化（<3 档）→
    # 回退 Platt；readmission cal 段出样本走真 isotonic（见
    # test_readmission_uses_isotonic_calibration）。
    method = artifact["calibration"]["method"]
    if method == "isotonic":
        from itertools import pairwise

        xs = artifact["calibration"]["x_thresholds"]
        ys = artifact["calibration"]["y_scores"]
        assert len(xs) == len(ys) and len(xs) >= 2
        assert all(a < b for a, b in pairwise(xs))
        assert all(a <= b for a, b in pairwise(ys))
    else:
        assert method == "platt"
        assert artifact["calibration"]["fallback_reason"] == "isotonic_resolution"
    assert artifact["calibration"]["fitted_on"]

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


def test_platt_artifact_still_loads(tmp_path):
    """旧 Platt 工件兼容（Ruling R-P7-5）：解析与打分路径不变。"""
    from app.config import Settings
    from app.domain import ClinicalEvent
    from app.risk import RiskEngine

    path = _train(tmp_path)
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    artifact["calibration"] = {
        "method": "platt",
        "a": 1.2,
        "b": -0.3,
        "fitted_on": "train_segment",
    }
    platt_path = tmp_path / "platt.json"
    platt_path.write_text(json.dumps(artifact), encoding="utf-8")
    engine = RiskEngine(Settings(model_path=str(platt_path)))
    assert engine.calibration["method"] == "platt"
    score = engine.score(
        ClinicalEvent(
            event_type="vital_signs",
            patient_id="p",
            encounter_id="e",
            payload={"age": 70, "heart_rate": 120},
        )
    )
    assert 0.0 <= score.score <= 1.0


def test_corrupt_isotonic_block_rejected(tmp_path):
    import pytest

    from app.config import Settings
    from app.risk import RiskEngine

    path = _train(tmp_path)
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    artifact["calibration"] = {
        "method": "isotonic",
        "x_thresholds": [1.0, 0.5],  # 非升序 → 拒绝
        "y_scores": [0.1, 0.9],
    }
    bad = tmp_path / "bad-cal.json"
    bad.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises((RuntimeError, ValueError)):
        RiskEngine(Settings(model_path=str(bad), app_env="test", model_signature_required=False))


def test_readmission_uses_isotonic_calibration(tmp_path):
    """Ruling R-P7-5a：readmission 的 cal 段双类且出样本 → 真 isotonic（WBS T7.3）。"""
    path = _train(tmp_path, extra=["--target-id", "readmission_30d"])
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    calibration = artifact["calibration"]
    assert calibration["method"] == "isotonic"
    xs = calibration["x_thresholds"]
    ys = calibration["y_scores"]
    from itertools import pairwise

    assert len(xs) == len(ys) and len({round(v, 6) for v in ys}) >= 3
    assert all(a < b for a, b in pairwise(xs))
    assert all(a <= b for a, b in pairwise(ys))
    assert calibration["fitted_on"] == "cal_segment"

    # P7 deferred 6：真 isotonic 分支此前从未被 Brier 不劣化钉住（deterioration
    # 走的是 R-P7-5a Platt 回退）。评估数据与拟合数据同源（同 seed 同行数重建
    # → 恰为 cal 段拟合样本）；isotonic 在拟合样本上对任意单调分数不劣化 Brier。
    import copy

    from app.config import Settings
    from app.domain import ClinicalEvent
    from app.features import READMISSION_PAYLOAD_KEYS
    from app.risk import RiskEngine
    from scripts.evaluate_model import binary_metrics
    from scripts.train_model import build_demo_dataset

    payloads, _matrix, raw_labels = build_demo_dataset("readmission_30d", 480)
    calibrated_engine = RiskEngine(
        Settings(readmission_model_path=path), target_id="readmission_30d"
    )
    raw_artifact = copy.deepcopy(artifact)
    raw_artifact["calibration"] = None
    raw_path = tmp_path / "raw_readmission.json"
    raw_path.write_text(json.dumps(raw_artifact), encoding="utf-8")
    raw_engine = RiskEngine(
        Settings(readmission_model_path=str(raw_path)), target_id="readmission_30d"
    )

    cal_scores: list[float] = []
    raw_scores: list[float] = []
    for i, item in enumerate(payloads):
        payload = {key: item[key] for key in READMISSION_PAYLOAD_KEYS if key in item}
        event = ClinicalEvent(
            event_type="discharge",
            patient_id=f"iso-p-{i}",
            encounter_id=f"iso-e-{i}",
            payload=payload,
        )
        cal_scores.append(calibrated_engine.score(event).score)
        raw_scores.append(raw_engine.score(event).score)
    brier_cal = binary_metrics(raw_labels, cal_scores, calibrated_engine.thresholds["high"])[
        "brier"
    ]
    brier_raw = binary_metrics(raw_labels, raw_scores, raw_engine.thresholds["high"])["brier"]
    assert brier_cal <= brier_raw + 1e-9


def test_load_dataset_csv_matches_features_before(tmp_path):
    """规模 CSV 的 21 维映射必须与 serving 同源 _features_before 逐位一致（R-RS-12）。"""
    import math

    from app.synthetic import FEATURES
    from scripts.baselines import news2_score
    from scripts.train_model import _features_before, load_dataset_csv

    present = {
        "age": 70.0,
        "respiratory_rate": 18.0,
        "systolic_bp": 110.0,
        "temperature": 37.2,
        "spo2": 96.0,
        "wbc": 9.1,
        "creatinine": 1.3,
        "prior_admissions_30d": 1.0,
        # heart_rate、lactate 缺失（CSV 留空）
    }
    vitals_subset = {
        key: present[key]
        for key in ("respiratory_rate", "spo2", "systolic_bp", "heart_rate", "temperature")
        if key in present
    }
    news2 = news2_score(vitals_subset)[0]
    csv_path = tmp_path / "det.csv"
    header = ["encounter_id", "patient_id", "index_time", "label", *FEATURES, "news2_score"]
    values = ["" if feature not in present else str(present[feature]) for feature in FEATURES]
    csv_path.write_text(
        ",".join(header)
        + "\n"
        + ",".join(
            [
                "enc-001",
                "pat-1",
                "2025-06-01T08:00:00+00:00",
                "1",
                *values,
                str(news2),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    times, matrix, labels = load_dataset_csv(csv_path, "deterioration_24h")
    expected = _features_before(dict(present))
    actual = matrix[0]
    assert len(actual) == 21 == len(expected)
    for got, want in zip(actual, expected):
        if math.isnan(want):
            assert math.isnan(got)
        else:
            assert got == want
    assert labels == [1]
    assert times == [{"index_time": "2025-06-01T08:00:00+00:00"}]


def test_load_dataset_csv_readmission_encodes_payload(tmp_path):
    import csv as csv_module
    from io import StringIO

    from app.features import READMISSION_PAYLOAD_KEYS, encode_readmission_payload
    from scripts.train_model import load_dataset_csv

    csv_path = tmp_path / "rm.csv"
    header = ["encounter_id", "patient_id", "index_time", "label", *READMISSION_PAYLOAD_KEYS]
    row = [
        "enc-1",
        "pat-1",
        "2025-06-01T10:00:00+00:00",
        "1",
        "3.5",
        "2",
        "4.5",
        "1",
        '["D00","D01","D02"]',
        "snf",
    ]
    # csv.writer 正确引用含逗号的 JSON 字段（与 G1 产物同一写法）
    buffer = StringIO()
    writer = csv_module.writer(buffer)
    writer.writerow(header)
    writer.writerow(row)
    csv_path.write_text(buffer.getvalue(), encoding="utf-8")
    times, matrix, labels = load_dataset_csv(csv_path, "readmission_30d")
    raw = dict(zip(READMISSION_PAYLOAD_KEYS, row[4:], strict=True))
    assert matrix[0] == encode_readmission_payload(raw)
    assert labels == [1]
    assert times == [{"index_time": "2025-06-01T10:00:00+00:00"}]


def test_train_cli_with_input_writes_artifact(tmp_path):
    """--input 规模路径：读 dataset CSV 训练，工件契约与 demo 路径一致。"""
    from app.synthetic import FEATURES

    csv_path = tmp_path / "ds.csv"
    header = ["encounter_id", "patient_id", "index_time", "label", *FEATURES, "news2_score"]
    lines = [",".join(header)]
    for i in range(40):
        cells = []
        for feature in FEATURES:
            if feature == "heart_rate" and i % 7 == 0:
                cells.append("")  # 制造缺失 → 指示列与 imputer 路径
            elif feature == "age":
                cells.append(str(40 + i % 45))
            elif feature == "prior_admissions_30d":
                cells.append(str(i % 3))
            else:
                cells.append(str(round(50 + (i * 7 % 60) + len(feature), 1)))
        lines.append(
            ",".join(
                [
                    f"enc-{i + 1:05d}",
                    f"pat-{i + 1:05d}",
                    f"2025-03-01T{i % 24:02d}:00:00+00:00",
                    str(int(i % 3 == 0)),
                    *cells,
                    str(3 + i % 5),
                ]
            )
        )
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output = str(tmp_path / "model.json")
    old_argv = sys.argv
    sys.argv = [
        "train_model.py",
        "--input",
        str(csv_path),
        "--output",
        output,
        "--target-id",
        "deterioration_24h",
        "--version",
        "scale-in-v1",
    ]
    try:
        with contextlib.redirect_stdout(open(str(tmp_path / "train.log"), "w", encoding="utf-8")):
            from scripts.train_model import main as train_main

            train_main()
    finally:
        sys.argv = old_argv
    artifact = json.loads(Path(output).read_text(encoding="utf-8"))
    assert artifact["model_version"] == "scale-in-v1"
    # P6 deferred 1：training_rows 只计实际 fit 的 60% 训练段，不再记全量。
    assert artifact["training_rows"] == 24  # int(40 * 0.6)
    assert artifact["feature_order"] == [
        *FEATURES,
        *[f"{feature}__missing" for feature in FEATURES],
        "news2_score",
    ]


def test_calibration_block_carries_sample_audit(tmp_path):
    """P6 deferred 10：校准审计链除 fitted_on 外还须有样本量与阳性数。"""
    path = _train(tmp_path)
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    cal = artifact["calibration"]
    assert cal["n_samples"] > 0
    assert 0 < cal["n_positives"] <= cal["n_samples"]
    if cal["fitted_on"] == "train_segment":
        # 校准拟合自训练段 → 样本数与 training_rows 同源（两项修复互相钉住）。
        assert cal["n_samples"] == artifact["training_rows"]


def test_threshold_defaults_have_single_source():
    """P6 deferred 3：train_model 不得再持有独立的阈值字面量副本。"""
    from app.risk import DEFAULT_THRESHOLDS
    from scripts import train_model

    assert not hasattr(train_model, "TARGET_THRESHOLDS")
    assert DEFAULT_THRESHOLDS["deterioration_24h"] == {
        "medium": 0.2,
        "high": 0.45,
        "critical": 0.75,
    }


def test_record_fold_auprc_skips_positive_free_fold():
    """P6 deferred 4：测试折无阳性 → 跳过该折，不记 0（记 0 会压低均值扭曲 C 选择）。"""
    from scripts.train_model import _record_fold_auprc

    bucket: list[float] = []
    _record_fold_auprc(bucket, [0, 0], [0.4, 0.6])  # 无阳性 → auprc=None
    assert bucket == []
    _record_fold_auprc(bucket, [0, 1], [0.4, 0.6])
    assert len(bucket) == 1 and 0.0 < bucket[0] <= 1.0


def test_make_golden_default_artifacts_track_settings_defaults():
    """P6 deferred 8：make_golden 的工件路径不得与 app.config 默认值双写。"""
    import scripts.make_golden as make_golden_module
    from app.config import Settings
    from scripts.make_golden import DEFAULT_ARTIFACTS

    assert DEFAULT_ARTIFACTS["deterioration_24h"] == Path(
        Settings.model_fields["model_path"].default
    )
    assert DEFAULT_ARTIFACTS["readmission_30d"] == Path(
        Settings.model_fields["readmission_model_path"].default
    )
    # 结构性钉子：路径字面量不得回流 make_golden（改 config 默认值必须只改一处）
    source = Path(make_golden_module.__file__).read_text(encoding="utf-8")
    assert 'Path("data/model.json")' not in source
    assert 'Path("data/readmission_model.json")' not in source
