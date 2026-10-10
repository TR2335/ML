import json
import sys
from pathlib import Path


def _write_gate_rows(csv_path: Path, invert_labels: bool) -> None:
    """16 行留出集、25% 阳性：空阳性 bootstrap 抽样 0.75^16≈1.0% < 2.5% 分位，
    保证 CI lo 可用；news2 分量恒定 → binary_metrics 元组平局按 label 降序 →
    AUPRC=1.0（既有行为，见台账 Task 3 Ruling）。"""
    header = "creatinine,heart_rate,respiratory_rate,systolic_bp,spo2,temperature,label"
    lines = [header]
    for i in range(16):
        creatinine = 3.0 if i % 4 == 0 else 0.8
        label = (1 if creatinine > 2 else 0) ^ (1 if invert_labels else 0)
        lines.append(f"{creatinine},70,16,120,98,36.8,{label}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gate_artifact(path: Path, version: str) -> None:
    """两点定线：creatinine=3 → z≈4.6（p≈0.99）、0.8 → z≈-2.87（p≈0.05）——ECE≈0.043。"""
    from app.synthetic import FEATURES

    weights = {feature: 0.0 for feature in FEATURES}
    weights["creatinine"] = 3.396
    path.write_text(
        json.dumps(
            {
                "model_version": version,
                "target_id": "deterioration_24h",
                "algorithm": "logistic_regression_sklearn",
                "feature_names": FEATURES,
                "weights": weights,
                "bias": -5.588,
                "thresholds": {"medium": 0.2, "high": 0.45, "critical": 0.75},
            }
        ),
        encoding="utf-8",
    )


def test_promote_gate_rejects_low_metric_artifact(tmp_path):
    import pytest

    from scripts.manage_model import main as manage_main

    artifact = tmp_path / "bad.json"
    holdout = tmp_path / "holdout.csv"
    _write_gate_artifact(artifact, "bad-v1")
    _write_gate_rows(holdout, invert_labels=True)  # 模型排序与标签反向
    registry = tmp_path / "registry"
    old_argv = sys.argv
    sys.argv = [
        "manage_model.py",
        "--registry-dir",
        str(registry),
        "promote",
        "--artifact",
        str(artifact),
        "--target-id",
        "deterioration_24h",
        "--input",
        str(holdout),
    ]
    try:
        with pytest.raises(SystemExit):
            manage_main()
    finally:
        sys.argv = old_argv
    assert not (registry / "manifest.json").exists()  # 拒绝后不得留下 manifest


def test_promote_accepts_full_v2_deterioration_artifact(tmp_path):
    """Review C2/Ruling R-P8-9：真实 21 维工件不得被 load_artifact 拒绝。"""
    from scripts.manage_model import main as manage_main
    from scripts.train_model import FEATURES_V2

    artifact = tmp_path / "v2.json"
    holdout = tmp_path / "holdout.csv"
    weights = {feature: 0.0 for feature in FEATURES_V2}
    weights["creatinine"] = 3.396
    artifact.write_text(
        json.dumps(
            {
                "model_version": "v2-shape",
                "target_id": "deterioration_24h",
                "algorithm": "logistic_regression_sklearn",
                "feature_names": FEATURES_V2,
                "feature_order": FEATURES_V2,
                "preprocess": {
                    feature: {"median": 0.0, "mean": 0.0, "std": 1.0} for feature in FEATURES_V2
                },
                "calibration": None,
                "weights": weights,
                "bias": -5.588,
                "thresholds": {"medium": 0.2, "high": 0.45, "critical": 0.75},
            }
        ),
        encoding="utf-8",
    )
    _write_gate_rows(holdout, invert_labels=False)
    registry = tmp_path / "registry"
    old_argv = sys.argv
    sys.argv = [
        "manage_model.py",
        "--registry-dir",
        str(registry),
        "promote",
        "--artifact",
        str(artifact),
        "--target-id",
        "deterioration_24h",
        "--input",
        str(holdout),
    ]
    try:
        manage_main()
    finally:
        sys.argv = old_argv
    assert (registry / "manifest.json").exists()
    eval_path = registry / "deterioration_24h" / "v2-shape" / "eval.json"
    assert json.loads(eval_path.read_text(encoding="utf-8"))["passed"] is True


def test_promote_gate_accepts_and_records_eval(tmp_path):
    from scripts.manage_model import main as manage_main

    artifact = tmp_path / "good.json"
    holdout = tmp_path / "holdout.csv"
    _write_gate_artifact(artifact, "good-v1")
    _write_gate_rows(holdout, invert_labels=False)
    registry = tmp_path / "registry"
    old_argv = sys.argv
    sys.argv = [
        "manage_model.py",
        "--registry-dir",
        str(registry),
        "promote",
        "--artifact",
        str(artifact),
        "--target-id",
        "deterioration_24h",
        "--input",
        str(holdout),
    ]
    try:
        manage_main()
    finally:
        sys.argv = old_argv
    eval_path = registry / "deterioration_24h" / "good-v1" / "eval.json"
    report = json.loads(eval_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["ece"] < 0.05
    assert report["auprc_ci"]["lo"] >= report["news2_auprc"]
    assert (registry / "manifest.json").exists()
