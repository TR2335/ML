import json
import sys
from pathlib import Path


def _write_gate_rows(csv_path: Path, invert_labels: bool) -> None:
    """16 行留出集、25% 阳性：空阳性 bootstrap 抽样 0.75^16≈1.0% < 2.5% 分位，
    保证 CI lo 可用；news2 分量恒定 → binary_metrics 元组平局按 label 降序 →
    AUPRC=1.0（既有行为，见台账 Task 3 Ruling）。"""
    header = (
        "encounter_id,creatinine,heart_rate,respiratory_rate,systolic_bp,spo2,temperature,label"
    )
    lines = [header]
    for i in range(16):
        creatinine = 3.0 if i % 4 == 0 else 0.8
        label = (1 if creatinine > 2 else 0) ^ (1 if invert_labels else 0)
        lines.append(f"enc-{i + 1:04d},{creatinine},70,16,120,98,36.8,{label}")
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
    assert report["auprc_ci"]["lo"] >= report["baseline_auprc"]
    assert report["baseline"] == "news2"
    assert (registry / "manifest.json").exists()


def _write_lace_rows(csv_path: Path, invert_labels: bool) -> None:
    """24 行出院留出集：连续 comorbidity 决定标签（LACE 的分箱在此损失排序信息）。"""
    header = (
        "encounter_id,los_days,prior_utilization_12m,comorbidity_index,ed_visits_6m,"
        "diagnosis_codes,discharge_destination,label"
    )
    lines = [header]
    for i in range(24):
        comorbidity = round(0.25 * i, 2)  # 0.00..5.75 连续
        label = int(comorbidity >= 3.0) ^ (1 if invert_labels else 0)
        lines.append(f"enc-{i + 1:04d},2.0,0,{comorbidity},0,[],home,{label}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_lace_artifact(path: Path, version: str) -> None:
    """逻辑回归在连续 comorbidity 上两点定线：c≥3 → p≈0.99、c=0 → p≈0.02。"""
    from app.features import READMISSION_FEATURES

    weights = {feature: 0.0 for feature in READMISSION_FEATURES}
    weights["comorbidity_index"] = 6.0
    path.write_text(
        json.dumps(
            {
                "model_version": version,
                "target_id": "readmission_30d",
                "algorithm": "logistic_regression_sklearn",
                "feature_names": READMISSION_FEATURES,
                "feature_order": READMISSION_FEATURES,
                "preprocess": {
                    feature: {"median": 0.0, "mean": 0.0, "std": 1.0}
                    for feature in READMISSION_FEATURES
                },
                "calibration": None,
                "weights": weights,
                "bias": -17.9,
                "thresholds": {"medium": 0.15, "high": 0.3, "critical": 0.55},
            }
        ),
        encoding="utf-8",
    )


def test_promote_gate_readmission_rejects_worse_than_lace(tmp_path):
    import pytest

    from scripts.manage_model import main as manage_main

    artifact = tmp_path / "bad-readmission.json"
    holdout = tmp_path / "lace-holdout.csv"
    _write_lace_artifact(artifact, "rm-bad-v1")
    _write_lace_rows(holdout, invert_labels=True)
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
        "readmission_30d",
        "--input",
        str(holdout),
    ]
    try:
        with pytest.raises(SystemExit):
            manage_main()
    finally:
        sys.argv = old_argv
    assert not (registry / "manifest.json").exists()


def test_promote_gate_readmission_accepts_and_records_eval(tmp_path):
    from scripts.manage_model import main as manage_main

    artifact = tmp_path / "good-readmission.json"
    holdout = tmp_path / "lace-holdout.csv"
    _write_lace_artifact(artifact, "rm-good-v1")
    _write_lace_rows(holdout, invert_labels=False)
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
        "readmission_30d",
        "--input",
        str(holdout),
    ]
    try:
        manage_main()
    finally:
        sys.argv = old_argv
    eval_path = registry / "readmission_30d" / "rm-good-v1" / "eval.json"
    report = json.loads(eval_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["ece"] < 0.05
    assert report["auprc_ci"]["lo"] >= report["baseline_auprc"]
    assert report["baseline"] == "lace"
    assert (registry / "manifest.json").exists()


def test_gate_rejects_subgroup_sensitivity_imbalance(tmp_path):
    """WBS T7.3 验收：合成亚组失衡数据触发门禁失败（CI/ECE 过、仅 subgroup 拒）。"""
    import pytest

    from scripts.manage_model import main as manage_main

    artifact = tmp_path / "imbalance.json"
    holdout = tmp_path / "subgroup.csv"
    from app.synthetic import FEATURES

    weights = {feature: 0.0 for feature in FEATURES}
    weights["creatinine"] = 3.396
    # bias −4.128：低肌酐 p≈0.081 ≈ bucket0 经验占比（ECE≈0.001，台账 Task 4 Ruling）
    artifact.write_text(
        json.dumps(
            {
                "model_version": "imb-v1",
                "target_id": "deterioration_24h",
                "algorithm": "logistic_regression_sklearn",
                "feature_names": FEATURES,
                "weights": weights,
                "bias": -4.128,
                "thresholds": {"medium": 0.2, "high": 0.45, "critical": 0.75},
            }
        ),
        encoding="utf-8",
    )
    header = (
        "encounter_id,creatinine,heart_rate,respiratory_rate,systolic_bp,spo2,temperature,age,label"
    )
    lines = [header]
    for i in range(12):  # 老组阳性：高肌酐 → 模型高分命中（sens=1.0）
        lines.append(f"enc-{i + 1:04d},3.0,70,16,120,98,36.8,75,1")
    for i in range(12):  # 年轻组阳性：低肌酐 → 模型低分漏检（sens=0.0）
        lines.append(f"enc-{i + 13:04d},0.5,70,16,120,98,36.8,40,1")
    for i in range(136):  # 阴性（与年轻组阳性同特征 → 同分桶）
        lines.append(f"enc-{i + 25:04d},0.5,70,16,120,98,36.8,40,0")
    holdout.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
        with pytest.raises(SystemExit) as excinfo:
            manage_main()
    finally:
        sys.argv = old_argv
    assert "subgroup" in str(excinfo.value)
    assert not (registry / "manifest.json").exists()


def test_gate_rules_degrade_when_columns_absent(tmp_path):
    """既有 fixture 无 age/facility 列 → 亚组/院区规则跳过，门禁语义不变（Ruling R-P7-6）。"""
    from scripts.manage_model import main as manage_main

    artifact = tmp_path / "good.json"
    holdout = tmp_path / "holdout.csv"
    _write_gate_artifact(artifact, "deg-v1")
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
    report = json.loads(
        (registry / "deterioration_24h" / "deg-v1" / "eval.json").read_text(encoding="utf-8")
    )
    assert report["passed"] is True
    assert report["rules"]["subgroup"]["status"] == "skipped"
    assert report["rules"]["facility"]["status"] == "skipped"
    assert report["rules"]["decision_curve"]["status"] in ("pass", "skipped")


def test_promote_gate_records_patient_split_mode(tmp_path):
    """患者级口径（spec §4.2 接入点 3）：eval.json 记录 split_mode=patient。"""
    from scripts.manage_model import main as manage_main

    artifact = tmp_path / "good.json"
    holdout = tmp_path / "holdout.csv"
    _write_gate_artifact(artifact, "split-v1")
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
    report = json.loads(
        (registry / "deterioration_24h" / "split-v1" / "eval.json").read_text(encoding="utf-8")
    )
    assert report["split_mode"] == "patient"
    assert report["group_column"] == "encounter_id"
    # 小 holdout 走全量 bootstrap（>50000 行才子采样，R-RS-10）
    assert report["auprc_ci"]["n"] == 1000
    assert report["auprc_ci"]["replicate_subsample"] is None


def test_promote_gate_rejects_missing_group_keys(tmp_path):
    """缺分组键的 holdout 在患者级口径下响亮拒绝，不留 manifest（R-RS-9）。"""
    import pytest

    from scripts.manage_model import main as manage_main

    artifact = tmp_path / "good.json"
    holdout = tmp_path / "bare.csv"
    _write_gate_artifact(artifact, "nokey-v1")
    holdout.write_text(
        "creatinine,label\n3.0,1\n0.8,0\n" * 8,
        encoding="utf-8",
    )
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
        with pytest.raises(SystemExit, match="promotion gate rejected"):
            manage_main()
    finally:
        sys.argv = old_argv
    assert not (registry / "manifest.json").exists()
