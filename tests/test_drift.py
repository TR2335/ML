import json
import sys


def test_psi_flags_synthetic_drift():
    from scripts.monitor_drift import compute_psi

    reference = {"lo": 50.0, "hi": 150.0, "proportions": [0.1] * 10, "missing_rate": 0.0}
    stable = [55.0 + i for i in range(100)]
    assert compute_psi(reference, stable) < 0.2
    shifted = [140.0 + (i % 10) for i in range(100)]  # 整体漂到高端
    assert compute_psi(reference, shifted) > 0.2


def test_monitor_drift_cli_alerts_and_exits(tmp_path):
    import pytest

    from scripts.monitor_drift import main as monitor_main

    artifact = tmp_path / "model.json"
    artifact.write_text(
        json.dumps(
            {
                "model_version": "d-v1",
                "target_id": "deterioration_24h",
                "feature_names": ["heart_rate"],
                "feature_order": ["heart_rate"],
                "weights": {"heart_rate": 1.0},
                "bias": -1.0,
                "thresholds": {"medium": 0.2, "high": 0.45, "critical": 0.75},
                "drift_reference": {
                    "heart_rate": {
                        "lo": 50.0,
                        "hi": 150.0,
                        "proportions": [0.1] * 10,
                        "missing_rate": 0.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    csv_path = tmp_path / "batch.csv"
    lines = ["heart_rate,label"]
    for i in range(40):
        lines.append(f"{140 + i % 10},{i % 2}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    old_argv = sys.argv
    sys.argv = ["monitor_drift.py", "--artifact", str(artifact), "--input", str(csv_path)]
    try:
        with pytest.raises(SystemExit) as excinfo:
            monitor_main()
    finally:
        sys.argv = old_argv
    assert excinfo.value.code == 1
    # 无告警路径 exit 0：同一工件喂稳定分布
    stable_csv = tmp_path / "stable.csv"
    lines = ["heart_rate,label"]
    for i in range(100):
        lines.append(f"{55 + i % 90},{i % 2}")
    stable_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.argv = ["monitor_drift.py", "--artifact", str(artifact), "--input", str(stable_csv)]
    try:
        with pytest.raises(SystemExit) as excinfo:
            monitor_main()
    finally:
        sys.argv = old_argv
    assert excinfo.value.code == 0


def test_recalibrate_only_updates_calibration_and_gates(tmp_path):
    """Ruling R-P7-10 两半主旨：只动 calibration；门禁不过则原件分毫不动。

    用 readmission 做过门禁半场：字段驱动标签 → 模型排序强于 LACE 分箱（demo
    deterioration 的模型打不过 NEWS2 基线，门禁会正确拒绝——见台账 Task 5）。
    拒绝半场用全零标签的 --gate-input（"holdout needs at least one positive"，
    确定性 SystemExit）。
    """
    import csv as _csv

    import pytest

    from app.features import READMISSION_PAYLOAD_KEYS
    from scripts.recalibrate import main as recalibrate_main
    from scripts.train_model import build_demo_dataset
    from scripts.train_model import main as train_main

    artifact = tmp_path / "rm.json"
    old_argv = sys.argv
    sys.argv = [
        "train_model.py",
        "--output",
        str(artifact),
        "--rows",
        "800",
        "--version",
        "rc-v1",
        "--target-id",
        "readmission_30d",
    ]
    try:
        train_main()
    finally:
        sys.argv = old_argv
    before = json.loads(artifact.read_text(encoding="utf-8"))
    assert before["calibration"]["method"] in ("isotonic", "platt")

    rows, _matrix, labels = build_demo_dataset("readmission_30d", 800)
    holdout = tmp_path / "cal.csv"
    fields = [*READMISSION_PAYLOAD_KEYS, "label"]
    with holdout.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row, label in zip(rows, labels):
            out = {}
            for key in READMISSION_PAYLOAD_KEYS:
                value = row.get(key, "")
                out[key] = json.dumps(value) if isinstance(value, list) else str(value)
            out["label"] = str(label)
            writer.writerow(out)

    sys.argv = [
        "recalibrate.py",
        "--artifact",
        str(artifact),
        "--input",
        str(holdout),
        "--target-id",
        "readmission_30d",
    ]
    try:
        recalibrate_main()
    finally:
        sys.argv = old_argv
    after = json.loads(artifact.read_text(encoding="utf-8"))
    changed = {key for key in before if before[key] != after.get(key)}
    assert changed <= {"calibration"}
    assert after["calibration"]["fitted_on"] == "recalibration_set"

    # 不过路径：gate 输入无阳性 → SystemExit 且原件分毫不动
    no_positive = tmp_path / "no-positive.csv"
    with no_positive.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in list(rows)[:40]:
            out = {}
            for key in READMISSION_PAYLOAD_KEYS:
                value = row.get(key, "")
                out[key] = json.dumps(value) if isinstance(value, list) else str(value)
            out["label"] = "0"
            writer.writerow(out)
    before_text = artifact.read_text(encoding="utf-8")
    sys.argv = [
        "recalibrate.py",
        "--artifact",
        str(artifact),
        "--input",
        str(holdout),
        "--target-id",
        "readmission_30d",
        "--gate-input",
        str(no_positive),
    ]
    try:
        with pytest.raises(SystemExit):
            recalibrate_main()
    finally:
        sys.argv = old_argv
    assert artifact.read_text(encoding="utf-8") == before_text


def test_linear_margin_matches_sigmoid_inverse(tmp_path):
    import math

    from app.config import Settings
    from app.domain import ClinicalEvent
    from app.risk import RiskEngine
    from scripts.train_model import main as train_main

    artifact = tmp_path / "m.json"
    old_argv = sys.argv
    sys.argv = [
        "train_model.py",
        "--output",
        str(artifact),
        "--rows",
        "480",
        "--version",
        "m-v1",
    ]
    try:
        train_main()
    finally:
        sys.argv = old_argv
    # margin 是校准前 logit——剥掉校准块后 sigmoid(margin) == serving 分数。
    raw_artifact = json.loads(artifact.read_text(encoding="utf-8"))
    raw_artifact["calibration"] = None
    artifact.write_text(json.dumps(raw_artifact), encoding="utf-8")
    engine = RiskEngine(Settings(model_path=str(artifact)))
    event = ClinicalEvent(
        event_type="vital_signs",
        patient_id="p",
        encounter_id="e",
        payload={"age": 70, "heart_rate": 120},
    )
    margin = engine.linear_margin(event)
    score = engine.score(event).score
    assert abs(1 / (1 + math.exp(-margin)) - score) < 2e-4
