import json
import sys

import pytest


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
    拒绝半场用全零标签的 --gate-input（P7 deferred 5 后由 recalibrate 预检拒绝：
    "recalibrate: gate-input needs both classes"，确定性 SystemExit）。
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
    fields = ["encounter_id", *READMISSION_PAYLOAD_KEYS, "label"]
    with holdout.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for index, (row, label) in enumerate(zip(rows, labels), start=1):
            out = {"encounter_id": f"enc-{index:04d}"}
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
    # P6 deferred 10：再校准块同样携带样本审计。
    recal_rows = list(_csv.DictReader(holdout.open(newline="", encoding="utf-8")))
    assert after["calibration"]["n_samples"] == len(recal_rows)
    assert after["calibration"]["n_positives"] == sum(
        1 for row in recal_rows if row["label"] == "1"
    )

    # 不过路径：gate 输入无阳性 → SystemExit 且原件分毫不动。
    # P7 deferred 5：拒绝发生在 recalibrate 的 gate-input 预检处，语出 recalibrate。
    no_positive = tmp_path / "no-positive.csv"
    with no_positive.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(list(rows)[:40], start=1):
            out = {"encounter_id": f"enc-{index:04d}"}
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
        with pytest.raises(SystemExit, match="recalibrate: gate-input needs both classes"):
            recalibrate_main()
    finally:
        sys.argv = old_argv
    assert artifact.read_text(encoding="utf-8") == before_text


@pytest.mark.parametrize("bad_label", ["2", "-1", "0.5", "1.0"])
def test_recalibrate_rejects_non_binary_labels(tmp_path, bad_label):
    """P7 deferred 8：int("2")==2 曾被放行进校准拟合——严格只收 "0"/"1"。"""
    from scripts.recalibrate import main as recalibrate_main

    artifact = tmp_path / "rm.json"
    artifact.write_text(json.dumps({"target_id": "readmission_30d"}), encoding="utf-8")
    inp = tmp_path / "in.csv"
    inp.write_text(
        "encounter_id,label\n"
        f"enc-0001,{bad_label}\n"
        "enc-0002,0\n"
        "enc-0003,1\n",
        encoding="utf-8",
    )
    old_argv = sys.argv
    sys.argv = [
        "recalibrate.py",
        "--artifact",
        str(artifact),
        "--input",
        str(inp),
        "--target-id",
        "readmission_30d",
    ]
    try:
        with pytest.raises(SystemExit, match="needs label 0/1"):
            recalibrate_main()
    finally:
        sys.argv = old_argv


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


def test_recent_scores_and_band_share_gauges(tmp_path):
    """Review I4：T7.4 指标半件回归钉——recent_scores 序、gauge 渲染、/api/metrics 聚合。"""
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.domain import ClinicalEvent
    from app.main import create_app
    from app.observability import metrics
    from app.risk import RiskEngine
    from app.store import Store

    # 1) recent_scores 按写入倒序返回
    settings = Settings(model_path=str(tmp_path / "missing.json"))
    with Store(str(tmp_path / "store.db")) as store:
        store.seed(
            [{"patient_id": "p1", "encounter_id": "e1", "display_name": "指标"}],
            [
                ClinicalEvent(
                    event_type="vital_signs",
                    patient_id="p1",
                    encounter_id="e1",
                    payload={"heart_rate": 80},
                )
            ],
        )
        engine = RiskEngine(settings)
        store.upsert_score(
            engine.score(
                ClinicalEvent(
                    event_type="vital_signs",
                    patient_id="p1",
                    encounter_id="e1",
                    payload={"heart_rate": 80},
                )
            )
        )
        recent = store.recent_scores(limit=10)
        assert len(recent) >= 1
        assert recent[0].encounter_id == "e1"

    # 2) set_share 渲染合法 gauge 行
    metrics.set_share("medintel_band_share", {"low": 0.75, "high": 0.25}, label="band")
    rendered = metrics.render()
    assert "# TYPE medintel_band_share gauge" in rendered
    assert 'medintel_band_share{band="low"} 0.750000' in rendered

    # 3) /api/metrics 现场聚合（dev 种子已有分数 → 两条 gauge 必在）
    application = create_app(
        Settings(store_path=str(tmp_path / "app.db"), model_path=str(tmp_path / "missing.json"))
    )
    with TestClient(application) as client:
        text = client.get("/api/metrics").text
    assert "medintel_band_share" in text
    assert "medintel_missing_rate" in text


def test_compute_ks_hand_computed():
    from scripts.monitor_drift import compute_ks

    # Fa over [1,2,3,4] vs Fb over [2.5,3.5,4.5,5]：max|Fa−Fb| = 0.5（区间 [2.5,3)）
    assert compute_ks([1, 2, 3, 4], [2.5, 3.5, 4.5, 5.0]) == 0.5
    assert compute_ks([1, 2, 3], [1, 2, 3]) == 0.0  # 同分布
    assert compute_ks([1, 2, 3], [7, 8, 9]) == 1.0  # 完全分离
    assert compute_ks([1, 1, 2, 2], [1, 2]) == 0.0  # 平局重复值


def test_compute_ks_skips_when_reference_missing():
    from scripts.monitor_drift import compute_ks

    assert compute_ks(None, [1, 2]) is None
    assert compute_ks([], [1, 2]) is None
    assert compute_ks([1, 2], []) is None
    assert compute_ks([1, 2], ["", None, "not-a-number"]) is None  # 全无效 → None


def test_drift_alert_truth_table():
    from scripts.monitor_drift import PSI_THRESHOLD, drift_alert

    assert drift_alert(0.3, None) is True  # psi 高独立触发（ks 不可用不阻断）
    assert drift_alert(0.1, 0.05) is False  # 双低
    assert drift_alert(0.1, 0.2) is True  # psi 低但 ks 高 → OR 触发
    assert drift_alert(0.1, 0.2, ks_threshold=0.3) is False  # 阈值可调
    assert drift_alert(PSI_THRESHOLD, None) is False  # 恰等阈值不触发（> 严格）
    assert drift_alert(0.1, None) is False  # 无 ks 不误报


def _drift_artifact(path, sample):
    path.write_text(
        json.dumps(
            {
                "model_version": "ks-v1",
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
                        "sample": sample,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_monitor_drift_cli_reports_ks(tmp_path):
    import pytest

    from scripts.monitor_drift import main as monitor_main

    artifact = tmp_path / "model.json"
    reference_sample = list(range(50, 150))  # 100 个值，与 10 箱均匀参考一致
    _drift_artifact(artifact, reference_sample)

    # 稳定输入（与参考同源）→ ks≈0、psi 低 → exit 0
    stable = tmp_path / "stable.csv"
    stable.write_text(
        "heart_rate,label\n" + "\n".join(f"{v},0" for v in reference_sample) + "\n",
        encoding="utf-8",
    )
    old_argv = sys.argv
    sys.argv = ["monitor_drift.py", "--artifact", str(artifact), "--input", str(stable)]
    try:
        with pytest.raises(SystemExit) as excinfo:
            monitor_main()
    finally:
        sys.argv = old_argv
    assert excinfo.value.code == 0

    # 剧偏输入（全部 > 参考上界）→ ks=1.0 > 0.1 → exit 1
    shifted = tmp_path / "shifted.csv"
    shifted.write_text(
        "heart_rate,label\n" + "\n".join(f"{200 + i},0" for i in range(60)) + "\n",
        encoding="utf-8",
    )
    sys.argv = ["monitor_drift.py", "--artifact", str(artifact), "--input", str(shifted)]
    try:
        with pytest.raises(SystemExit) as excinfo:
            monitor_main()
    finally:
        sys.argv = old_argv
    assert excinfo.value.code == 1


def test_monitor_drift_legacy_artifact_without_sample(tmp_path):
    """旧工件无 sample → ks=None + reason，不误报（向后兼容，exit 由 psi 决定）。"""
    import pytest

    from scripts.monitor_drift import main as monitor_main

    artifact = tmp_path / "model.json"
    _drift_artifact(artifact, [])
    artifact.write_text(
        json.dumps(
            {
                **json.loads(artifact.read_text(encoding="utf-8")),
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
    stable = tmp_path / "rows.csv"
    lines = ["heart_rate,label"] + [f"{50 + i % 100},{i % 2}" for i in range(100)]
    stable.write_text("\n".join(lines) + "\n", encoding="utf-8")
    old_argv = sys.argv
    sys.argv = ["monitor_drift.py", "--artifact", str(artifact), "--input", str(stable)]
    try:
        with pytest.raises(SystemExit) as excinfo:
            monitor_main()
    finally:
        sys.argv = old_argv
    assert excinfo.value.code == 0  # 无 sample 且 psi 低 → 不告警


def test_drift_reference_includes_reference_sample(tmp_path):
    """训练写入确定性参考样本（≤2000/特征，R-RS-13）：KS 检验的数据源。"""
    from scripts.train_model import main as train_main

    def train(output):
        old_argv = sys.argv
        sys.argv = [
            "train_model.py",
            "--output",
            str(output),
            "--rows",
            "800",
            "--target-id",
            "deterioration_24h",
            "--version",
            "ks-ref-v1",
        ]
        try:
            import contextlib
            import io

            with contextlib.redirect_stdout(io.StringIO()):
                train_main()
        finally:
            sys.argv = old_argv

    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    train(first)
    train(second)
    ref_a = json.loads(first.read_text(encoding="utf-8"))["drift_reference"]
    ref_b = json.loads(second.read_text(encoding="utf-8"))["drift_reference"]
    assert ref_a == ref_b  # 确定性
    assert ref_a
    for name, entry in ref_a.items():
        assert "sample" in entry, name
        assert 0 < len(entry["sample"]) <= 2000, name
        assert all(isinstance(value, (int, float)) for value in entry["sample"]), name


def test_monitor_drift_derives_readmission_features_from_source_columns(tmp_path):
    """输入 CSV 只有出院源字段（无派生列）→ monitor 即时编码，不全 missing 误报（R-RS-15）。"""
    import pytest

    from app.features import READMISSION_FEATURES
    from scripts.monitor_drift import main as monitor_main

    artifact = tmp_path / "model.json"
    # 参考分布：两派生特征取值 0..3 均匀（10 箱下命中箱 0/3/6/9）
    uniform = [0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25]
    artifact.write_text(
        json.dumps(
            {
                "model_version": "derive-v1",
                "target_id": "readmission_30d",
                "feature_names": list(READMISSION_FEATURES),
                "feature_order": list(READMISSION_FEATURES),
                "weights": {name: 0.0 for name in READMISSION_FEATURES},
                "bias": -1.0,
                "thresholds": {"medium": 0.15, "high": 0.3, "critical": 0.55},
                "drift_reference": {
                    "diagnosis_code_count": {
                        "lo": 0.0,
                        "hi": 3.0,
                        "proportions": uniform,
                        "missing_rate": 0.0,
                    },
                    "discharge_destination_code": {
                        "lo": 0.0,
                        "hi": 3.0,
                        "proportions": uniform,
                        "missing_rate": 0.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    import csv as csv_module
    from io import StringIO

    destinations = ["home", "home_health", "snf", "other"]
    buffer = StringIO()
    writer = csv_module.writer(buffer)
    writer.writerow(
        [
            "encounter_id",
            "los_days",
            "prior_utilization_12m",
            "comorbidity_index",
            "ed_visits_6m",
            "diagnosis_codes",
            "discharge_destination",
            "label",
        ]
    )
    for i in range(80):
        codes = json.dumps([f"D{j:02d}" for j in range(i % 4)])
        writer.writerow([f"enc-{i + 1:04d}", 2.0, 1, 1.5, 0, codes, destinations[i % 4], 0])
    batch = tmp_path / "batch.csv"
    batch.write_text(buffer.getvalue(), encoding="utf-8")
    old_argv = sys.argv
    sys.argv = ["monitor_drift.py", "--artifact", str(artifact), "--input", str(batch)]
    try:
        with pytest.raises(SystemExit) as excinfo:
            monitor_main()
    finally:
        sys.argv = old_argv
    assert excinfo.value.code == 0  # 派生后同分布 → 不告警（旧行为：missing 爆表 exit 1）
