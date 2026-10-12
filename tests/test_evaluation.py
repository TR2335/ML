import sys

from scripts.evaluate_model import binary_metrics


def test_binary_metrics_rank_and_threshold_outputs():
    result = binary_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], 0.5)
    assert result["auroc"] == 1.0
    assert result["auprc"] == 1.0
    assert result["sensitivity"] == 1.0
    assert result["specificity"] == 1.0


def test_news2_score_matches_manual_example():
    from scripts.baselines import news2_score

    # 官方权重手算：RR 25→3、SpO2 92→2、SBP 95→2、HR 125→2、T 39.5→2，合计 11 → high
    score, band = news2_score(
        {
            "respiratory_rate": 25,
            "spo2": 92,
            "systolic_bp": 95,
            "heart_rate": 125,
            "temperature": 39.5,
        }
    )
    assert score == 11
    assert band == "high"


def test_labeled_csv_three_way_comparison(tmp_path):
    import json

    from app.config import Settings
    from app.risk import RiskEngine
    from app.synthetic import FEATURES
    from scripts.evaluate_model import comparison_metrics

    path = tmp_path / "model.json"
    path.write_text(
        json.dumps(
            {
                "model_version": "cmp-v1",
                "feature_names": FEATURES,
                "weights": {feature: 0.0 for feature in FEATURES},
                "bias": 0.5,
                "thresholds": {"medium": 0.2, "high": 0.45, "critical": 0.75},
            }
        ),
        encoding="utf-8",
    )
    engine = RiskEngine(Settings(model_path=str(path)))
    rows = [
        {
            "label": "1",
            "heart_rate": "130",
            "respiratory_rate": "28",
            "systolic_bp": "92",
            "spo2": "91",
            "temperature": "39.2",
            "news2_score": "12",
        },
        {
            "label": "0",
            "heart_rate": "72",
            "respiratory_rate": "16",
            "systolic_bp": "124",
            "spo2": "98",
            "temperature": "36.6",
            "news2_score": "1",
        },
        {"label": "1", "heart_rate": "118", "news2_score": "7"},
        {"label": "0", "heart_rate": "70", "news2_score": "0"},
        {"label": "1", "heart_rate": "135", "news2_score": "13"},
        {"label": "0", "heart_rate": "66", "news2_score": "2"},
    ]
    report = comparison_metrics(rows, engine, "label")
    assert set(report) == {"model", "news2", "model_with_news2"}
    for metrics in report.values():
        assert {"auroc", "auprc", "brier"} <= set(metrics)


def test_comparison_news2_median_imputation_on_v2_artifact(tmp_path):
    """11 特征工件：model 列缺 news2 → median 填补；model_with_news2 注入后分数不同。"""
    import json

    from app.config import Settings
    from app.risk import RiskEngine
    from app.synthetic import FEATURES
    from scripts.evaluate_model import comparison_metrics

    order = [*FEATURES, "news2_score"]
    path = tmp_path / "model.json"
    path.write_text(
        json.dumps(
            {
                "model_version": "v2-news2",
                "feature_names": order,
                "feature_order": order,
                "preprocess": {
                    **{feature: {"median": 0.0, "mean": 0.0, "std": 1.0} for feature in FEATURES},
                    "news2_score": {"median": 6.0, "mean": 6.0, "std": 2.0},
                },
                "weights": {**{feature: 0.0 for feature in FEATURES}, "news2_score": 2.0},
                "bias": 0.0,
                "thresholds": {"medium": 0.2, "high": 0.45, "critical": 0.75},
            }
        ),
        encoding="utf-8",
    )
    engine = RiskEngine(Settings(model_path=str(path)))
    rows = [
        {"label": "1", "news2_score": "12"},
        {"label": "0", "news2_score": "1"},
    ]
    report = comparison_metrics(rows, engine, "label")
    # model 列缺 news2 → median=6 填补 → z=0 → 两行同为 sigmoid(0)=0.5 → brier 恰 0.25
    assert report["model"]["brier"] == 0.25
    # model_with_news2 注入：12→z=3→logit 6；1→z=-2.5→logit -5 → 分数贴合标签
    assert report["model_with_news2"]["brier"] < 0.01
    # 三列共用同一 label 向量
    assert report["model"]["n"] == report["model_with_news2"]["n"] == 2
    assert report["model"]["prevalence"] == report["model_with_news2"]["prevalence"]


def test_labeled_csv_rejects_subgroup_analysis(tmp_path, monkeypatch):
    """Plan Ruling R12：labeled_csv 与 --group-column 互斥。"""
    import pytest

    from scripts.evaluate_model import main

    csv_path = tmp_path / "rows.csv"
    csv_path.write_text(
        "encounter_id,index_time,label\nenc-1,2026-01-01T00:00:00+00:00,1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_model.py",
            "--input",
            str(csv_path),
            "--input-format",
            "labeled_csv",
            "--group-column",
            "department",
        ],
    )
    with pytest.raises(SystemExit, match="subgroup analysis requires"):
        main()


def test_labeled_csv_rejected_for_readmission(tmp_path, monkeypatch):
    """Review I1：labeled_csv 三列 NEWS2 对比对 readmission 显式拒绝，不产出退化指标。"""
    import pytest

    from scripts.evaluate_model import main

    csv_path = tmp_path / "holdout.csv"
    csv_path.write_text(
        "encounter_id,los_days,prior_utilization_12m,comorbidity_index,ed_visits_6m,label\n"
        "enc-1,2.0,0,1.0,0,1\nenc-2,2.0,0,0.0,0,0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_model.py",
            "--input",
            str(csv_path),
            "--input-format",
            "labeled_csv",
            "--target-id",
            "readmission_30d",
        ],
    )
    with pytest.raises(SystemExit, match="labeled_csv"):
        main()


def test_evaluate_cli_uses_explicit_model_for_readmission(tmp_path, monkeypatch):
    """Review I1 同族：--model 必须对 readmission 目标生效（不得静默读默认路径）。"""
    import json

    from app.synthetic import FEATURES
    from scripts.evaluate_model import main

    artifact = tmp_path / "probe.json"
    weights = {feature: 0.0 for feature in FEATURES}
    artifact.write_text(
        json.dumps(
            {
                "model_version": "probe-rm-v1",
                "target_id": "readmission_30d",
                "feature_names": FEATURES,
                "weights": weights,
                "bias": -2.0,
                "thresholds": {"medium": 0.15, "high": 0.3, "critical": 0.55},
            }
        ),
        encoding="utf-8",
    )
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text("encounter_id,los_days,label\nenc-1,1.0,1\nenc-2,2.0,0\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_model.py",
            "--input",
            str(csv_path),
            "--target-id",
            "readmission_30d",
            "--model",
            str(artifact),
        ],
    )
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        main()
    report = json.loads(buffer.getvalue())
    assert report["model_version"] == "probe-rm-v1"


def test_evaluate_cli_patient_mode_reports_split(tmp_path, monkeypatch):
    """默认 patient 模式：解析分组键并写入报告（spec §4.2 接入点 2）。"""
    import contextlib
    import io
    import json

    from scripts.evaluate_model import main

    csv_path = tmp_path / "rows.csv"
    csv_path.write_text(
        "patient_id,encounter_id,index_time,label\n"
        "pat-1,enc-1,2026-01-01T00:00:00+00:00,1\n"
        "pat-2,enc-2,2026-01-02T00:00:00+00:00,0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_model.py", "--input", str(csv_path), "--split", "patient"],
    )
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        main()
    report = json.loads(buffer.getvalue())
    assert report["split_mode"] == "patient"
    assert report["group_column"] == "patient_id"


def test_evaluate_cli_patient_mode_loud_without_group_keys(tmp_path, monkeypatch):
    import pytest

    from scripts.evaluate_model import main

    csv_path = tmp_path / "bare.csv"
    csv_path.write_text("los_days,label\n1.0,1\n2.0,0\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_model.py", "--input", str(csv_path), "--split", "patient"],
    )
    with pytest.raises(SystemExit, match="patient-level"):
        main()


def _quadratic_auprc_reference(labels, scores) -> float:
    """旧 O(n²) 前缀和实现——等价性测试的对照（R-RS-11）。"""
    positives = sum(labels)
    pairs = sorted(zip(scores, labels), reverse=True)
    precision_recall = []
    for index, (_, label) in enumerate(pairs, start=1):
        true_positive = sum(item_label for _, item_label in pairs[:index])
        precision_recall.append(
            (true_positive / index, true_positive / positives if positives else 0)
        )
    auprc = 0.0
    previous_recall = 0.0
    for precision, recall in precision_recall:
        auprc += precision * max(recall - previous_recall, 0)
        previous_recall = recall
    return auprc


def test_binary_metrics_auprc_matches_quadratic_reference():
    from random import Random

    from scripts.evaluate_model import binary_metrics

    rng = Random(7)
    for trial in range(25):
        labels = [rng.randrange(2) for _ in range(rng.randint(10, 300))]
        if not sum(labels):
            continue
        scores = [round(rng.random(), 1) for _ in labels]  # 大量平局
        new = binary_metrics(labels, scores, 0.5)["auprc"]
        old = _quadratic_auprc_reference(labels, scores)
        assert new == old, f"trial {trial}: {new!r} != {old!r}"  # 整数前缀和 → 位级一致


def test_binary_metrics_handles_100k_rows():
    """O(n²) 旧实现在此规模必挂（1e10 级前缀和）；O(n log n) 新实现秒级完成。"""
    from scripts.evaluate_model import binary_metrics

    n = 100_000
    labels = [1 if index % 7 == 0 else 0 for index in range(n)]
    scores = [(index % 997) / 997 for index in range(n)]
    metrics = binary_metrics(labels, scores, 0.5)
    assert metrics["n"] == n
    assert metrics["auprc"] is not None and 0 < metrics["auprc"] <= 1
