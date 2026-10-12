"""自动模型报告卡（真实规模优化 G3）：章节完整性、指标口径防漂移、双产物。"""

import contextlib
import csv
import io
import json
import sys
from pathlib import Path

FEATURE_HEADER = [
    "encounter_id",
    "patient_id",
    "index_time",
    "label",
    "age",
    "facility_id",
    "heart_rate",
    "respiratory_rate",
    "systolic_bp",
    "temperature",
    "spo2",
    "wbc",
    "creatinine",
    "lactate",
    "prior_admissions_30d",
]


def _write_artifact(path: Path) -> None:
    from app.synthetic import FEATURES

    weights = {feature: 0.0 for feature in FEATURES}
    weights["creatinine"] = 3.0
    path.write_text(
        json.dumps(
            {
                "model_version": "rc-test-v1",
                "target_id": "deterioration_24h",
                "algorithm": "logistic_regression_sklearn",
                "feature_names": FEATURES,
                "feature_order": FEATURES,
                "weights": weights,
                "bias": -4.0,
                "thresholds": {"medium": 0.2, "high": 0.45, "critical": 0.75},
            }
        ),
        encoding="utf-8",
    )


def _write_rows(path: Path, with_age: bool = True) -> None:
    """24 行 / 8 患者（每患者 3 行）：creatinine 决定标签 → 6 阳性（25%）。"""
    header = [name for name in FEATURE_HEADER if with_age or name != "age"]
    lines = [",".join(header)]
    for i in range(24):
        creatinine = 3.0 if i % 4 == 0 else 0.8
        label = int(creatinine > 2)
        cells = [
            f"enc-{i + 1:04d}",
            f"pat-{i // 3:03d}",
            f"2025-05-01T{i:02d}:00:00+00:00",
            str(label),
        ]
        if with_age:
            cells.append(str(40 + (i * 3) % 45))
        cells += [
            "hospital-01" if i % 2 == 0 else "hospital-02",
            "70",
            "16",
            "120",
            "36.8",
            "98",
            "6.5",
            str(creatinine),
            "1.2",
            str(i % 3),
        ]
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_report_card(tmp_path: Path, with_age: bool = True, target: str = "deterioration_24h"):
    from scripts.report_card import main as report_main

    artifact = tmp_path / "model.json"
    rows_csv = tmp_path / "rows.csv"
    _write_artifact(artifact)
    _write_rows(rows_csv, with_age=with_age)
    output = tmp_path / "report_card"
    old_argv = sys.argv
    sys.argv = [
        "report_card.py",
        "--input",
        str(rows_csv),
        "--target-id",
        target,
        "--model",
        str(artifact),
        "--output",
        str(output),
        "--n-splits",
        "3",
    ]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            report_main()
    finally:
        sys.argv = old_argv
    report = json.loads(Path(str(output) + ".json").read_text(encoding="utf-8"))
    return report, Path(str(output) + ".md")


def _section(report: dict, name: str) -> dict:
    for section in report["sections"]:
        if section["name"] == name:
            return section
    raise AssertionError(f"missing section: {name}")


def test_report_card_sections_complete_or_justified(tmp_path):
    report, _md = _run_report_card(tmp_path)
    names = [section["name"] for section in report["sections"]]
    assert names == [
        "data_profile",
        "split_strategy",
        "discrimination",
        "calibration",
        "baseline_comparison",
        "subgroup_facility",
        "decision_curve",
        "threshold_table",
        "drift_reference",
        "runtime",
    ]
    for section in report["sections"]:
        assert section["status"] in ("ok", "skipped")
        if section["status"] == "skipped":
            assert section.get("reason"), section["name"]  # 跳过必须注明原因


def test_report_card_discrimination_equals_direct_metrics(tmp_path):
    """防口径漂移：报告卡 discrimination == 直接调用既有函数的 OOF 结果（R-RS-12）。"""
    from app.config import Settings
    from app.risk import RiskEngine
    from scripts.evaluate_model import binary_metrics, evaluate_rows
    from scripts.validate import patient_rolling_time_split

    report, _md = _run_report_card(tmp_path)
    artifact = str(tmp_path / "model.json")
    engine = RiskEngine(Settings(model_path=artifact, app_env="dev"))
    with (tmp_path / "rows.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    folds = patient_rolling_time_split(rows, 3)
    oof_labels: list[int] = []
    oof_scores: list[float] = []
    for _train, test in folds:
        labels, scores = evaluate_rows(test, engine, "label")
        oof_labels.extend(labels)
        oof_scores.extend(scores)
    expected = binary_metrics(oof_labels, oof_scores, engine.thresholds["high"])
    discrimination = _section(report, "discrimination")
    assert discrimination["data"]["out_of_fold"] == expected
    # 校准章节与判别章节同源（同一次 binary_metrics 结果）
    calibration = _section(report, "calibration")
    assert calibration["data"]["ece"] == expected["ece"]
    assert calibration["data"]["brier"] == expected["brier"]


def test_report_card_writes_md_and_json(tmp_path):
    report, md_path = _run_report_card(tmp_path)
    assert md_path.exists()
    text = md_path.read_text(encoding="utf-8")
    assert "# " in text
    assert "discrimination" in text  # 章节标题落入 md
    assert report["target_id"] == "deterioration_24h"
    runtime = _section(report, "runtime")
    assert runtime["data"]["seed"] == 20261012
    assert "generated_at" in runtime["data"]
    assert runtime["data"]["elapsed_seconds"] >= 0


def test_report_card_skips_subgroup_without_age(tmp_path):
    report, _md = _run_report_card(tmp_path, with_age=False)
    subgroup = _section(report, "subgroup_facility")
    assert subgroup["status"] == "skipped"
    assert "age" in subgroup["reason"]


def test_report_card_decision_curve_and_thresholds_present(tmp_path):
    report, _md = _run_report_card(tmp_path)
    decision = _section(report, "decision_curve")
    assert decision["status"] == "ok"
    assert decision["data"]["mean_gain"] is not None
    assert len(decision["data"]["table"]) == 17  # 0.10..0.90 步 0.05
    thresholds = _section(report, "threshold_table")
    assert thresholds["status"] == "ok"
    assert set(thresholds["data"]["bands"]) == {"medium", "high", "critical"}
    for band in thresholds["data"]["bands"].values():
        assert {"threshold", "sensitivity", "specificity", "ppv"} <= set(band)


def test_report_card_split_strategy_discloses_fit_window(tmp_path):
    """终审 Important-2：OOF 折与 fit 窗（前 60% 时间行）的重叠逐折留痕（R-FIX-2）。"""
    report, _md = _run_report_card(tmp_path)
    split = _section(report, "split_strategy")
    fit = split["data"]["fit_window"]
    assert "60%" in fit["definition"]
    assert fit["fit_cutoff"]
    assert "holdout" in fit["note"]
    assert "interleave" in split["data"]["ordering"]  # 患者时点序 ≠ 行级时间序（M3）
    for fold in split["data"]["folds"]:
        in_fit = fold["test_rows_in_fit_window"]
        beyond = fold["test_rows_beyond_fit_window"]
        assert in_fit + beyond == fold["test_rows"]
    # 报告卡 bootstrap 记录 seed（M5：与 validate 20260928 的差异可机读比对）
    discrimination = _section(report, "discrimination")
    assert discrimination["data"]["bootstrap_ci"]["seed"] == 20261012


def _write_readmission_artifact(path: Path) -> None:
    from app.features import READMISSION_FEATURES

    weights = {feature: 0.0 for feature in READMISSION_FEATURES}
    weights["comorbidity_index"] = 6.0
    preprocess = {
        feature: {"median": 0.0, "mean": 0.0, "std": 1.0} for feature in READMISSION_FEATURES
    }
    path.write_text(
        json.dumps(
            {
                "model_version": "rc-rm-v1",
                "target_id": "readmission_30d",
                "algorithm": "logistic_regression_sklearn",
                "feature_names": READMISSION_FEATURES,
                "feature_order": READMISSION_FEATURES,
                "preprocess": preprocess,
                "calibration": None,
                "weights": weights,
                "bias": -17.9,
                "thresholds": {"medium": 0.15, "high": 0.3, "critical": 0.55},
            }
        ),
        encoding="utf-8",
    )


def test_report_card_readmission_lace_baseline_normalized(tmp_path):
    """终审 Important-1：LACE 归一化 [0,1]（/19）+ 阈值 4/19，ece/brier 恢复可读（R-FIX-1）。

    终审实证：修复前真实 25k 报告卡 ece=5.32（ECE>1 定义上不可能）、brier=38。
    """
    import csv as csv_module
    from io import StringIO

    from scripts.report_card import main as report_main

    artifact = tmp_path / "rm-model.json"
    rows_csv = tmp_path / "rm-rows.csv"
    _write_readmission_artifact(artifact)
    buffer = StringIO()
    writer = csv_module.writer(buffer)
    writer.writerow(
        [
            "encounter_id",
            "patient_id",
            "index_time",
            "label",
            "los_days",
            "prior_utilization_12m",
            "comorbidity_index",
            "ed_visits_6m",
            "diagnosis_codes",
            "discharge_destination",
            "age",
        ]
    )
    for i in range(24):
        comorbidity = round(0.25 * i, 2)
        label = int(comorbidity >= 3.0)
        writer.writerow(
            [
                f"enc-{i + 1:04d}",
                f"pat-{i // 3:03d}",
                f"2025-05-01T{i:02d}:00:00+00:00",
                label,
                2.0,
                1,
                comorbidity,
                1,
                "[]",
                "home",
                50 + i,
            ]
        )
    rows_csv.write_text(buffer.getvalue(), encoding="utf-8")
    output = tmp_path / "rm-report"
    old_argv = sys.argv
    sys.argv = [
        "report_card.py",
        "--input",
        str(rows_csv),
        "--target-id",
        "readmission_30d",
        "--model",
        str(artifact),
        "--output",
        str(output),
        "--n-splits",
        "3",
    ]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            report_main()
    finally:
        sys.argv = old_argv
    report = json.loads(Path(str(output) + ".json").read_text(encoding="utf-8"))
    baseline = _section(report, "baseline_comparison")["data"]["baseline"]
    assert baseline["name"] == "lace"
    metrics = baseline["metrics"]
    assert metrics["ece"] is not None and 0.0 <= metrics["ece"] <= 1.0
    assert metrics["brier"] is not None and 0.0 <= metrics["brier"] <= 1.0
    assert metrics["sensitivity"] is not None and 0.0 <= metrics["sensitivity"] <= 1.0
    assert 0.0 <= (metrics["auprc"] or 0.0) <= 1.0
    # 分数源归一化：LACE 原始分（0–19）/19 后进入 0–1 区间（终审 R-FIX-1）
    from scripts.baselines import lace_score
    from scripts.evaluate_model import payload_from_row

    with rows_csv.open(newline="", encoding="utf-8") as handle:
        first = next(csv_module.DictReader(handle))
    expected = lace_score(payload_from_row(first)) / 19
    # 阈值口径：LACE≥4 文献截断 → 4/19（而非 NEWS2 的 7/15）
    from scripts.report_card import BASELINE_THRESHOLDS

    assert BASELINE_THRESHOLDS["lace"] == 4 / 19
    assert 0.0 <= expected <= 1.0
