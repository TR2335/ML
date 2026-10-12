"""端到端规模演练（真实规模优化 G5）：--scale 1500 冒烟，全链路一次跑通。"""

import json
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_scale_rehearsal_smoke(tmp_path):
    from scripts.scale_rehearsal import main as rehearsal_main

    work = tmp_path / "scale"
    prefix = tmp_path / "rehearsal_report"
    old_argv = sys.argv
    sys.argv = [
        "scale_rehearsal.py",
        "--scale",
        "1500",
        "--seed",
        "20261012",
        "--work-dir",
        str(work),
        "--report-prefix",
        str(prefix),
    ]
    try:
        rehearsal_main()
    finally:
        sys.argv = old_argv

    report = json.loads(Path(str(prefix) + ".json").read_text(encoding="utf-8"))
    assert report["scale"] == 1500
    assert report["seed"] == 20261012

    # 步骤与分组预算字段齐全
    assert report["steps"], "no steps recorded"
    for step in report["steps"]:
        assert {"name", "group", "seconds", "exit_code", "ok"} <= set(step), step["name"]
        assert step["ok"], f"{step['name']} failed: {step.get('stderr_tail', '')[-500:]}"
    group_names = [group["name"] for group in report["groups"]]
    assert group_names == [
        "generate_cohort",
        "train",
        "validate",
        "evaluate",
        "gate",
        "report",
        "drift",
    ]
    for group in report["groups"]:
        assert {"name", "seconds", "budget", "budget_exceeded"} <= set(group)
        assert group["budget"] > 0

    # 通过标准：泄漏断言、门禁、指标 sanity、规模 sanity 全过
    checks = report["checks"]
    assert checks["leakage"]["status"] == "pass"
    assert checks["gates"]["deterioration"] is True
    assert checks["gates"]["readmission"] is True
    assert checks["metrics_sanity"]["deterioration_event_rate"]["status"] == "pass"
    assert checks["metrics_sanity"]["readmission_rate"]["status"] == "pass"
    assert checks["scale_sanity"]["status"] == "pass"
    assert report["passed"] is True

    # 产物落盘：报告 md + 各步产物
    assert Path(str(prefix) + ".md").exists()
    assert (work / "cohort_stats.json").exists()
    assert (work / "model_deterioration.json").exists()
    assert (work / "model_readmission.json").exists()
    assert (work / "report_card_deterioration.json").exists()
    assert (work / "report_card_deterioration.md").exists()
    assert (work / "report_card_readmission.json").exists()
    assert (work / "holdout_deterioration.csv").exists()
    assert (work / "holdout_readmission.csv").exists()
