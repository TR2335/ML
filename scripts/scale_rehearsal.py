"""端到端规模演练编排（真实规模优化 G5）：逐段计时 + 机读/人读报告。

按 spec §4.5 顺序编排：生成 → 训练（两目标）→ 患者级验证 → 评估 → promote 门禁 →
报告卡 → 漂移自检。时间预算按组判定，超预算记 budget_exceeded（告警不阻断）。
门禁为 spec 步骤表漏列项（通过标准 2 要求），作为独立 gate 组补入——合计预算
1380s（spec 原表 1200s + 门禁 180s，R-RHR-1）。
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

# 直接以 `python scripts/x.py` 运行时根目录不在 sys.path（Ruling R14）。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_SCALE = 25000
DEFAULT_SEED = 20261012

# 组 → (预算秒, 步骤键列表)；预算来源 spec §4.5（gate 组为补入项）
GROUPS: list[tuple[str, int, list[str]]] = [
    ("generate_cohort", 180, ["generate"]),
    ("train", 480, ["train_deterioration", "train_readmission"]),
    ("validate", 300, ["validate_deterioration", "validate_readmission"]),
    ("evaluate", 120, ["evaluate_deterioration", "evaluate_readmission"]),
    ("gate", 180, ["gate_deterioration", "gate_readmission"]),
    ("report", 60, ["report_deterioration", "report_readmission"]),
    ("drift", 60, ["drift_deterioration", "drift_readmission"]),
]


def _split_holdout(source: Path, destination: Path, start_fraction: float = 0.8) -> int:
    """按 index_time 排序取尾部时间段写 holdout（训练/校准只用前 80% 时间段）。"""
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    rows.sort(key=lambda row: row["index_time"])
    tail = rows[int(len(rows) * start_fraction) :]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(tail)
    return len(tail)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_json_from_stdout(stdout: str) -> dict | None:
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return None


def run(args: argparse.Namespace) -> dict:
    root = Path(__file__).resolve().parents[1]
    work = Path(args.work_dir)
    if not work.is_absolute():
        work = root / work
    work.mkdir(parents=True, exist_ok=True)
    prefix = Path(args.report_prefix)
    if not prefix.is_absolute():
        prefix = root / prefix

    det_data = work / "dataset_deterioration.csv"
    rm_data = work / "dataset_readmission.csv"
    det_model = work / "model_deterioration.json"
    rm_model = work / "model_readmission.json"
    det_holdout = work / "holdout_deterioration.csv"
    rm_holdout = work / "holdout_readmission.csv"
    registry = work / "registry"
    cohort_stats_path = work / "cohort_stats.json"

    steps: list[dict] = []
    started = time.perf_counter()

    group_budget = {name: budget for name, budget, _keys in GROUPS}

    def run_step(name: str, group: str, argv: list[str]):
        step_started = time.perf_counter()
        # 挂死保护（M6）：预算仅事后标记，超时硬中断（组预算×2，下限 10 分钟）。
        timeout = max(group_budget.get(group, 300) * 2, 600)
        try:
            proc = subprocess.run(
                [sys.executable, *argv],
                capture_output=True,
                text=True,
                cwd=root,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stderr = exc.stderr or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            steps.append(
                {
                    "name": name,
                    "group": group,
                    "seconds": round(time.perf_counter() - step_started, 2),
                    "exit_code": None,
                    "ok": False,
                    "error": f"timeout after {timeout}s",
                    "stderr_tail": stderr[-2000:],
                }
            )
            raise SystemExit(f"scale_rehearsal: step {name} timed out after {timeout}s") from exc
        entry: dict = {
            "name": name,
            "group": group,
            "seconds": round(time.perf_counter() - step_started, 2),
            "exit_code": proc.returncode,
            "ok": proc.returncode == 0,
        }
        if proc.returncode != 0:
            entry["stderr_tail"] = proc.stderr[-2000:]
        steps.append(entry)
        return proc

    run_step(
        "generate",
        "generate_cohort",
        [
            "scripts/generate_scale_data.py",
            "--scale",
            str(args.scale),
            "--seed",
            str(args.seed),
            "--out-dir",
            str(work),
        ],
    )
    holdout_rows: dict[str, int] = {}
    if steps[-1]["ok"]:
        holdout_rows["deterioration"] = _split_holdout(det_data, det_holdout)
        holdout_rows["readmission"] = _split_holdout(rm_data, rm_holdout)

    run_step(
        "train_deterioration",
        "train",
        [
            "scripts/train_model.py",
            "--input",
            str(det_data),
            "--target-id",
            "deterioration_24h",
            "--output",
            str(det_model),
            "--seed",
            str(args.seed),
        ],
    )
    run_step(
        "train_readmission",
        "train",
        [
            "scripts/train_model.py",
            "--input",
            str(rm_data),
            "--target-id",
            "readmission_30d",
            "--output",
            str(rm_model),
            "--seed",
            str(args.seed),
        ],
    )

    validate_reports: dict[str, dict | None] = {}
    for target, data_path, key in (
        ("deterioration_24h", det_data, "deterioration"),
        ("readmission_30d", rm_data, "readmission"),
    ):
        proc = run_step(
            f"validate_{key}",
            "validate",
            [
                "scripts/validate.py",
                "--input",
                str(data_path),
                "--target-id",
                target,
                "--split",
                "patient",
            ],
        )
        validate_reports[key] = (
            _read_json_from_stdout(proc.stdout) if proc.returncode == 0 else None
        )

    for target, data_path, key in (
        ("deterioration_24h", det_data, "deterioration"),
        ("readmission_30d", rm_data, "readmission"),
    ):
        argv = [
            "scripts/evaluate_model.py",
            "--input",
            str(data_path),
            "--target-id",
            target,
            "--split",
            "patient",
        ]
        if target == "deterioration_24h":
            argv += ["--input-format", "labeled_csv"]
        run_step(f"evaluate_{key}", "evaluate", argv)

    gate_reports: dict[str, dict | None] = {}
    for target, model_path, holdout_path, key in (
        ("deterioration_24h", det_model, det_holdout, "deterioration"),
        ("readmission_30d", rm_model, rm_holdout, "readmission"),
    ):
        run_step(
            f"gate_{key}",
            "gate",
            [
                "scripts/manage_model.py",
                "--registry-dir",
                str(registry),
                "promote",
                "--artifact",
                str(model_path),
                "--target-id",
                target,
                "--input",
                str(holdout_path),
            ],
        )
        artifact = _read_json(model_path)
        version = (artifact or {}).get("model_version", "unknown")
        gate_reports[key] = _read_json(registry / target / version / "eval.json")

    for target, data_path, model_path, key in (
        ("deterioration_24h", det_data, det_model, "deterioration"),
        ("readmission_30d", rm_data, rm_model, "readmission"),
    ):
        run_step(
            f"report_{key}",
            "report",
            [
                "scripts/report_card.py",
                "--input",
                str(data_path),
                "--target-id",
                target,
                "--model",
                str(model_path),
                "--output",
                str(work / f"report_card_{key}"),
                "--cohort-stats",
                str(cohort_stats_path),
            ],
        )

    for _target, model_path, holdout_path, key in (
        ("deterioration_24h", det_model, det_holdout, "deterioration"),
        ("readmission_30d", rm_model, rm_holdout, "readmission"),
    ):
        run_step(
            f"drift_{key}",
            "drift",
            [
                "scripts/monitor_drift.py",
                "--artifact",
                str(model_path),
                "--input",
                str(holdout_path),
            ],
        )

    # —— 分组预算 ——
    seconds_by_group: dict[str, float] = {}
    for step in steps:
        seconds_by_group[step["group"]] = seconds_by_group.get(step["group"], 0.0) + step["seconds"]
    groups = [
        {
            "name": name,
            "seconds": round(seconds_by_group.get(name, 0.0), 2),
            "budget": budget,
            "budget_exceeded": seconds_by_group.get(name, 0.0) > budget,
        }
        for name, budget, _keys in GROUPS
    ]

    # —— 通过标准（spec §4.5）——
    cohort = _read_json(cohort_stats_path) or {}
    leakage_ok = all(
        validate_reports[key] is not None and validate_reports[key].get("split_mode") == "patient"
        for key in ("deterioration", "readmission")
    )
    checks: dict = {
        "leakage": {
            "status": "pass" if leakage_ok else "fail",
            # violations 恒为 int（机读 schema 稳定，M7）；失败原因放 reason。
            "violations": 0 if leakage_ok else None,
            **({} if leakage_ok else {"reason": "patient split unavailable or wrong mode"}),
        },
        "gates": {
            key: bool((gate_reports[key] or {}).get("passed"))
            for key in ("deterioration", "readmission")
        },
    }
    det_event_rate = (cohort.get("deterioration") or {}).get("event_rate")
    rm_rate = (cohort.get("readmission") or {}).get("positive_rate")
    checks["metrics_sanity"] = {
        # deterioration 按遭遇级事件率口径（R-RS-5）；readmission 行级=患者级
        "deterioration_event_rate": {
            "value": det_event_rate,
            "status": "pass"
            if det_event_rate is not None and 0.01 <= det_event_rate <= 0.07
            else "fail",
        },
        "readmission_rate": {
            "value": rm_rate,
            "status": "pass" if rm_rate is not None and 0.09 <= rm_rate <= 0.15 else "fail",
        },
        "ece": {
            key: (gate_reports[key] or {}).get("ece") for key in ("deterioration", "readmission")
        },
    }
    det_rows = (cohort.get("deterioration") or {}).get("rows")
    facilities = cohort.get("facilities") or {}
    patients = cohort.get("patients")
    # 患者数核对（spec §4.5 第 4 条，M4）：round(scale × 0.82)，±1 容差（R-RS-1 公式）。
    expected_patients = round(args.scale * 0.82)
    scale_ok = (
        cohort.get("scale") == args.scale
        and det_rows == args.scale * 16
        and patients is not None
        and abs(patients - expected_patients) <= 1
        and len(facilities) == 3
        and all(count > 0 for count in facilities.values())
    )
    checks["scale_sanity"] = {
        "status": "pass" if scale_ok else "fail",
        "scale": cohort.get("scale"),
        "deterioration_rows": det_rows,
        "patients": patients,
        "expected_patients": expected_patients,
        "facilities": facilities,
        "holdout_rows": holdout_rows,
    }

    steps_ok = all(step["ok"] for step in steps)
    gates_ok = all(checks["gates"].values())
    sanity_ok = (
        checks["leakage"]["status"] == "pass"
        and checks["metrics_sanity"]["deterioration_event_rate"]["status"] == "pass"
        and checks["metrics_sanity"]["readmission_rate"]["status"] == "pass"
        and checks["scale_sanity"]["status"] == "pass"
    )
    report = {
        "report_type": "scale_rehearsal",
        "scale": args.scale,
        "seed": args.seed,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "steps": steps,
        "groups": groups,
        "checks": checks,
        "budget_exceeded": [group["name"] for group in groups if group["budget_exceeded"]],
        "passed": bool(steps_ok and gates_ok and sanity_ok),
    }

    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(str(prefix) + ".json")
    md_path = Path(str(prefix) + ".md")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "passed": report["passed"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return report


def _render_markdown(report: dict) -> str:
    lines = [
        "# Scale Rehearsal Report",
        "",
        f"- scale: {report['scale']}",
        f"- seed: {report['seed']}",
        f"- elapsed: {report['elapsed_seconds']}s",
        f"- passed: {report['passed']}",
        "",
        "## Steps",
        "",
        "| step | group | seconds | exit | ok |",
        "|---|---|---|---|---|",
    ]
    for step in report["steps"]:
        lines.append(
            f"| {step['name']} | {step['group']} | {step['seconds']} "
            f"| {step['exit_code']} | {step['ok']} |"
        )
    lines += [
        "",
        "## Budget Groups",
        "",
        "| group | seconds | budget | exceeded |",
        "|---|---|---|---|",
    ]
    for group in report["groups"]:
        lines.append(
            f"| {group['name']} | {group['seconds']} | {group['budget']} "
            f"| {group['budget_exceeded']} |"
        )
    lines += ["", "## Checks", ""]
    lines.append(f"- leakage: {json.dumps(report['checks']['leakage'], ensure_ascii=False)}")
    lines.append(f"- gates: {json.dumps(report['checks']['gates'], ensure_ascii=False)}")
    lines.append(
        "- metrics_sanity: " + json.dumps(report["checks"]["metrics_sanity"], ensure_ascii=False)
    )
    lines.append(
        "- scale_sanity: " + json.dumps(report["checks"]["scale_sanity"], ensure_ascii=False)
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end scale rehearsal with budgets")
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--work-dir", default="data/scale")
    parser.add_argument("--report-prefix", default="data/rehearsal_report")
    args = parser.parse_args()
    report = run(args)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
