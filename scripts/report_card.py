"""自动模型报告卡（真实规模优化 G3）：单命令产出 .json（机读）+ .md（人读）。

硬约束：指标只允许复用既有函数（binary_metrics / evaluate_rows / bootstrap_metric /
decision_curve_table / subgroup_sensitivity_diff / per_facility_auprc）——禁止在
报告卡里重算一遍指标（防口径漂移）。缺数据的章节标 skipped 并注明原因，不静默省略。
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# 直接以 `python scripts/x.py` 运行时根目录不在 sys.path（Ruling R14）。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.risk import RiskEngine
from app.synthetic import FEATURES
from scripts.baselines import lace_score, news2_score
from scripts.evaluate_model import (
    _row_time,
    binary_metrics,
    evaluate_rows,
    payload_from_row,
    resolve_group_column,
)
from scripts.validate import (
    bootstrap_budget,
    bootstrap_metric,
    decision_curve_table,
    patient_rolling_time_split,
    per_facility_auprc,
    rolling_time_split,
    subgroup_sensitivity_diff,
)

SECTION_NAMES = (
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
)
DEFAULT_SEED = 20261012
# 基线判阳阈值（R-FIX-1）：NEWS2 high=7/15；LACE≥4 为 Austin 2010 常用截断（/19 归一化）。
# 基线分数必须在 [0,1]——原始 0–19 分直入 binary_metrics 会让 ECE/Brier/灵敏度全部失真。
BASELINE_THRESHOLDS = {"news2": 7 / 15, "lace": 4 / 19}
LACE_MAX = 19
META_PROFILE_EXCLUDE = {
    "label",
    "patient_id",
    "encounter_id",
    "index_time",
    "offline_score",
    "serving_score",
    "score",
    "uncertainty",
}


def _section(name: str, data: dict | None = None, status: str = "ok", reason: str | None = None):
    entry: dict = {"name": name, "status": status}
    if reason is not None:
        entry["reason"] = reason
    entry["data"] = data if data is not None else {}
    return entry


def _labels_from(rows: list[dict], label_column: str) -> list[int]:
    labels: list[int] = []
    for index, row in enumerate(rows, start=1):
        try:
            label = int(row[label_column])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"report_card: row {index}: {label_column} must be 0 or 1") from exc
        if label not in (0, 1):
            raise SystemExit(f"report_card: row {index}: {label_column} must be 0 or 1")
        labels.append(label)
    return labels


def _distinct_groups(rows: list[dict]) -> int | None:
    if not rows:
        return None
    try:
        column = resolve_group_column(rows)
    except ValueError:
        return None
    return len({str(row.get(column, "") or "") for row in rows})


def _decile_calibration(labels: list[int], scores: list[float], bins: int = 10) -> list[dict]:
    """按分位的校准展示表（展示用，非门禁指标——故允许在此构造）。"""
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    per = max(1, len(order) // bins)
    table: list[dict] = []
    for bucket in range(bins):
        chunk = (
            order[bucket * per :]
            if bucket == bins - 1
            else order[bucket * per : (bucket + 1) * per]
        )
        if not chunk:
            continue
        table.append(
            {
                "bucket": bucket + 1,
                "n": len(chunk),
                "mean_score": round(sum(scores[i] for i in chunk) / len(chunk), 6),
                "observed_rate": round(sum(labels[i] for i in chunk) / len(chunk), 6),
            }
        )
    return table


def _baseline_scores(oof_rows: list[dict], target_id: str) -> tuple[str, list[float]] | None:
    """OOF 行集上的基线分：deterioration → NEWS2，readmission → LACE。"""
    if target_id == "deterioration_24h":
        scores = []
        for row in oof_rows:
            payload = {
                feature: float(row[feature])
                for feature in FEATURES
                if row.get(feature, "") not in ("", None)
            }
            scores.append(news2_score(payload)[0] / 15)
        return "news2", scores
    try:
        # 归一化到 [0,1]（R-FIX-1）：与 NEWS2/15 同一量纲，非概率分数的口径与既有 NEWS2 列一致。
        return "lace", [float(lace_score(payload_from_row(row))) / LACE_MAX for row in oof_rows]
    except ValueError:
        return None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def build_report_card(
    rows: list[dict],
    engine: RiskEngine,
    *,
    target_id: str,
    label_column: str = "label",
    split_mode: str = "patient",
    n_splits: int = 3,
    artifact: dict | None = None,
    cohort_stats: dict | None = None,
    seed: int = DEFAULT_SEED,
) -> dict:
    """构建 10 章节报告（每章节 ok 或 skipped+reason）。"""
    started = time.perf_counter()
    if not rows:
        raise SystemExit("report_card: empty input")
    labels_all = _labels_from(rows, label_column)
    threshold = engine.thresholds["high"]
    sections: list[dict] = []

    # 1. 数据画像
    missing_rates = {
        name: round(sum(1 for row in rows if row.get(name, "") in ("", None)) / len(rows), 4)
        for name in rows[0]
        if name not in META_PROFILE_EXCLUDE
    }
    facilities: dict[str, int] = {}
    departments: dict[str, int] = {}
    for row in rows:
        facility = str(row.get("facility_id", "") or "")
        if facility:
            facilities[facility] = facilities.get(facility, 0) + 1
        department = str(row.get("department", "") or "")
        if department:
            departments[department] = departments.get(department, 0) + 1
    profile: dict = {
        "rows": len(rows),
        "patients": _distinct_groups(rows),
        "encounters": len({str(row.get("encounter_id", "") or "") for row in rows})
        if any(row.get("encounter_id") for row in rows)
        else None,
        "positive_rate": round(sum(labels_all) / len(labels_all), 4),
        "missing_rates": missing_rates,
    }
    if facilities:
        profile["facilities"] = facilities
    if departments:
        profile["departments"] = departments
    if cohort_stats is not None:
        profile["cohort_stats"] = cohort_stats
    sections.append(_section("data_profile", profile))

    # 2. 切分策略 + OOF 打分（单一数据源，各章节复用）
    if split_mode == "patient":
        try:
            folds = patient_rolling_time_split(rows, n_splits)
        except ValueError as exc:
            raise SystemExit(f"report_card: {exc}") from exc
    else:
        try:
            folds = rolling_time_split(rows, n_splits)
        except ValueError as exc:
            raise SystemExit(f"report_card: {exc}") from exc
    oof_rows: list[dict] = []
    oof_labels: list[int] = []
    oof_scores: list[float] = []
    fold_details: list[dict] = []
    # fit 窗重叠披露（终审 R-FIX-2）：train_model fit 于全量行按 index_time 的前 60%
    # （P6 R4）；OOF 折与其重叠的 test 行是部分代入（in-sample），判别指标偏乐观。
    # promote 门禁与演练 sanity 用尾 20% holdout（fit/cal 之外），passed 判定不受影响。
    ordered_times = sorted(_row_time(row) for row in rows)
    fit_cutoff = ordered_times[int(len(ordered_times) * 0.6)]
    for train, test in folds:
        fold_labels, fold_scores = evaluate_rows(test, engine, label_column)
        oof_rows.extend(test)
        oof_labels.extend(fold_labels)
        oof_scores.extend(fold_scores)
        in_fit = sum(1 for row in test if _row_time(row) <= fit_cutoff)
        fold_details.append(
            {
                "train_rows": len(train),
                "test_rows": len(test),
                "train_patients": _distinct_groups(train),
                "test_patients": _distinct_groups(test),
                "test_rows_in_fit_window": in_fit,
                "test_rows_beyond_fit_window": len(test) - in_fit,
                "evaluation": binary_metrics(fold_labels, fold_scores, threshold),
            }
        )
    sections.append(
        _section(
            "split_strategy",
            {
                "mode": split_mode,
                "n_splits": n_splits,
                "fit_window": {
                    "definition": (
                        "train_model fits the earliest 60% of rows by index_time "
                        "(P6 R4); overlapping OOF test rows are partially in-sample"
                    ),
                    "fit_cutoff": fit_cutoff.isoformat(),
                    "note": (
                        "promotion gate uses the tail-20% holdout (fully unseen); "
                        "report-card discrimination over the full input is optimistic "
                        "where test rows fall inside the fit window"
                    ),
                },
                "ordering": (
                    "patient-time ordering (patient first index_time); row-level times "
                    "may interleave across folds — repeat admissions of the same "
                    "patient stay together by design"
                ),
                "folds": fold_details,
            },
        )
    )

    # 3. 判别性能：OOF 指标 + bootstrap CI（规模预算）
    oof_metrics = binary_metrics(oof_labels, oof_scores, threshold)
    n_bootstrap, max_rows = bootstrap_budget(len(oof_labels))

    def _auprc(lab: list[int], sco: list[float]) -> float:
        return binary_metrics(lab, sco, threshold)["auprc"] or 0.0

    point, lo, hi = bootstrap_metric(
        oof_labels, oof_scores, _auprc, n=n_bootstrap, seed=seed, max_rows=max_rows
    )
    sections.append(
        _section(
            "discrimination",
            {
                "out_of_fold": oof_metrics,
                "bootstrap_ci": {
                    "metric": "auprc",
                    "point": point,
                    "lo": lo,
                    "hi": hi,
                    "n": n_bootstrap,
                    "replicate_subsample": max_rows,
                    # validate.py 固定 seed=20260928（历史口径）；报告卡用运行 seed——
                    # 两者 CI 数字允许不同源，机读字段可比对（M5）。
                    "seed": seed,
                },
            },
        )
    )

    # 4. 校准
    sections.append(
        _section(
            "calibration",
            {
                "ece": oof_metrics["ece"],
                "brier": oof_metrics["brier"],
                "deciles": _decile_calibration(oof_labels, oof_scores),
            },
        )
    )

    # 5. 基线对比（model 列即 OOF 复用；model_with_news2 不在此二次打分）
    baseline = _baseline_scores(oof_rows, target_id)
    comparison: dict = {"model": oof_metrics}
    if baseline is None:
        comparison["baseline"] = {
            "status": "skipped",
            "reason": "lace fields missing/invalid in input rows",
        }
    else:
        baseline_name, baseline_scores = baseline
        baseline_metrics = binary_metrics(
            oof_labels, baseline_scores, BASELINE_THRESHOLDS[baseline_name]
        )
        comparison["baseline"] = {
            "name": baseline_name,
            "threshold": BASELINE_THRESHOLDS[baseline_name],
            "metrics": baseline_metrics,
        }
    comparison["model_with_news2"] = {
        "status": "skipped",
        "reason": (
            "report card reuses OOF scores for the model column; incremental-gain "
            "columns are produced by evaluate_model --input-format labeled_csv "
            "(avoids a second full pass of engine scoring)"
        ),
    }
    sections.append(_section("baseline_comparison", comparison))

    # 6. 亚组与院区（无 age 列 → 整章 skipped 并注明）
    ages = [row.get("age", "") for row in oof_rows]
    if baseline is not None:
        facility_data: dict = per_facility_auprc(oof_rows, oof_labels, oof_scores, baseline_scores)
    else:
        facility_data = {"checked": False, "reason": "baseline unavailable", "failures": []}
    if not all(age not in ("", None) for age in ages):
        sections.append(
            _section(
                "subgroup_facility",
                {"facility": facility_data},
                status="skipped",
                reason="age column absent; subgroup sensitivity not evaluated",
            )
        )
    else:
        try:
            groups = ["older" if float(age) >= 65 else "younger" for age in ages]
        except (TypeError, ValueError):
            sections.append(
                _section(
                    "subgroup_facility",
                    {"facility": facility_data},
                    status="skipped",
                    reason="age column present but non-numeric",
                )
            )
        else:
            diff = subgroup_sensitivity_diff(
                oof_labels, oof_scores, groups, threshold, min_positives=10
            )
            subgroup: dict = (
                {"status": "skipped", "reason": "insufficient positives per group"}
                if diff is None
                else {"status": "ok", "sensitivity_diff": diff}
            )
            sections.append(
                _section(
                    "subgroup_facility",
                    {"subgroup": subgroup, "facility": facility_data},
                )
            )

    # 7. 决策曲线
    try:
        curve = decision_curve_table(oof_labels, oof_scores)
        sections.append(
            _section(
                "decision_curve",
                {
                    "mean_gain": sum(row["net_gain"] for row in curve) / len(curve),
                    "table": curve,
                },
            )
        )
    except ValueError as exc:
        sections.append(_section("decision_curve", status="skipped", reason=str(exc)))

    # 8. 阈值表（工件三档 × OOF 口径）
    bands: dict[str, dict] = {}
    for name, cut in sorted(engine.thresholds.items()):
        band_metrics = binary_metrics(oof_labels, oof_scores, cut)
        bands[name] = {
            "threshold": cut,
            "sensitivity": band_metrics["sensitivity"],
            "specificity": band_metrics["specificity"],
            "ppv": band_metrics["ppv"],
        }
    sections.append(_section("threshold_table", {"bands": bands}))

    # 9. 漂移参考摘要
    reference = (artifact or {}).get("drift_reference")
    if isinstance(reference, dict) and reference:
        sections.append(
            _section(
                "drift_reference",
                {
                    name: {
                        "lo": entry.get("lo"),
                        "hi": entry.get("hi"),
                        "missing_rate": entry.get("missing_rate"),
                        "reference_sample_size": len(entry.get("sample") or []),
                    }
                    for name, entry in reference.items()
                },
            )
        )
    else:
        sections.append(
            _section(
                "drift_reference",
                status="skipped",
                reason="artifact has no drift_reference block",
            )
        )

    # 10. 运行元数据
    sections.append(
        _section(
            "runtime",
            {
                "seed": seed,
                "git_commit": _git_commit(),
                "generated_at": datetime.now(UTC).isoformat(),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "python": sys.version.split()[0],
            },
        )
    )

    return {
        "report_type": "model_report_card",
        "target_id": target_id,
        "model_version": engine.version,
        "split_mode": split_mode,
        "sections": sections,
    }


def _render_data(data: dict, indent: int = 0) -> list[str]:
    pad = "  " * indent
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{pad}- **{key}**:")
            lines.extend(_render_data(value, indent + 1))
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            lines.append(f"{pad}- **{key}**:")
            headers = list(value[0])
            lines.append(f"{pad}  | " + " | ".join(headers) + " |")
            lines.append(f"{pad}  |" + "|".join(["---"] * len(headers)) + "|")
            for item in value[:50]:
                lines.append(
                    f"{pad}  | "
                    + " | ".join(str(item.get(header, "")) for header in headers)
                    + " |"
                )
            if len(value) > 50:
                lines.append(f"{pad}  - … ({len(value)} rows total)")
        elif isinstance(value, list):
            preview = ", ".join(str(v) for v in value[:20])
            suffix = ", …" if len(value) > 20 else ""
            lines.append(f"{pad}- **{key}**: {preview}{suffix}")
        elif value is None:
            lines.append(f"{pad}- **{key}**: n/a")
        else:
            lines.append(f"{pad}- **{key}**: {value}")
    return lines


def render_markdown(report: dict) -> str:
    lines = [
        f"# Model Report Card — {report['target_id']}",
        "",
        f"- model_version: `{report['model_version']}`",
        f"- split_mode: {report['split_mode']}",
        f"- sections: {len(report['sections'])}",
        "",
    ]
    for section in report["sections"]:
        lines.append(f"## {section['name']}")
        if section["status"] == "skipped":
            lines.append(f"_skipped_: {section['reason']}")
        else:
            lines.extend(_render_data(section["data"]))
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model report card (md + json)")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--target-id", choices=["deterioration_24h", "readmission_30d"], required=True
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output", required=True, help="输出前缀（产 <prefix>.json 与 .md）")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--split", choices=["patient", "encounter"], default="patient")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cohort-stats", type=Path, default=None)
    args = parser.parse_args()

    model_path = args.model or Path(
        "data/model.json"
        if args.target_id == "deterioration_24h"
        else "data/readmission_model.json"
    )
    engine = RiskEngine(
        Settings(
            model_path=str(model_path),
            readmission_model_path=str(model_path),
            app_env="dev",
        ),
        target_id=args.target_id,
    )
    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    artifact = json.loads(model_path.read_text(encoding="utf-8"))
    cohort_stats = (
        json.loads(args.cohort_stats.read_text(encoding="utf-8")) if args.cohort_stats else None
    )
    report = build_report_card(
        rows,
        engine,
        target_id=args.target_id,
        label_column=args.label_column,
        split_mode=args.split,
        n_splits=args.n_splits,
        artifact=artifact,
        cohort_stats=cohort_stats,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(str(output) + ".json")
    md_path = Path(str(output) + ".md")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
