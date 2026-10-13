"""Promote, verify and roll back model artifacts.

The registry is intentionally filesystem based so the workflow is runnable in
the demo environment. A hospital deployment can replace the copy operation
with an approved artifact store while preserving the manifest shape.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

# 直接以 `python scripts/x.py` 运行时根目录不在 sys.path（Ruling R14）。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.risk import RiskEngine
from app.synthetic import FEATURES
from scripts.baselines import news2_score
from scripts.evaluate_model import binary_metrics, evaluate_rows, resolve_group_column
from scripts.validate import (
    bootstrap_budget,
    bootstrap_metric,
    decision_curve_mean_gain,
    per_facility_auprc,
    subgroup_sensitivity_diff,
)

TARGETS = {"deterioration_24h", "readmission_30d"}
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_artifact(path: Path, target_id: str, version: str | None = None) -> dict[str, Any]:
    from scripts.train_model import FEATURES_V2

    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read model artifact: {path}") from exc
    # 按目标白名单接受已知工件形态（Review C2 / Ruling R-P8-9）：
    # deterioration 10 维旧形态或 21 维 v2；readmission 10 维旧形态或出院快照编码。
    from app.features import READMISSION_FEATURES

    allowed = (
        {tuple(FEATURES), tuple(FEATURES_V2)}
        if target_id == "deterioration_24h"
        else {tuple(FEATURES), tuple(READMISSION_FEATURES)}
    )
    if not isinstance(artifact, dict) or tuple(artifact.get("feature_names") or ()) not in allowed:
        raise ValueError("model feature signature is invalid")
    if artifact.get("target_id") != target_id:
        raise ValueError("model target signature is invalid")
    artifact_version = str(artifact.get("model_version", ""))
    if not VERSION_PATTERN.fullmatch(artifact_version) or (version and artifact_version != version):
        raise ValueError("model version does not match requested version")
    return artifact


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "models": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("models", {}), dict):
        raise TypeError("registry manifest is invalid")
    return value


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_promotion_gate(
    artifact_path: Path, target_id: str, input_csv: Path, split_mode: str = "patient"
) -> dict:
    # 双路径都指向被 promote 的工件（readmission 引擎按 readmission_model_path 选
    # 路径）；strict 模式（app_env != dev）+ 显式关闭验签：工件加载失败直接抛错，
    # 禁止 dev 静默回落用默认权重打分（Review I1）。
    engine = RiskEngine(
        Settings(
            model_path=str(artifact_path),
            readmission_model_path=str(artifact_path),
            app_env="test",
            model_signature_required=False,
        ),
        target_id=target_id,
    )
    try:
        with input_csv.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise SystemExit(f"promotion gate rejected: cannot read holdout CSV: {exc}") from exc
    gate_group: str | None = None
    if split_mode == "patient":
        # 门禁数字一律以患者级口径评估（spec §4.2 接入点 3，R-RS-9）：
        # holdout 缺分组键 → 响亮拒绝，不静默按行级放行。
        try:
            gate_group = resolve_group_column(rows)
        except ValueError as exc:
            raise SystemExit(f"promotion gate rejected: {exc}") from exc
    try:
        labels, scores = evaluate_rows(rows, engine, "label")
    except (ValueError, KeyError) as exc:
        # 输入类错误（缺 label 列/非数字/无效 payload）统一拒绝语，不走 traceback
        # （P6 deferred 7）。
        raise SystemExit(f"promotion gate rejected: {exc}") from exc
    if not labels or not sum(labels):
        raise SystemExit("promotion gate rejected: holdout needs labels and at least one positive")

    def auprc_metric(lab: list[int], sco: list[float]) -> float:
        return binary_metrics(lab, sco, engine.thresholds["high"])["auprc"] or 0.0

    n_boot, max_rows = bootstrap_budget(len(labels))
    point, lo, hi = bootstrap_metric(
        labels, scores, auprc_metric, n=n_boot, seed=20260928, max_rows=max_rows
    )
    if target_id == "deterioration_24h":
        baseline_name = "news2"
        try:
            baseline_scores = [
                news2_score({f: float(row[f]) for f in FEATURES if row.get(f, "") not in ("", None)})[0]
                / 15
                for row in rows
            ]
        except (TypeError, ValueError, KeyError) as exc:
            raise SystemExit(
                f"promotion gate rejected: news2 baseline needs numeric vitals: {exc}"
            ) from exc
    else:
        # readmission 基线 = LACE（WBS T8.3，Ruling R-P8-6）。
        baseline_name = "lace"
        from scripts.baselines import lace_score
        from scripts.evaluate_model import payload_from_row

        try:
            baseline_scores = [float(lace_score(payload_from_row(row))) for row in rows]
        except ValueError as exc:
            raise SystemExit(f"promotion gate rejected: {exc}") from exc
    baseline_auprc = binary_metrics(labels, baseline_scores, 7 / 15)["auprc"] or 0.0
    ece = binary_metrics(labels, scores, engine.thresholds["high"])["ece"]
    # 三条新规则按数据可用性降级（Ruling R-P7-6）：前置不足 → skipped 不判失败。
    rules: dict[str, dict] = {}
    try:
        gain = decision_curve_mean_gain(labels, scores)
        rules["decision_curve"] = {"status": "pass" if gain >= 0 else "fail", "mean_gain": gain}
    except ValueError as exc:
        rules["decision_curve"] = {"status": "skipped", "reason": str(exc)}
    ages = [row.get("age", "") for row in rows]
    numeric_ages: list[float] = []
    age_complete = True
    for age in ages:
        if age in ("", None):
            age_complete = False
            break
        try:
            numeric_ages.append(float(age))
        except (TypeError, ValueError):
            age_complete = False
            break
    if age_complete:
        groups = ["older" if age >= 65 else "younger" for age in numeric_ages]
        diff = subgroup_sensitivity_diff(
            labels, scores, groups, engine.thresholds["high"], min_positives=10
        )
        rules["subgroup"] = (
            {"status": "skipped", "reason": "insufficient positives per group"}
            if diff is None
            else {"status": "pass" if diff <= 0.10 else "fail", "sensitivity_diff": diff}
        )
    elif any(age not in ("", None) for age in ages):
        # 列在但值缺失/非数值——与「整列缺失」措辞区分（P7 deferred 1）。
        rules["subgroup"] = {
            "status": "skipped",
            "reason": "age column present but values missing/non-numeric",
        }
    else:
        rules["subgroup"] = {"status": "skipped", "reason": "age column absent"}
    facility = per_facility_auprc(rows, labels, scores, baseline_auprc)
    facility["status"] = (
        "skipped" if not facility["checked"] else ("fail" if facility["failures"] else "pass")
    )
    rules["facility"] = facility
    failing_rules = [name for name, rule in rules.items() if rule.get("status") == "fail"]
    report = {
        "passed": bool(
            lo >= baseline_auprc and ece is not None and ece < 0.05 and not failing_rules
        ),
        "split_mode": split_mode,
        "group_column": gate_group,
        "auprc_ci": {
            "point": point,
            "lo": lo,
            "hi": hi,
            "n": n_boot,
            "seed": 20260928,
            "replicate_subsample": max_rows,
        },
        "baseline": baseline_name,
        "baseline_auprc": baseline_auprc,
        "ece": ece,
        "rules": rules,
        "gate": {
            "rule": (
                f"auprc_ci.lo >= baseline_auprc({baseline_name}) and ece < 0.05 and no rule fails"
            ),
            "max_ece": 0.05,
        },
    }
    if not report["passed"]:
        rules_suffix = f"; failing rules: {', '.join(failing_rules)}" if failing_rules else ""
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(
            f"promotion gate rejected: auprc CI lo {lo:.4f} vs {baseline_name} "
            f"{baseline_auprc:.4f}; ece {None if ece is None else round(ece, 4)} "
            f"(need < 0.05){rules_suffix}"
        )
    return report


def promote(args: argparse.Namespace) -> None:
    artifact_path = Path(args.artifact)
    artifact = load_artifact(artifact_path, args.target_id, args.version)
    gate_report = run_promotion_gate(
        artifact_path, args.target_id, Path(args.input), split_mode=args.split
    )
    version = str(artifact["model_version"])
    registry = Path(args.registry_dir)
    destination = registry / args.target_id / f"{version}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(artifact_path, destination)
    file_hash = digest(destination)
    manifest_path = registry / "manifest.json"
    manifest = read_manifest(manifest_path)
    manifest.setdefault("models", {})[args.target_id] = {
        "active_version": version,
        "artifact": destination.relative_to(registry).as_posix(),
        "sha256": file_hash,
    }
    atomic_write(manifest_path, manifest)
    eval_dir = registry / args.target_id / version
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "eval.json").write_text(
        json.dumps(gate_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.serving_path:
        serving_path = Path(args.serving_path)
        serving_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(destination, serving_path)
    print(
        json.dumps(
            {"target_id": args.target_id, "version": version, "sha256": file_hash},
            ensure_ascii=False,
        )
    )


def rollback(args: argparse.Namespace) -> None:
    if not VERSION_PATTERN.fullmatch(args.version):
        raise ValueError("model version is invalid")
    registry = Path(args.registry_dir)
    source = registry / args.target_id / f"{args.version}.json"
    artifact = load_artifact(source, args.target_id, args.version)
    file_hash = digest(source)
    manifest_path = registry / "manifest.json"
    manifest = read_manifest(manifest_path)
    manifest.setdefault("models", {})[args.target_id] = {
        "active_version": args.version,
        "artifact": source.relative_to(registry).as_posix(),
        "sha256": file_hash,
    }
    atomic_write(manifest_path, manifest)
    if args.serving_path:
        serving_path = Path(args.serving_path)
        serving_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, serving_path)
    print(
        json.dumps(
            {
                "target_id": args.target_id,
                "version": artifact["model_version"],
                "sha256": file_hash,
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-dir", default="model-registry")
    subparsers = parser.add_subparsers(dest="command", required=True)
    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--artifact", required=True)
    promote_parser.add_argument("--target-id", choices=sorted(TARGETS), required=True)
    promote_parser.add_argument(
        "--input", required=True, help="labeled holdout CSV for the promotion gate"
    )
    promote_parser.add_argument("--version")
    promote_parser.add_argument("--serving-path")
    promote_parser.add_argument(
        "--split",
        choices=["patient", "encounter"],
        default="patient",
        help="门禁分组键校验口径；encounter 保留旧行为供对照（spec §4.2）",
    )
    promote_parser.set_defaults(handler=promote)
    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("--target-id", choices=sorted(TARGETS), required=True)
    rollback_parser.add_argument("--version", required=True)
    rollback_parser.add_argument("--serving-path")
    rollback_parser.set_defaults(handler=rollback)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
