"""Monthly recalibration: refit the calibration block only, gated (WBS T7.4).

只更新工件的 `calibration` 块；门禁（T6.3/T7.3b 口径）在临时候选工件上先跑，
不过则原工件分毫不动（Ruling R-P7-10）。同输入校准+门禁的 in-sample 乐观性
已披露——真实数据请用独立的 `--gate-input` 留出集。
"""

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

# 直接以 `python scripts/x.py` 运行时根目录不在 sys.path（Ruling R14）。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.domain import ClinicalEvent
from app.risk import RiskEngine
from scripts.evaluate_model import payload_from_row
from scripts.manage_model import run_promotion_gate
from scripts.train_model import _fit_isotonic_block


def _margins(engine: RiskEngine, rows: list[dict]) -> list[float]:
    """当前工件的校准前 logit（线性或树路径）。"""
    out: list[float] = []
    for index, row in enumerate(rows):
        payload = payload_from_row(row)
        event = ClinicalEvent(
            event_type="vital_signs",
            patient_id=str(row.get("patient_id", f"rc-p-{index}")),
            encounter_id=str(row.get("encounter_id", f"rc-e-{index}")),
            payload=payload,
        )
        if engine.algorithm == "lightgbm" and engine.trees:
            out.append(engine.tree_raw_margin(event))
        else:
            out.append(engine.linear_margin(event))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Refit calibration block, gated")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--input", required=True, help="labeled CSV for refitting calibration")
    parser.add_argument(
        "--target-id",
        required=True,
        choices=["deterioration_24h", "readmission_30d"],
    )
    parser.add_argument(
        "--gate-input",
        default=None,
        help="independent holdout for the gate (default: same as --input)",
    )
    args = parser.parse_args()
    artifact_path = Path(args.artifact)
    original = artifact_path.read_text(encoding="utf-8")
    artifact = json.loads(original)
    with Path(args.input).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    labels = []
    for index, row in enumerate(rows):
        try:
            labels.append(int(row["label"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"recalibrate: row {index + 1} needs label 0/1") from exc
    if not labels or not sum(labels) or sum(labels) == len(labels):
        raise SystemExit("recalibrate: need both classes")
    # 双路径指向给定工件 + 目标显式校验（同 P6 修复轮口径）。
    settings = Settings(
        model_path=str(artifact_path),
        readmission_model_path=str(artifact_path),
        app_env="dev",
    )
    engine = RiskEngine(settings, target_id=args.target_id)
    if engine.degraded or engine.degraded_reason:
        raise SystemExit(f"recalibrate: model failed to load: {engine.degraded_reason}")
    if artifact.get("target_id") != args.target_id:
        raise SystemExit(
            f"recalibrate: target mismatch — artifact target_id is "
            f"{artifact.get('target_id')!r}, requested {args.target_id!r}"
        )
    margins = _margins(engine, rows)
    calibration_block = _fit_isotonic_block(margins, labels, fitted_on="recalibration_set")
    candidate = dict(artifact)
    candidate["calibration"] = calibration_block
    # 门禁在临时工件上跑（Ruling R-P7-10）；不过 → 原件分毫不动。
    with tempfile.TemporaryDirectory() as tmp:
        candidate_path = Path(tmp) / "candidate.json"
        candidate_path.write_text(json.dumps(candidate, ensure_ascii=False), encoding="utf-8")
        gate_report = run_promotion_gate(
            candidate_path, args.target_id, Path(args.gate_input or args.input)
        )
    artifact_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "updated": "calibration",
                "gate": {"passed": gate_report["passed"], "baseline": gate_report["baseline"]},
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
