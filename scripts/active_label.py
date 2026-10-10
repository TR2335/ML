"""Uncertainty sampling for human labeling (WBS T6.4).

只生成待标注清单——标签必须由人工填写，脚本绝不回写标签列。
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# 直接以 `python scripts/x.py` 运行时根目录不在 sys.path（Ruling R14）。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.domain import ClinicalEvent
from app.risk import RiskEngine
from app.synthetic import FEATURES


def _payload(row: dict) -> dict[str, float]:
    payload: dict[str, float] = {}
    for feature in FEATURES:
        raw = row.get(feature, "")
        if raw not in ("", None):
            payload[feature] = float(raw)
    if row.get("news2_score", "") not in ("", None):
        payload["news2_score"] = float(row["news2_score"])
    return payload


def select_for_labeling(rows: list[dict], engine: RiskEngine, top_n: int) -> list[dict]:
    """按 |p-0.5| 升序取 top-N；平局按 (encounter_id, index_time) 稳定排序。"""
    scored: list[dict] = []
    for index, row in enumerate(rows):
        event = ClinicalEvent(
            event_type="vital_signs",
            patient_id=str(row.get("patient_id", f"al-patient-{index}")),
            encounter_id=str(row.get("encounter_id", f"al-encounter-{index}")),
            payload=_payload(row),
        )
        probability = engine.score(event).score
        enriched = dict(row)
        enriched["score"] = probability
        enriched["uncertainty"] = abs(probability - 0.5)
        scored.append(enriched)
    scored.sort(
        key=lambda item: (
            item["uncertainty"],
            str(item.get("encounter_id", "")),
            str(item.get("index_time", "")),
        )
    )
    return scored[: max(top_n, 0)]


def _engine_for(model_path: str, target_id: str) -> RiskEngine:
    """双路径都指向给定工件：readmission 引擎按 readmission_model_path 选路径，
    否则 --model 被默认路径静默覆盖（Review I1）。加载前显式校验目标。"""
    try:
        artifact = json.loads(Path(model_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"model failed to load for {target_id}: {exc}") from exc
    if artifact.get("target_id") != target_id:
        raise SystemExit(
            f"model failed to load for {target_id}: artifact target_id is "
            f"{artifact.get('target_id')!r}"
        )
    settings = Settings(model_path=model_path, readmission_model_path=model_path, app_env="dev")
    engine = RiskEngine(settings, target_id=target_id)
    if engine.degraded or engine.degraded_reason:
        raise SystemExit(
            f"model failed to load for {target_id}: {engine.degraded_reason or 'degraded'}"
        )
    return engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank unlabeled rows for human review")
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-id", default="deterioration_24h")
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    engine = _engine_for(args.model, args.target_id)
    with Path(args.input).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    picked = select_for_labeling(rows, engine, args.top)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out_fields = [*fields, "score", "uncertainty"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(picked)
    print(f"wrote {output} ({len(picked)} of {len(rows)} rows; labels remain human-owned)")


if __name__ == "__main__":
    main()
