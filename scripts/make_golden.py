"""Freeze offline/serving score pairs into golden fixtures (WBS T6.5)."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from random import Random

# 直接以 `python scripts/x.py` 运行时根目录不在 sys.path（Ruling R14）。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.domain import ClinicalEvent
from app.risk import RiskEngine
from app.synthetic import FEATURES
from scripts.evaluate_model import evaluate_rows
from scripts.train_model import FEATURES_V2, build_demo_dataset

SAMPLE_ROWS = 200
SAMPLE_SEED = 99
TOLERANCE = 1e-9
DEFAULT_ARTIFACTS = {
    "deterioration_24h": Path("data/model.json"),
    "readmission_30d": Path("data/readmission_model.json"),
}


def _sample_rows(target_id: str) -> list[dict]:
    times, matrix, labels = build_demo_dataset(target_id, 9600)
    rng = Random(SAMPLE_SEED)
    picked = sorted(rng.sample(range(len(times)), min(SAMPLE_ROWS, len(times))))
    rows: list[dict] = []
    for i in picked:
        row = {name: matrix[i][j] for j, name in enumerate(FEATURES_V2)}
        row["label"] = str(labels[i])
        rows.append(row)
    return rows


def _payload_from_row(row: dict) -> dict[str, float]:
    payload = {
        feature: float(row[feature])
        for feature in FEATURES
        if row.get(feature) is not None and not math.isnan(float(row[feature]))
    }
    news2 = row.get("news2_score")
    if news2 is not None and not math.isnan(float(news2)):
        payload["news2_score"] = float(news2)
    return payload


def _serving_scores(engine: RiskEngine, rows: list[dict]) -> list[float]:
    return [
        engine.score(
            ClinicalEvent(
                event_type="vital_signs",
                patient_id=f"gold-p-{i}",
                encounter_id=f"gold-e-{i}",
                payload=_payload_from_row(row),
            )
        ).score
        for i, row in enumerate(rows)
    ]


def build_fixture(target_id: str, artifact_path: Path) -> list[dict]:
    engine = RiskEngine(Settings(model_path=str(artifact_path), app_env="dev"), target_id=target_id)
    rows = _sample_rows(target_id)
    _, offline = evaluate_rows(rows, engine, "label")
    serving = _serving_scores(engine, rows)
    for row, off, serv in zip(rows, offline, serving):
        row["offline_score"] = repr(off)
        row["serving_score"] = repr(serv)
    return rows


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):  # NaN → 空
        return ""
    return str(value)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [*FEATURES_V2, "label", "offline_score", "serving_score"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in fields})


def _rows_equal(frozen: list[dict], fresh: list[dict]) -> str | None:
    if len(frozen) != len(fresh):
        return f"row count {len(frozen)} != {len(fresh)}"
    for i, (a, b) in enumerate(zip(frozen, fresh)):
        for key in (*FEATURES_V2, "label"):
            va, vb = a.get(key, ""), b.get(key, "")
            if key == "label":
                if str(va) != str(vb):
                    return f"row {i} {key}: {va} != {vb}"
                continue
            fa = float(va) if va not in ("", None) else None
            fb = float(vb) if vb not in ("", None) else None
            if (fa is None) != (fb is None):
                return f"row {i} {key}: {va!r} != {vb!r}"
            if fa is not None and abs(fa - fb) > TOLERANCE:
                return f"row {i} {key}: {va} != {vb}"
        for key in ("offline_score", "serving_score"):
            if abs(float(a[key]) - float(b[key])) > TOLERANCE:
                return f"row {i} {key}: {a[key]} != {b[key]}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze golden fixtures (or --check them)")
    parser.add_argument("--check", action="store_true", help="compare instead of write")
    args = parser.parse_args()
    drift: list[str] = []
    for target_id, artifact in DEFAULT_ARTIFACTS.items():
        if not artifact.exists():
            raise SystemExit(f"missing artifact {artifact}; train first")
        rows = build_fixture(target_id, artifact)
        fixture_dir = Path("tests/fixtures") / target_id
        csv_path = fixture_dir / "golden_dataset.csv"
        if args.check:
            if not csv_path.exists():
                drift.append(f"{target_id}: fixture missing")
                continue
            with csv_path.open(newline="", encoding="utf-8") as handle:
                frozen = list(csv.DictReader(handle))
            reason = _rows_equal(frozen, rows)
            if reason:
                drift.append(f"{target_id}: {reason}")
        else:
            fixture_dir.mkdir(parents=True, exist_ok=True)
            _write_csv(csv_path, rows)
            (fixture_dir / "model.json").write_text(
                json.dumps(
                    json.loads(artifact.read_text(encoding="utf-8")),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"froze {len(rows)} rows -> {csv_path}")
    if drift:
        print("golden drift detected (run make_golden.py to regenerate):")
        for line in drift:
            print(f"  - {line}")
        raise SystemExit(1)
    if args.check:
        print("golden fixtures match")


if __name__ == "__main__":
    main()
