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
from scripts.evaluate_model import evaluate_rows
from scripts.train_model import FEATURES_V2, build_demo_dataset

SAMPLE_ROWS = 200
SAMPLE_SEED = 99
TOLERANCE = 1e-9
# 路径单一来源 = app.config 字段默认（P6 deferred 8：改 config 默认值时
# make_golden 不再静默指向旧工件）。用类字段默认而非实例，避免环境变量污染。
DEFAULT_ARTIFACTS = {
    "deterioration_24h": Path(Settings.model_fields["model_path"].default),
    "readmission_30d": Path(Settings.model_fields["readmission_model_path"].default),
}


def golden_columns(target_id: str) -> list[str]:
    if target_id == "deterioration_24h":
        return list(FEATURES_V2)
    from app.features import READMISSION_PAYLOAD_KEYS

    return list(READMISSION_PAYLOAD_KEYS)


def _sample_rows(target_id: str) -> list[dict]:
    times, matrix, labels = build_demo_dataset(target_id, 9600)
    rng = Random(SAMPLE_SEED)
    picked = sorted(rng.sample(range(len(times)), min(SAMPLE_ROWS, len(times))))
    if target_id == "deterioration_24h":
        source = [{name: matrix[i][j] for j, name in enumerate(FEATURES_V2)} for i in picked]
    else:
        from app.features import READMISSION_PAYLOAD_KEYS

        source = [
            {key: times[i][key] for key in READMISSION_PAYLOAD_KEYS if key in times[i]}
            for i in picked
        ]
    rows: list[dict] = []
    for row, i in zip(source, picked):
        row["label"] = str(labels[i])
        rows.append(row)
    return rows


def _payload_from_row(row: dict) -> dict:
    # 双目标共享的行→payload 转换（Ruling R-P8-7）。
    from scripts.evaluate_model import payload_from_row

    return payload_from_row(row)


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
    if isinstance(value, (list, dict)):  # JSON 序列化，保证 CSV 回读可解析
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    fields = [*columns, "label", "offline_score", "serving_score"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in fields})


def _cell_equal(key: str, va, vb) -> bool:
    """类型感知比较：两侧都可数值化 → 容差比较；否则字符串相等；空值互通。"""
    if key == "label":
        return str(va) == str(vb)
    empty_a = va in ("", None)
    empty_b = vb in ("", None)
    if empty_a or empty_b:
        return empty_a == empty_b
    # 列表/字典统一 JSON 序列化后再比（CSV 侧是 JSON 字符串，内存侧是原生对象）。
    if isinstance(va, (list, dict)):
        va = json.dumps(va, ensure_ascii=False)
    if isinstance(vb, (list, dict)):
        vb = json.dumps(vb, ensure_ascii=False)
    try:
        fa, fb = float(va), float(vb)
    except (TypeError, ValueError):
        return str(va) == str(vb)
    if math.isnan(fa) or math.isnan(fb):
        return math.isnan(fa) and math.isnan(fb)
    return abs(fa - fb) <= TOLERANCE


def _rows_equal(frozen: list[dict], fresh: list[dict], columns: list[str]) -> str | None:
    if len(frozen) != len(fresh):
        return f"row count {len(frozen)} != {len(fresh)}"
    for i, (a, b) in enumerate(zip(frozen, fresh)):
        for key in (*columns, "label"):
            if not _cell_equal(key, a.get(key, ""), b.get(key, "")):
                return f"row {i} {key}: {a.get(key, '')!r} != {b.get(key, '')!r}"
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
        columns = golden_columns(target_id)
        fixture_dir = Path("tests/fixtures") / target_id
        csv_path = fixture_dir / "golden_dataset.csv"
        if args.check:
            if not csv_path.exists():
                drift.append(f"{target_id}: fixture missing")
                continue
            with csv_path.open(newline="", encoding="utf-8") as handle:
                frozen = list(csv.DictReader(handle))
            reason = _rows_equal(frozen, rows, columns)
            if reason:
                drift.append(f"{target_id}: {reason}")
        else:
            fixture_dir.mkdir(parents=True, exist_ok=True)
            _write_csv(csv_path, rows, columns)
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
