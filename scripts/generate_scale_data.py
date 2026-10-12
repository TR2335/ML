"""生成 2–3 万规模合成出院/入院队列数据集（真实规模优化 G1）。

逐遭遇流式构建：生命体征 + 结局 → derive_labels（按遭遇分批）→ build_landmarks →
立即写盘丢弃——内存峰值保持在单遭遇 + 写缓冲级别。
"""

import argparse
import csv
import json
import shutil
import sys
from datetime import timedelta
from pathlib import Path

# 直接以 `python scripts/x.py` 运行时根目录不在 sys.path（Ruling R14）。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cohort import (
    BASE_VITALS,
    DEFAULT_HORIZON_DAYS,
    DEFAULT_SEED,
    LANDMARK_HOURS,
    SCALE_DEFAULT,
    SCALE_MAX,
    SCALE_MIN,
    deterioration_outcome,
    generate_encounters,
    readmission_snapshot,
    vital_events,
)
from app.features import READMISSION_PAYLOAD_KEYS, TUPLE_FEATURE_ORDER, build_landmarks
from app.synthetic import FEATURES
from scripts.baselines import news2_score
from scripts.derive_labels import _latest_payload_before, derive_labels

ENCOUNTER_FIELDS = [
    "patient_id",
    "encounter_id",
    "admitted_at",
    "discharged_at",
    "facility_id",
    "department",
    "age",
    "sex",
    "los_days",
]
LABEL_DETERIORATION_FIELDS = ["encounter_id", "patient_id", "index_time", "label", *FEATURES]
# demo_dataset 契约列序 + patient_id 第 2 列 + 性别/院区/科室尾列（R-RS-3）
DATASET_DETERIORATION_FIELDS = [
    "encounter_id",
    "patient_id",
    "index_time",
    "label",
    *FEATURES,
    *TUPLE_FEATURE_ORDER,
    "news2_score",
    "sex",
    "facility_id",
    "department",
]
LABEL_READMISSION_FIELDS = ["encounter_id", "patient_id", "index_time", "label"]
DATASET_READMISSION_FIELDS = [
    "encounter_id",
    "patient_id",
    "index_time",
    "label",
    *READMISSION_PAYLOAD_KEYS,
    "age",
    "sex",
    "facility_id",
    "department",
]
NEWS2_VITALS = ("respiratory_rate", "spo2", "systolic_bp", "heart_rate", "temperature")
DYNAMIC_VITALS = tuple(BASE_VITALS)


def _news2_before(events, index_time) -> int | str:
    """窗口内最近一条生命体征的 NEWS2（与 build_features 语义同源：最新值口径）。"""
    prior = [event for event in events if event.occurred_at < index_time]
    if not prior:
        return ""
    latest = max(prior, key=lambda event: event.occurred_at)
    vitals = {key: latest.payload[key] for key in NEWS2_VITALS if key in latest.payload}
    return news2_score(vitals)[0] if vitals else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 20k-30k scale synthetic cohort CSVs")
    parser.add_argument("--scale", type=int, default=SCALE_DEFAULT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument("--out-dir", default="data/scale")
    args = parser.parse_args()
    if not SCALE_MIN <= args.scale <= SCALE_MAX:
        raise SystemExit(
            f"generate_scale_data: --scale must be in [{SCALE_MIN}, {SCALE_MAX}], got {args.scale}"
        )
    if args.horizon_days < 90:
        raise SystemExit("generate_scale_data: --horizon-days must be >= 90")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free
    if free < 500 * 1024**2:
        raise SystemExit(
            f"generate_scale_data: insufficient free disk at {out} "
            f"({free} bytes free, need >= 500MB)"
        )

    encounters = generate_encounters(args.scale, args.seed, args.horizon_days)
    landmark_count = len(LANDMARK_HOURS)
    det_positives = 0
    det_events = 0
    rm_positives = 0
    missing_slots = 0
    vital_slots = 0
    facility_counts: dict[str, int] = {}
    department_counts: dict[str, int] = {}

    with (
        (out / "encounters.csv").open("w", newline="", encoding="utf-8") as enc_handle,
        (out / "labels_deterioration.csv").open("w", newline="", encoding="utf-8") as ldet_handle,
        (out / "dataset_deterioration.csv").open("w", newline="", encoding="utf-8") as ddet_handle,
        (out / "labels_readmission.csv").open("w", newline="", encoding="utf-8") as lrm_handle,
        (out / "dataset_readmission.csv").open("w", newline="", encoding="utf-8") as drm_handle,
    ):
        enc_writer = csv.writer(enc_handle)
        enc_writer.writerow(ENCOUNTER_FIELDS)
        ldet_writer = csv.writer(ldet_handle)
        ldet_writer.writerow(LABEL_DETERIORATION_FIELDS)
        ddet_writer = csv.writer(ddet_handle)
        ddet_writer.writerow(DATASET_DETERIORATION_FIELDS)
        lrm_writer = csv.writer(lrm_handle)
        lrm_writer.writerow(LABEL_READMISSION_FIELDS)
        drm_writer = csv.writer(drm_handle)
        drm_writer.writerow(DATASET_READMISSION_FIELDS)

        for enc in encounters:
            enc_writer.writerow(
                [
                    enc.patient_id,
                    enc.encounter_id,
                    enc.admitted_at.isoformat(),
                    enc.discharged_at.isoformat(),
                    enc.facility_id,
                    enc.department,
                    enc.age,
                    enc.sex,
                    enc.los_days,
                ]
            )
            facility_counts[enc.facility_id] = facility_counts.get(enc.facility_id, 0) + 1
            department_counts[enc.department] = department_counts.get(enc.department, 0) + 1

            events = vital_events(enc, args.seed)
            vital_slots += len(events) * len(DYNAMIC_VITALS)
            missing_slots += len(events) * len(DYNAMIC_VITALS) - sum(
                1 for event in events for name in DYNAMIC_VITALS if name in event.payload
            )

            # —— deterioration：6h 网格 16 格 landmark + 24h horizon ——
            outcome = deterioration_outcome(enc, args.seed)
            if outcome is not None:
                det_events += 1
            index_times = [
                (enc.encounter_id, enc.admitted_at + timedelta(hours=hour))
                for hour in LANDMARK_HOURS
            ]
            labeled = derive_labels(
                events, [outcome] if outcome else [], index_times, horizon_hours=24
            )
            landmarks = {
                row["index_time"]: row
                for row in build_landmarks({enc.encounter_id: events}, index_times, 24)
            }
            for item in labeled:
                latest = _latest_payload_before(events, item.encounter_id, item.index_time)
                label_row = [
                    enc.encounter_id,
                    enc.patient_id,
                    item.index_time.isoformat(),
                    item.label,
                    *[latest.get(name, "") for name in FEATURES],
                ]
                ldet_writer.writerow(label_row)
                landmark = landmarks[item.index_time.isoformat()]
                tuple_values = [
                    "" if landmark.get(col) is None else landmark[col]
                    for col in TUPLE_FEATURE_ORDER
                ]
                ddet_writer.writerow(
                    [
                        *label_row,
                        *tuple_values,
                        _news2_before(events, item.index_time),
                        enc.sex,
                        enc.facility_id,
                        enc.department,
                    ]
                )
                det_positives += item.label

            # —— readmission：出院时刻单行 + 720h horizon ——
            snapshot, rm_outcome = readmission_snapshot(enc, args.seed)
            rm_labeled = derive_labels(
                events,
                [rm_outcome] if rm_outcome else [],
                [(enc.encounter_id, enc.discharged_at)],
                horizon_hours=720,
            )
            item = rm_labeled[0]
            lrm_writer.writerow(
                [enc.encounter_id, enc.patient_id, item.index_time.isoformat(), item.label]
            )
            drm_writer.writerow(
                [
                    enc.encounter_id,
                    enc.patient_id,
                    item.index_time.isoformat(),
                    item.label,
                    snapshot["los_days"],
                    snapshot["prior_utilization_12m"],
                    snapshot["comorbidity_index"],
                    snapshot["ed_visits_6m"],
                    json.dumps(snapshot["diagnosis_codes"]),
                    snapshot["discharge_destination"],
                    enc.age,
                    enc.sex,
                    enc.facility_id,
                    enc.department,
                ]
            )
            rm_positives += item.label

    det_rows = args.scale * landmark_count
    stats = {
        "scale": args.scale,
        "seed": args.seed,
        "horizon_days": args.horizon_days,
        "patients": len({enc.patient_id for enc in encounters}),
        "deterioration": {
            "rows": det_rows,
            "positives": det_positives,
            "positive_rate": round(det_positives / det_rows, 4),
            "event_encounters": det_events,
            "event_rate": round(det_events / args.scale, 4),
        },
        "readmission": {
            "rows": args.scale,
            "positives": rm_positives,
            "positive_rate": round(rm_positives / args.scale, 4),
        },
        "missing_rate_overall": round(missing_slots / max(vital_slots, 1), 4),
        "facilities": facility_counts,
        "departments": department_counts,
    }
    (out / "cohort_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "out_dir": str(out),
                "scale": args.scale,
                "patients": stats["patients"],
                "deterioration_rows": det_rows,
                "deterioration_event_rate": stats["deterioration"]["event_rate"],
                "readmission_rate": stats["readmission"]["positive_rate"],
                "missing_rate": stats["missing_rate_overall"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
