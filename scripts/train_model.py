"""Train the demo logistic model with scikit-learn (P6).

Data comes from the same synthetic pipeline the demo/import paths use
(P5 T5.1); serving stays dependency-free because the artifact carries
its own preprocess block (T5.3 v2).
"""

import argparse
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from random import Random

# 直接以 `python scripts/x.py` 运行时根目录不在 sys.path（Ruling R14）。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.domain import ClinicalEvent
from app.synthetic import FEATURES, demo_patients_with_outcomes
from scripts.baselines import news2_score
from scripts.derive_labels import _latest_payload_before, demo_index_times, derive_labels
from scripts.validate import rolling_time_split

FEATURES_V2 = [
    *FEATURES,
    *[f"{feature}__missing" for feature in FEATURES],
    "news2_score",
]

TARGET_THRESHOLDS = {
    "deterioration_24h": {"medium": 0.2, "high": 0.45, "critical": 0.75},
    "readmission_30d": {"medium": 0.15, "high": 0.3, "critical": 0.55},
}
READMISSION_DISCHARGE_HOURS = 72
READMISSION_POSITIVE_RATE = 0.08


def _features_before(payload: dict) -> list[float]:
    base = [float(payload[f]) if payload.get(f) is not None else math.nan for f in FEATURES]
    indicators = [1.0 if math.isnan(value) else 0.0 for value in base]
    present = {k: v for k, v in payload.items() if v is not None}
    news2 = float(news2_score(present)[0]) if present else math.nan
    return [*base, *indicators, news2]


def build_demo_dataset(
    target_id: str, target_rows: int
) -> tuple[list[dict], list[list[float]], list[int]]:
    """T5.1 同源管道生成 (index_time 行, 21 维特征, 标签)（Plan Ruling R3/R9）。"""
    # readmission 每就诊一个出院 landmark，行数即就诊数（入账裁定）。
    patients = max(
        18, target_rows if target_id == "readmission_30d" else math.ceil(target_rows / 96)
    )
    _, events, outcomes = demo_patients_with_outcomes(count=patients, seed=7)
    if target_id == "deterioration_24h":
        index_times = demo_index_times(events)
        labeled = derive_labels(events, outcomes, index_times, horizon_hours=24)
    else:
        first_seen: dict[str, datetime] = {}
        for event in sorted(events, key=lambda item: item.occurred_at):
            first_seen.setdefault(event.encounter_id, event.occurred_at)
        # derive_labels 要求 list[tuple[str, datetime]]（Plan 草稿的 dict 形式已改，入账）。
        # 所有 demo 就诊首事件同刻——按就诊序号加 0..6 天 LOS 抖动，产生 >=4 个不同
        # 时间戳供 rolling_time_split 使用（入账裁定）。
        index_times = [
            (
                encounter_id,
                at
                + timedelta(hours=READMISSION_DISCHARGE_HOURS)
                + timedelta(days=int(encounter_id.split("-")[1]) % 7),
            )
            for encounter_id, at in first_seen.items()
        ]
        rng = Random(42)
        readmissions = [
            {
                "encounter_id": encounter_id,
                "occurred_at": (at + timedelta(days=2 + rng.randrange(27))).isoformat(),
                "outcome_type": "unplanned_readmission",
            }
            for encounter_id, at in first_seen.items()
            if rng.random() < READMISSION_POSITIVE_RATE
        ]
        labeled = derive_labels(events, readmissions, index_times, horizon_hours=720)
    events_by_encounter: dict[str, list[ClinicalEvent]] = {}
    for event in events:
        events_by_encounter.setdefault(event.encounter_id, []).append(event)
    rows: list[dict] = []
    matrix: list[list[float]] = []
    labels: list[int] = []
    for item in labeled:
        payload = _latest_payload_before(
            events_by_encounter.get(item.encounter_id, []), item.encounter_id, item.index_time
        )
        rows.append({"index_time": item.index_time.isoformat()})
        matrix.append(_features_before(payload))
        labels.append(item.label)
    return rows, matrix, labels


def _fit_pipeline(
    times: list[dict], matrix: list[list[float]], labels: list[int]
) -> tuple[Pipeline, float]:
    """C 网格 × rolling_time_split 时间交叉验证，按平均 AUPRC 选（WBS T6.1）。"""
    from scripts.evaluate_model import binary_metrics

    fold_rows = [
        {"index_time": t["index_time"], "_i": i, "label": str(labels[i])}
        for i, t in enumerate(times)
    ]
    folds = rolling_time_split(fold_rows, 3)

    def make_pipeline(c: float) -> Pipeline:
        return Pipeline(
            [
                # keep_empty_features：全空列不丢，保 21 维对齐（R2 关键）
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                (
                    "clf",
                    # sklearn>=1.8 弃用显式 penalty；默认即 L2。
                    LogisticRegression(C=c, class_weight="balanced", max_iter=1000),
                ),
            ]
        )

    # C 选择的 CV 跑在全量行上（最终模型仍只 fit train 60%，R4 留校准/验证段）。
    c_grid = (0.01, 0.1, 1.0)
    best_c, best_score = c_grid[0], -1.0
    matrix_array = np.asarray(matrix, dtype=float)
    labels_array = np.asarray(labels, dtype=int)
    for c in c_grid:
        scores = []
        for train_rows, test_rows in folds:
            train_idx = [row["_i"] for row in train_rows]
            test_idx = [row["_i"] for row in test_rows]
            train_labels = [labels[i] for i in train_idx]
            if len(set(train_labels)) < 2:
                # 早期时间窗可能全阴性（demo 阳性集中在 25-48h）——单类无法拟合，
                # 跳过该折（入账裁定）。
                continue
            pipe = make_pipeline(c)
            pipe.fit(matrix_array[train_idx], labels_array[train_idx])
            pred = pipe.predict_proba(matrix_array[test_idx])[:, 1]
            test_labels = [labels[i] for i in test_idx]
            value = binary_metrics(test_labels, list(map(float, pred)), 0.5)["auprc"]
            scores.append(value or 0.0)
        if not scores:
            continue
        mean_auprc = sum(scores) / len(scores)
        if mean_auprc > best_score + 1e-12:
            best_c, best_score = c, mean_auprc
    return make_pipeline(best_c), best_c


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/model.json")
    parser.add_argument("--rows", type=int, default=2500, help="目标训练行数（Ruling R9）")
    parser.add_argument("--version", default="baseline-logistic-sklearn-v1")
    parser.add_argument(
        "--target-id",
        choices=["deterioration_24h", "readmission_30d"],
        default="deterioration_24h",
    )
    args = parser.parse_args()
    times, matrix, labels = build_demo_dataset(args.target_id, args.rows)
    order = sorted(range(len(times)), key=lambda i: times[i]["index_time"])
    n = len(order)
    train_idx = order[: int(n * 0.6)]
    cal_window = order[int(n * 0.6) : int(n * 0.8)]
    x = np.asarray(matrix, dtype=float)
    y = np.asarray(labels, dtype=int)
    pipeline, best_c = _fit_pipeline(times, matrix, labels)
    pipeline.fit(x[train_idx], y[train_idx])
    if len(set(y[cal_window])) < 2:
        # demo 标签在时间上集中（RRT 固定 48h），60-80% 时间段可能恒为单类——
        # 回退全量 Platt，保留正确患病率（Plan Ruling：None 回退与其自身测试矛盾）。
        cal_idx = order
    else:
        cal_idx = cal_window
    z_cal = pipeline.decision_function(x[cal_idx]).reshape(-1, 1)
    platt = LogisticRegression()
    platt.fit(z_cal, y[cal_idx])
    calibration_block = {
        "method": "platt",
        "a": float(platt.coef_[0][0]),
        "b": float(platt.intercept_[0]),
    }
    imputer = pipeline.named_steps["impute"]
    scaler = pipeline.named_steps["scale"]
    clf = pipeline.named_steps["clf"]
    artifact = {
        "model_version": args.version,
        "target_id": args.target_id,
        "algorithm": "logistic_regression_sklearn",
        "feature_names": FEATURES_V2,
        "feature_order": FEATURES_V2,
        "preprocess": {
            name: {
                "median": float(imputer.statistics_[i]),
                "mean": float(scaler.mean_[i]),
                "std": float(scaler.scale_[i]) or 1.0,
            }
            for i, name in enumerate(FEATURES_V2)
        },
        "calibration": calibration_block,
        "bias": float(clf.intercept_[0]),
        "weights": {name: float(clf.coef_[0][i]) for i, name in enumerate(FEATURES_V2)},
        "training_rows": len(y),
        "best_C": best_c,
        "label_definition": f"synthetic_{args.target_id}",
        "thresholds": TARGET_THRESHOLDS[args.target_id],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output} ({args.version}, C={best_c}, rows={len(y)})")


if __name__ == "__main__":
    main()
