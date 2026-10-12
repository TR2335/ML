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
from app.features import READMISSION_FEATURES, encode_readmission_payload
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


def feature_names_for(target_id: str) -> list[str]:
    if target_id == "deterioration_24h":
        return list(FEATURES_V2)
    return list(READMISSION_FEATURES)


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
        readmissions = []
        raw_by_encounter: dict[str, dict] = {}
        for encounter_id, at in first_seen.items():
            ordinal = int(encounter_id.split("-")[1])
            field_rng = Random(7000 + ordinal)
            los_days = round(0.5 + min(field_rng.expovariate(0.45), 40.0), 1)
            comorbidity = round(min(14.0, field_rng.expovariate(0.5)), 1)
            prior_util = min(6, int(field_rng.expovariate(0.8)))
            ed_visits = min(8, int(field_rng.expovariate(1.0)))
            destination = field_rng.choices(
                ["home", "home_health", "snf", "other"], weights=[55, 20, 15, 10]
            )[0]
            code_count = int(comorbidity) + field_rng.randrange(3)
            discharge_payload = {
                "los_days": los_days,
                "prior_utilization_12m": prior_util,
                "comorbidity_index": comorbidity,
                "ed_visits_6m": ed_visits,
                "diagnosis_codes": [f"D{i:02d}" for i in range(code_count)],
                "discharge_destination": destination,
            }
            encoded = encode_readmission_payload(discharge_payload)
            # 字段驱动标签（Ruling R-P8-8）：否则逻辑回归学不到信号、无法打赢 LACE。
            latent = (
                -4.6
                + 0.10 * encoded[0]
                + 0.38 * encoded[1]
                + 0.26 * encoded[2]
                + 0.22 * encoded[3]
                + 0.30 * encoded[4]
                + (0.0 if destination == "home" else 0.45)
            )
            probability = 1 / (1 + math.exp(-latent))
            if rng.random() < probability:
                readmissions.append(
                    {
                        "encounter_id": encounter_id,
                        "occurred_at": (at + timedelta(days=2 + rng.randrange(27))).isoformat(),
                        "outcome_type": "unplanned_readmission",
                    }
                )
            raw_by_encounter[encounter_id] = discharge_payload
        labeled = derive_labels(events, readmissions, index_times, horizon_hours=720)
    events_by_encounter: dict[str, list[ClinicalEvent]] = {}
    for event in events:
        events_by_encounter.setdefault(event.encounter_id, []).append(event)
    rows: list[dict] = []
    matrix: list[list[float]] = []
    labels: list[int] = []
    for item in labeled:
        if target_id == "deterioration_24h":
            payload = _latest_payload_before(
                events_by_encounter.get(item.encounter_id, []), item.encounter_id, item.index_time
            )
            rows.append({"index_time": item.index_time.isoformat()})
            matrix.append(_features_before(payload))
        else:
            raw = raw_by_encounter[item.encounter_id]
            rows.append({"index_time": item.index_time.isoformat(), **raw})
            matrix.append(encode_readmission_payload(raw))
        labels.append(item.label)
    return rows, matrix, labels


def load_dataset_csv(path: Path, target_id: str) -> tuple[list[dict], list[list[float]], list[int]]:
    """规模 dataset CSV → build_demo_dataset 同构的 (times, matrix, labels)（R-RS-12）。

    deterioration：21 维 = 10 原始体征列（空→NaN）+ 由空值推导的 __missing 指示
    （serving R2 同语义）+ news2_score 列——与 _features_before 逐位一致。
    readmission：6 源字段列 → encode_readmission_payload（训练/serving 唯一编码源）。
    """
    import csv

    from app.features import READMISSION_PAYLOAD_KEYS, encode_readmission_payload

    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"train_model: empty input CSV: {path}")
    times: list[dict] = []
    matrix: list[list[float]] = []
    labels: list[int] = []
    for index, row in enumerate(rows, start=1):
        raw_label = row.get("label", "")
        try:
            label = int(raw_label)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"train_model: row {index}: label must be 0 or 1") from exc
        if label not in (0, 1):
            raise SystemExit(f"train_model: row {index}: label must be 0 or 1")
        index_time = row.get("index_time", "")
        if not index_time:
            raise SystemExit(f"train_model: row {index}: missing index_time")
        times.append({"index_time": index_time})
        labels.append(label)
        if target_id == "deterioration_24h":
            base: list[float] = []
            for feature in FEATURES:
                raw = row.get(feature, "")
                try:
                    base.append(float(raw) if raw not in ("", None) else math.nan)
                except (TypeError, ValueError) as exc:
                    raise SystemExit(
                        f"train_model: row {index}: {feature} must be numeric or empty"
                    ) from exc
            indicators = [1.0 if math.isnan(value) else 0.0 for value in base]
            raw_news2 = row.get("news2_score", "")
            try:
                news2 = float(raw_news2) if raw_news2 not in ("", None) else math.nan
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    f"train_model: row {index}: news2_score must be numeric or empty"
                ) from exc
            matrix.append([*base, *indicators, news2])
        else:
            payload = {name: row.get(name, "") for name in READMISSION_PAYLOAD_KEYS}
            matrix.append(encode_readmission_payload(payload))
    return times, matrix, labels


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


MONOTONE_CONSTRAINTS_BY_TARGET: dict[str, list[int]] = {
    # Ruling R-P7-2：按 DEFAULT_WEIGHTS 符号 + 临床方向（21 维 deterioration 序）。
    "deterioration_24h": [
        1,
        1,
        1,
        -1,
        1,
        -1,
        1,
        1,
        1,
        1,  # 10 原始体征
        *[0] * 10,  # __missing 指示列不约束
        1,  # news2_score
    ],
    "readmission_30d": [1, 1, 1, 1, 1, 0],  # 出院字段；去向序数不约束
}


def _flatten_structure(structure: dict, nodes: list[dict]) -> int:
    """递归拍平嵌套 tree_structure，返回根下标（就地写入 nodes）。"""
    index = len(nodes)
    if "leaf_value" in structure:
        nodes.append(
            {
                "feature_order_index": -1,
                "threshold": 0.0,
                "left": -1,
                "right": -1,
                "value": float(structure["leaf_value"]),
                "default_left": True,
                "leaf": True,
            }
        )
        return index
    nodes.append(
        {
            "feature_order_index": int(structure["split_feature"]),
            "threshold": float(structure["threshold"]),
            "left": -1,
            "right": -1,
            "value": float(structure.get("internal_value", 0.0)),
            "default_left": bool(structure.get("default_left", True)),
            "leaf": False,
        }
    )
    nodes[index]["left"] = _flatten_structure(structure["left_child"], nodes)
    nodes[index]["right"] = _flatten_structure(structure["right_child"], nodes)
    return index


def _serialize_trees(booster) -> list[dict]:
    """dump_model 的嵌套 tree_structure 拍平为 Ruling R-P7-3 节点列表。

    LightGBM dump：叶子键 leaf_value、内部键 split_feature/threshold/
    left_child/right_child/internal_value；预测即 Σ leaf_value（无需再乘
    shrinkage——tree_info.shrinkage 实测为 1）。
    """
    dump = booster.dump_model()
    trees: list[dict] = []
    for tree in dump["tree_info"]:
        nodes: list[dict] = []
        _flatten_structure(tree["tree_structure"], nodes)
        trees.append({"nodes": nodes, "shrinkage": float(tree.get("shrinkage", 1.0))})
    return trees


def _drift_reference(
    matrix: list[list[float]],
    names: list[str],
    *,
    sample_size: int = 2000,
    seed: int = 20261010,
) -> dict:
    """每特征 lo/hi + 10 等宽箱占比 + 缺失率 + ≤sample_size 确定性参考样本（Ruling
    R-P7-9 + G4 KS 数据源；样本上限 2000——spec 5000 的工件体积修正，R-RS-13）。"""
    rng = Random(seed)
    reference: dict[str, dict] = {}
    for col, name in enumerate(names):
        values = [row[col] for row in matrix]
        finite = [v for v in values if not math.isnan(v)]
        lo, hi = (min(finite), max(finite)) if finite else (0.0, 1.0)
        if hi <= lo:
            hi = lo + 1.0
        width = (hi - lo) / 10
        counts = [0] * 10
        missing = 0
        for value in values:
            if math.isnan(value):
                missing += 1
            else:
                counts[min(int((value - lo) / width), 9)] += 1
        total = max(len(values), 1)
        sample = rng.sample(finite, sample_size) if len(finite) > sample_size else list(finite)
        reference[name] = {
            "lo": lo,
            "hi": hi,
            "proportions": [c / total for c in counts],
            "missing_rate": missing / total,
            "sample": sample,
        }
    return reference


def _fit_lightgbm(
    times: list[dict],
    matrix: list[list[float]],
    labels: list[int],
    target_id: str,
    trials: int,
    seed: int,
):
    """Optuna 时间 CV：目标 = 平均 AUPRC − 0.1×ECE（WBS T7.1，Ruling R-P7-4）。"""
    import optuna
    from lightgbm import LGBMClassifier

    from scripts.evaluate_model import binary_metrics

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fold_rows = [
        {"index_time": t["index_time"], "_i": i, "label": str(labels[i])}
        for i, t in enumerate(times)
    ]
    folds = rolling_time_split(fold_rows, 3)
    x = np.asarray(matrix, dtype=float)
    y = np.asarray(labels, dtype=int)
    monotone = MONOTONE_CONSTRAINTS_BY_TARGET[target_id]

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        auprcs: list[float] = []
        eces: list[float] = []
        for train_rows, test_rows in folds:
            train_idx = [r["_i"] for r in train_rows]
            test_idx = [r["_i"] for r in test_rows]
            if len({labels[i] for i in train_idx}) < 2:
                continue
            model = LGBMClassifier(
                objective="binary",
                monotone_constraints=monotone,
                deterministic=True,
                n_jobs=1,
                verbose=-1,
                **params,
            )
            model.fit(x[train_idx], y[train_idx])
            pred = model.predict_proba(x[test_idx])[:, 1]
            metrics = binary_metrics([labels[i] for i in test_idx], list(map(float, pred)), 0.5)
            auprcs.append(metrics["auprc"] or 0.0)
            eces.append(metrics["ece"] or 0.0)
        if not auprcs:
            return -1.0
        return sum(auprcs) / len(auprcs) - 0.1 * (sum(eces) / len(eces))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = LGBMClassifier(
        objective="binary",
        monotone_constraints=monotone,
        deterministic=True,
        n_jobs=1,
        verbose=-1,
        **study.best_params,
    )
    return best, study


def _fit_isotonic_block(margins, labels: list[int], fitted_on: str) -> dict:
    """Isotonic 校准块（Ruling R-P7-5；两算法共用）。

    分辨率退化时回退 Platt（Ruling R-P7-5a）：唯一 y 水平 <3 的 isotonic 会把
    分数压成二值，直接摧毁阈值阶梯与分带——Platt 保留分级并记 fallback_reason。
    """
    from sklearn.isotonic import IsotonicRegression

    margin_array = np.asarray(margins, dtype=float).reshape(-1)
    label_array = np.asarray(labels, dtype=int)
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(margin_array, label_array)
    y_scores = [float(v) for v in isotonic.y_thresholds_]
    if len({round(v, 6) for v in y_scores}) < 3:
        platt = LogisticRegression()
        platt.fit(margin_array.reshape(-1, 1), label_array)
        return {
            "method": "platt",
            "a": float(platt.coef_[0][0]),
            "b": float(platt.intercept_[0]),
            "fitted_on": fitted_on,
            "fallback_reason": "isotonic_resolution",
        }
    return {
        "method": "isotonic",
        "x_thresholds": [float(v) for v in isotonic.X_thresholds_],
        "y_scores": y_scores,
        "fitted_on": fitted_on,
    }


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
    parser.add_argument(
        "--algorithm",
        choices=["logistic", "lightgbm"],
        default="logistic",
        help="挑战者算法（Ruling R-P7-1：默认保持 logistic，golden 不换代）",
    )
    parser.add_argument("--trials", type=int, default=100, help="Optuna 试验数（Ruling R-P7-4）")
    parser.add_argument("--seed", type=int, default=20261010, help="Optuna/采样种子")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="规模 dataset CSV（G1 产物）；缺省走 build_demo_dataset 演示管道，--rows 被忽略",
    )
    args = parser.parse_args()
    if args.input is not None:
        times, matrix, labels = load_dataset_csv(args.input, args.target_id)
    else:
        times, matrix, labels = build_demo_dataset(args.target_id, args.rows)
    order = sorted(range(len(times)), key=lambda i: times[i]["index_time"])
    n = len(order)
    train_idx = order[: int(n * 0.6)]
    cal_window = order[int(n * 0.6) : int(n * 0.8)]
    x = np.asarray(matrix, dtype=float)
    y = np.asarray(labels, dtype=int)
    feature_names = feature_names_for(args.target_id)
    drift_reference = _drift_reference(matrix, feature_names)
    # 校准段选择（Review I2 链）：60-80% 时间段单类（demo 阳性集中在前半段）时回退
    # train 段——val 段（80-100%）标签始终不进校准器；再单类才兜底全量。
    if len(set(y[cal_window])) < 2:
        if len(set(y[train_idx])) < 2:
            cal_idx = order
            fitted_on = "full"
        else:
            cal_idx = train_idx
            fitted_on = "train_segment"
    else:
        cal_idx = cal_window
        fitted_on = "cal_segment"
    if args.algorithm == "lightgbm":
        model, study = _fit_lightgbm(times, matrix, labels, args.target_id, args.trials, args.seed)
        model.fit(x[train_idx], y[train_idx])
        booster = model.booster_
        artifact = {
            "model_version": args.version,
            "target_id": args.target_id,
            "algorithm": "lightgbm",
            "feature_names": feature_names,
            "feature_order": feature_names,
            "preprocess": None,
            "calibration": _fit_isotonic_block(
                booster.predict(x[cal_idx], raw_score=True), y[cal_idx], fitted_on
            ),
            "bias": 0.0,
            "weights": {name: 0.0 for name in feature_names},
            "trees": _serialize_trees(booster),
            "booster_text": booster.model_to_string(),
            "drift_reference": drift_reference,
            "optuna": {
                "trials": args.trials,
                "seed": args.seed,
                "best_params": study.best_params,
                "best_value": study.best_value,
            },
            "training_rows": len(y),
            "label_definition": f"synthetic_{args.target_id}",
            "thresholds": TARGET_THRESHOLDS[args.target_id],
        }
    else:
        pipeline, best_c = _fit_pipeline(times, matrix, labels)
        pipeline.fit(x[train_idx], y[train_idx])
        calibration_block = _fit_isotonic_block(
            pipeline.decision_function(x[cal_idx]), y[cal_idx], fitted_on
        )
        imputer = pipeline.named_steps["impute"]
        scaler = pipeline.named_steps["scale"]
        clf = pipeline.named_steps["clf"]
        artifact = {
            "model_version": args.version,
            "target_id": args.target_id,
            "algorithm": "logistic_regression_sklearn",
            "feature_names": feature_names,
            "feature_order": feature_names,
            "preprocess": {
                name: {
                    "median": float(imputer.statistics_[i]),
                    "mean": float(scaler.mean_[i]),
                    "std": float(scaler.scale_[i]) or 1.0,
                }
                for i, name in enumerate(feature_names)
            },
            "calibration": calibration_block,
            "bias": float(clf.intercept_[0]),
            "weights": {name: float(clf.coef_[0][i]) for i, name in enumerate(feature_names)},
            "training_rows": len(y),
            "best_C": best_c,
            "drift_reference": drift_reference,
            "label_definition": f"synthetic_{args.target_id}",
            "thresholds": TARGET_THRESHOLDS[args.target_id],
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    detail = f"C={best_c}" if args.algorithm == "logistic" else f"trials={args.trials}"
    print(f"wrote {output} ({args.version}, {detail}, rows={len(y)})")


if __name__ == "__main__":
    main()
