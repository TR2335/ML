"""Train the demo logistic model without a heavyweight ML runtime.

The production pipeline should replace this script with a governed training job,
but the same artifact shape is intentionally kept stable for the serving layer.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from random import Random

# 直接以 `python scripts/x.py` 运行时根目录不在 sys.path，补一次以便跨 scripts 导入。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FEATURES = [
    "age",
    "heart_rate",
    "respiratory_rate",
    "systolic_bp",
    "temperature",
    "spo2",
    "wbc",
    "creatinine",
    "lactate",
    "prior_admissions_30d",
]
BASE_WEIGHTS = [0.012, 0.028, 0.09, -0.035, 0.55, -0.11, 0.11, 0.34, 0.44, 0.35]
READMISSION_WEIGHTS = [0.018, 0.006, 0.01, -0.006, 0.02, -0.012, 0.02, 0.12, 0.06, 0.9]
TARGET_CONFIG = {
    "deterioration_24h": {
        "weights": BASE_WEIGHTS,
        "bias": -7.1,
        "news2_weight": 0.06,
        "thresholds": {"medium": 0.2, "high": 0.45, "critical": 0.75},
    },
    "readmission_30d": {
        "weights": READMISSION_WEIGHTS,
        "bias": -3.0,
        "news2_weight": 0.02,
        "thresholds": {"medium": 0.15, "high": 0.3, "critical": 0.55},
    },
}

FEATURE_ORDER_V2 = [*FEATURES, "news2_score"]


def transform(row: list[float]) -> list[float]:
    (
        age,
        heart_rate,
        respiratory_rate,
        systolic_bp,
        temperature,
        spo2,
        wbc,
        creatinine,
        lactate,
        prior,
    ) = row
    return [
        max(age - 50, 0),
        heart_rate,
        respiratory_rate,
        systolic_bp - 120,
        abs(temperature - 37.0),
        spo2 - 96,
        wbc,
        creatinine,
        lactate,
        prior,
    ]


def column_stats(rows: list[list[float]]) -> list[dict]:
    """Median/mean/std per raw column (population std; zero std guarded to 1)."""
    stats = []
    for column in zip(*rows):
        ordered = sorted(column)
        count = len(ordered)
        mean = sum(ordered) / count
        variance = sum((value - mean) ** 2 for value in ordered) / count
        median = (
            ordered[count // 2]
            if count % 2
            else (ordered[count // 2 - 1] + ordered[count // 2]) / 2
        )
        std = math.sqrt(variance) or 1.0
        stats.append({"median": median, "mean": mean, "std": std})
    return stats


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-max(min(value, 30), -30)))


def make_row(rng: Random, target_id: str) -> tuple[list[float], float, int]:
    row = [
        rng.uniform(18, 95),
        rng.uniform(55, 150),
        rng.uniform(10, 40),
        rng.uniform(70, 165),
        rng.uniform(35.5, 40),
        rng.uniform(82, 100),
        rng.uniform(2, 25),
        rng.uniform(0.4, 4.5),
        rng.uniform(0.4, 6),
        rng.uniform(0, 4),
    ]
    from scripts.baselines import news2_score

    transformed = transform(row)
    config = TARGET_CONFIG[target_id]
    total, _ = news2_score(dict(zip(FEATURES, row)))
    latent = (
        config["bias"]
        + sum(weight * value for weight, value in zip(config["weights"], transformed))
        + config["news2_weight"] * total
    )
    probability = sigmoid(latent)
    return row, float(total), int(rng.random() < probability)


def train(
    rows: list[list[float]], labels: list[int], epochs: int = 900, learning_rate: float = 0.004
) -> tuple[float, list[float]]:
    weights = [0.0] * len(rows[0])  # 宽度由向量决定（v2 为 11 维：10 体征 + news2）
    bias = 0.0
    for _ in range(epochs):
        grad_w = [0.0] * len(weights)
        grad_b = 0.0
        for row, label in zip(rows, labels):
            features = row  # 调用方传入已标准化的行（v2：z 分数空间训练）
            prediction = sigmoid(
                bias + sum(weight * value for weight, value in zip(weights, features))
            )
            error = prediction - label
            grad_b += error
            for index, value in enumerate(features):
                grad_w[index] += error * value
        scale = learning_rate / max(len(rows), 1)
        bias -= scale * grad_b
        weights = [weight - scale * gradient for weight, gradient in zip(weights, grad_w)]
    return bias, weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/model.json")
    parser.add_argument("--rows", type=int, default=2500)
    parser.add_argument("--version", default="baseline-logistic-v1")
    parser.add_argument(
        "--target-id", choices=["deterioration_24h", "readmission_30d"], default="deterioration_24h"
    )
    args = parser.parse_args()
    rng = Random(42)
    generated = [make_row(rng, args.target_id) for _ in range(args.rows)]
    raw_rows = [item[0] for item in generated]
    news2_values = [item[1] for item in generated]
    labels = [item[2] for item in generated]
    stats = column_stats([row + [news2] for row, news2 in zip(raw_rows, news2_values)])
    vectors = [
        [(value - column["mean"]) / column["std"] for value, column in zip([*row, news2], stats)]
        for row, news2 in zip(raw_rows, news2_values)
    ]
    bias, weights = train(vectors, labels)
    artifact = {
        "model_version": args.version,
        "target_id": args.target_id,
        "algorithm": "logistic_regression_gradient_descent",
        "feature_names": FEATURE_ORDER_V2,
        "feature_order": FEATURE_ORDER_V2,
        "preprocess": {
            feature: {
                "median": round(entry["median"], 8),
                "mean": round(entry["mean"], 8),
                "std": round(entry["std"], 8),
            }
            for feature, entry in zip(FEATURE_ORDER_V2, stats)
        },
        "bias": round(bias, 8),
        "weights": {
            feature: round(weight, 8) for feature, weight in zip(FEATURE_ORDER_V2, weights)
        },
        "training_rows": args.rows,
        "label_definition": f"synthetic_{args.target_id}",
        "thresholds": TARGET_CONFIG[args.target_id]["thresholds"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output} ({args.version})")


if __name__ == "__main__":
    main()
