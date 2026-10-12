"""规模队列生成（真实规模优化 G1）：确定性、数据画像窗口、列序契约。"""

import csv
import json
import sys


def _generate(tmp_path, scale: int = 1000, seed: int = 20261012):
    from scripts.generate_scale_data import main as gen_main

    out = tmp_path / "scale"
    old_argv = sys.argv
    sys.argv = [
        "generate_scale_data.py",
        "--scale",
        str(scale),
        "--seed",
        str(seed),
        "--horizon-days",
        "540",
        "--out-dir",
        str(out),
    ]
    try:
        gen_main()
    finally:
        sys.argv = old_argv
    return out


def test_scale_generation_is_deterministic(tmp_path):
    first = _generate(tmp_path / "a")
    second = _generate(tmp_path / "b")
    for name in (
        "encounters.csv",
        "labels_deterioration.csv",
        "labels_readmission.csv",
        "dataset_deterioration.csv",
        "dataset_readmission.csv",
        "cohort_stats.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_cohort_rates_within_spec_windows(tmp_path):
    stats = json.loads((_generate(tmp_path) / "cohort_stats.json").read_text(encoding="utf-8"))
    # 遭遇级 24h 恶化事件率 4% ±3pp（spec §2；sanity 按遭遇级口径）
    assert 0.01 <= stats["deterioration"]["event_rate"] <= 0.07
    # 阳性 landmark 必须真实存在（6h 网格 × 事件前 24h 窗口）
    assert stats["deterioration"]["positives"] > 0
    # 30 天非计划再入院 12% ±3pp（行级 = 患者级）
    assert 0.09 <= stats["readmission"]["positive_rate"] <= 0.15
    # 连续体征 MAR 缺失注入率 5%–12%
    assert 0.05 <= stats["missing_rate_overall"] <= 0.12
    # 院区占比 45/35/20 ±5pp
    total = stats["scale"]
    assert 0.40 <= stats["facilities"]["hospital-01"] / total <= 0.50
    assert 0.30 <= stats["facilities"]["hospital-02"] / total <= 0.40
    assert 0.15 <= stats["facilities"]["hospital-03"] / total <= 0.25


def test_patient_structure_follows_formula(tmp_path):
    out = _generate(tmp_path)
    with (out / "encounters.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1000
    patients: dict[str, int] = {}
    for row in rows:
        patients[row["patient_id"]] = patients.get(row["patient_id"], 0) + 1
    # 唯一患者 = round(scale × 0.82)；多次患者各恰好 2 次；多次患者占比 ≈22%
    assert len(patients) == round(1000 * 0.82)
    multi = [count for count in patients.values() if count > 1]
    assert all(count == 2 for count in multi)
    assert 0.20 <= len(multi) / len(patients) <= 0.24


def test_dataset_column_order_matches_demo_contract(tmp_path):
    from app.features import READMISSION_PAYLOAD_KEYS, TUPLE_FEATURE_ORDER
    from app.synthetic import FEATURES

    out = _generate(tmp_path)
    with (out / "dataset_deterioration.csv").open(newline="", encoding="utf-8") as handle:
        det_header = next(csv.reader(handle))
    # demo_dataset 契约列序（encounter_id, index_time, label, *FEATURES, *七元组, news2）
    # + patient_id 插第 2 列、性别/院区/科室尾列（元数据供亚组与逐院区门禁）
    assert det_header == [
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
    with (out / "dataset_readmission.csv").open(newline="", encoding="utf-8") as handle:
        rm_header = next(csv.reader(handle))
    assert rm_header == [
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


def test_row_counts_match_grid(tmp_path):
    out = _generate(tmp_path)  # scale=1000

    def count(path):
        with path.open(newline="", encoding="utf-8") as handle:
            return sum(1 for _ in csv.DictReader(handle))

    assert count(out / "encounters.csv") == 1000
    # deterioration：16 格 landmark/遭遇（6h 网格 6..96h）
    assert count(out / "dataset_deterioration.csv") == 1000 * 16
    assert count(out / "labels_deterioration.csv") == 1000 * 16
    # readmission：出院时刻单行/遭遇
    assert count(out / "dataset_readmission.csv") == 1000
    assert count(out / "labels_readmission.csv") == 1000


def test_scale_out_of_range_fails_loudly(tmp_path):
    import pytest

    from scripts.generate_scale_data import main as gen_main

    old_argv = sys.argv
    sys.argv = ["generate_scale_data.py", "--scale", "500", "--out-dir", str(tmp_path / "x")]
    try:
        with pytest.raises(SystemExit):
            gen_main()
    finally:
        sys.argv = old_argv
