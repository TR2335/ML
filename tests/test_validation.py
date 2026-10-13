def _timed_rows(count: int = 40) -> list[dict]:
    from datetime import UTC, datetime, timedelta

    base = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        {
            "encounter_id": f"enc-{index:03d}",
            "index_time": (base + timedelta(hours=index)).isoformat(),
            "label": str(index % 2),
        }
        for index in range(count)
    ]


def test_rolling_time_split_ordered_and_disjoint():
    from scripts.validate import rolling_time_split

    rows = list(reversed(_timed_rows()))  # 输入故意乱序
    folds = rolling_time_split(rows, n_splits=4)
    assert len(folds) == 4
    seen_test_ids: set[str] = set()
    for train, test in folds:
        train_times = sorted(row["index_time"] for row in train)
        test_times = sorted(row["index_time"] for row in test)
        assert train_times[-1] < test_times[0]  # train 全部早于 test（无泄漏）
        test_ids = {row["encounter_id"] for row in test}
        assert test_ids.isdisjoint(seen_test_ids)  # test 块互不重叠
        seen_test_ids |= test_ids
    # 首块只做初始 train：40 行分 5 块 × 8 行，test 合计 4 块
    assert len(seen_test_ids) == 32


def test_bootstrap_ci_coverage_sane():
    from random import Random

    from scripts.validate import bootstrap_metric

    def mean_metric(labels, scores):
        return sum(labels) / len(labels)

    # 点估计 == 样本均值（确定性）
    labels = [1, 0, 1, 1, 0]
    point, lo, hi = bootstrap_metric(labels, labels, mean_metric, n=200, seed=7)
    assert point == 0.6
    assert lo <= point <= hi
    # 覆盖率：对已知总体 p=0.3 做 100 次「抽样 + bootstrap」，95% 名义覆盖 ≥85%
    covered = 0
    reps = 100
    for rep in range(reps):
        data_rng = Random(rep)
        sample = [1 if data_rng.random() < 0.3 else 0 for _ in range(200)]
        _, lo_r, hi_r = bootstrap_metric(sample, sample, mean_metric, n=200, seed=rep + 10_000)
        covered += int(lo_r <= 0.3 <= hi_r)
    assert covered >= 85, f"覆盖率 {covered}/{reps} 过低"


def test_rolling_time_split_keeps_duplicate_timestamps_on_one_side():
    from datetime import UTC, datetime, timedelta

    from scripts.validate import rolling_time_split

    base = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for tick in range(50):  # 50 个时刻 × 每刻 3 行（真实数据常见同刻多 landmark）
        for copy in range(3):
            rows.append(
                {
                    "encounter_id": f"enc-{tick:03d}-{copy}",
                    "index_time": (base + timedelta(hours=tick)).isoformat(),
                    "label": str((tick + copy) % 2),
                }
            )
    folds = rolling_time_split(rows, n_splits=3)
    assert len(folds) == 3
    for train, test in folds:
        train_max = max(row["index_time"] for row in train)
        test_min = min(row["index_time"] for row in test)
        # 同刻行必须整体落同一侧：train 严格早于 test（不允许 TIE）
        assert train_max < test_min, f"边界 TIE: train_max={train_max} test_min={test_min}"


def test_subgroup_sensitivity_diff_flags_imbalance():
    from scripts.validate import subgroup_sensitivity_diff

    # 老组 10 阳性全检出（sens=1.0）；年轻组 10 阳性只检出 2（sens=0.2）→ 差 0.8
    labels = [1] * 20
    scores = [0.9] * 10 + [0.9, 0.9] + [0.1] * 8
    groups = ["older"] * 10 + ["younger"] * 10
    diff = subgroup_sensitivity_diff(labels, scores, groups, threshold=0.5, min_positives=10)
    assert diff is not None and abs(diff - 0.8) < 1e-9
    # 前置不足 → None（降级跳过）
    assert (
        subgroup_sensitivity_diff([1] * 5, [0.9] * 5, ["older"] * 5, 0.5, min_positives=10) is None
    )
    assert (
        subgroup_sensitivity_diff([1] * 20, [0.9] * 20, ["same"] * 20, 0.5, min_positives=10)
        is None  # 单组无从比较
    )


def test_decision_curve_mean_gain_both_sides():
    from scripts.validate import decision_curve_mean_gain

    labels = [1, 1, 0, 0]
    perfect = [0.9, 0.8, 0.2, 0.1]
    inverted = [0.1, 0.2, 0.8, 0.9]
    assert decision_curve_mean_gain(labels, perfect) > 0
    assert decision_curve_mean_gain(labels, inverted) < 0


def test_per_facility_auprc_requires_two_facilities():
    import pytest

    from scripts.validate import per_facility_auprc

    rows = [{"facility_id": "h1"}] * 30
    labels = [1] * 10 + [0] * 20
    scores = [0.9] * 10 + [0.1] * 20
    baseline = [0.5] * 30  # 恒分基线 → 本院基线点估计 = 患病率 1/3
    single = per_facility_auprc(rows, labels, scores, baseline)
    assert single["checked"] is False
    rows2 = [{"facility_id": "h1" if i < 30 else "h2"} for i in range(60)]
    labels2 = labels + labels
    both = per_facility_auprc(rows2, labels2, scores + scores, baseline + baseline)
    assert both["checked"] is True and not both["failures"]
    # h2 反向排序：10 阳性压到 20 阴性之后 → 模型 AUPRC≈0.2055 <
    # 同院区基线（prevalence 1/3）→ 拒；baseline 记录的是本院点估计而非全局值。
    inverted2 = scores + [0.01 * i for i in range(10)] + [0.9] * 20
    bad = per_facility_auprc(rows2, labels2, inverted2, baseline + baseline)
    assert bad["checked"] is True and bad["failures"]
    assert bad["failures"][0]["baseline"] == pytest.approx(1 / 3)
    # 缺 facility 列 → 降级跳过
    missing = per_facility_auprc([{}] * 60, labels2, scores + scores, baseline + baseline)
    assert missing["checked"] is False


def test_per_facility_auprc_skips_facilities_below_positive_floor():
    """R-P7-6 对齐口径：合格院区须 ≥10 阳性——8 阳性的小院区不参与裁决
    （AUPRC 点估计在个位数阳性下方差无意义），其余院区照常评估。"""
    from scripts.validate import per_facility_auprc

    rows = [{"facility_id": "h1"}] * 30 + [{"facility_id": "h3"}] * 30
    labels = [1] * 10 + [0] * 20 + [1] * 8 + [0] * 22
    scores = [0.9] * 10 + [0.1] * 20 + [0.5] * 30  # h3 恒分（会被 floor 跳过）
    baseline = [0.5] * 60
    result = per_facility_auprc(rows, labels, scores, baseline)
    assert result["checked"] is True
    assert result["failures"] == []
    assert result["worst"] is not None


def _staggered_patient_rows() -> list[dict]:
    """6 患者 × 6 行；患者时点按 48h 错开，患者内行连续 1h。"""
    from datetime import UTC, datetime, timedelta

    base = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for patient in range(6):
        for tick in range(6):
            rows.append(
                {
                    "patient_id": f"pat-{patient:02d}",
                    "encounter_id": f"enc-{patient:02d}-{tick:02d}",
                    "index_time": (base + timedelta(hours=patient * 48 + tick)).isoformat(),
                    "label": str((patient + tick) % 2),
                }
            )
    return rows


def test_patient_split_never_splits_a_patient():
    from scripts.validate import patient_rolling_time_split

    rows = list(reversed(_staggered_patient_rows()))  # 输入故意乱序
    folds = patient_rolling_time_split(rows, n_splits=3)
    assert len(folds) == 3
    total_test_rows = 0
    for train, test in folds:
        train_patients = {row["patient_id"] for row in train}
        test_patients = {row["patient_id"] for row in test}
        assert train_patients.isdisjoint(test_patients)  # 核心不变量：整患者不跨折
        # 行级时间序同样保持（fixture 患者块时间错开）
        assert max(row["index_time"] for row in train) < min(row["index_time"] for row in test)
        total_test_rows += len(test)
    # expanding window：首块只做初始 train，其余行恰好出现在一个 test 折
    assert total_test_rows + len(folds[0][0]) == len(rows)


def test_patient_split_keeps_simultaneous_patients_on_one_side():
    from datetime import UTC, datetime, timedelta

    from scripts.validate import patient_rolling_time_split

    base = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for patient, start in (("pat-a", 0), ("pat-b", 0), ("pat-c", 72), ("pat-d", 72)):
        for tick in range(4):
            rows.append(
                {
                    "patient_id": patient,
                    "encounter_id": f"{patient}-{tick}",
                    "index_time": (base + timedelta(hours=start + tick)).isoformat(),
                    "label": "1",
                }
            )
    train, _test = patient_rolling_time_split(rows, n_splits=1)[0]
    train_patients = {row["patient_id"] for row in train}
    # 同一患者时点的多个患者必须整体同侧（时间戳不跨折裁定）
    assert ("pat-a" in train_patients) == ("pat-b" in train_patients)
    assert ("pat-c" in train_patients) == ("pat-d" in train_patients)
    assert train_patients in ({"pat-a", "pat-b"}, {"pat-c", "pat-d"})


def test_patient_split_missing_group_keys_raises():
    import pytest

    from scripts.validate import patient_rolling_time_split

    rows = [{"label": "1", "index_time": "2026-01-01T00:00:00+00:00"}] * 4
    with pytest.raises(ValueError, match="patient_id"):
        patient_rolling_time_split(rows, n_splits=2)


def test_patient_split_falls_back_to_encounter_id():
    from scripts.validate import patient_rolling_time_split

    rows = _staggered_patient_rows()
    for row in rows:
        del row["patient_id"]  # 只剩 encounter_id
    folds = patient_rolling_time_split(rows, n_splits=3)
    for train, test in folds:
        train_keys = {row["encounter_id"] for row in train}
        test_keys = {row["encounter_id"] for row in test}
        assert train_keys.isdisjoint(test_keys)


def test_resolve_group_column_order():
    import pytest

    from scripts.evaluate_model import resolve_group_column

    assert resolve_group_column([{"patient_id": "p", "encounter_id": "e"}]) == "patient_id"
    assert resolve_group_column([{"encounter_id": "e"}]) == "encounter_id"
    with pytest.raises(ValueError, match="patient_id"):
        resolve_group_column([{"label": "1"}])


def test_validate_cli_patient_mode_reports_split(tmp_path, monkeypatch):
    import csv
    import json

    from scripts.validate import main as validate_main

    csv_path = tmp_path / "rows.csv"
    rows = _staggered_patient_rows()
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(
        "sys.argv",
        ["validate.py", "--input", str(csv_path), "--split", "patient", "--n-splits", "3"],
    )
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        validate_main()
    report = json.loads(buffer.getvalue())
    assert report["split_mode"] == "patient"
    assert len(report["folds"]) == 3


def test_validate_cli_patient_mode_loud_without_group_keys(tmp_path, monkeypatch):
    import pytest

    from scripts.validate import main as validate_main

    csv_path = tmp_path / "bare.csv"
    csv_path.write_text(
        "index_time,label\n2026-01-01T00:00:00+00:00,1\n2026-01-02T00:00:00+00:00,0\n"
        "2026-01-03T00:00:00+00:00,1\n2026-01-04T00:00:00+00:00,0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["validate.py", "--input", str(csv_path), "--split", "patient", "--n-splits", "3"],
    )
    with pytest.raises(SystemExit, match="patient-level"):
        validate_main()


def test_validate_cli_encounter_mode_keeps_row_semantics(tmp_path, monkeypatch):
    import csv
    import json

    from scripts.validate import main as validate_main

    csv_path = tmp_path / "rows.csv"
    rows = _staggered_patient_rows()
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(
        "sys.argv",
        ["validate.py", "--input", str(csv_path), "--split", "encounter", "--n-splits", "3"],
    )
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        validate_main()
    report = json.loads(buffer.getvalue())
    assert report["split_mode"] == "encounter"


def test_bootstrap_budget_thresholds():
    from scripts.validate import bootstrap_budget

    assert bootstrap_budget(49_999) == (1000, None)
    assert bootstrap_budget(50_001) == (500, 20_000)
    assert bootstrap_budget(400_000) == (500, 20_000)


def test_bootstrap_max_rows_limits_replicate_size():
    from scripts.validate import bootstrap_metric

    sizes: list[int] = []

    def probe(labels, scores):
        sizes.append(len(labels))
        return sum(labels) / len(labels)

    labels = list(range(100))
    point, lo, hi = bootstrap_metric(labels, labels, probe, n=10, seed=1, max_rows=7)
    assert sizes[0] == 100  # 点估计用全量
    assert sizes[1:] == [7] * 10  # 每次 replicate 只抽 max_rows 行（规模裁定）
    assert point == 49.5
    assert lo <= hi
    # 缺省（max_rows=None）保持全量
    sizes.clear()
    bootstrap_metric(labels, labels, probe, n=3, seed=1)
    assert sizes == [100] * 4  # 点估计 + 3 次全量 replicate
