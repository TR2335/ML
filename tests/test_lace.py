import pytest


def test_lace_matches_hand_computed_table():
    from scripts.baselines import lace_score

    # 手算（Ruling R-P8-5）：L(4-6天)=2 + A(1次)=2 + C(2-3)=4 + E(2次)=2 = 10
    assert (
        lace_score(
            {
                "los_days": 5.0,
                "prior_utilization_12m": 1,
                "comorbidity_index": 3.0,
                "ed_visits_6m": 2,
            }
        )
        == 10
    )
    # 全零下界
    assert (
        lace_score(
            {
                "los_days": 0.5,
                "prior_utilization_12m": 0,
                "comorbidity_index": 0.0,
                "ed_visits_6m": 0,
            }
        )
        == 0
    )
    # 上界：L(≥7)=3 + A(≥3)=7 + C(≥4)=6 + E(≥3)=3 = 19
    assert (
        lace_score(
            {
                "los_days": 12.0,
                "prior_utilization_12m": 5,
                "comorbidity_index": 9.0,
                "ed_visits_6m": 4,
            }
        )
        == 19
    )


def test_lace_bins_are_boundary_exact():
    from scripts.baselines import lace_score

    base = {"prior_utilization_12m": 0, "comorbidity_index": 0.0, "ed_visits_6m": 0}
    assert lace_score({**base, "los_days": 1.0}) == 0  # ≤1
    assert lace_score({**base, "los_days": 1.5}) == 1  # 2–3 档
    assert lace_score({**base, "los_days": 3.9}) == 1
    assert lace_score({**base, "los_days": 4.0}) == 2
    assert lace_score({**base, "los_days": 6.9}) == 2
    assert lace_score({**base, "los_days": 7.0}) == 3
    admission = {"los_days": 1.0, "comorbidity_index": 0.0, "ed_visits_6m": 0}
    assert lace_score({**admission, "prior_utilization_12m": 1}) == 2
    assert lace_score({**admission, "prior_utilization_12m": 2}) == 4
    assert lace_score({**admission, "prior_utilization_12m": 3}) == 7
    util = {"los_days": 1.0, "prior_utilization_12m": 0, "ed_visits_6m": 0}
    assert lace_score({**util, "comorbidity_index": 1.0}) == 2
    assert lace_score({**util, "comorbidity_index": 2.0}) == 4
    assert lace_score({**util, "comorbidity_index": 4.0}) == 6
    comorbid = {"los_days": 1.0, "prior_utilization_12m": 0, "comorbidity_index": 0.0}
    assert lace_score({**comorbid, "ed_visits_6m": 1}) == 1
    assert lace_score({**comorbid, "ed_visits_6m": 3}) == 3


def test_lace_missing_field_raises_with_name():
    from scripts.baselines import lace_score

    with pytest.raises(ValueError, match="ed_visits_6m"):
        lace_score({"los_days": 2.0, "prior_utilization_12m": 0, "comorbidity_index": 0.0})
    with pytest.raises(ValueError, match="los_days"):
        lace_score(
            {
                "los_days": "long",
                "prior_utilization_12m": 0,
                "comorbidity_index": 0.0,
                "ed_visits_6m": 0,
            }
        )
