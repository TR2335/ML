from scripts.evaluate_model import binary_metrics


def test_binary_metrics_rank_and_threshold_outputs():
    result = binary_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], 0.5)
    assert result["auroc"] == 1.0
    assert result["auprc"] == 1.0
    assert result["sensitivity"] == 1.0
    assert result["specificity"] == 1.0
