from .domain import RiskScore


def fallback_score(score: RiskScore, reason: str) -> RiskScore:
    """Mark a model failure as non-actionable while retaining auditability."""
    score.band = "low"
    score.score = 0.0
    score.status = "new"
    score.human_review = f"模型不可用，需人工/规则流程：{reason}"
    score.top_factors = []
    score.missing_features = list(dict.fromkeys([*score.missing_features, "model_unavailable"]))
    return score
