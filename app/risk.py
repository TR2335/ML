import hashlib
import hmac
import json
import logging
import math
from pathlib import Path

from .config import Settings
from .degrade import fallback_score
from .domain import ClinicalEvent, RiskFactor, RiskScore, TargetId
from .synthetic import FEATURES

DEFAULT_WEIGHTS = {
    "age": 0.012,
    "heart_rate": 0.028,
    "respiratory_rate": 0.09,
    "systolic_bp": -0.035,
    "temperature": 0.55,
    "spo2": -0.11,
    "wbc": 0.11,
    "creatinine": 0.34,
    "lactate": 0.44,
    "prior_admissions_30d": 0.35,
}
DEFAULT_BIAS = -7.1
LOGGER = logging.getLogger(__name__)
LABELS = {
    "age": "高龄",
    "heart_rate": "心率升高",
    "respiratory_rate": "呼吸频率升高",
    "systolic_bp": "收缩压偏低",
    "temperature": "体温异常",
    "spo2": "血氧偏低",
    "wbc": "白细胞异常",
    "creatinine": "肌酐升高",
    "lactate": "乳酸升高",
    "prior_admissions_30d": "近期住院次数",
}


TARGET_LABELS = {
    "deterioration_24h": "未来24小时临床恶化",
    "readmission_30d": "出院后30天再入院",
}
DEFAULT_THRESHOLDS = {
    "deterioration_24h": {"medium": 0.2, "high": 0.45, "critical": 0.75},
    "readmission_30d": {"medium": 0.15, "high": 0.3, "critical": 0.55},
}


class RiskEngine:
    def __init__(self, settings: Settings, target_id: TargetId = "deterioration_24h"):
        self.target_id = target_id
        self.version = settings.model_version
        self._app_env = settings.app_env
        self.weights = DEFAULT_WEIGHTS.copy()
        self.bias = DEFAULT_BIAS
        self.algorithm = "logistic_regression"
        self.label_definition = f"demo_{target_id}"
        self.thresholds = DEFAULT_THRESHOLDS[target_id].copy()
        self.degraded = False
        self.degraded_reason: str | None = None
        self._model_path = model_path = (
            settings.model_path
            if target_id == "deterioration_24h"
            else settings.readmission_model_path
        )
        self._signature_required = settings.model_signature_required
        self._signature_path = settings.model_signature_path
        self._model_fallback_enabled = settings.model_fallback_enabled
        self._load(model_path)

    def _load(self, path: str) -> None:
        model_path = Path(path)
        if not model_path.exists():
            reason = f"model artifact not found: {model_path}"
            if self._signature_required or (
                self._strict_model_mode() and not self._fallback_enabled()
            ):
                raise RuntimeError(f"model artifact not found: {model_path}")
            if self._fallback_enabled():
                self._mark_degraded(reason)
            return
        try:
            self._verify_signature(model_path)
            artifact = json.loads(model_path.read_text(encoding="utf-8"))
            weights = artifact.get("weights")
            if artifact.get("feature_names") != FEATURES or not isinstance(weights, dict):
                raise ValueError("model feature signature is invalid")
            candidate_weights = {feature: float(weights[feature]) for feature in FEATURES}
            candidate_bias = float(artifact["bias"])
            if not math.isfinite(candidate_bias) or any(
                not math.isfinite(value) for value in candidate_weights.values()
            ):
                raise ValueError("model parameters must be finite")
            self.version = str(artifact["model_version"])
            self.algorithm = str(artifact.get("algorithm", self.algorithm))
            artifact_target = artifact.get("target_id")
            if artifact_target is None and self._strict_model_mode():
                raise ValueError("model target signature is invalid")
            if artifact_target is not None and artifact_target != self.target_id:
                raise ValueError("model target signature is invalid")
            self.label_definition = str(artifact.get("label_definition", self.label_definition))
            thresholds = artifact.get("thresholds", self.thresholds)
            candidate_thresholds = {
                name: float(thresholds[name]) for name in ("medium", "high", "critical")
            }
            if not (
                0
                < candidate_thresholds["medium"]
                < candidate_thresholds["high"]
                < candidate_thresholds["critical"]
                < 1
            ):
                raise ValueError("model thresholds are invalid")
            self.thresholds = candidate_thresholds
            self.weights.update(candidate_weights)
            self.bias = candidate_bias
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            if self._signature_required or (
                self._strict_model_mode() and not self._fallback_enabled()
            ):
                raise RuntimeError(f"invalid model artifact: {model_path}") from exc
            if self._fallback_enabled():
                self._mark_degraded(str(exc))
            else:
                LOGGER.warning("using development fallback model: %s", exc)

    def _fallback_enabled(self) -> bool:
        return bool(getattr(self, "_model_fallback_enabled", False))

    def _mark_degraded(self, reason: str) -> None:
        self.degraded = True
        self.degraded_reason = reason
        LOGGER.error("model is degraded; scores will be marked non-actionable: %s", reason)

    def _verify_signature(self, model_path: Path) -> None:
        if not self._signature_required:
            return
        signature_path = Path(self._signature_path) if self._signature_path else None
        if signature_path is None or not signature_path.exists():
            raise ValueError("model signature manifest is required")
        raw = signature_path.read_text(encoding="utf-8").strip()
        expected: str | None = None
        try:
            manifest = json.loads(raw)
        except json.JSONDecodeError:
            manifest = None
        if isinstance(manifest, dict):
            entry = manifest.get("models", {}).get(self.target_id, manifest)
            if isinstance(entry, dict):
                expected = entry.get("sha256") or entry.get("hash")
            elif isinstance(entry, str):
                expected = entry
        elif raw:
            expected = raw.split()[0]
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("model signature is invalid")
        actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if not hmac_compare(actual, expected.lower()):
            raise ValueError("model signature mismatch")

    def _strict_model_mode(self) -> bool:
        return self._app_env != "dev"

    def score(self, event: ClinicalEvent) -> RiskScore:
        if self.degraded:
            return self._fallback(event, self.degraded_reason or "model_unavailable")
        try:
            return self._score(event)
        except Exception as exc:
            if not self._fallback_enabled():
                raise
            self._mark_degraded(str(exc))
            return self._fallback(event, str(exc))

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1 / (1 + math.exp(-max(min(value, 30), -30)))

    def _score(self, event: ClinicalEvent) -> RiskScore:
        payload = event.payload
        missing_features = [feature for feature in FEATURES if payload.get(feature) is None]
        contributions: list[tuple[str, float, float | None]] = []
        linear = self.bias
        for feature in FEATURES:
            raw = payload.get(feature)
            if raw is None:
                continue
            value = float(raw)
            contribution = self.weights.get(feature, 0) * value
            # Low blood pressure and oxygen saturation have negative raw values in the model.
            # Their clinical risk direction is represented by the signed contribution after centering.
            if feature == "systolic_bp":
                contribution = self.weights[feature] * (value - 120)
            elif feature == "spo2":
                contribution = self.weights[feature] * (value - 96)
            elif feature == "age":
                contribution = self.weights[feature] * max(value - 50, 0)
            elif feature == "temperature":
                contribution = self.weights[feature] * abs(value - 37.0)
            linear += contribution
            contributions.append((feature, contribution, value))
        score = round(self._sigmoid(linear), 4)
        band = (
            "critical"
            if score >= self.thresholds["critical"]
            else "high"
            if score >= self.thresholds["high"]
            else "medium"
            if score >= self.thresholds["medium"]
            else "low"
        )
        factors = sorted(contributions, key=lambda item: abs(item[1]), reverse=True)[:4]
        return RiskScore(
            patient_id=event.patient_id,
            encounter_id=event.encounter_id,
            tenant_id=event.tenant_id,
            facility_id=event.facility_id,
            model_version=self.version,
            target_id=self.target_id,
            score=score,
            band=band,
            top_factors=[
                RiskFactor(feature=f, label=LABELS[f], contribution=round(c, 3), value=v)
                for f, c, v in factors
            ],
            missing_features=missing_features,
            as_of=event.occurred_at,
        )

    def _fallback(self, event: ClinicalEvent, reason: str) -> RiskScore:
        return fallback_score(
            RiskScore(
                patient_id=event.patient_id,
                encounter_id=event.encounter_id,
                tenant_id=event.tenant_id,
                facility_id=event.facility_id,
                model_version=self.version,
                target_id=self.target_id,
                score=0.0,
                band="low",
                as_of=event.occurred_at,
            ),
            reason,
        )


def hmac_compare(left: str, right: str) -> bool:
    """Constant-time comparison for artifact digests."""
    return hmac.compare_digest(left, right)
