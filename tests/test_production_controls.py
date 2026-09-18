import hashlib
import json
import sys

import pytest

from app.config import Settings
from app.domain import ClinicalEvent
from app.risk import RiskEngine
from app.synthetic import FEATURES
from scripts.manage_model import main as manage_model_main


def write_model(path, target="deterioration_24h", version="approved-v1"):
    path.write_text(
        json.dumps(
            {
                "model_version": version,
                "target_id": target,
                "feature_names": FEATURES,
                "weights": {feature: 0.0 for feature in FEATURES},
                "bias": -2.0,
            }
        ),
        encoding="utf-8",
    )


def test_required_model_signature_is_verified(tmp_path):
    model = tmp_path / "model.json"
    write_model(model)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "models": {
                    "deterioration_24h": {"sha256": hashlib.sha256(model.read_bytes()).hexdigest()}
                }
            }
        ),
        encoding="utf-8",
    )
    engine = RiskEngine(
        Settings(
            app_env="prod",
            model_path=str(model),
            model_signature_required=True,
            model_signature_path=str(manifest),
        )
    )
    assert engine.version == "approved-v1"
    model.write_text(model.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid model artifact"):
        RiskEngine(
            Settings(
                app_env="prod",
                model_path=str(model),
                model_signature_required=True,
                model_signature_path=str(manifest),
            )
        )


def test_fallback_is_explicitly_degraded(tmp_path):
    settings = Settings(
        app_env="prod",
        model_path=str(tmp_path / "missing.json"),
        model_fallback_enabled=True,
    )
    engine = RiskEngine(settings)
    score = engine.score(ClinicalEvent(event_type="vital_signs", patient_id="p", encounter_id="e"))
    assert engine.degraded is True
    assert score.score == 0
    assert "model_unavailable" in score.missing_features
    assert "模型不可用" in (score.human_review or "")


def test_model_registry_promote_and_rollback(tmp_path, monkeypatch):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_model(first, version="v1")
    write_model(second, version="v2")
    registry = tmp_path / "registry"
    serving = tmp_path / "serving.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manage_model.py",
            "--registry-dir",
            str(registry),
            "promote",
            "--artifact",
            str(first),
            "--target-id",
            "deterioration_24h",
            "--serving-path",
            str(serving),
        ],
    )
    manage_model_main()
    assert (
        json.loads((registry / "manifest.json").read_text())["models"]["deterioration_24h"][
            "active_version"
        ]
        == "v1"
    )
    assert json.loads(serving.read_text())["model_version"] == "v1"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manage_model.py",
            "--registry-dir",
            str(registry),
            "promote",
            "--artifact",
            str(second),
            "--target-id",
            "deterioration_24h",
            "--serving-path",
            str(serving),
        ],
    )
    manage_model_main()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manage_model.py",
            "--registry-dir",
            str(registry),
            "rollback",
            "--target-id",
            "deterioration_24h",
            "--version",
            "v1",
            "--serving-path",
            str(serving),
        ],
    )
    manage_model_main()
    assert json.loads(serving.read_text())["model_version"] == "v1"
