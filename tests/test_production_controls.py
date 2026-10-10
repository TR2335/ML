import hashlib
import json
import sys

import pytest

from app.config import Settings
from app.domain import ClinicalEvent
from app.risk import RiskEngine
from app.security import validate_auth_configuration
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
    from tests.test_manage_model import _write_gate_artifact, _write_gate_rows

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_gate_artifact(first, "v1")
    _write_gate_artifact(second, "v2")
    holdout = tmp_path / "holdout.csv"
    _write_gate_rows(holdout, invert_labels=False)
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
            "--input",
            str(holdout),
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
            "--input",
            str(holdout),
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


def test_non_dev_rejects_demo_auth():
    with pytest.raises(RuntimeError, match="demo authentication"):
        validate_auth_configuration(Settings(app_env="prod", demo_auth_enabled=True))


def test_dev_demo_auth_still_works():
    validate_auth_configuration(Settings(app_env="dev", demo_auth_enabled=True))


def test_prod_requires_signature_by_default(tmp_path):
    model = tmp_path / "model.json"
    write_model(model)
    # 非 dev 且未显式配置（None → fail-closed 视为 True）：无签名清单必须拒绝
    with pytest.raises(RuntimeError, match="invalid model artifact"):
        RiskEngine(Settings(app_env="prod", model_path=str(model)))


def test_dev_skips_signature_by_default(tmp_path):
    model = tmp_path / "model.json"
    write_model(model, version="dev-v1")
    engine = RiskEngine(Settings(app_env="dev", model_path=str(model)))
    assert engine.version == "dev-v1"


def test_empty_string_signature_setting_is_unset():
    from app.config import Settings as SettingsCls

    assert SettingsCls(model_signature_required="").model_signature_required is None
    assert SettingsCls(model_signature_required=True).model_signature_required is True


def test_invalid_thresholds_leaves_version_unchanged(tmp_path):
    model = tmp_path / "model.json"
    write_model(model)  # version=approved-v1
    engine = RiskEngine(Settings(model_path=str(model)))
    assert engine.version == "approved-v1"
    artifact = json.loads(model.read_text(encoding="utf-8"))
    artifact["model_version"] = "polluted-v99"
    artifact["thresholds"] = {"medium": 0.9, "high": 0.5, "critical": 0.95}  # 非递增 → 非法
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(artifact), encoding="utf-8")
    engine._load(str(bad))  # dev 非严格：仅告警，不抛
    assert engine.version == "approved-v1"  # 拒绝的工件不得污染任何状态
    assert engine.thresholds["medium"] == 0.2  # 默认阈值保持不变


def test_missing_artifact_reports_reason_in_health(tmp_path):
    engine = RiskEngine(Settings(model_path=str(tmp_path / "missing.json")))
    assert engine.degraded is False  # 不改分数行为
    assert engine.degraded_reason is not None
    assert "not found" in engine.degraded_reason
    from app.domain import ClinicalEvent

    score = engine.score(
        ClinicalEvent(event_type="vital_signs", patient_id="pat-1", encounter_id="enc-1")
    )
    assert 0.0 <= score.score <= 1.0  # 默认模型仍可打分


def test_prod_rejects_default_database_url():
    from app.config import validate_production_configuration

    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        validate_production_configuration(Settings(app_env="prod", storage_backend="postgres"))
    # dev 不检查；显式改过的 URL 通过；sqlite 后端不检查
    validate_production_configuration(Settings(app_env="dev", storage_backend="postgres"))
    validate_production_configuration(
        Settings(
            app_env="prod",
            storage_backend="postgres",
            database_url="postgresql://user:pass@db.example:5432/other",
        )
    )
    validate_production_configuration(Settings(app_env="prod", storage_backend="sqlite"))


def test_prod_rejects_unknown_setting(tmp_path, monkeypatch):
    from pydantic import ValidationError

    from app.config import get_settings

    (tmp_path / ".env").write_text("APP_ENV=prod\nTYPOED_SETTING=1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    try:
        with pytest.raises(ValidationError):
            get_settings()
    finally:
        get_settings.cache_clear()


def test_dev_env_file_allows_extra_entries(tmp_path, monkeypatch):
    from app.config import get_settings

    (tmp_path / ".env").write_text(
        "APP_ENV=dev\nSOME_TOOL_VAR=1\nMLFLOW_TRACKING_URI=http://mlflow:5000\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.app_env == "dev"
        assert settings.mlflow_tracking_uri == "http://mlflow:5000"
    finally:
        get_settings.cache_clear()


def test_bearer_prefix_case_insensitive():
    from app.security import authenticate

    settings = Settings(app_env="dev", demo_auth_enabled=True)
    principal = authenticate("bearer demo-clinician-token", settings)
    assert principal.subject == "demo-clinician"


@pytest.mark.skipif(
    __import__("importlib.util", fromlist=["find_spec"]).find_spec("jwt") is None,
    reason="环境性守卫：PyJWT 属 [prod] extra，本地未安装；leeway 行为由代码审阅 + prod 环境集成覆盖",
)
def test_jwt_leeway_allows_25s_clock_skew(monkeypatch):
    from datetime import UTC, datetime, timedelta

    import jwt as pyjwt

    from app import security

    secret = "test-secret"
    settings = Settings(
        app_env="prod",
        demo_auth_enabled=False,
        oidc_enabled=True,
        oidc_issuer="https://issuer.example",
        oidc_audience="medintel",
        oidc_jwks_url="https://issuer.example/jwks",
    )

    class _Key:
        key = secret

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def get_signing_key_from_jwt(self, token):
            return _Key()

    monkeypatch.setattr(pyjwt, "PyJWKClient", _Client)
    authenticator = security.OIDCAuthenticator(settings)
    token = pyjwt.encode(
        {
            "sub": "clinician-1",
            "roles": ["clinician"],
            "tenant_id": "group-demo",
            "facility_id": "hospital-01",
            "exp": datetime.now(UTC) - timedelta(seconds=25),
        },
        secret,
        algorithm="HS256",
    )
    principal = authenticator.authenticate(token)  # 25s 过期但 leeway=30 → 通过
    assert principal.subject == "clinician-1"


def test_non_dev_hides_schema_docs(tmp_path, monkeypatch):
    """Plan Ruling R5 回归：非 dev 关闭 /docs 与 /redoc，dev 保持可用。"""
    from fastapi.testclient import TestClient

    import app.main as main_mod
    from app.config import Settings

    # prod 全链路校验需要 OIDC/jwt（[prod] extra）；本测试只关心 docs 路由，
    # 故短路认证校验（docs 路由本身无鉴权依赖）。
    monkeypatch.setattr(main_mod, "validate_auth_configuration", lambda settings: None)
    prod = main_mod.create_app(
        Settings(
            app_env="prod",
            demo_auth_enabled=False,
            model_fallback_enabled=True,
            store_path=str(tmp_path / "prod.db"),
            model_path=str(tmp_path / "missing.json"),
        )
    )
    with TestClient(prod) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404

    dev = main_mod.create_app(
        Settings(store_path=str(tmp_path / "dev.db"), model_path=str(tmp_path / "missing.json"))
    )
    with TestClient(dev) as client:
        assert client.get("/docs").status_code == 200
        assert client.get("/redoc").status_code == 200


def test_env_postgres_password_allowed_in_prod(tmp_path, monkeypatch):
    """compose 插值取值放在 .env 是最自然的路径；strict 模式不得拒绝该专用变量。"""
    from app.config import get_settings

    (tmp_path / ".env").write_text(
        "APP_ENV=prod\nPOSTGRES_PASSWORD=secret\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.postgres_password == "secret"
    finally:
        get_settings.cache_clear()


def test_dev_env_selector_typo_raises(tmp_path, monkeypatch):
    """.env 里 APP_ENV 键名拼错（如 APP_ENB）不得静默回落 dev（fail-open）。"""
    from app.config import get_settings

    (tmp_path / ".env").write_text("APP_ENB=prod\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="APP_ENB"):
            get_settings()
    finally:
        get_settings.cache_clear()
