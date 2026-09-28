from dataclasses import dataclass
from threading import Lock

from fastapi import Header, HTTPException

from .config import Settings
from .domain import AuditEvent
from .store import Store

ROLE_PERMISSIONS = {
    "clinician": {"risk:read", "risk:review"},
    "quality_manager": {"risk:read", "risk:review", "audit:read", "model:read"},
    "admin": {"risk:read", "risk:review", "audit:read", "model:read"},
}


@dataclass(frozen=True)
class Principal:
    subject: str
    role: str
    tenant_id: str
    facility_id: str | None


def principal_from_claims(claims: dict, settings: Settings) -> Principal:
    """Map verified OIDC claims to the application's least-privilege scope."""
    subject = claims.get("sub")
    tenant_id = claims.get(settings.oidc_tenant_claim)
    facility_id = claims.get(settings.oidc_facility_claim)
    raw_roles = claims.get(settings.oidc_role_claim, claims.get("role", []))
    if isinstance(raw_roles, str):
        roles = [raw_roles]
    elif isinstance(raw_roles, (list, tuple)):
        roles = list(raw_roles)
    else:
        roles = []
    role = next((candidate for candidate in roles if candidate in ROLE_PERMISSIONS), None)
    if not isinstance(subject, str) or not subject:
        raise ValueError("OIDC subject is missing")
    if not isinstance(tenant_id, str) or not tenant_id:
        raise ValueError("OIDC tenant scope is missing")
    if role is None:
        raise ValueError("OIDC role is not allowed")
    if role == "clinician" and (not isinstance(facility_id, str) or not facility_id):
        raise ValueError("clinician facility scope is missing")
    return Principal(
        subject, role, tenant_id, facility_id if isinstance(facility_id, str) else None
    )


class OIDCAuthenticator:
    def __init__(self, settings: Settings):
        if not all((settings.oidc_issuer, settings.oidc_audience, settings.oidc_jwks_url)):
            raise RuntimeError("OIDC_ISSUER, OIDC_AUDIENCE and OIDC_JWKS_URL are required")
        try:
            import jwt
        except ModuleNotFoundError as exc:
            raise RuntimeError("OIDC requires the 'prod' dependency extra") from exc
        self._jwt = jwt
        self._issuer = settings.oidc_issuer
        self._audience = settings.oidc_audience
        self._client = jwt.PyJWKClient(settings.oidc_jwks_url)
        self._settings = settings

    def authenticate(self, token: str) -> Principal:
        try:
            signing_key = self._client.get_signing_key_from_jwt(token).key
            claims = self._jwt.decode(
                token,
                signing_key,
                algorithms=["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"],
                audience=self._audience,
                issuer=self._issuer,
                options={"require": ["exp", "sub"]},
            )
            return principal_from_claims(claims, self._settings)
        except Exception as exc:
            raise HTTPException(
                status_code=401,
                detail="invalid_token",
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc


_OIDC_AUTHENTICATORS: dict[tuple[str, str, str, str, str, str], OIDCAuthenticator] = {}
_OIDC_LOCK = Lock()


def _oidc_authenticator(settings: Settings) -> OIDCAuthenticator:
    key = (
        settings.oidc_issuer,
        settings.oidc_audience,
        settings.oidc_jwks_url,
        settings.oidc_role_claim,
        settings.oidc_tenant_claim,
        settings.oidc_facility_claim,
    )
    with _OIDC_LOCK:
        authenticator = _OIDC_AUTHENTICATORS.get(key)
        if authenticator is None:
            authenticator = OIDCAuthenticator(settings)
            _OIDC_AUTHENTICATORS[key] = authenticator
        return authenticator


def validate_auth_configuration(settings: Settings) -> None:
    if settings.app_env != "dev" and settings.demo_auth_enabled:
        raise RuntimeError(
            "demo authentication must be disabled outside dev (set DEMO_AUTH_ENABLED=false)"
        )
    if settings.app_env != "dev" and not settings.oidc_enabled:
        raise RuntimeError("production environments require OIDC authentication")
    if settings.oidc_enabled:
        _oidc_authenticator(settings)


def _demo_principal(token: str, settings: Settings) -> Principal | None:
    if not settings.demo_auth_enabled or settings.app_env != "dev":
        return None
    principals = {
        settings.demo_clinician_token: Principal(
            "demo-clinician", "clinician", "group-demo", "hospital-01"
        ),
        settings.demo_quality_token: Principal(
            "demo-quality", "quality_manager", "group-demo", "hospital-01"
        ),
        settings.demo_admin_token: Principal("demo-admin", "admin", "group-demo", None),
    }
    return principals.get(token)


def authenticate(authorization: str | None, settings: Settings) -> Principal:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="bearer_token_required", headers={"WWW-Authenticate": "Bearer"}
        )
    token = authorization.removeprefix("Bearer ").strip()
    if settings.oidc_enabled:
        return _oidc_authenticator(settings).authenticate(token)
    principal = _demo_principal(token, settings)
    if principal:
        return principal
    raise HTTPException(
        status_code=401, detail="invalid_token", headers={"WWW-Authenticate": "Bearer"}
    )


def require_permission(permission: str, settings: Settings):
    def dependency(authorization: str | None = Header(default=None)) -> Principal:
        principal = authenticate(authorization, settings)
        if permission not in ROLE_PERMISSIONS.get(principal.role, set()):
            raise HTTPException(status_code=403, detail="insufficient_role")
        return principal

    return dependency


def audit(
    store: Store,
    principal: Principal,
    action: str,
    resource_type: str,
    resource_id: str,
    detail: dict,
    resource_facility_id: str | None = None,
) -> None:
    actor_facility = principal.facility_id or "*"
    store.add_audit(
        AuditEvent(
            action=action,
            actor=principal.subject,
            resource_type=resource_type,
            resource_id=resource_id,
            tenant_id=principal.tenant_id,
            facility_id=actor_facility,
            actor_facility_id=actor_facility,
            resource_facility_id=resource_facility_id or actor_facility,
            detail={**detail, "role": principal.role},
        )
    )
