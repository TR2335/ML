from contextlib import asynccontextmanager
from pathlib import Path
from time import monotonic

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles

from .api import build_router
from .config import get_settings
from .observability import metrics
from .risk import RiskEngine
from .security import validate_auth_configuration
from .storage import create_store
from .synthetic import demo_patients


def create_app(settings_override=None, store_override=None, engine_override=None) -> FastAPI:
    local_settings = settings_override or get_settings()
    validate_auth_configuration(local_settings)
    owns_store = store_override is None
    local_store = store_override or create_store(local_settings)
    local_engine = engine_override or RiskEngine(local_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if local_settings.app_env == "dev":
            patients, events = demo_patients()
            local_store.seed(patients, events)
            readmission_engine = RiskEngine(local_settings, target_id="readmission_30d")
            for event in events:
                if not local_store.get_score(
                    event.encounter_id, event.tenant_id, event.facility_id
                ):
                    local_store.upsert_score(local_engine.score(event))
                if not local_store.get_score(
                    event.encounter_id,
                    event.tenant_id,
                    event.facility_id,
                    target_id="readmission_30d",
                ):
                    local_store.upsert_score(readmission_engine.score(event))
        try:
            yield
        finally:
            if owns_store:
                local_store.close()

    application = FastAPI(title="MedIntel Clinical Safety API", version="0.1.0", lifespan=lifespan)

    @application.middleware("http")
    async def observe_requests(request, call_next):
        started = monotonic()
        try:
            response = await call_next(request)
            if local_settings.metrics_enabled:
                metrics.inc(f"medintel_http_responses_total_{response.status_code}")
            return response
        finally:
            if local_settings.metrics_enabled:
                metrics.observe("medintel_http_request_seconds", monotonic() - started)

    @application.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        if request.url.path == "/":
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline';"
                " object-src 'none'; base-uri 'self'",
            )
            response.headers.setdefault("X-Content-Type-Options", "nosniff")
        return response

    application.add_middleware(
        CORSMiddleware,
        allow_origins=local_settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.mount(
        "/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static"
    )
    application.include_router(build_router(local_store, local_engine, local_settings))

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    return application


settings = get_settings()
app = create_app(settings)
