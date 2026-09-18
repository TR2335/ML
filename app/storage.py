from .config import Settings
from .store import Store


def create_store(settings: Settings):
    """Create the configured storage adapter.

    PostgreSQL dependencies are intentionally optional for the local demo;
    selecting that backend without installing the production extra fails with
    an actionable error during startup.
    """
    if settings.storage_backend == "sqlite":
        return Store(settings.store_path)
    if settings.storage_backend == "postgres":
        try:
            from .postgres_store import PostgresStore
        except ModuleNotFoundError as exc:
            raise RuntimeError("PostgreSQL backend requires the 'prod' dependency extra") from exc
        return PostgresStore(settings.database_url)
    raise ValueError(f"unsupported storage backend: {settings.storage_backend}")
