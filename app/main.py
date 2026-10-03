from fastapi import FastAPI

from app.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    in_production = settings.env == "production"
    app = FastAPI(
        title="Somajji",
        # /docs is off in production (access rule).
        docs_url=None if in_production else "/docs",
        redoc_url=None,
        openapi_url=None if in_production else "/openapi.json",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        # No database call on purpose: platforms poll this, it must stay cheap.
        return {"status": "ok"}

    return app


app = create_app()
