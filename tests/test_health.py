from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def make_client(env: str) -> TestClient:
    settings = Settings(_env_file=None, database_url="postgresql+psycopg://x:y@localhost/z", env=env)
    return TestClient(create_app(settings))


def test_health_returns_ok():
    response = make_client("development").get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_docs_are_on_in_development():
    assert make_client("development").get("/docs").status_code == 200


def test_docs_are_off_in_production():
    client = make_client("production")
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
