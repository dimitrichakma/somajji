import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.db.session import get_session


def test_session_can_select_one():
    session = next(get_session())
    try:
        assert session.execute(text("SELECT 1")).scalar() == 1
    except OperationalError:
        pytest.fail("Cannot reach Postgres. Start it with: docker compose up -d")
    finally:
        session.close()
