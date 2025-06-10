import pytest
from sqlmodel import create_engine, Session
from ocht.core.models import Setting
from ocht.repositories.setting import (
    create_setting,
    get_setting_by_key,
    get_all_settings,
    update_setting,
    delete_setting
)

# Use a temporary SQLite database for testing
engine = create_engine("sqlite:///./test.db")

def setup_module():
    """Create database tables before tests."""
    from sqlmodel import SQLModel
    SQLModel.metadata.create_all(engine)

def teardown_module():
    """Drop database tables after tests."""
    from sqlmodel import SQLModel
    SQLModel.metadata.drop_all(engine)

@pytest.fixture
def db_session():
    """Provide a database session for each test."""
    with Session(engine) as session:
        yield session
        session.rollback()

def test_create_setting(db_session):
    """Test creating a new setting."""
    setting = create_setting(db_session, "test_key", "test_value")
    db_session.commit()

    assert setting.setting_key == "test_key"
    assert setting.setting_value == "test_value"

def test_get_setting_by_key(db_session):
    """Test retrieving a setting by key."""
    create_setting(db_session, "test_key", "test_value")
    db_session.commit()

    result = get_setting_by_key(db_session, "test_key")
    assert result is not None
    assert result.setting_key == "test_key"

def test_get_all_settings(db_session):
    """Test retrieving all settings with pagination."""
    create_setting(db_session, "key1", "value1")
    create_setting(db_session, "key2", "value2")
    db_session.commit()

    settings = get_all_settings(db_session, limit=1)
    assert len(settings) == 1

    settings = get_all_settings(db_session, limit=1, offset=1)
    assert len(settings) == 1

def test_update_setting(db_session):
    """Test updating a setting's value or key."""
    create_setting(db_session, "old_key", "old_value")
    db_session.commit()

    # Update value
    updated = update_setting(db_session, "old_key", value="new_value")
    assert updated.setting_value == "new_value"

    # Update key
    updated = update_setting(db_session, "old_key", new_key="new_key")
    assert updated.setting_key == "new_key"
    assert get_setting_by_key(db_session, "old_key") is None

def test_delete_setting(db_session):
    """Test deleting a setting."""
    create_setting(db_session, "test_key", "test_value")
    db_session.commit()

    assert delete_setting(db_session, "test_key") is True
    assert get_setting_by_key(db_session, "test_key") is None