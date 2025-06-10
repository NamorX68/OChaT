import os
from pathlib import Path
import tempfile

import pytest
from sqlalchemy.engine import Engine
from sqlmodel import select

from ocht.core.db import get_database_url, create_db_engine, init_db, get_session
from ocht.core.models import Workspace, LLMProviderConfig  # Hinweis: passe den Import-Pfad ggf. an


def test_get_database_url():
    # Test mit SET DATABASE_URL
    os.environ["DATABASE_URL"] = "sqlite:///test.db"
    assert get_database_url() == "sqlite:///test.db"

    # Test ohne SET DATABASE_URL
    if "DATABASE_URL" in os.environ:
        del os.environ["DATABASE_URL"]
    assert get_database_url().endswith("data/ocht.db")


def test_create_db_engine():
    engine = create_db_engine()
    assert isinstance(engine, Engine)


def test_init_db():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
        engine = create_db_engine()
        init_db(engine)
        # Hier könnten Sie weitere Überprüfungen hinzufügen, um sicherzustellen,
        # dass die Tabellen korrekt erstellt wurden.


def test_get_session():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
        engine = create_db_engine()
        init_db(engine)
        session = next(get_session(engine))
        assert session is not None


def test_get_session_and_crud(tmp_path, monkeypatch):
    # Prüfe, ob Session funktioniert und CRUD-Befehle möglich sind
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    engine = create_db_engine()
    init_db(engine)

    # get_session liefert einen Generator, also hole dir eine Session
    session = next(get_session(engine))
    assert session is not None

    # Zuerst einen LLMProviderConfig erstellen
    provider_config = LLMProviderConfig(
        prov_name="openai",
        prov_api_key="test-key",
        prov_default_model="gpt-4"
    )
    session.add(provider_config)
    session.commit()
    session.refresh(provider_config)  # Um die automatisch generierte ID zu erhalten

    # Jetzt Workspace mit der korrekten prov_id erstellen
    new_ws = Workspace(work_name="TestWS", work_default_model=str(provider_config.prov_id))
    session.add(new_ws)
    session.commit()

    # Query über die Session
    result = session.exec(select(Workspace).where(Workspace.work_name == "TestWS")).one_or_none()
    assert result is not None
    assert result.work_default_model == str(provider_config.prov_id)