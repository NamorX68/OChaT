import os
from pathlib import Path
from typing import Generator

from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

# Default database path as fallback
DEFAULT_DB_PATH = "data/ocht.db"


def get_database_url() -> str:
    """
    Returns the database URL.
    First checks the DATABASE_URL environment variable.
    If not available, uses the default path.
    """
    # Check for environment variable
    database_url = os.environ.get("DATABASE_URL")

    if database_url:
        return database_url

    # Fallback: use default path
    data_dir = Path(DEFAULT_DB_PATH).parent
    data_dir.mkdir(exist_ok=True, parents=True)

    return f"sqlite:///{DEFAULT_DB_PATH}"


def create_db_engine() -> Engine:
    """
    Creates and configures the database engine.
    
    Returns:
        A SQLAlchemy engine instance configured for SQLite.
    """
    database_url = get_database_url()
    return create_engine(
        database_url,
        echo=False,
        connect_args={"check_same_thread": False}
    )


def init_db(engine: Engine = None) -> None:
    """
    Initializes the database by creating all tables.
    
    Args:
        engine: Optional, the engine to use.
               If not provided, create_db_engine() will be called.
    """
    if engine is None:
        engine = create_db_engine()
    SQLModel.metadata.create_all(engine)


def get_session(engine: Engine = None) -> Generator[Session, None, None]:
    """
    Creates a new database session.
    Recommended to use as a context manager.
    
    Args:
        engine: Optional, the engine to use.
               If not provided, create_db_engine() will be called.
    
    Example:
        with get_session() as session:
            # Perform database operations
    
    Yields:
        A SQLModel Session object.
    """
    if engine is None:
        engine = create_db_engine()

    with Session(engine) as session:
        yield session
