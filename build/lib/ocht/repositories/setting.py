# CRUD functions for Setting
from typing import Optional, Sequence

from sqlmodel import Session, select

from ocht.core.models import Setting


def create_setting(db: Session, key: str, value: str) -> Setting:
    """
    Creates a new setting.

    Args:
        db (Session): The database session.
        key (str): The key of the setting.
        value (str): The value of the setting.

    Returns:
        Setting: The newly created setting object.
    """
    db_setting = Setting(setting_key=key, setting_value=value)
    db.add(db_setting)
    db.commit()
    db.refresh(db_setting)

    return db_setting


def get_setting_by_key(db: Session, key: str) -> Optional[Setting]:
    """
    Retrieves a setting by its key.

    Args:
        db (Session): The database session.
        key (str): The key of the setting.

    Returns:
        Optional[Setting]: The setting object with the specified key or None if not found.
    """
    statement = select(Setting).where(Setting.setting_key == key)
    result = db.exec(statement)
    return result.one_or_none()


def get_all_settings(db: Session, limit: Optional[int] = None, offset: int = 0) -> Sequence[Setting]:
    """
    Retrieves all settings with optional limitation and offset.

    Args:
        db (Session): The database session.
        limit (Optional[int], optional): The maximum number of settings to return. Default is None.
        offset (int, optional): The offset for the query. Default is 0.

    Returns:
        Sequence[Setting]: A list of setting objects.
    """
    if limit is not None and limit < 0:
        raise ValueError("Limit cannot be negative.")
    if offset < 0:
        raise ValueError("Offset cannot be negative.")

    statement = select(Setting).offset(offset)
    if limit is not None:
        statement = statement.limit(limit)

    settings = db.exec(statement).all()
    return settings


def update_setting(db: Session, setting_key: str, new_key: Optional[str] = None, value: Optional[str] = None) -> Optional[Setting]:
    """
    Updates an existing setting.

    Args:
        db (Session): The database session.
        setting_key (str): The key of the setting to be updated.
        new_key (Optional[str]): New key for the setting. Default is None.
        value (Optional[str]): New value for the setting. Default is None.

    Returns:
        Optional[Setting]: The updated setting object or None if not found.
    """
    db_setting = get_setting_by_key(db, setting_key)
    if not db_setting:
        return None

    if new_key is not None:
        # Create a copy of the current setting with the new key
        new_setting = Setting(setting_key=new_key, setting_value=db_setting.setting_value,
                             setting_workspace_id=db_setting.setting_workspace_id)
        db.add(new_setting)

        # Delete old setting
        db.delete(db_setting)
    elif value is not None:
        db_setting.setting_value = value

    db.commit()
    if new_key is not None:
        return new_setting
    else:
        db.refresh(db_setting)
        return db_setting


def delete_setting(db: Session, setting_key: str) -> bool:
    """
    Deletes a setting.

    Args:
        db (Session): The database session.
        setting_key (str): The key of the setting to be deleted.

    Returns:
        bool: True if the deletion was successful, False otherwise.
    """
    db_setting = get_setting_by_key(db, setting_key)
    if not db_setting:
        return False

    db.delete(db_setting)
    db.commit()

    return True
