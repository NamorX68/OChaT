import pytest
from unittest import mock
from sqlalchemy.orm import Session
from ocht.repositories.prompt_template import (
    create_prompt_template,
    get_prompt_template_by_id,
    get_all_prompt_templates,
    update_prompt_template,
    delete_prompt_template
)
from ocht.core.models import PromptTemplate

@pytest.fixture
def mock_db():
    return mock.create_autospec(Session)

def test_create_prompt_template(mock_db):
    # Arrange
    db = mock_db
    name = "test-template"
    text = "This is a test template"
    description = "Test description"

    # Act
    result = create_prompt_template(db, name, text, description)

    # Assert
    db.add.assert_called_once()
    db.commit.assert_called_once()
    db.refresh.assert_called_once()
    assert result.templ_name == name
    assert result.templ_text == text
    assert result.templ_description == description

def test_get_prompt_template_by_id_found(mock_db):
    # Arrange
    db = mock_db
    template_id = 1
    mock_template = mock.create_autospec(PromptTemplate)
    db.exec.return_value.first.return_value = mock_template

    # Act
    result = get_prompt_template_by_id(db, template_id)

    # Assert
    assert result == mock_template

def test_get_prompt_template_by_id_not_found(mock_db):
    # Arrange
    db = mock_db
    template_id = 999
    db.exec.return_value.first.return_value = None

    # Act
    result = get_prompt_template_by_id(db, template_id)

    # Assert
    assert result is None

def test_get_all_prompt_templates(mock_db):
    # Arrange
    db = mock_db
    mock_templates = [mock.create_autospec(PromptTemplate) for _ in range(3)]
    db.exec.return_value.all.return_value = mock_templates

    # Act
    result = get_all_prompt_templates(db)

    # Assert
    assert len(result) == 3
    assert all(isinstance(m, PromptTemplate) for m in result)

def test_update_prompt_template(mock_db):
    # Arrange
    db = mock_db
    template_id = 1
    new_name = "updated-name"
    new_description = "Updated description"
    new_text = "Updated template text"
    mock_template = mock.create_autospec(PromptTemplate)
    db.exec.return_value.first.return_value = mock_template

    # Act
    result = update_prompt_template(db, template_id, name=new_name, description=new_description, text=new_text)

    # Assert
    assert result.templ_name == new_name
    assert result.templ_description == new_description
    assert result.templ_text == new_text
    db.commit.assert_called_once()
    db.refresh.assert_called_once()

def test_delete_prompt_template(mock_db):
    # Arrange
    db = mock_db
    template_id = 1
    mock_template = mock.create_autospec(PromptTemplate)
    db.exec.return_value.first.return_value = mock_template

    # Act
    result = delete_prompt_template(db, template_id)

    # Assert
    assert result is True
    db.delete.assert_called_once()
    db.commit.assert_called_once()

def test_delete_prompt_template_not_found(mock_db):
    # Arrange
    db = mock_db
    template_id = 999
    db.exec.return_value.first.return_value = None

    # Act
    result = delete_prompt_template(db, template_id)

    # Assert
    assert result is False