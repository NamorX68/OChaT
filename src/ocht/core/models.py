from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field


class Workspace(SQLModel, table=True):
    """
    Represents a chat workspace.

    Attributes:
        work_id (Optional[int]): Primary key of the workspace.
        work_name (str): Name of the workspace.
        work_default_model (str): Default model for new chats, references Model.name.
        work_created_at (datetime): Creation timestamp.
        work_updated_at (datetime): Last update timestamp.
        work_description (Optional[str]): Optional description about the workspace.
    """
    work_id: Optional[int] = Field(default=None, primary_key=True)
    work_name: str
    work_default_model: str = Field(foreign_key="llmproviderconfig.prov_id")
    work_created_at: datetime = Field(default_factory=datetime.now)
    work_updated_at: datetime = Field(default_factory=datetime.now)
    work_description: Optional[str] = None


class Message(SQLModel, table=True):
    """
    Represents a message in a chat workspace.

    Attributes:
        msg_id (Optional[int]): Primary key of the message.
        msg_workspace_id (int): Foreign key linking to Workspace.work_id.
        msg_role (str): Role of the message (e.g., 'user', 'assistant', 'system').
        msg_content (str): The text content of the message.
        msg_created_at (datetime): Creation timestamp.
        msg_updated_at (Optional[datetime]): Timestamp of last update, if edited.
        msg_parent_id (Optional[int]): Parent message ID for threaded replies.
        msg_token_count (Optional[int]): Token count of the message.
        msg_metadata (Optional[str]): Additional metadata stored as JSON string.
    """
    msg_id: Optional[int] = Field(default=None, primary_key=True)
    msg_workspace_id: int = Field(foreign_key="workspace.work_id")
    msg_role: str
    msg_content: str
    msg_created_at: datetime = Field(default_factory=datetime.now)
    msg_updated_at: Optional[datetime] = None
    msg_parent_id: Optional[int] = Field(default=None, foreign_key="message.msg_id")
    msg_token_count: Optional[int] = None
    msg_metadata: Optional[str] = None


class LLMProviderConfig(SQLModel, table=True):
    """
    Represents a configuration for an LLM provider.

    Attributes:
        prov_id (Optional[int]): Primary key of the configuration.
        prov_name (str): Name of the provider (e.g., 'openai', 'ollama', 'claude').
        prov_api_key (str): API key or credentials for accessing the LLM service.
        prov_endpoint (Optional[str]): URL or host for the API endpoint, if different from default.
        prov_default_model (Optional[str]): Model name, if this provider offers multiple models.
        prov_params (Optional[str]): JSON string for additional parameters (temperature, top-p, max-tokens, etc.).
        prov_created_at (datetime): Creation timestamp.
        prov_updated_at (datetime): Last update timestamp.
    """
    prov_id: Optional[int] = Field(default=None, primary_key=True)
    prov_name: str
    prov_api_key: str
    prov_endpoint: Optional[str] = None
    prov_default_model: Optional[str] = None
    prov_params: Optional[str] = None
    prov_created_at: datetime = Field(default_factory=datetime.now)
    prov_updated_at: datetime = Field(default_factory=datetime.now)


class Model(SQLModel, table=True):
    """
    Represents an available LLM model.

    Attributes:
        model_name (str): Primary key of the model (e.g., 'gpt-4').
        model_provider_id (int): Foreign key linking to LLMProviderConfig.prov_id.
        model_description (Optional[str]): Optional description of the model.
        model_version (Optional[str]): Version identifier of the model.
        model_created_at (datetime): Timestamp when the model entry was created.
        model_updated_at (datetime): Timestamp when the model entry was last updated.
        model_params (Optional[str]): JSON string with default parameters (e.g., temperature).
    """
    model_name: str = Field(primary_key=True)
    model_provider_id: int = Field(foreign_key="llmproviderconfig.prov_id")
    model_description: Optional[str] = None
    model_version: Optional[str] = None
    model_created_at: datetime = Field(default_factory=datetime.now)
    model_updated_at: datetime = Field(default_factory=datetime.now)
    model_params: Optional[str] = None


class Setting(SQLModel, table=True):
    """
    Represents a general key-value setting.

    Attributes:
        setting_key (str): Primary key name of the setting.
        setting_value (str): Value of the setting, stored as a string (use JSON if needed).
        setting_workspace_id (Optional[int]): Foreign key linking to Workspace.work_id, for workspace-specific settings.
        setting_created_at (datetime): Timestamp when the setting was created.
        setting_updated_at (datetime): Timestamp when the setting was last updated.
    """
    setting_key: str = Field(primary_key=True)
    setting_value: str
    setting_workspace_id: Optional[int] = Field(default=None, foreign_key="workspace.work_id")
    setting_created_at: datetime = Field(default_factory=datetime.now)
    setting_updated_at: datetime = Field(default_factory=datetime.now)


class PromptTemplate(SQLModel, table=True):
    """
    Represents a reusable prompt template.

    Attributes:
        templ_id (Optional[int]): Primary key of the template.
        templ_name (str): Unique name of the template.
        templ_description (Optional[str]): Short description of the template.
        templ_text (str): The prompt text, potentially containing placeholders.
        templ_created_at (datetime): Timestamp when the template was created.
        templ_updated_at (datetime): Timestamp when the template was last updated.
    """
    templ_id: Optional[int] = Field(default=None, primary_key=True)
    templ_name: str
    templ_description: Optional[str] = None
    templ_text: str
    templ_created_at: datetime = Field(default_factory=datetime.now)
    templ_updated_at: datetime = Field(default_factory=datetime.now)
