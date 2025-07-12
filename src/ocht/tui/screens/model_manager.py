from textual.widgets import Static, DataTable, Button, Input, Label, Header, Footer, Select
from textual.containers import Vertical, Horizontal
from textual.screen import Screen, ModalScreen
from textual.binding import Binding
from typing import List, Optional
from ocht.core.models import Model, LLMProviderConfig
from ocht.services.model_manager import (
    get_models_with_provider_info,
    create_model_with_validation,
    update_model_with_validation,
    delete_model_with_checks
)
from ocht.services.provider_manager import get_available_providers


class ModelEditScreen(ModalScreen):
    """Modal screen for editing/creating models."""

    CSS_PATH = "../styles/model_edit.tcss"

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "save", "Save"),
    ]

    def __init__(self, model: Optional[Model] = None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.is_edit_mode = model is not None
        self.providers: List[LLMProviderConfig] = []

    def compose(self):
        title = "Edit Model" if self.is_edit_mode else "Create New Model"

        # Load available providers for model assignment
        try:
            self.providers = get_available_providers()
        except Exception as e:
            self.providers = []

        # Create provider selection options for model assignment
        provider_options = [(f"{provider.prov_name} (ID: {provider.prov_id})", provider.prov_id) for provider in self.providers]

        yield Vertical(
            Static(f"🤖 {title}", classes="modal-title"),
            Horizontal(
                Label("Name:", classes="form-label"),
                Input(
                    value=self.model.model_name if self.model else "",
                    placeholder="Model name (e.g., 'gpt-4', 'llama2')",
                    id="model-name"
                ),
                classes="form-row"
            ),
            Horizontal(
                Label("Model Provider:", classes="form-label"),
                Select(
                    options=provider_options,
                    value=self.model.model_provider_id if self.model else (provider_options[0][1] if provider_options else None),
                    id="model-provider"
                ),
                classes="form-row"
            ),
            Horizontal(
                Label("Description:", classes="form-label"),
                Input(
                    value=self.model.model_description if self.model and self.model.model_description else "",
                    placeholder="Optional: Model description",
                    id="model-description"
                ),
                classes="form-row"
            ),
            Horizontal(
                Label("Version:", classes="form-label"),
                Input(
                    value=self.model.model_version if self.model and self.model.model_version else "",
                    placeholder="Optional: Model version",
                    id="model-version"
                ),
                classes="form-row"
            ),
            Horizontal(
                Label("Parameters:", classes="form-label"),
                Input(
                    value=self.model.model_params if self.model and self.model.model_params else "",
                    placeholder="Optional: JSON parameters (e.g., '{\"temperature\": 0.7}')",
                    id="model-params"
                ),
                classes="form-row"
            ),
            Horizontal(
                Button("Save", variant="primary", id="save-btn"),
                Button("Cancel", variant="default", id="cancel-btn"),
                classes="button-row"
            ),
            classes="model-edit-modal"
        )

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "save-btn":
            self.action_save()

    def action_cancel(self):
        """Cancel model editing."""
        self.dismiss(None)

    def action_save(self):
        """Save the model."""
        self.save_model()

    def save_model(self):
        """Save the model data."""
        name = self.query_one("#model-name", Input).value.strip()
        selected_provider_id = self.query_one("#model-provider", Select).value
        description = self.query_one("#model-description", Input).value.strip() or None
        version = self.query_one("#model-version", Input).value.strip() or None
        params = self.query_one("#model-params", Input).value.strip() or None

        if not name:
            self.notify("Model name is required", severity="error")
            return

        if selected_provider_id is None:
            self.notify("Model provider must be selected", severity="error")
            return

        try:
            if self.is_edit_mode:
                # Update existing model using service function
                updated_model = update_model_with_validation(
                    self.model.model_name,
                    new_name=name,
                    provider_id=selected_provider_id,
                    description=description,
                    version=version,
                    params=params
                )
                if updated_model:
                    self.dismiss(updated_model)
                else:
                    self.notify("Failed to update model", severity="error")
            else:
                # Create new model using service function
                new_model = create_model_with_validation(
                    name,
                    selected_provider_id,
                    description=description,
                    version=version,
                    params=params
                )
                self.dismiss(new_model)
        except ValueError as e:
            self.notify(str(e), severity="error")
        except Exception as e:
            self.notify(f"Error saving model: {str(e)}", severity="error")


class ModelManagerScreen(Screen):
    """Screen for managing models."""

    CSS_PATH = "../styles/model_manager.tcss"

    BINDINGS = [
        ("escape", "back", "Back to Chat"),
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+n", "add_model", "Add Model"),
        ("ctrl+e", "edit_model", "Edit Model"),
        ("ctrl+d", "delete_model", "Delete Model"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.models: List[Model] = []
        self.providers: List[LLMProviderConfig] = []

    def compose(self):
        """Compose the model manager screen."""
        yield Header(show_clock=True)
        yield Vertical(
            Static("Model Management - Use Ctrl+N to add, Ctrl+E to edit, Ctrl+D to delete, ESC to go back", classes="help-text"),
            DataTable(id="model-table"),
            Horizontal(
                Button("➕ Add Model", variant="primary", id="add-model-btn"),
                Button("✏️ Edit", variant="default", id="edit-model-btn"),
                Button("🗑️ Delete", variant="error", id="delete-model-btn"),
                Button("🔄 Refresh", variant="default", id="refresh-btn"),
                classes="model-toolbar"
            ),
            classes="model-manager-screen"
        )
        yield Footer()

    def on_mount(self):
        """Initialize the model table when screen is mounted."""
        self.setup_table()
        self.load_models()
        # Set focus on the table after mounting
        self.query_one("#model-table", DataTable).focus()

    def setup_table(self):
        """Setup the data table columns."""
        table = self.query_one("#model-table", DataTable)
        table.add_columns("Name", "Model Provider", "Description", "Version", "Created")

    def load_models(self):
        """Load models from database and populate the table."""
        try:
            # Get models with provider info using service function
            model_data = get_models_with_provider_info()

            # Extract models for internal use
            self.models = [data['model'] for data in model_data]

            table = self.query_one("#model-table", DataTable)
            table.clear()

            for data in model_data:
                model = data['model']
                provider_name = data['provider_name']
                table.add_row(
                    model.model_name,
                    provider_name,
                    model.model_description or "None",
                    model.model_version or "None",
                    model.model_created_at.strftime("%Y-%m-%d %H:%M")
                )
        except Exception as e:
            self.notify(f"Error loading models: {str(e)}", severity="error")

    def on_button_pressed(self, event: Button.Pressed):
        """Handle button presses."""
        if event.button.id == "add-model-btn":
            self.add_model()
        elif event.button.id == "edit-model-btn":
            self.edit_model()
        elif event.button.id == "delete-model-btn":
            self.delete_model()
        elif event.button.id == "refresh-btn":
            self.load_models()

    def action_back(self):
        """Go back to the main chat screen."""
        self.app.pop_screen()

    def action_add_model(self):
        """Add a new model."""
        self.add_model()

    def action_edit_model(self):
        """Edit selected model."""
        self.edit_model()

    def action_delete_model(self):
        """Delete selected model."""
        self.delete_model()

    def add_model(self):
        """Show modal to add new model."""
        def handle_result(result):
            if result:
                self.load_models()
                self.notify(f"Model '{result.model_name}' created successfully", severity="information")

        self.app.push_screen(ModelEditScreen(), handle_result)

    def edit_model(self):
        """Show modal to edit selected model."""
        table = self.query_one("#model-table", DataTable)
        if table.cursor_row is None:
            self.notify("Please select a model to edit", severity="warning")
            return

        selected_model = self.models[table.cursor_row]

        def handle_result(result):
            if result:
                self.load_models()
                self.notify(f"Model '{result.model_name}' updated successfully", severity="information")

        self.app.push_screen(ModelEditScreen(selected_model), handle_result)

    def delete_model(self):
        """Delete selected model."""
        table = self.query_one("#model-table", DataTable)
        if table.cursor_row is None:
            self.notify("Please select a model to delete", severity="warning")
            return

        selected_model = self.models[table.cursor_row]

        # Simple confirmation - in a real app you might want a proper confirmation dialog
        try:
            if delete_model_with_checks(selected_model.model_name):
                self.load_models()
                self.notify(f"Model '{selected_model.model_name}' deleted successfully", severity="information")
            else:
                self.notify("Failed to delete model", severity="error")
        except ValueError as e:
            self.notify(str(e), severity="error")
        except Exception as e:
            self.notify(f"Error deleting model: {str(e)}", severity="error")
