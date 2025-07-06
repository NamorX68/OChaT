from textual.widgets import Static, DataTable, Button, Input, Label, Header, Footer
from textual.containers import Vertical, Horizontal
from textual.screen import Screen, ModalScreen
from textual.binding import Binding
from typing import List, Optional
from ocht.core.models import LLMProviderConfig
from ocht.core.db import get_session
from ocht.repositories.llm_provider_config import (
    get_all_llm_provider_configs,
    create_llm_provider_config,
    update_llm_provider_config,
    delete_llm_provider_config
)


class ProviderEditScreen(ModalScreen):
    """Modal screen for editing/creating providers."""

    CSS_PATH = "../styles/provider_edit.tcss"

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "save", "Save"),
    ]

    def __init__(self, provider: Optional[LLMProviderConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.provider = provider
        self.is_edit_mode = provider is not None

    def compose(self):
        title = "Edit Provider" if self.is_edit_mode else "Create New Provider"
        yield Vertical(
            Static(f"🔧 {title}", classes="modal-title"),
            Horizontal(
                Label("Name:", classes="form-label"),
                Input(
                    value=self.provider.prov_name if self.provider else "",
                    placeholder="Provider name (e.g., 'OpenAI', 'Ollama')",
                    id="provider-name"
                ),
                classes="form-row"
            ),
            Horizontal(
                Label("API Key:", classes="form-label"),
                Input(
                    value=self.provider.prov_api_key if self.provider else "",
                    placeholder="API key or credentials",
                    password=True,
                    id="provider-api-key"
                ),
                classes="form-row"
            ),
            Horizontal(
                Label("Endpoint:", classes="form-label"),
                Input(
                    value=self.provider.prov_endpoint if self.provider and self.provider.prov_endpoint else "",
                    placeholder="Optional: Custom endpoint URL",
                    id="provider-endpoint"
                ),
                classes="form-row"
            ),
            Horizontal(
                Label("Default Model:", classes="form-label"),
                Input(
                    value=self.provider.prov_default_model if self.provider and self.provider.prov_default_model else "",
                    placeholder="Optional: Default model name",
                    id="provider-default-model"
                ),
                classes="form-row"
            ),
            Horizontal(
                Button("Save", variant="primary", id="save-btn"),
                Button("Cancel", variant="default", id="cancel-btn"),
                classes="button-row"
            ),
            classes="provider-edit-modal"
        )

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "save-btn":
            self.action_save()

    def action_cancel(self):
        """Cancel provider editing."""
        self.dismiss(None)

    def action_save(self):
        """Save the provider."""
        self.save_provider()

    def save_provider(self):
        """Save the provider data."""
        name = self.query_one("#provider-name", Input).value.strip()
        api_key = self.query_one("#provider-api-key", Input).value.strip()
        endpoint = self.query_one("#provider-endpoint", Input).value.strip() or None
        default_model = self.query_one("#provider-default-model", Input).value.strip() or None

        if not name:
            self.notify("Provider name is required", severity="error")
            return

        if not api_key:
            self.notify("API key is required", severity="error")
            return

        try:
            for session in get_session():
                if self.is_edit_mode:
                    # Update existing provider using repository function
                    updated_provider = update_llm_provider_config(
                        session, 
                        self.provider.prov_id, 
                        name=name, 
                        api_key=api_key,
                        endpoint=endpoint,
                        default_model=default_model
                    )
                    if updated_provider:
                        self.dismiss(updated_provider)
                    else:
                        self.notify("Failed to update provider", severity="error")
                else:
                    # Create new provider using repository function
                    new_provider = create_llm_provider_config(
                        session, 
                        name, 
                        api_key, 
                        endpoint=endpoint, 
                        default_model=default_model
                    )
                    self.dismiss(new_provider)
        except Exception as e:
            self.notify(f"Error saving provider: {str(e)}", severity="error")


class ProviderManagerScreen(Screen):
    """Screen for managing providers."""

    CSS_PATH = "../styles/provider_manager.tcss"

    BINDINGS = [
        ("escape", "back", "Back to Chat"),
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+n", "add_provider", "Add Provider"),
        ("ctrl+e", "edit_provider", "Edit Provider"),
        ("ctrl+d", "delete_provider", "Delete Provider"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.providers: List[LLMProviderConfig] = []

    def compose(self):
        """Compose the provider manager screen."""
        yield Header(show_clock=True)
        yield Vertical(
            Static("Provider Management - Use Ctrl+N to add, Ctrl+E to edit, Ctrl+D to delete, ESC to go back", classes="help-text"),
            DataTable(id="provider-table"),
            Horizontal(
                Button("➕ Add Provider", variant="primary", id="add-provider-btn"),
                Button("✏️ Edit", variant="default", id="edit-provider-btn"),
                Button("🗑️ Delete", variant="error", id="delete-provider-btn"),
                Button("🔄 Refresh", variant="default", id="refresh-btn"),
                classes="provider-toolbar"
            ),
            classes="provider-manager-screen"
        )
        yield Footer()

    def on_mount(self):
        """Initialize the provider table when screen is mounted."""
        self.setup_table()
        self.load_providers()
        # Set focus on the table after mounting
        self.query_one("#provider-table", DataTable).focus()

    def setup_table(self):
        """Setup the data table columns."""
        table = self.query_one("#provider-table", DataTable)
        table.add_columns("ID", "Name", "Endpoint", "Default Model", "Created")

    def load_providers(self):
        """Load providers from database and populate the table."""
        try:
            for session in get_session():
                self.providers = get_all_llm_provider_configs(session)

            table = self.query_one("#provider-table", DataTable)
            table.clear()

            for provider in self.providers:
                table.add_row(
                    str(provider.prov_id),
                    provider.prov_name,
                    provider.prov_endpoint or "Default",
                    provider.prov_default_model or "None",
                    provider.prov_created_at.strftime("%Y-%m-%d %H:%M")
                )
        except Exception as e:
            self.notify(f"Error loading providers: {str(e)}", severity="error")

    def on_button_pressed(self, event: Button.Pressed):
        """Handle button presses."""
        if event.button.id == "add-provider-btn":
            self.add_provider()
        elif event.button.id == "edit-provider-btn":
            self.edit_provider()
        elif event.button.id == "delete-provider-btn":
            self.delete_provider()
        elif event.button.id == "refresh-btn":
            self.load_providers()

    def action_back(self):
        """Go back to the main chat screen."""
        self.app.pop_screen()

    def action_add_provider(self):
        """Add a new provider."""
        self.add_provider()

    def action_edit_provider(self):
        """Edit selected provider."""
        self.edit_provider()

    def action_delete_provider(self):
        """Delete selected provider."""
        self.delete_provider()

    def add_provider(self):
        """Show modal to add new provider."""
        def handle_result(result):
            if result:
                self.load_providers()
                self.notify(f"Provider '{result.prov_name}' created successfully", severity="information")

        self.app.push_screen(ProviderEditScreen(), handle_result)

    def edit_provider(self):
        """Show modal to edit selected provider."""
        table = self.query_one("#provider-table", DataTable)
        if table.cursor_row is None:
            self.notify("Please select a provider to edit", severity="warning")
            return

        selected_provider = self.providers[table.cursor_row]

        def handle_result(result):
            if result:
                self.load_providers()
                self.notify(f"Provider '{result.prov_name}' updated successfully", severity="information")

        self.app.push_screen(ProviderEditScreen(selected_provider), handle_result)

    def delete_provider(self):
        """Delete selected provider."""
        table = self.query_one("#provider-table", DataTable)
        if table.cursor_row is None:
            self.notify("Please select a provider to delete", severity="warning")
            return

        selected_provider = self.providers[table.cursor_row]

        # Simple confirmation - in a real app you might want a proper confirmation dialog
        try:
            for session in get_session():
                if delete_llm_provider_config(session, selected_provider.prov_id):
                    self.load_providers()
                    self.notify(f"Provider '{selected_provider.prov_name}' deleted successfully", severity="information")
                else:
                    self.notify("Failed to delete provider", severity="error")
        except Exception as e:
            self.notify(f"Error deleting provider: {str(e)}", severity="error")
