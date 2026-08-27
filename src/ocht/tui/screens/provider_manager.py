"""TUI screens for creating, editing, and managing LLM provider configurations."""
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, Static

from ocht.core.models import LLMProviderConfig
from ocht.services.provider_manager import (
    create_provider_with_validation,
    delete_provider_with_checks,
    get_providers_with_info,
    update_provider_with_validation,
)


class ProviderEditScreen(ModalScreen):
    """Modal screen for editing/creating providers."""

    CSS_PATH = "../styles/provider_edit.tcss"

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "save", "Save"),
    ]

    def __init__(self, provider: LLMProviderConfig | None = None, **kwargs):
        """Initializes the screen in create mode, or edit mode if a provider is given.

        Args:
            provider: Existing provider to edit, or None to create a new provider.
            **kwargs: Additional keyword arguments forwarded to `ModalScreen`.
        """
        super().__init__(**kwargs)
        self.provider = provider
        self.is_edit_mode = provider is not None

    def compose(self):
        """Build the modal form for creating or editing a provider."""
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
                    value=(
                        self.provider.prov_default_model if self.provider and self.provider.prov_default_model else ""
                    ),
                    placeholder="Optional: Default model name",
                    id="provider-default-model"
                ),
                classes="form-row"
            ),
            Horizontal(
                Label("Routing Params:", classes="form-label"),
                Input(
                    value=self.provider.prov_params if self.provider and self.provider.prov_params else "",
                    placeholder='Optional JSON, e.g. {"quantizations": ["fp8"], "preferred_min_throughput": 40}',
                    id="provider-params"
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
        """Dispatch save/cancel button presses to their respective actions."""
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
        # Not `or None` on purpose: the field always reflects the intended final value, and an
        # explicit "" is how update_provider_with_validation() knows to clear an existing value
        # rather than leaving it untouched (see its docstring).
        params = self.query_one("#provider-params", Input).value.strip()

        if not name:
            self.notify("Provider name is required", severity="error")
            return

        if not api_key:
            self.notify("API key is required", severity="error")
            return

        try:
            if self.is_edit_mode:
                # Update existing provider using service function
                updated_provider = update_provider_with_validation(
                    self.provider.prov_id,
                    name=name,
                    api_key=api_key,
                    endpoint=endpoint,
                    default_model=default_model,
                    params=params
                )
                if updated_provider:
                    self.dismiss(updated_provider)
                else:
                    self.notify("Failed to update provider", severity="error")
            else:
                # Create new provider using service function
                new_provider = create_provider_with_validation(
                    name,
                    api_key=api_key,
                    endpoint=endpoint,
                    default_model=default_model,
                    params=params or None
                )
                self.dismiss(new_provider)
        except ValueError as e:
            self.notify(str(e), severity="error")
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
        """Initializes the screen with an empty provider list.

        Args:
            **kwargs: Additional keyword arguments forwarded to `Screen`.
        """
        super().__init__(**kwargs)
        self.providers: list[LLMProviderConfig] = []

    def compose(self):
        """Compose the provider manager screen."""
        yield Header(show_clock=True)
        yield Vertical(
            Static(
                "Provider Management - Use Ctrl+N to add, Ctrl+E to edit, Ctrl+D to delete, ESC to go back",
                classes="help-text",
            ),
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
            # Get providers with info using service function
            provider_data = get_providers_with_info()

            # Extract providers for internal use
            self.providers = [data['provider'] for data in provider_data]

            table = self.query_one("#provider-table", DataTable)
            table.clear()

            for data in provider_data:
                provider = data['provider']
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
            if delete_provider_with_checks(selected_provider.prov_id):
                self.load_providers()
                self.notify(f"Provider '{selected_provider.prov_name}' deleted successfully", severity="information")
            else:
                self.notify("Failed to delete provider", severity="error")
        except ValueError as e:
            self.notify(str(e), severity="error")
        except Exception as e:
            self.notify(f"Error deleting provider: {str(e)}", severity="error")
