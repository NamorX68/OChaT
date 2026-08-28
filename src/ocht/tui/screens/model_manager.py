"""TUI screens for creating, editing, and managing LLM models."""
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, Select, Static

from ocht.core.models import LLMProviderConfig, Model
from ocht.services.health_check import run_health_check
from ocht.services.model_manager import (
    create_model_with_validation,
    delete_model_with_checks,
    get_models_with_provider_info,
    split_model_params_for_editing,
    update_model_with_validation,
)
from ocht.services.provider_manager import get_available_providers


class ModelEditScreen(ModalScreen):
    """Modal screen for editing/creating models."""

    CSS_PATH = "../styles/model_edit.tcss"

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "save", "Save"),
    ]

    def __init__(self, model: Model | None = None, **kwargs):
        """Initializes the screen in create mode, or edit mode if a model is given.

        Args:
            model: Existing model to edit, or None to create a new model.
            **kwargs: Additional keyword arguments forwarded to `ModalScreen`.
        """
        super().__init__(**kwargs)
        self.model = model
        self.is_edit_mode = model is not None
        self.providers: list[LLMProviderConfig] = []

    def _provider_name(self, provider_id: int | None) -> str | None:
        """Looks up a loaded provider's name by ID.

        Args:
            provider_id: The provider ID to look up, or None.

        Returns:
            The provider's `prov_name`, or None if not found/not given.
        """
        return next((p.prov_name for p in self.providers if p.prov_id == provider_id), None)

    def compose(self):
        """Build the modal form for creating or editing a model."""
        title = "Edit Model" if self.is_edit_mode else "Create New Model"

        # Load available providers for model assignment
        try:
            self.providers = get_available_providers()
        except Exception:
            self.providers = []

        # Create provider selection options for model assignment
        provider_options = [
            (f"{provider.prov_name} (ID: {provider.prov_id})", provider.prov_id) for provider in self.providers
        ]
        initial_provider_id = (
            self.model.model_provider_id if self.model else (provider_options[0][1] if provider_options else None)
        )
        initial_provider_name = self._provider_name(initial_provider_id) or ""

        # Split the model's existing model_params JSON into the typed fields plus whatever's left
        # over (e.g. temperature/top_p/top_k) - see split_model_params_for_editing()'s docstring.
        max_tokens, context_window, advanced_params = split_model_params_for_editing(
            self.model.model_params if self.model else None, initial_provider_name
        )
        context_window_is_ollama = initial_provider_name.lower() == "ollama"

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
                    value=initial_provider_id,
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
                Label("Max Tokens:", classes="form-label"),
                Input(
                    value=str(max_tokens) if max_tokens is not None else "",
                    placeholder="Optional: max response length",
                    id="model-max-tokens"
                ),
                classes="form-row"
            ),
            Horizontal(
                Label("Context Window:", classes="form-label"),
                Input(
                    value=str(context_window) if context_window is not None else "",
                    placeholder=(
                        "Optional: Ollama only" if context_window_is_ollama else "Not supported by this provider"
                    ),
                    disabled=not context_window_is_ollama,
                    id="model-context-window"
                ),
                classes="form-row"
            ),
            Horizontal(
                Label("Advanced Params:", classes="form-label"),
                Input(
                    value=advanced_params or "",
                    placeholder="Optional: extra JSON params (e.g., '{\"temperature\": 0.7}')",
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
        """Dispatch save/cancel button presses to their respective actions."""
        if event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "save-btn":
            self.action_save()

    def on_select_changed(self, event: Select.Changed):
        """Enables the Context Window field only while the selected provider is Ollama.

        Args:
            event: The provider `Select` widget's change event.
        """
        if event.select.id != "model-provider":
            return
        is_ollama = (self._provider_name(event.value) or "").lower() == "ollama"
        context_window_input = self.query_one("#model-context-window", Input)
        context_window_input.disabled = not is_ollama
        context_window_input.placeholder = "Optional: Ollama only" if is_ollama else "Not supported by this provider"
        if not is_ollama:
            context_window_input.value = ""

    def action_cancel(self):
        """Cancel model editing."""
        self.dismiss(None)

    def action_save(self):
        """Save the model."""
        self.save_model()

    def _parse_optional_int(self, field_id: str, label: str) -> int | None:
        """Reads an `Input` field as an optional positive integer.

        Args:
            field_id: The widget ID of the `Input` to read.
            label: Human-readable field name used in the error notification.

        Returns:
            The parsed integer, or None if the field is blank.

        Raises:
            ValueError: If the field is non-blank but not a valid positive integer.
        """
        raw = self.query_one(f"#{field_id}", Input).value.strip()
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError as e:
            raise ValueError(f"{label} must be a whole number") from e
        if value <= 0:
            raise ValueError(f"{label} must be a positive number")
        return value

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
            max_tokens = self._parse_optional_int("model-max-tokens", "Max Tokens")
            context_window = self._parse_optional_int("model-context-window", "Context Window")

            if self.is_edit_mode:
                # Update existing model using service function
                updated_model = update_model_with_validation(
                    self.model.model_name,
                    new_name=name,
                    provider_id=selected_provider_id,
                    description=description,
                    version=version,
                    params=params,
                    max_tokens=max_tokens,
                    context_window=context_window
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
                    params=params,
                    max_tokens=max_tokens,
                    context_window=context_window
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
        ("ctrl+h", "health_check", "Health Check"),
    ]

    def __init__(self, **kwargs):
        """Initializes the screen with empty model and provider lists.

        Args:
            **kwargs: Additional keyword arguments forwarded to `Screen`.
        """
        super().__init__(**kwargs)
        self.models: list[Model] = []
        self.providers: list[LLMProviderConfig] = []

    def compose(self):
        """Compose the model manager screen."""
        yield Header(show_clock=True)
        yield Vertical(
            Static(
                "Model Management - Use Ctrl+N to add, Ctrl+E to edit, Ctrl+D to delete, "
                "Ctrl+H to health-check, ESC to go back",
                classes="help-text",
            ),
            DataTable(id="model-table"),
            Horizontal(
                Button("➕ Add Model", variant="primary", id="add-model-btn"),
                Button("✏️ Edit", variant="default", id="edit-model-btn"),
                Button("🗑️ Delete", variant="error", id="delete-model-btn"),
                Button("💓 Health Check", variant="default", id="health-check-btn"),
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
        table.add_columns("Name", "Model Provider", "Description", "Version", "Params", "Health", "Created")

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
                    self._format_params_preview(model.model_params),
                    self._format_health_preview(model),
                    model.model_created_at.strftime("%Y-%m-%d %H:%M")
                )
        except Exception as e:
            self.notify(f"Error loading models: {str(e)}", severity="error")

    def _format_health_preview(self, model: Model, max_length: int = 40) -> str:
        """Formats a model's last health check result for the overview table.

        Mirrors `_format_params_preview()`'s truncation style, so the table stays scannable even
        for long error messages.

        Args:
            model: The model whose `last_checked`/`last_check_*` fields to summarize.
            max_length: Maximum number of characters to show before truncating with an ellipsis.

        Returns:
            "— Not checked" if no health check has ever run, "❌ <truncated error>" if the last
            one failed, otherwise "✅" plus whichever of latency/tokens-per-second are available.
        """
        if model.last_checked is None:
            return "— Not checked"

        if model.last_check_error:
            prefix = "❌ "
            remaining = max_length - len(prefix)
            error = model.last_check_error
            if len(error) > remaining:
                error = error[:remaining - 1] + "…"
            return prefix + error

        parts = []
        if model.last_check_latency_ms is not None:
            parts.append(f"{model.last_check_latency_ms:.0f}ms")
        if model.last_check_tokens_per_second is not None:
            parts.append(f"{model.last_check_tokens_per_second:.1f} tok/s")
        return "✅ " + " / ".join(parts) if parts else "✅"

    def _format_params_preview(self, params: str | None, max_length: int = 40) -> str:
        """Formats model_params for the overview table, truncating long JSON for readability.

        Mirrors `ProviderManagerScreen._format_params_preview()`, which does the same for
        `LLMProviderConfig.prov_params` - both fields only surfaced their raw JSON in the edit
        form before this, with no way to see at a glance whether a row even had params set.

        Args:
            params: The raw JSON string of model generation parameters, or None if unset.
            max_length: Maximum number of characters to show before truncating with an ellipsis.

        Returns:
            "None" if unset, otherwise the JSON string truncated to `max_length` characters.
        """
        if not params:
            return "None"
        if len(params) <= max_length:
            return params
        return params[:max_length - 1] + "…"

    def on_button_pressed(self, event: Button.Pressed):
        """Handle button presses."""
        if event.button.id == "add-model-btn":
            self.add_model()
        elif event.button.id == "edit-model-btn":
            self.edit_model()
        elif event.button.id == "delete-model-btn":
            self.delete_model()
        elif event.button.id == "health-check-btn":
            self.run_worker(self.health_check_selected())
        elif event.button.id == "refresh-btn":
            self.load_models()

    def action_back(self):
        """Go back to the main chat screen."""
        self.app.pop_screen()

    def action_add_model(self):
        """Add a new model."""
        self.add_model()

    def action_health_check(self):
        """Health-check the selected model (Ctrl+H binding)."""
        self.run_worker(self.health_check_selected())

    async def health_check_selected(self):
        """Runs a real completion against the selected model and refreshes its row.

        Builds its own throwaway adapter (via `run_health_check()` -> `build_adapter()`) rather
        than touching the active chat adapter - selecting a row here and health-checking it never
        switches what the user is actually chatting with, see `services/health_check.py`.
        """
        table = self.query_one("#model-table", DataTable)
        if table.cursor_row is None:
            self.notify("Please select a model to check", severity="warning")
            return

        model = self.models[table.cursor_row]
        self.notify(f"Checking {model.model_name}…")

        try:
            result = await run_health_check(model.model_provider_id, model.model_name)
        except Exception as e:
            self.notify(f"Error running health check: {str(e)}", severity="error")
            return

        self.load_models()
        if result.is_available:
            message = f"{model.model_name}: OK via {result.path}"
            if result.tokens_per_second:
                message += f" ({result.tokens_per_second:.1f} tok/s)"
            self.notify(message, severity="information")
        else:
            self.notify(f"{model.model_name}: FAILED - {result.error}", severity="error")

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
