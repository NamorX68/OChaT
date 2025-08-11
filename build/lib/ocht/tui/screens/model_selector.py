from textual.widgets import Static, ListItem, ListView, Button
from textual.containers import Vertical, Horizontal
from textual.screen import ModalScreen
from textual.binding import Binding
from typing import List, Optional
from ocht.core.models import Model
from ocht.services.model_manager import list_llm_models
from ocht.services.model_manager import restore_model
from ocht.tui.widgets.confirmation_dialog import ConfirmationDialog

class ModelSelectorModal(ModalScreen):
    """Modal dialog for selecting models."""

    CSS_PATH = "../styles/selector_modal.tcss"

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "select", "Select"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.models: List[Model] = []
        self.selected_model: Optional[Model] = None

    def compose(self):
        """Compose the model selector modal."""
        yield Vertical(
            Static("🔧 Select a model", classes="modal-title"),
            ListView(id="model-list", classes="selector-list"),
            Horizontal(
                Button("OK", variant="primary", id="ok-btn"),
                Button("Cancel", variant="default", id="cancel-btn"),
                classes="button-row"
            ),
            classes="selector-modal"
        )

    def on_mount(self):
        """Load models when modal is mounted."""
        try:
            self.load_models()
            # Set focus on the model list after mounting
            self.query_one("#model-list", ListView).focus()
        except Exception as e:
            self.notify(f"Error loading models: {e}", severity="error")

    def load_models(self):
        """Load models from database and populate the list."""
        try:
            # Get models using service function
            self.models = list_llm_models()

            model_list = self.query_one("#model-list", ListView)
            model_list.clear()

            if not self.models:
                model_list.append(ListItem(Static("No models found. Use model management to add models.")))
                return

            for model in self.models:
                if model.is_available:
                    item_text = f"🔧 {model.model_name}"
                else:
                    item_text = f"❌ {model.model_name} (nicht verfügbar)"
                model_list.append(ListItem(Static(item_text)))

            # Automatically select the first AVAILABLE model if models exist
            if self.models:
                # Find first available model
                first_available_index = None
                for i, model in enumerate(self.models):
                    if model.is_available:
                        first_available_index = i
                        break
                
                if first_available_index is not None:
                    model_list.index = first_available_index
                    self.selected_model = self.models[first_available_index]
                else:
                    # No available models, select first one but don't set selected_model
                    model_list.index = 0
                    self.selected_model = None

        except Exception as e:
            model_list = self.query_one("#model-list", ListView)
            model_list.clear()
            model_list.append(ListItem(Static(f"❌ Error loading models: {str(e)}")))

    def on_list_view_selected(self, event: ListView.Selected):
        """Handle model selection."""
        if event.list_view.id == "model-list" and self.models:
            selected_index = event.list_view.index
            if 0 <= selected_index < len(self.models):
                self.selected_model = self.models[selected_index]

    def on_key(self, event):
        """Handle key events, specifically Enter key when ListView has focus."""
        if event.key == "enter":
            model_list = self.query_one("#model-list", ListView)
            if model_list.has_focus:
                self.action_select()
                event.prevent_default()
                event.stop()  # Stop event propagation completely
                return

    def on_button_pressed(self, event: Button.Pressed):
        """Handle button presses."""
        if event.button.id  == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "ok-btn":
            self.action_select()

    def action_cancel(self):
        """Cancel model selection."""
        self.dismiss(None)

    def action_select(self):
        """Select the currently highlighted model."""
        model_list = self.query_one("#model-list", ListView)
        current_index = model_list.index
        
        if current_index is not None and 0 <= current_index < len(self.models):
            selected_model = self.models[current_index]
            
            if not selected_model.is_available:
                self._download_and_select_model(selected_model)
                return
            
            self.dismiss(selected_model)
        else:
            self.notify("Please select a model first", severity="warning")

    def _download_and_select_model(self, model: Model):
        """Download unavailable model and select it."""
        # Show confirmation dialog first
        def show_confirmation():
            dialog = ConfirmationDialog(
                title="Model herunterladen",
                message=f"Das Model '{model.model_name}' ist nicht verfügbar.\n\nMöchten Sie es jetzt von Ollama herunterladen?\n\nDies kann einige Minuten dauern.",
                confirm_text="Ja, herunterladen",
                cancel_text="Abbrechen",
                confirm_variant="primary"
            )
            self.app.push_screen(dialog, self._handle_download_confirmation)
        
        show_confirmation()

    def _handle_download_confirmation(self, confirmed: bool):
        """Handle the result of the download confirmation dialog."""
        if not confirmed:
            return
            
        # Get the currently selected model for download
        model = self.selected_model
        if not model:
            model_list = self.query_one("#model-list", ListView)
            if model_list.index is not None and 0 <= model_list.index < len(self.models):
                model = self.models[model_list.index]
            else:
                return

        try:
            self.notify(f"Lade Model '{model.model_name}' herunter...", severity="information")
            result = restore_model(model.model_name)
            
            if result['success']:
                self.notify(f"Model '{model.model_name}' erfolgreich heruntergeladen!", severity="information")
                # Refresh the model list to update availability status
                self.load_models()
                self.dismiss(model)
            else:
                self.notify(f"Fehler beim Herunterladen: {result.get('message', 'Unbekannter Fehler')}", severity="error")
        except Exception as e:
            self.notify(f"Fehler beim Herunterladen von '{model.model_name}': {str(e)}", severity="error")