from textual.widgets import Static, ListItem, ListView, Button
from textual.containers import Vertical, Horizontal
from textual.screen import ModalScreen
from textual.binding import Binding
from typing import List, Optional
from ocht.core.models import Model
from ocht.core.db import get_session
from ocht.repositories.model import get_all_models

class ModelSelectorModal(ModalScreen):
    """Modal dialog for selecting models."""

    CSS_PATH = "../styles/model_selector.tcss"

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
            ListView(id="model-list", classes="model-list"),
            Horizontal(
                Button("OK", variant="primary", id="ok-btn"),
                Button("Cancel", variant="default", id="cancel-btn"),
                classes="button-row"
            ),
            classes="model-selector-modal"
        )

    def on_mount(self):
        """Load model when modal is mounted."""
        self.load_model()
        # Set focus on the provider list after mounting
        self.query_one("#model-list", ListView).focus()

    def load_model(self):
        """Load models from database and populate the list."""
        try:
            for session in get_session():
                self.models = get_all_models(session)

            model_list = self.query_one("#model-list", ListView)
            model_list.clear()

            if not self.models:
                model_list.append(ListItem(Static("No models found. Use model management to add models.")))
                return

            for model in self.models:
                item_text = f"🔧 {model.model_name}"
                model_list.append(ListItem(Static(item_text)))

            # Automatically select the first element if models exist
            if self.models:
                model_list.index = 0
                # Also set the selected_model to the first one
                self.selected_model = self.models[0]
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
        if self.selected_model:
            self.dismiss(self.selected_model)
        else:
            model_list = self.query_one("#model-list", ListView)
            if model_list.index is not None and 0 <= model_list.index < len(self.models):
                selected_model = self.models[model_list.index]
                self.dismiss(selected_model)
            else:
                self.notify("Please select a model first", severity="warning")