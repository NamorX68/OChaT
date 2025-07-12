from textual.widgets import Static, ListItem, ListView, Button
from textual.containers import Vertical, Horizontal
from textual.screen import ModalScreen
from textual.binding import Binding
from typing import List, Optional
from ocht.core.models import LLMProviderConfig
from ocht.services.provider_manager import get_available_providers


class ProviderSelectorModal(ModalScreen):
    """Modal dialog for selecting LLM providers."""

    CSS_PATH = "../styles/provider_selector.tcss"

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "select", "Select"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.providers: List[LLMProviderConfig] = []
        self.selected_provider: Optional[LLMProviderConfig] = None

    def compose(self):
        """Compose the provider selector modal."""
        yield Vertical(
            Static("🔧 Select a Provider", classes="modal-title"),
            ListView(id="provider-list", classes="provider-list"),
            Horizontal(
                Button("OK", variant="primary", id="ok-btn"),
                Button("Cancel", variant="default", id="cancel-btn"),
                classes="button-row"
            ),
            classes="provider-selector-modal"
        )

    def on_mount(self):
        """Load providers when modal is mounted."""
        self.load_providers()
        # Set focus on the provider list after mounting
        self.query_one("#provider-list", ListView).focus()

    def load_providers(self):
        """Load providers from database and populate the list."""
        try:
            # Get providers using service function
            self.providers = get_available_providers()

            provider_list = self.query_one("#provider-list", ListView)
            provider_list.clear()

            if not self.providers:
                provider_list.append(ListItem(Static("No providers found. Use provider management to add providers.")))
                return

            for provider in self.providers:
                item_text = f"🔧 {provider.prov_name}"
                if provider.prov_default_model:
                    item_text += f" (Default: {provider.prov_default_model})"
                provider_list.append(ListItem(Static(item_text)))

            # Automatically select the first element if providers exist
            if self.providers:
                provider_list.index = 0
                # Also set the selected_provider to the first one
                self.selected_provider = self.providers[0]

        except Exception as e:
            provider_list = self.query_one("#provider-list", ListView)
            provider_list.clear()
            provider_list.append(ListItem(Static(f"❌ Error loading providers: {str(e)}")))

    def on_list_view_selected(self, event: ListView.Selected):
        """Handle provider selection."""
        if event.list_view.id == "provider-list" and self.providers:
            selected_index = event.list_view.index
            if 0 <= selected_index < len(self.providers):
                self.selected_provider = self.providers[selected_index]

    def on_key(self, event):
        """Handle key events, specifically Enter key when ListView has focus."""
        if event.key == "enter":
            # Check if the provider list has focus
            provider_list = self.query_one("#provider-list", ListView)
            if provider_list.has_focus:
                # Trigger the select action when Enter is pressed on ListView
                self.action_select()
                event.prevent_default()
                return
        # Let other keys be handled normally by the default behavior

    def on_button_pressed(self, event: Button.Pressed):
        """Handle button presses."""
        if event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "ok-btn":
            self.action_select()

    def action_cancel(self):
        """Cancel provider selection."""
        self.dismiss(None)

    def action_select(self):
        """Select the currently highlighted provider."""
        if self.selected_provider:
            self.dismiss(self.selected_provider)
        else:
            # If no provider is explicitly selected, try to get the highlighted one
            provider_list = self.query_one("#provider-list", ListView)
            if provider_list.index is not None and 0 <= provider_list.index < len(self.providers):
                selected_provider = self.providers[provider_list.index]
                self.dismiss(selected_provider)
            else:
                self.notify("Please select a provider first", severity="warning")
