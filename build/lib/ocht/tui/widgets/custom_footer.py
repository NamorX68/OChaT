from textual.widgets import Static
from textual.reactive import reactive
from textual.app import ComposeResult
from textual.containers import Horizontal


class CustomFooter(Static):
    """Custom footer widget that displays adapter information alongside keybindings."""
    
    DEFAULT_CSS = """
    CustomFooter {
        dock: bottom;
        height: 1;
        background: $surface-lighten-1;
        color: $text-muted;
    }
    
    CustomFooter Horizontal {
        width: 100%;
        height: 100%;
        align: left middle;
    }
    
    CustomFooter .footer-keys {
        width: 1fr;
        text-align: left;
        content-align: left middle;
        padding: 0 1;
    }
    
    CustomFooter .footer-adapter {
        width: auto;
        text-align: right;
        content-align: right middle;
        padding: 0 1;
        color: $accent;
    }
    """
    
    adapter_info = reactive("No adapter configured")
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compose(self) -> ComposeResult:
        """Compose the footer with keybindings and adapter info."""
        with Horizontal():
            # Keybindings on the left
            yield Static("^C Quit  ^L Clear  ESC Focus", classes="footer-keys")
            # Adapter info on the right
            yield Static(self.adapter_info, id="footer-adapter", classes="footer-adapter")
    
    def on_mount(self) -> None:
        """Update keybindings when mounted."""
        self._update_keybindings()
    
    def _update_keybindings(self) -> None:
        """Update the keybindings display using app's bindings."""
        try:
            # Get keybinding text from the app's bindings
            binding_text = ""
            if hasattr(self.app, "BINDINGS"):
                bindings = []
                for binding in self.app.BINDINGS:
                    if len(binding) >= 3:
                        key, action, description = binding[0], binding[1], binding[2]
                        if key == "ctrl+c":
                            bindings.append("^C Quit")
                        elif key == "ctrl+l":
                            bindings.append("^L Clear")
                        elif key == "escape":
                            bindings.append("ESC Focus")
                binding_text = "  ".join(bindings)
            
            if not binding_text:
                binding_text = "^C Quit  ^L Clear  ESC Focus"
            
            keys_widget = self.query_one(".footer-keys", Static)
            keys_widget.update(binding_text)
        except Exception:
            # Fallback - keys widget might not exist yet
            pass
    
    def update_adapter_info(self, provider_name: str = "", model_name: str = "") -> None:
        """Update the adapter information display.
        
        Args:
            provider_name: Name of the current provider
            model_name: Name of the current model
        """
        if provider_name and model_name:
            self.adapter_info = f"{provider_name} | {model_name}"
        else:
            self.adapter_info = "No adapter configured"
        
        # Update the static widget if it exists
        try:
            adapter_widget = self.query_one("#footer-adapter", Static)
            adapter_widget.update(self.adapter_info)
        except Exception:
            # Widget might not exist yet during initialization
            pass