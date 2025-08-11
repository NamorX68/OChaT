from textual.widgets import Static, Button
from textual.containers import Vertical, Horizontal
from textual.screen import ModalScreen
from textual.binding import Binding
from typing import Optional, Callable


class ConfirmationDialog(ModalScreen):
    """Reusable confirmation dialog modal."""

    CSS_PATH = "../styles/confirmation_dialog.tcss"

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("y", "confirm", "Yes"),
        Binding("n", "cancel", "No"),
    ]

    def __init__(self, 
                 title: str = "Bestätigung", 
                 message: str = "Sind Sie sicher?",
                 confirm_text: str = "Ja",
                 cancel_text: str = "Nein",
                 confirm_variant: str = "primary",
                 **kwargs):
        """
        Initialize confirmation dialog.
        
        Args:
            title: Dialog title
            message: Confirmation message
            confirm_text: Text for confirm button
            cancel_text: Text for cancel button
            confirm_variant: Button variant for confirm button (primary, success, warning, error)
        """
        super().__init__(**kwargs)
        self.title = title
        self.message = message
        self.confirm_text = confirm_text
        self.cancel_text = cancel_text
        self.confirm_variant = confirm_variant

    def compose(self):
        """Compose the confirmation dialog."""
        yield Vertical(
            Static(self.title, classes="dialog-title"),
            Static(self.message, classes="dialog-message"),
            Horizontal(
                Button(self.confirm_text, variant=self.confirm_variant, id="confirm-btn"),
                Button(self.cancel_text, variant="default", id="cancel-btn"),
                classes="button-row"
            ),
            classes="confirmation-dialog"
        )

    def on_mount(self):
        """Set focus on cancel button by default for safety."""
        # Small delay to avoid immediate enter key events
        self.call_later(self._delayed_focus)
        
    def _delayed_focus(self):
        """Set focus with small delay to avoid enter key conflicts."""
        self.query_one("#cancel-btn", Button).focus()

    def on_button_pressed(self, event: Button.Pressed):
        """Handle button presses."""
        if event.button.id == "confirm-btn":
            self.action_confirm()
        elif event.button.id == "cancel-btn":
            self.action_cancel()

    def on_key(self, event):
        """Handle key events."""
        if event.key == "enter":
            # Find which button has focus and activate it
            confirm_btn = self.query_one("#confirm-btn", Button)
            cancel_btn = self.query_one("#cancel-btn", Button) 
            
            if confirm_btn.has_focus:
                self.action_confirm()
                event.prevent_default()
            elif cancel_btn.has_focus:
                self.action_cancel()
                event.prevent_default()

    def action_confirm(self):
        """Confirm action."""
        self.dismiss(True)

    def action_cancel(self):
        """Cancel action."""
        self.dismiss(False)


class MessageDialog(ModalScreen):
    """Simple message dialog for information/warnings/errors."""

    CSS_PATH = "../styles/confirmation_dialog.tcss"

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("enter", "close", "Close"),
    ]

    def __init__(self, 
                 title: str = "Information", 
                 message: str = "",
                 message_type: str = "info",  # info, warning, error, success
                 button_text: str = "OK",
                 **kwargs):
        """
        Initialize message dialog.
        
        Args:
            title: Dialog title
            message: Message to display
            message_type: Type of message (info, warning, error, success)
            button_text: Text for the button
        """
        super().__init__(**kwargs)
        self.title = title
        self.message = message
        self.message_type = message_type
        self.button_text = button_text
    def compose(self):
        """Compose the message dialog."""
        # Choose icon based on message type
        icons = {
            "info": "ℹ️",
            "warning": "⚠️", 
            "error": "❌",
            "success": "✅"
        }
        icon = icons.get(self.message_type, "ℹ️")
        
        yield Vertical(
            Static(f"{icon} {self.title}", classes="dialog-title"),
            Static(self.message, classes="dialog-message"),
            Horizontal(
                Button(self.button_text, variant="primary", id="ok-btn"),
                classes="button-row button-row-center"
            ),
            classes="message-dialog"
        )

    def on_mount(self):
        """Set focus on OK button."""
        self.query_one("#ok-btn", Button).focus()

    def on_button_pressed(self, event: Button.Pressed):
        """Handle button press."""
        if event.button.id == "ok-btn":
            self.action_close()

    def action_close(self):
        """Close dialog."""
        self.dismiss(None)