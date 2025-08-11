from textual.widgets import Static, ListItem, ListView, Button
from textual.containers import Vertical, Horizontal
from textual.screen import ModalScreen
from textual.binding import Binding
from typing import List, Optional
from ocht.core.models import Workspace
from ocht.services.workspace_manager import get_available_workspaces


class WorkspaceSelectorModal(ModalScreen):
    """Modal dialog for selecting workspaces."""

    CSS_PATH = "../styles/selector_modal.tcss"

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "select", "Select"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.workspaces: List[Workspace] = []
        self.selected_workspace: Optional[Workspace] = None

    def compose(self):
        """Compose the workspace selector modal."""
        yield Vertical(
            Static("📁 Select a Workspace", classes="modal-title"),
            ListView(id="workspace-list", classes="selector-list"),
            Horizontal(
                Button("OK", variant="primary", id="ok-btn"),
                Button("Cancel", variant="default", id="cancel-btn"),
                classes="button-row"
            ),
            classes="selector-modal"
        )

    def on_mount(self):
        """Load workspaces when modal is mounted."""
        self.load_workspaces()
        # Set focus on the workspace list after mounting
        self.query_one("#workspace-list", ListView).focus()

    def load_workspaces(self):
        """Load workspaces from database and populate the list."""
        try:
            # Get workspaces using service function
            self.workspaces = get_available_workspaces()

            workspace_list = self.query_one("#workspace-list", ListView)
            workspace_list.clear()

            if not self.workspaces:
                workspace_list.append(ListItem(Static("No workspaces found. Use workspace management to add workspaces.")))
                return

            for workspace in self.workspaces:
                item_text = f"📁 {workspace.work_name}"
                if workspace.work_description:
                    item_text += f" - {workspace.work_description}"
                if workspace.work_default_model:
                    item_text += f" (Model: {workspace.work_default_model})"
                workspace_list.append(ListItem(Static(item_text)))

            # Automatically select the first element if workspaces exist
            if self.workspaces:
                workspace_list.index = 0
                # Also set the selected_workspace to the first one
                self.selected_workspace = self.workspaces[0]

        except Exception as e:
            workspace_list = self.query_one("#workspace-list", ListView)
            workspace_list.clear()
            workspace_list.append(ListItem(Static(f"❌ Error loading workspaces: {str(e)}")))

    def on_list_view_selected(self, event: ListView.Selected):
        """Handle workspace selection."""
        if event.list_view.id == "workspace-list" and self.workspaces:
            selected_index = event.list_view.index
            if 0 <= selected_index < len(self.workspaces):
                self.selected_workspace = self.workspaces[selected_index]

    def on_key(self, event):
        """Handle key events, specifically Enter key when ListView has focus."""
        if event.key == "enter":
            # Check if the workspace list has focus
            workspace_list = self.query_one("#workspace-list", ListView)
            if workspace_list.has_focus:
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
        """Cancel workspace selection."""
        self.dismiss(None)

    def action_select(self):
        """Select the currently highlighted workspace."""
        if self.selected_workspace:
            self.dismiss(self.selected_workspace)
        else:
            # If no workspace is explicitly selected, try to get the highlighted one
            workspace_list = self.query_one("#workspace-list", ListView)
            if workspace_list.index is not None and 0 <= workspace_list.index < len(self.workspaces):
                selected_workspace = self.workspaces[workspace_list.index]
                self.dismiss(selected_workspace)
            else:
                self.notify("Please select a workspace first", severity="warning")