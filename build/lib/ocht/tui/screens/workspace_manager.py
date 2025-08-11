from textual.widgets import Static, DataTable, Button, Input, Label, Header, Footer
from textual.containers import Vertical, Horizontal
from textual.screen import Screen, ModalScreen
from textual.binding import Binding
from typing import List, Optional
from ocht.core.models import Workspace
from ocht.services.workspace_manager import (
    get_workspaces_with_info,
    create_workspace_with_validation,
    update_workspace_with_validation,
    delete_workspace_with_checks
)


class WorkspaceEditScreen(ModalScreen):
    """Modal screen for editing/creating workspaces."""

    CSS_PATH = "../styles/workspace_edit.tcss"

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "save", "Save"),
    ]

    def __init__(self, workspace: Optional[Workspace] = None, **kwargs):
        super().__init__(**kwargs)
        self.workspace = workspace
        self.is_edit_mode = workspace is not None

    def compose(self):
        title = "Edit Workspace" if self.is_edit_mode else "Create New Workspace"
        yield Vertical(
            Static(f"📁 {title}", classes="modal-title"),
            Horizontal(
                Label("Name:", classes="form-label"),
                Input(
                    value=self.workspace.work_name if self.workspace else "",
                    placeholder="Workspace name (e.g., 'Project Alpha', 'Research')",
                    id="workspace-name"
                ),
                classes="form-row"
            ),
            Horizontal(
                Label("Default Model:", classes="form-label"),
                Input(
                    value=self.workspace.work_default_model if self.workspace else "",
                    placeholder="Default model for this workspace",
                    id="workspace-default-model"
                ),
                classes="form-row"
            ),
            Horizontal(
                Label("Description:", classes="form-label"),
                Input(
                    value=self.workspace.work_description if self.workspace and self.workspace.work_description else "",
                    placeholder="Optional: Description of workspace purpose",
                    id="workspace-description"
                ),
                classes="form-row"
            ),
            Horizontal(
                Button("Save", variant="primary", id="save-btn"),
                Button("Cancel", variant="default", id="cancel-btn"),
                classes="button-row"
            ),
            classes="workspace-edit-modal"
        )

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "save-btn":
            self.action_save()

    def action_cancel(self):
        """Cancel workspace editing."""
        self.dismiss(None)

    def action_save(self):
        """Save the workspace."""
        self.save_workspace()

    def save_workspace(self):
        """Save the workspace data."""
        name = self.query_one("#workspace-name", Input).value.strip()
        default_model = self.query_one("#workspace-default-model", Input).value.strip()
        description = self.query_one("#workspace-description", Input).value.strip() or None

        if not name:
            self.notify("Workspace name is required", severity="error")
            return

        if not default_model:
            self.notify("Default model is required", severity="error")
            return

        try:
            if self.is_edit_mode:
                # Update existing workspace using service function
                updated_workspace = update_workspace_with_validation(
                    self.workspace.work_id,
                    name=name,
                    default_model=default_model,
                    description=description
                )
                if updated_workspace:
                    self.dismiss(updated_workspace)
                else:
                    self.notify("Failed to update workspace", severity="error")
            else:
                # Create new workspace using service function
                new_workspace = create_workspace_with_validation(
                    name,
                    default_model=default_model,
                    description=description
                )
                self.dismiss(new_workspace)
        except ValueError as e:
            self.notify(str(e), severity="error")
        except Exception as e:
            self.notify(f"Error saving workspace: {str(e)}", severity="error")


class WorkspaceManagerScreen(Screen):
    """Screen for managing workspaces."""

    CSS_PATH = "../styles/workspace_manager.tcss"

    BINDINGS = [
        ("escape", "back", "Back to Chat"),
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+n", "add_workspace", "Add Workspace"),
        ("ctrl+e", "edit_workspace", "Edit Workspace"),
        ("ctrl+d", "delete_workspace", "Delete Workspace"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.workspaces: List[Workspace] = []

    def compose(self):
        """Compose the workspace manager screen."""
        yield Header(show_clock=True)
        yield Vertical(
            Static("Workspace Management - Use Ctrl+N to add, Ctrl+E to edit, Ctrl+D to delete, ESC to go back", classes="help-text"),
            DataTable(id="workspace-table"),
            Horizontal(
                Button("➕ Add Workspace", variant="primary", id="add-workspace-btn"),
                Button("✏️ Edit", variant="default", id="edit-workspace-btn"),
                Button("🗑️ Delete", variant="error", id="delete-workspace-btn"),
                Button("🔄 Refresh", variant="default", id="refresh-btn"),
                classes="workspace-toolbar"
            ),
            classes="workspace-manager-screen"
        )
        yield Footer()

    def on_mount(self):
        """Initialize the workspace table when screen is mounted."""
        self.setup_table()
        self.load_workspaces()
        # Set focus on the table after mounting
        self.query_one("#workspace-table", DataTable).focus()

    def setup_table(self):
        """Setup the data table columns."""
        table = self.query_one("#workspace-table", DataTable)
        table.add_columns("ID", "Name", "Default Model", "Description", "Created")

    def load_workspaces(self):
        """Load workspaces from database and populate the table."""
        try:
            # Get workspaces with info using service function
            workspace_data = get_workspaces_with_info()

            # Extract workspaces for internal use
            self.workspaces = [data['workspace'] for data in workspace_data]

            table = self.query_one("#workspace-table", DataTable)
            table.clear()

            for data in workspace_data:
                workspace = data['workspace']
                table.add_row(
                    str(workspace.work_id),
                    workspace.work_name,
                    workspace.work_default_model,
                    workspace.work_description or "No description",
                    workspace.work_created_at.strftime("%Y-%m-%d %H:%M")
                )
        except Exception as e:
            self.notify(f"Error loading workspaces: {str(e)}", severity="error")

    def on_button_pressed(self, event: Button.Pressed):
        """Handle button presses."""
        if event.button.id == "add-workspace-btn":
            self.add_workspace()
        elif event.button.id == "edit-workspace-btn":
            self.edit_workspace()
        elif event.button.id == "delete-workspace-btn":
            self.delete_workspace()
        elif event.button.id == "refresh-btn":
            self.load_workspaces()

    def action_back(self):
        """Go back to the main chat screen."""
        self.app.pop_screen()

    def action_add_workspace(self):
        """Add a new workspace."""
        self.add_workspace()

    def action_edit_workspace(self):
        """Edit selected workspace."""
        self.edit_workspace()

    def action_delete_workspace(self):
        """Delete selected workspace."""
        self.delete_workspace()

    def add_workspace(self):
        """Show modal to add new workspace."""
        def handle_result(result):
            if result:
                self.load_workspaces()
                self.notify(f"Workspace '{result.work_name}' created successfully", severity="information")

        self.app.push_screen(WorkspaceEditScreen(), handle_result)

    def edit_workspace(self):
        """Show modal to edit selected workspace."""
        table = self.query_one("#workspace-table", DataTable)
        if table.cursor_row is None:
            self.notify("Please select a workspace to edit", severity="warning")
            return

        selected_workspace = self.workspaces[table.cursor_row]

        def handle_result(result):
            if result:
                self.load_workspaces()
                self.notify(f"Workspace '{result.work_name}' updated successfully", severity="information")

        self.app.push_screen(WorkspaceEditScreen(selected_workspace), handle_result)

    def delete_workspace(self):
        """Delete selected workspace."""
        table = self.query_one("#workspace-table", DataTable)
        if table.cursor_row is None:
            self.notify("Please select a workspace to delete", severity="warning")
            return

        selected_workspace = self.workspaces[table.cursor_row]

        # Simple confirmation - in a real app you might want a proper confirmation dialog
        try:
            if delete_workspace_with_checks(selected_workspace.work_id):
                self.load_workspaces()
                self.notify(f"Workspace '{selected_workspace.work_name}' deleted successfully", severity="information")
            else:
                self.notify("Failed to delete workspace", severity="error")
        except ValueError as e:
            self.notify(str(e), severity="error")
        except Exception as e:
            self.notify(f"Error deleting workspace: {str(e)}", severity="error")