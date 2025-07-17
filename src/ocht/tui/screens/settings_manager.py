from textual.widgets import Static, DataTable, Button, Input, Label, Header, Footer
from textual.containers import Vertical, Horizontal
from textual.screen import Screen, ModalScreen
from textual.binding import Binding
from typing import List, Optional
from ocht.core.models import Setting
from ocht.services.settings_manager import (
    get_all_settings_with_info,
    create_setting_with_validation,
    update_setting_with_validation,
    delete_setting_with_checks
)


class SettingEditScreen(ModalScreen):
    """Modal screen for editing/creating settings."""

    CSS_PATH = "../styles/setting_edit.tcss"

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "save", "Save"),
    ]

    def __init__(self, setting: Optional[Setting] = None, **kwargs):
        super().__init__(**kwargs)
        self.setting = setting
        self.is_edit_mode = setting is not None

    def compose(self):
        title = "Edit Setting" if self.is_edit_mode else "Create New Setting"
        yield Vertical(
            Static(f"⚙️ {title}", classes="modal-title"),
            Horizontal(
                Label("Key:", classes="form-label"),
                Input(
                    value=self.setting.setting_key if self.setting else "",
                    placeholder="Setting key (e.g., 'theme', 'language')",
                    id="setting-key",
                    disabled=self.is_edit_mode  # Don't allow key changes in edit mode
                ),
                classes="form-row"
            ),
            Horizontal(
                Label("Value:", classes="form-label"),
                Input(
                    value=self.setting.setting_value if self.setting else "",
                    placeholder="Setting value",
                    id="setting-value"
                ),
                classes="form-row"
            ),
            Horizontal(
                Button("Save", variant="primary", id="save-btn"),
                Button("Cancel", variant="default", id="cancel-btn"),
                classes="button-row"
            ),
            classes="setting-edit-modal"
        )

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "save-btn":
            self.action_save()

    def action_cancel(self):
        """Cancel setting editing."""
        self.dismiss(None)

    def action_save(self):
        """Save the setting."""
        self.save_setting()

    def save_setting(self):
        """Save the setting data."""
        key = self.query_one("#setting-key", Input).value.strip()
        value = self.query_one("#setting-value", Input).value.strip()

        if not key:
            self.notify("Setting key is required", severity="error")
            return

        if not value:
            self.notify("Setting value is required", severity="error")
            return

        try:
            if self.is_edit_mode:
                # Update existing setting using service function
                updated_setting = update_setting_with_validation(
                    self.setting.setting_key,
                    value=value
                )
                if updated_setting:
                    self.dismiss(updated_setting)
                else:
                    self.notify("Failed to update setting", severity="error")
            else:
                # Create new setting using service function
                new_setting = create_setting_with_validation(key, value)
                self.dismiss(new_setting)
        except ValueError as e:
            self.notify(str(e), severity="error")
        except Exception as e:
            self.notify(f"Error saving setting: {str(e)}", severity="error")


class SettingsManagerScreen(Screen):
    """Screen for managing settings."""

    CSS_PATH = "../styles/settings_manager.tcss"

    BINDINGS = [
        ("escape", "back", "Back to Chat"),
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+n", "add_setting", "Add Setting"),
        ("ctrl+e", "edit_setting", "Edit Setting"),
        ("ctrl+d", "delete_setting", "Delete Setting"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.settings: List[Setting] = []

    def compose(self):
        """Compose the settings manager screen."""
        yield Header(show_clock=True)
        yield Vertical(
            Static("Settings Management - Use Ctrl+N to add, Ctrl+E to edit, Ctrl+D to delete, ESC to go back", classes="help-text"),
            DataTable(id="settings-table"),
            Horizontal(
                Button("➕ Add Setting", variant="primary", id="add-setting-btn"),
                Button("✏️ Edit", variant="default", id="edit-setting-btn"),
                Button("🗑️ Delete", variant="error", id="delete-setting-btn"),
                Button("🔄 Refresh", variant="default", id="refresh-btn"),
                classes="settings-toolbar"
            ),
            classes="settings-manager-screen"
        )
        yield Footer()

    def on_mount(self):
        """Initialize the settings table when screen is mounted."""
        self.setup_table()
        self.load_settings()
        # Set focus on the table after mounting
        self.query_one("#settings-table", DataTable).focus()

    def setup_table(self):
        """Setup the data table columns."""
        table = self.query_one("#settings-table", DataTable)
        table.add_columns("Key", "Value", "Workspace Scoped", "Created", "Updated")

    def load_settings(self):
        """Load settings from database and populate the table."""
        try:
            # Get settings with info using service function
            settings_data = get_all_settings_with_info()

            # Extract settings for internal use
            self.settings = [data['setting'] for data in settings_data]

            table = self.query_one("#settings-table", DataTable)
            table.clear()

            for data in settings_data:
                setting = data['setting']
                workspace_scoped = "Yes" if data['workspace_scoped'] else "No"
                
                # Truncate long values for display
                display_value = setting.setting_value
                if len(display_value) > 50:
                    display_value = display_value[:47] + "..."
                
                table.add_row(
                    setting.setting_key,
                    display_value,
                    workspace_scoped,
                    setting.setting_created_at.strftime("%Y-%m-%d %H:%M"),
                    setting.setting_updated_at.strftime("%Y-%m-%d %H:%M")
                )
        except Exception as e:
            self.notify(f"Error loading settings: {str(e)}", severity="error")

    def on_button_pressed(self, event: Button.Pressed):
        """Handle button presses."""
        if event.button.id == "add-setting-btn":
            self.add_setting()
        elif event.button.id == "edit-setting-btn":
            self.edit_setting()
        elif event.button.id == "delete-setting-btn":
            self.delete_setting()
        elif event.button.id == "refresh-btn":
            self.load_settings()

    def action_back(self):
        """Go back to the main chat screen."""
        self.app.pop_screen()

    def action_add_setting(self):
        """Add a new setting."""
        self.add_setting()

    def action_edit_setting(self):
        """Edit selected setting."""
        self.edit_setting()

    def action_delete_setting(self):
        """Delete selected setting."""
        self.delete_setting()

    def add_setting(self):
        """Show modal to add new setting."""
        def handle_result(result):
            if result:
                self.load_settings()
                self.notify(f"Setting '{result.setting_key}' created successfully", severity="information")

        self.app.push_screen(SettingEditScreen(), handle_result)

    def edit_setting(self):
        """Show modal to edit selected setting."""
        table = self.query_one("#settings-table", DataTable)
        if table.cursor_row is None:
            self.notify("Please select a setting to edit", severity="warning")
            return

        selected_setting = self.settings[table.cursor_row]

        def handle_result(result):
            if result:
                self.load_settings()
                self.notify(f"Setting '{result.setting_key}' updated successfully", severity="information")

        self.app.push_screen(SettingEditScreen(selected_setting), handle_result)

    def delete_setting(self):
        """Delete selected setting."""
        table = self.query_one("#settings-table", DataTable)
        if table.cursor_row is None:
            self.notify("Please select a setting to delete", severity="warning")
            return

        selected_setting = self.settings[table.cursor_row]

        try:
            if delete_setting_with_checks(selected_setting.setting_key):
                self.load_settings()
                self.notify(f"Setting '{selected_setting.setting_key}' deleted successfully", severity="information")
            else:
                self.notify("Failed to delete setting", severity="error")
        except ValueError as e:
            self.notify(str(e), severity="error")
        except Exception as e:
            self.notify(f"Error deleting setting: {str(e)}", severity="error")