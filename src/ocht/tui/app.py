import asyncio
import os
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input
from textual.containers import VerticalScroll, Horizontal
from ocht.tui.widgets.chat_bubble import ChatBubble
from ocht.tui.widgets.custom_footer import CustomFooter
from ocht.tui.screens.provider_manager import ProviderManagerScreen
from ocht.tui.screens.provider_selector import ProviderSelectorModal
from ocht.tui.screens.model_selector import ModelSelectorModal
from ocht.tui.screens.model_manager import ModelManagerScreen
from ocht.tui.screens.settings_manager import SettingsManagerScreen
from ocht.tui.screens.workspace_manager import WorkspaceManagerScreen
from ocht.tui.screens.workspace_selector import WorkspaceSelectorModal
from ocht.tui.widgets.confirmation_dialog import ConfirmationDialog
from ocht.services.adapter_manager import adapter_manager


class ChatApp(App):
    """Elegant Chat Terminal User Interface"""

    TITLE = "OChaT"

    CSS_PATH = "styles/app.tcss"

    # Configure mouse and input handling to prevent escape sequences
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_chat", "Clear chat"),
        ("escape", "focus_input", "Focus input"),
    ]

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the ChatApp.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.adapter = None
        self.notifications = []

    def compose(self) -> ComposeResult:
        """Compose the UI components.

        Returns:
            ComposeResult: The result containing the UI components.
        """
        yield Header(show_clock=True)
        yield VerticalScroll(id="chat-container")
        yield Input(
            placeholder="💬 Write your message... (ESC to focus)", id="chat-input"
        )
        yield CustomFooter()

    async def on_mount(self) -> None:
        """App start: Focus input and initialize adapter."""
        self.query_one("#chat-input", Input).focus()

        # Try to load settings on startup
        if adapter_manager.load_settings_on_startup():
            self.adapter = adapter_manager.get_current_adapter()
            self._update_footer_adapter_info()
            self._add_message(
                "👋 Hello! I am your AI Assistant. Type `/help` for help.", "bot"
            )
        else:
            # No settings found, need to configure first
            self._add_message("⚙️ Welcome! Let's set up your AI assistant first.", "bot")

            # Check what needs to be configured
            if adapter_manager.requires_provider_selection():
                await self._show_initial_provider_selection()
            elif adapter_manager.requires_model_selection():
                await self._show_initial_model_selection()
                
            # Update footer even if no adapter is configured yet
            self._update_footer_adapter_info()

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        """Handle input submission with mouse escape sequence filtering.

        Args:
            message (Input.Submitted): The submitted input message.
        """
        prompt = message.value.strip()
        message.input.value = ""

        if not prompt:
            return

        # Filter out mouse escape sequences and control characters
        if self._is_mouse_escape_sequence(prompt):
            # Ignore mouse escape sequences silently
            return

        if prompt.startswith("/"):
            await self._handle_command(prompt)
        else:
            await self._process_prompt(prompt)

    def _is_mouse_escape_sequence(self, text: str) -> bool:
        """Check if text contains mouse escape sequences or control characters.

        Args:
            text (str): The text to check.

        Returns:
            bool: True if text contains mouse escape sequences.
        """
        # Allow disabling mouse filtering via environment variable for debugging
        if os.getenv("OCHT_DISABLE_MOUSE_FILTER", "").lower() == "true":
            return False

        # Common mouse escape sequence patterns
        mouse_patterns = [
            "[<",  # Mouse SGR format like [<35;1341;-1M
            "\x1b[M",  # Mouse X10 format
            "\x1b[<",  # Mouse SGR format with ESC
            ";-1M",  # Mouse release pattern
            ";1M",  # Mouse press pattern
        ]

        # Check for mouse escape sequences
        for pattern in mouse_patterns:
            if pattern in text:
                return True

        # Check for control characters (except normal ones like tab, newline)
        if any(ord(c) < 32 and c not in "\t\n\r" for c in text):
            return True

        # Check for escape sequences starting with ESC
        if text.startswith("\x1b") or "\x1b" in text:
            return True

        return False

    async def _handle_command(self, command: str) -> None:
        """Handle chat commands with match statement.

        Args:
            command (str): The command to handle.
        """
        match command:
            case "/bye" | "/quit" | "/exit":
                self._add_message("👋 Goodbye!", "bot", "success")
                await asyncio.sleep(0.5)
                self.exit()

            case "/clear":
                await self.action_clear_chat()

            case "/provider":
                await self._handle_provider_change()

            case "/provider-manage":
                await self.push_screen(ProviderManagerScreen())

            case "/model":
                await self._handle_model_change()

            case "/model-manage":
                await self.push_screen(ModelManagerScreen())

            case "/settings":
                await self.push_screen(SettingsManagerScreen())

            case "/workspace":

                def handle_workspace_selection(result):
                    if result:
                        self.add_note(
                            f"✅ Selected Workspace: {result.work_name} (ID: {result.work_id})"
                        )

                await self.push_screen(
                    WorkspaceSelectorModal(), handle_workspace_selection
                )

            case "/workspace-manage":
                await self.push_screen(WorkspaceManagerScreen())

            case "/help":
                help_text = """# 🤖 OChaT Help

## Commands:
- `/bye`, `/quit`, `/exit` - End chat
- `/clear` - Clear chat history
- `/provider` - Select LLM provider
- `/provider-manage` - Manage LLM providers
- `/model` - Select LLM Model
- `/model-manage` - Manage LLM Models
- `/workspace` - Select workspace
- `/workspace-manage` - Manage workspaces
- `/settings` - Manage application settings
- `/help` - Show this help

## Keyboard shortcuts:
- `Ctrl+C` - Exit program
- `Ctrl+L` - Clear chat
- `ESC` - Focus input field"""
                self._add_message(help_text, "bot")

            case _:
                self._add_message(
                    f"❌ Unknown command: {command}\nType `/help` for help.",
                    "bot",
                    "error",
                )

    async def _process_prompt(self, prompt: str) -> None:
        """Process the user's prompt with streaming support.

        Args:
            prompt (str): The user's input prompt.
        """
        # Check if adapter is available
        if not self.adapter:
            self._add_message(
                "❌ Kein Adapter konfiguriert. Bitte wählen Sie zuerst einen Provider und ein Modell.",
                "bot",
                "error",
            )
            return

        # Add user message and scroll immediately
        self._add_message(prompt, "user")

        # Create streaming bot message bubble with thinking indicator
        bot_bubble = self._add_message("🤔 Thinking...", "bot", streaming=True)
        full_response = ""

        try:
            # Stream the response with live updates
            async for chunk in self.adapter.send_prompt_stream(prompt):
                full_response += chunk
                bot_bubble.update_content(full_response)

                # Auto-scroll to keep up with streaming content
                container = self.query_one("#chat-container", VerticalScroll)
                container.scroll_end(animate=False)

            # Finalize the message (remove typing indicator)
            bot_bubble.finalize()

        except Exception as e:
            # Handle streaming errors gracefully
            if full_response:
                # If we got partial content, finalize it first
                bot_bubble.finalize()
            else:
                # Remove empty bubble and show error
                await bot_bubble.parent.remove()

            error_msg = f"❌ **Error:** {str(e)}\n\nPlease check your configuration."
            self._add_message(error_msg, "bot", "error")

            # Fallback to async method if streaming fails
            if "stream" in str(e).lower():
                self.notify("Streaming failed, falling back to standard mode...")
                await self._process_prompt_fallback(prompt)

    async def _process_prompt_fallback(self, prompt: str) -> None:
        """Fallback method using async send_prompt_async instead of streaming.

        Args:
            prompt (str): The user's input prompt.
        """
        # Add typing indicator
        typing_bubble = self._add_message("🤔 *thinking...*", "bot", "typing")

        try:
            # Use async method instead of streaming
            answer = await self.adapter.send_prompt_async(prompt)
            await typing_bubble.parent.remove()
            self._add_message(answer, "bot")
        except Exception as e:
            await typing_bubble.parent.remove()
            error_msg = f"❌ **Error:** {str(e)}\n\nPlease check your configuration."
            self._add_message(error_msg, "bot", "error")

    def _add_message(
        self, message: str, sender: str, style: str = "", streaming: bool = False
    ) -> ChatBubble:
        """Add a new chat message and scroll immediately.

        Args:
            message (str): The message content to add.
            sender (str): The sender of the message ('user' or 'bot').
            style (str, optional): Additional style class for the message. Defaults to "".
            streaming (bool, optional): Enable streaming support for live updates. Defaults to False.

        Returns:
            ChatBubble: The chat bubble widget that was added.
        """
        container = self.query_one("#chat-container", VerticalScroll)

        # Additional CSS classes based on style
        extra_classes = f" {style}" if style else ""
        bubble = ChatBubble(message, sender + extra_classes, streaming=streaming)

        # Add bubble directly to chat container with sender-specific styling
        bubble.add_class(f"bubble-{sender}")
        container.mount(bubble)

        # Immediate scrolling without animation
        container.scroll_end(animate=False)

        return bubble

    async def action_clear_chat(self) -> None:
        """Clear the chat history."""
        container = self.query_one("#chat-container", VerticalScroll)
        await container.remove_children()
        self._add_message("✨ Chat history has been cleared.", "bot", "success")

    def action_focus_input(self) -> None:
        """Focus the input field."""
        self.query_one("#chat-input", Input).focus()

    def add_note(self, message: str) -> None:
        """Add a notification message to the chat.

        Args:
            message (str): The notification message to display.
        """
        self._add_message(message, "bot", "success")

    def _update_footer_adapter_info(self) -> None:
        """Update the footer with current adapter information."""
        try:
            footer = self.query_one(CustomFooter)
            adapter_info = adapter_manager.get_adapter_info()
            
            provider_name = adapter_info.get("provider_name", "")
            model_name = adapter_info.get("model_name", "")
            
            footer.update_adapter_info(provider_name, model_name)
        except Exception:
            # Footer might not exist yet or adapter info not available
            pass

    async def _show_initial_provider_selection(self) -> None:
        """Show provider selection during initial setup."""

        def handle_initial_provider_selection(result):
            if result:
                self._add_message(
                    f"✅ Provider selected: {result.prov_name}", "bot", "success"
                )
                # After provider selection, check if we need model selection
                if adapter_manager.requires_model_selection():
                    asyncio.create_task(self._show_initial_model_selection())
            else:
                self._add_message(
                    "❌ Setup cancelled. Please select a provider to continue.",
                    "bot",
                    "error",
                )

        await self.push_screen(
            ProviderSelectorModal(), handle_initial_provider_selection
        )

    async def _show_initial_model_selection(self) -> None:
        """Show model selection during initial setup."""

        def handle_initial_model_selection(result):
            if result:
                self._add_message(
                    f"✅ Model selected: {result.model_name}", "bot", "success"
                )
                # Try to create adapter with selected provider and model
                if adapter_manager.switch_adapter(
                    adapter_manager.get_current_provider_id()
                    or result.model_provider_id,
                    result.model_name,
                ):
                    self.adapter = adapter_manager.get_current_adapter()
                    self._update_footer_adapter_info()
                    self._add_message(
                        "🎉 Setup complete! You can now start chatting.",
                        "bot",
                        "success",
                    )
                else:
                    self._add_message(
                        "❌ Failed to initialize adapter. Please check your configuration.",
                        "bot",
                        "error",
                    )
            else:
                self._add_message(
                    "❌ Setup cancelled. Please select a model to continue.",
                    "bot",
                    "error",
                )

        await self.push_screen(ModelSelectorModal(), handle_initial_model_selection)

    async def _handle_provider_change(self) -> None:
        """Handle provider selection with chat loss warning."""

        # Show provider selection first
        def handle_provider_selection(result):
            if result:
                # Check if user selected a different provider
                current_provider_id = adapter_manager.get_current_provider_id()
                if current_provider_id and result.prov_id != current_provider_id:
                    # Different provider selected, show confirmation for chat loss

                    # Show confirmation dialog for chat loss
                    def handle_provider_confirmation(confirmed):
                        if not confirmed:
                            return

                        # Clear current chat when switching providers
                        if adapter_manager.has_active_chat():
                            asyncio.create_task(self.action_clear_chat())

                        # Update adapter manager and app adapter
                        if adapter_manager.switch_adapter(
                            result.prov_id,
                            adapter_manager.get_current_model_name() or "",
                        ):
                            self.adapter = adapter_manager.get_current_adapter()
                            self._update_footer_adapter_info()
                            self.add_note(f"✅ Provider gewechselt: {result.prov_name}")
                        else:
                            self.add_note("❌ Fehler beim Wechseln des Providers")

                    if adapter_manager.has_active_chat():
                        # Use callback pattern for ConfirmationDialog
                        self.push_screen(
                            ConfirmationDialog(
                                title="Provider wechseln",
                                message="Beim Wechsel des Providers geht der aktuelle Chat verloren.\nMöchten Sie trotzdem fortfahren?",
                                confirm_text="Ja, wechseln",
                                cancel_text="Abbrechen",
                                confirm_variant="warning",
                            ),
                            handle_provider_confirmation,
                        )
                    else:
                        # No active chat, switch directly
                        if adapter_manager.switch_adapter(
                            result.prov_id,
                            adapter_manager.get_current_model_name() or "",
                        ):
                            self.adapter = adapter_manager.get_current_adapter()
                            self.add_note(f"✅ Provider gewechselt: {result.prov_name}")
                        else:
                            self.add_note("❌ Fehler beim Wechseln des Providers")
                else:
                    # Same provider or no current provider, no confirmation needed
                    if adapter_manager.switch_adapter(
                        result.prov_id, adapter_manager.get_current_model_name() or ""
                    ):
                        self.adapter = adapter_manager.get_current_adapter()
                        self._update_footer_adapter_info()
                        self.add_note(f"✅ Provider ausgewählt: {result.prov_name}")
                    else:
                        self.add_note("❌ Fehler beim Auswählen des Providers")

        await self.push_screen(ProviderSelectorModal(), handle_provider_selection)

    async def _handle_model_change(self) -> None:
        """Handle model selection with chat loss warning."""

        def handle_model_selection(selected_model):
            if not selected_model:
                return

            # Check if same model
            current_model_name = adapter_manager.get_current_model_name()
            if current_model_name and selected_model.model_name == current_model_name:
                self.add_note(f"✅ Modell bereits aktiv: {selected_model.model_name}")
                return

            # Show confirmation dialog for chat loss
            def handle_confirmation(confirmed):
                if not confirmed:
                    return

                # Clear current chat when switching models
                if adapter_manager.has_active_chat():
                    asyncio.create_task(self.action_clear_chat())

                # Switch model
                provider_id = (
                    adapter_manager.get_current_provider_id()
                    or selected_model.model_provider_id
                )

                if adapter_manager.switch_adapter(
                    provider_id, selected_model.model_name
                ):
                    self.adapter = adapter_manager.get_current_adapter()
                    self._update_footer_adapter_info()
                    self.add_note(f"✅ Modell gewechselt: {selected_model.model_name}")
                else:
                    self.add_note("❌ Fehler beim Wechseln des Modells")

            if adapter_manager.has_active_chat():
                # Use callback pattern for ConfirmationDialog
                self.push_screen(
                    ConfirmationDialog(
                        title="Modell wechseln",
                        message="Beim Wechsel des Modells geht der aktuelle Chat verloren.\nMöchten Sie trotzdem fortfahren?",
                        confirm_text="Ja, wechseln",
                        cancel_text="Abbrechen",
                        confirm_variant="warning",
                    ),
                    handle_confirmation,
                )
            else:
                # No active chat, switch directly
                provider_id = (
                    adapter_manager.get_current_provider_id()
                    or selected_model.model_provider_id
                )

                if adapter_manager.switch_adapter(
                    provider_id, selected_model.model_name
                ):
                    self.adapter = adapter_manager.get_current_adapter()
                    self._update_footer_adapter_info()
                    self.add_note(f"✅ Modell gewechselt: {selected_model.model_name}")
                else:
                    self.add_note("❌ Fehler beim Wechseln des Modells")

        await self.push_screen(ModelSelectorModal(), handle_model_selection)
