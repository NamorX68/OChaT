import asyncio
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input
from textual.containers import VerticalScroll, Horizontal
from ocht.adapters.ollama import OllamaAdapter
from ocht.tui.widgets.chat_bubble import ChatBubble
from ocht.tui.screens.provider_manager import ProviderManagerScreen
from ocht.tui.screens.provider_selector import ProviderSelectorModal
from ocht.tui.screens.model_selector import ModelSelectorModal
from ocht.tui.screens.model_manager import ModelManagerScreen
from ocht.tui.screens.settings_manager import SettingsManagerScreen
from ocht.tui.screens.workspace_manager import WorkspaceManagerScreen
from ocht.tui.screens.workspace_selector import WorkspaceSelectorModal

class ChatApp(App):
    """Elegant Chat Terminal User Interface"""

    TITLE = "OChaT"

    CSS_PATH = "styles/app.tcss"

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
        self.adapter = OllamaAdapter(
            model="qwen3:30b-a3b",
            default_params={"temperature": 0.5}
        )
        self.notifications = []

    def compose(self) -> ComposeResult:
        """Compose the UI components.

        Returns:
            ComposeResult: The result containing the UI components.
        """
        yield Header(show_clock=True)
        yield VerticalScroll(id="chat-container")
        yield Input(
            placeholder="💬 Write your message... (ESC to focus)",
            id="chat-input"
        )
        yield Footer()

    def on_mount(self) -> None:
        """App start: Focus input and show greeting."""
        self.query_one("#chat-input", Input).focus()
        self._add_message("👋 Hello! I am your AI Assistant. Type `/help` for help.", "bot")

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        """Handle input submission.

        Args:
            message (Input.Submitted): The submitted input message.
        """
        prompt = message.value.strip()
        message.input.value = ""

        if not prompt:
            return

        if prompt.startswith("/"):
            await self._handle_command(prompt)
        else:
            await self._process_prompt(prompt)

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
                def handle_provider_selection(result):
                    if result:
                        self.add_note(f"✅ Selected Provider: {result.prov_name} (ID: {result.prov_id})")

                await self.push_screen(ProviderSelectorModal(), handle_provider_selection)

            case "/provider-manage":
                await self.push_screen(ProviderManagerScreen())

            case "/model":
                def handle_model_selection(result):
                    if result:
                        self.add_note(f"✅ Selected model: {result.model_name} (ID: {result.model_id})")

                await self.push_screen(ModelSelectorModal(), handle_model_selection)

            case "/model-manage":
                await self.push_screen(ModelManagerScreen())

            case "/settings":
                await self.push_screen(SettingsManagerScreen())

            case "/workspace":
                def handle_workspace_selection(result):
                    if result:
                        self.add_note(f"✅ Selected Workspace: {result.work_name} (ID: {result.work_id})")

                await self.push_screen(WorkspaceSelectorModal(), handle_workspace_selection)

            case "/workspace-manage":
                await self.push_screen(WorkspaceManagerScreen())

            case "/help":
                help_text = """🤖 **Commands:**

• `/bye`, `/quit`, `/exit` - End chat
• `/clear` - Clear chat history
• `/provider` - Select LLM provider
• `/provider-manage` - Manage LLM providers
• `/model` - Select LLM Model
• `/model-manage` - Manage LLM Models
• `/workspace` - Select workspace
• `/workspace-manage` - Manage workspaces
• `/settings` - Manage application settings
• `/help` - Show this help

**Keyboard shortcuts:**
• `Ctrl+C` - Exit program
• `Ctrl+L` - Clear chat
• `ESC` - Focus input field"""
                self._add_message(help_text, "bot")

            case _:
                self._add_message(
                    f"❌ Unknown command: {command}\nType `/help` for help.",
                    "bot",
                    "error"
                )

    async def _process_prompt(self, prompt: str) -> None:
        """Process the user's prompt.

        Args:
            prompt (str): The user's input prompt.
        """
        # Add user message and scroll immediately
        self._add_message(prompt, "user")

        # Add typing indicator and scroll immediately
        typing_bubble = self._add_message("🤔 *thinking...*", "bot", "typing")

        try:
            answer = await asyncio.to_thread(self.adapter.send_prompt, prompt)
            await typing_bubble.remove()
            self._add_message(answer, "bot")
        except Exception as e:
            await typing_bubble.remove()
            error_msg = f"❌ **Error:** {str(e)}\n\nPlease check your Ollama installation."
            self._add_message(error_msg, "bot", "error")

    def _add_message(self, message: str, sender: str, style: str="") -> Horizontal:
        """Add a new chat message and scroll immediately.

        Args:
            message (str): The message content to add.
            sender (str): The sender of the message ('user' or 'bot').
            style (str, optional): Additional style class for the message. Defaults to "".

        Returns:
            Horizontal: The message row container that was added.
        """
        container = self.query_one("#chat-container", VerticalScroll)

        # Additional CSS classes based on style
        extra_classes = f" {style}" if style else ""
        bubble = ChatBubble(message, sender + extra_classes)

        # Create container for the message row with the bubble inside
        message_row = Horizontal(bubble, classes=f"message-row {sender}")

        # Add message row to chat container
        container.mount(message_row)

        # Immediate scrolling without animation
        container.scroll_end(animate=False)

        return message_row

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
