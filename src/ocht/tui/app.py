import asyncio
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input
from textual.containers import VerticalScroll, Horizontal
from ocht.adapters.ollama import OllamaAdapter
from ocht.tui.widgets.chat_bubble import ChatBubble

class ChatApp(App):
    """Elegante Chat-TUI mit verbessertem Design."""

    TITLE = "OChaT"

    CSS_PATH = "styles/app.tcss"

    BINDINGS = [
        ("ctrl+c", "quit", "Beenden"),
        ("ctrl+l", "clear_chat", "Chat leeren"),
        ("escape", "focus_input", "Input fokussieren"),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.adapter = OllamaAdapter(
            model="devstral:24b-q8_0",
            default_params={"temperature": 0.5}
        )

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield VerticalScroll(id="chat-container")
        yield Input(
            placeholder="💬 Schreibe deine Nachricht... (ESC zum Fokussieren)",
            id="chat-input"
        )
        yield Footer()

    def on_mount(self) -> None:
        """App-Start: Input fokussieren und Begrüßung anzeigen."""
        self.query_one("#chat-input", Input).focus()
        self._add_message("👋 Hallo! Ich bin dein AI-Assistent. Tippe `/help` für Hilfe.", "bot")

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        prompt = message.value.strip()
        message.input.value = ""

        if not prompt:
            return

        if prompt.startswith("/"):
            await self._handle_command(prompt)
        else:
            await self._process_prompt(prompt)

    async def _handle_command(self, command: str) -> None:
        """Behandelt Chat-Kommandos mit match-Statement."""
        match command:
            case "/bye" | "/quit" | "/exit":
                self._add_message("👋 Auf Wiedersehen!", "bot", "success")
                await asyncio.sleep(0.5)
                self.exit()

            case "/clear":
                await self.action_clear_chat()

            case "/help":
                help_text = """🤖 **Verfügbare Kommandos:**

• `/bye`, `/quit`, `/exit` - Chat beenden
• `/clear` - Chat-Verlauf löschen
• `/help` - Diese Hilfe anzeigen

**Tastenkürzel:**
• `Ctrl+C` - Programm beenden
• `Ctrl+L` - Chat leeren
• `ESC` - Input fokussieren"""
                self._add_message(help_text, "bot")

            case _:
                self._add_message(
                    f"❌ Unbekanntes Kommando: {command}\nTippe `/help` für Hilfe.",
                    "bot",
                    "error"
                )

    async def _process_prompt(self, prompt: str) -> None:
        """Verarbeitet normalen Chat-Input."""
        # User-Message hinzufügen und sofort scrollen
        self._add_message(prompt, "user")

        # Typing-Indikator hinzufügen und sofort scrollen
        typing_bubble = self._add_message("🤔 *thinking...*", "bot", "typing")

        try:
            answer = await asyncio.to_thread(self.adapter.send_prompt, prompt)
            await typing_bubble.remove()
            self._add_message(answer, "bot")
        except Exception as e:
            await typing_bubble.remove()
            error_msg = f"❌ **Fehler:** {str(e)}\n\nBitte überprüfe deine Ollama-Installation."
            self._add_message(error_msg, "bot", "error")

    def _add_message(self, message: str, sender: str, style: str = "") -> Horizontal:
        """Fügt eine neue Chat-Nachricht hinzu und scrollt sofort."""
        container = self.query_one("#chat-container", VerticalScroll)

        # Zusätzliche CSS-Klassen basierend auf dem Style
        extra_classes = f" {style}" if style else ""
        bubble = ChatBubble(message, sender + extra_classes)

        # Container für die Message-Zeile mit der Bubble darin erstellen
        message_row = Horizontal(bubble, classes=f"message-row {sender}")

        # Message-Zeile zum Chat-Container hinzufügen
        container.mount(message_row)

        # Sofortiges Scrollen ohne Animation
        container.scroll_end(animate=False)

        return message_row

    async def action_clear_chat(self) -> None:
        """Leert den Chat-Verlauf."""
        container = self.query_one("#chat-container", VerticalScroll)
        await container.remove_children()
        self._add_message("✨ Chat-Verlauf wurde geleert.", "bot", "success")

    def action_focus_input(self) -> None:
        """Fokussiert das Eingabefeld."""
        self.query_one("#chat-input", Input).focus()