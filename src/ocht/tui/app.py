from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, TextLog
from ocht.adapters.ollama import OllamaAdapter


class ChatApp(App):
    """Simple Chat-TUI mit einem Input und einem Log-Fenster."""
    CSS = """
    Screen { layout: vertical; }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.adapter = OllamaAdapter(model="qwen3:30b-a3b", default_params={"temperature": 0.7})

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield TextLog(highlight=True, id="chat-log")
        yield Input(placeholder="Gib deine Frage ein…", id="chat-input")
        yield Footer()

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        prompt = message.value
        log = self.query_one("#chat-log", TextLog)
        log.write(f"[b]Du:[/b] {prompt}")
        adapter = OllamaAdapter()
        answer = self.adapter.send_prompt(prompt)
        log.write(f"[b]Bot:[/b] {answer}")
        # Input leeren
        message.input.value = ""
