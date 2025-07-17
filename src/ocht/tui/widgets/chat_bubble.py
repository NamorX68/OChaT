from textual.widgets import Markdown

class ChatBubble(Markdown):
    def __init__(self, text: str, sender: str, **kwargs):
        style = "bubble user" if sender == "user" else "bubble bot"
        super().__init__(text, classes=style, **kwargs)