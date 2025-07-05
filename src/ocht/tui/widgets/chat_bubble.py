from textual.widgets import Static

class ChatBubble(Static):
    def __init__(self, text: str, sender: str, **kwargs):
        style = "bubble user" if sender == "user" else "bubble bot"
        super().__init__(text, classes=style, markup=True, **kwargs)