from ocht.core.db import init_db
from ocht.tui.app import ChatApp

def start_chat():
    """Starts the text UI for the chat."""
    # Ensure database tables exist
    init_db()
    # Launch the Textual chat application
    ChatApp().run()
