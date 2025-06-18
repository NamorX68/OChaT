from ocht.core.db import init_db
from ocht.tui.app import ChatApp

def start_chat():
    """Startet die Text-UI für den Chat."""
    # Ensure database tables exist
    init_db()
    # Launch the Textual chat application
    ChatApp().run()