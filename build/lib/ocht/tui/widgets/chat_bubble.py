from textual.widgets import Markdown
from textual.containers import Container
from textual.binding import Binding
import re
import pyperclip

class ChatBubble(Container):
    """Chat message bubble widget with Markdown rendering AND text selection support."""
    
    BINDINGS = [
        Binding("ctrl+c", "copy_content", "Kopieren"),
        Binding("c", "toggle_copy_mode", "Kopiermodus"),
    ]
    
    def __init__(self, text: str, sender: str, streaming: bool = False, **kwargs):
        """
        Initialize chat bubble.
        
        Args:
            text: Initial message content
            sender: 'user' or 'bot'
            streaming: Whether this bubble supports live content updates
            **kwargs: Additional arguments passed to Container
        """
        super().__init__(**kwargs)
        
        # Set styling classes
        self.add_class("bubble")
        self.add_class("user" if sender == "user" else "bot")
        
        # Force compact sizing
        self.styles.height = "auto"
        self.styles.max_height = "20"
        
        self.sender = sender
        self.streaming = streaming
        self._content = text
        self._is_finalized = False
        self._copy_mode = False
        
        # Store initial text
        self._initial_text = text
    
    def compose(self):
        """Compose the chat bubble with Markdown rendering."""
        # Einfache Struktur nur mit Markdown
        yield Markdown(
            self._initial_text,
            id="bubble-markdown",
            classes=f"bubble-content {self.sender}-bubble-content"
        )
    
    def update_content(self, new_content: str) -> None:
        """
        Update the content of the bubble (for streaming).
        
        Args:
            new_content: New complete content to display
        """
        if not self.streaming or self._is_finalized:
            return
            
        self._content = new_content
        
        # Add typing indicator for bot messages while streaming
        display_content = new_content
        if self.sender == "bot" and not self._is_finalized:
            display_content += " ▋"  # Cursor indicator
        
        # Update Markdown content
        try:
            markdown_widget = self.query_one("#bubble-markdown", Markdown)
            markdown_widget.update(display_content)
        except Exception:
            # Markdown might not be composed yet
            self._initial_text = display_content
    
    def finalize(self) -> None:
        """
        Finalize the bubble content (remove typing indicators).
        """
        if not self.streaming:
            return
            
        self._is_finalized = True
        
        # Update Markdown with final content (no cursor indicator)
        try:
            markdown_widget = self.query_one("#bubble-markdown", Markdown)
            markdown_widget.update(self._content)
        except Exception:
            # Markdown might not be composed yet
            self._initial_text = self._content
    
    def get_content(self) -> str:
        """
        Get the current content without indicators.
        
        Returns:
            The actual message content
        """
        return self._content
    
    
    def action_copy_content(self):
        """Copies the entire bubble content to clipboard."""
        try:
            # Konvertiere Markdown zu Plain Text für bessere Kopierbarkeit
            plain_text = self._markdown_to_plain_text(self._content)
            pyperclip.copy(plain_text)
            self.app.notify("Text kopiert!", severity="information")
        except Exception as e:
            self.app.notify(f"Fehler beim Kopieren: {e}", severity="error")
    
    def action_toggle_copy_mode(self):
        """Toggles copy mode to show raw markdown."""
        self._copy_mode = not self._copy_mode
        
        if self._copy_mode:
            # Zeige Raw Markdown für einfaches Kopieren
            try:
                markdown_widget = self.query_one("#bubble-markdown", Markdown)
                # Temporär Raw Text anzeigen
                markdown_widget.update(f"```\n{self._content}\n```")
                self.app.notify("Copy-Modus: Raw Markdown", severity="information")
            except Exception:
                pass
        else:
            # Zurück zur normalen Markdown-Darstellung
            try:
                markdown_widget = self.query_one("#bubble-markdown", Markdown)
                markdown_widget.update(self._content)
                self.app.notify("Normal-Modus: Rendered Markdown", severity="information")
            except Exception:
                pass
    
    def _markdown_to_plain_text(self, markdown_text: str) -> str:
        """
        Konvertiert Markdown zu lesbarem Plain Text.
        Behält Code-Blöcke bei für bessere Kopierbarkeit.
        """
        # Bewahre Code-Blöcke
        code_blocks = []
        def preserve_code(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks)-1}__"
        
        # Extrahiere Code-Blöcke
        text = re.sub(r'```[\s\S]*?```', preserve_code, markdown_text)
        
        # Entferne andere Markdown-Formatierung
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Inline code
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # Headers
        text = re.sub(r'^\* ', '• ', text, flags=re.MULTILINE)  # Lists
        
        # Setze Code-Blöcke zurück
        for i, block in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{i}__", block)
        
        return text
    
    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        # Ensure Markdown has the correct initial content
        if hasattr(self, '_initial_text'):
            try:
                markdown_widget = self.query_one("#bubble-markdown", Markdown)
                markdown_widget.update(self._initial_text)
            except Exception:
                pass  # Markdown might not be ready yet