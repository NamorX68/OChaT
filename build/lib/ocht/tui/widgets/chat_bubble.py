from textual.widgets import Markdown

class ChatBubble(Markdown):
    """Chat message bubble widget with streaming support."""
    
    def __init__(self, text: str, sender: str, streaming: bool = False, **kwargs):
        """
        Initialize chat bubble.
        
        Args:
            text: Initial message content
            sender: 'user' or 'bot'
            streaming: Whether this bubble supports live content updates
            **kwargs: Additional arguments passed to Markdown
        """
        style = "bubble user" if sender == "user" else "bubble bot"
        super().__init__(text, classes=style, **kwargs)
        self.sender = sender
        self.streaming = streaming
        self._content = text
        self._is_finalized = False
    
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
            
        self.update(display_content)
    
    def finalize(self) -> None:
        """
        Finalize the bubble content (remove typing indicators).
        """
        if not self.streaming:
            return
            
        self._is_finalized = True
        self.update(self._content)  # Remove cursor indicator
    
    def get_content(self) -> str:
        """
        Get the current content without indicators.
        
        Returns:
            The actual message content
        """
        return self._content