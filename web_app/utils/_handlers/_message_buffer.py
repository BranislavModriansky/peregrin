from threading import local
from typing import List, Tuple, Optional

class MessageBuffer:
    """Thread-safe message buffer for collecting messages from worker threads."""
    
    _instance = None
    
    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(MessageBuffer, cls).__new__(cls)
            cls._instance._thread_local = local()
        return cls._instance
    
    def add(self, level: str, message: str, detail: Optional[str] = None):
        """Add a message to the current thread's buffer."""
        buffer = self._get_buffer()
        buffer.append((level, message, detail))
    
    def flush_to_queue(self, queue):
        """Flush all buffered messages to the queue (call from main thread)."""
        if hasattr(self._thread_local, 'messages') and self._thread_local.messages:
            for level, message, detail in self._thread_local.messages:
                queue.Report(level, message, detail)
            self.clear()
    
    def clear(self):
        """Clear the message buffer for the current thread."""
        if hasattr(self._thread_local, 'messages'):
            self._thread_local.messages = []
    
    def _get_buffer(self) -> List[Tuple[str, str, Optional[str]]]:
        """Get or create a message buffer for the current thread."""
        if not hasattr(self._thread_local, 'messages'):
            self._thread_local.messages = []
        return self._thread_local.messages

# Create the singleton instance
message_buffer = MessageBuffer()