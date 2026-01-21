"""
Local token counting utilities.

Uses tiktoken when available. Falls back to a simple whitespace-based
approximation if tiktoken cannot be imported.
"""

from typing import Iterable, Dict, Any

try:
    import tiktoken
except Exception:  # pragma: no cover - fallback for environments without tiktoken
    tiktoken = None


class TokenCounter:
    """Local token counter with a best-effort tokenizer."""

    def __init__(self, model: str = "cl100k_base") -> None:
        self.model = model
        self._encoder = None
        if tiktoken is not None:
            try:
                self._encoder = tiktoken.encoding_for_model(model)
            except Exception:
                self._encoder = tiktoken.get_encoding("cl100k_base")

    def count_text(self, text: str) -> int:
        """Count tokens in a text string."""
        if not text:
            return 0
        if self._encoder is None:
            return len(text.split())
        return len(self._encoder.encode(text))

    def count_messages(self, messages: Iterable[Dict[str, Any]]) -> int:
        """Count tokens in a list of OpenAI-style chat messages."""
        total = 0
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = msg.get("content", "")
            total += self.count_text(role)
            total += self.count_text(content)
        return total
