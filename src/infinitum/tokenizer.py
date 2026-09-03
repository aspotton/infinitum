from __future__ import annotations

from typing import Any

from .text import first_text_content


class TokenCounter:
    def __init__(self) -> None:
        self._encoding = None
        try:
            import tiktoken  # type: ignore

            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoding = None

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding is not None:
            return len(self._encoding.encode(text, disallowed_special=()))
        # Good enough for budgeting when no tokenizer is installed. It errs
        # slightly conservative for normal English/code mixtures.
        return max(1, (len(text) + 2) // 3)

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            total += 6  # role/framing overhead approximation
            total += self.count_text(first_text_content(msg.get("content")))
            if msg.get("name"):
                total += self.count_text(str(msg["name"]))
            if msg.get("tool_calls"):
                total += self.count_text(str(msg["tool_calls"]))
        return total
