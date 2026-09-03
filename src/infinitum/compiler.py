from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import AppConfig
from .database import Database
from .models import RequestContext, ScoredMemory
from .retrieval import MemoryRetriever
from .text import dedup_similarity, first_text_content, lexical_similarity
from .tokenizer import TokenCounter


@dataclass(slots=True)
class CompiledMemoryContext:
    text: str
    memories: list[ScoredMemory]
    memory_tokens: int
    available_memory_tokens: int


class ContextCompiler:
    def __init__(
        self,
        db: Database,
        retriever: MemoryRetriever,
        tokens: TokenCounter,
        config: AppConfig,
    ):
        self.db = db
        self.retriever = retriever
        self.tokens = tokens
        self.config = config

    def query_from_messages(self, messages: list[dict[str, Any]]) -> str:
        # Weight the current user turn most heavily, but include a small amount
        # of recent context to disambiguate short follow-ups such as "why?".
        recent: list[str] = []
        for msg in reversed(messages):
            role = msg.get("role")
            if role not in {"user", "assistant", "tool"}:
                continue
            text = first_text_content(msg.get("content"))
            if text:
                recent.append(text)
            if len(recent) >= 4:
                break
        recent.reverse()
        return "\n".join(recent)

    async def compile(
        self,
        messages: list[dict[str, Any]],
        request_context: RequestContext | None = None,
    ) -> CompiledMemoryContext:
        if not self.config.memory.enabled:
            return CompiledMemoryContext("", [], 0, 0)

        query = self.query_from_messages(messages)
        if not query:
            return CompiledMemoryContext("", [], 0, 0)

        existing_tokens = self.tokens.count_messages(messages)
        available = min(
            self.config.context.max_memory_tokens,
            max(
                0,
                self.config.context.model_context_window
                - self.config.context.reserve_output_tokens
                - self.config.context.reserve_free_tokens
                - existing_tokens,
            ),
        )
        if available <= 0:
            return CompiledMemoryContext("", [], 0, available)

        candidates = await self.retriever.search(query, request_context=request_context)
        selected: list[ScoredMemory] = []
        used_tokens = 0

        topic_blocks: list[str] = []
        covered_topics: set[str] = set()
        for topic in await self.db.list_topics(limit=100):
            relevance = lexical_similarity(query, f"{topic.topic} {topic.summary}")
            if relevance < 0.18:
                continue
            block = f"[topic-summary | topic={topic.topic} | memories={topic.memory_count}]\n{topic.summary}"
            cost = self.tokens.count_text(block)
            if used_tokens + cost > min(available, max(2048, available // 4)):
                continue
            topic_blocks.append(block)
            covered_topics.add(topic.topic)
            used_tokens += cost
            if len(topic_blocks) >= 3:
                break

        topic_detail_counts: dict[str, int] = {}
        for candidate in candidates:
            memory = candidate.memory
            if memory.topic in covered_topics and topic_detail_counts.get(memory.topic, 0) >= 5:
                continue
            if any(
                dedup_similarity(memory.content, chosen.memory.content)
                >= self.config.memory.dedup_similarity
                for chosen in selected
            ):
                continue

            rendered = self._render_memory(candidate)
            cost = self.tokens.count_text(rendered)
            if used_tokens + cost > available:
                continue
            selected.append(candidate)
            topic_detail_counts[memory.topic] = topic_detail_counts.get(memory.topic, 0) + 1
            used_tokens += cost
            if len(selected) >= self.config.memory.inject_max_memories:
                break

        if not selected and not topic_blocks:
            return CompiledMemoryContext("", [], 0, available)

        body_parts = list(topic_blocks) + [self._render_memory(item) for item in selected]
        body = "\n\n".join(body_parts)
        text = (
            "<infinitum_memory>\n"
            "The following is persistent memory derived from prior interactions. "
            "Treat active decisions and goals as current unless the user's present message explicitly changes them. "
            "Do not mention this memory block unless it is useful to the answer.\n\n"
            f"{body}\n"
            "</infinitum_memory>"
        )
        await self.db.touch_memories([item.memory.id for item in selected])
        return CompiledMemoryContext(text, selected, self.tokens.count_text(text), available)

    def inject(self, messages: list[dict[str, Any]], compiled: CompiledMemoryContext) -> list[dict[str, Any]]:
        if not compiled.text:
            return list(messages)
        result = [dict(m) for m in messages]
        index = 0
        while index < len(result) and result[index].get("role") in {"system", "developer"}:
            index += 1
        result.insert(
            index,
            {
                "role": self.config.context.memory_message_role,
                "content": compiled.text,
            },
        )
        return result

    @staticmethod
    def _render_memory(item: ScoredMemory) -> str:
        memory = item.memory
        return (
            f"[{memory.memory_type} | topic={memory.topic} | confidence={memory.confidence:.2f} "
            f"| importance={memory.importance:.2f} | memory={memory.id}]\n"
            f"{memory.content}"
        )
