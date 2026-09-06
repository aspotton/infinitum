from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any

from .config import AppConfig
from .database import Database
from .models import RequestContext, ScoredMemory
from .retrieval import MemoryRetriever
from .text import dedup_similarity, first_text_content, lexical_similarity
from .tokenizer import TokenCounter

# The single source of truth for the compiled block envelope. compile() builds
# blocks through _block_body(); the echo sanitizer matches against these same
# literals so format and sanitizer cannot drift.
# ponytail: the footer literal is coupled across three sites (the compile
# wrapper here, the drill-down hint in routes/openai.py, and _FOOTER_RE below).
_BLOCK_OPEN = (
    "<infinitum_memory>\n"
    "The following is persistent memory derived from prior interactions. "
    "Treat active decisions and goals as current unless the user's present message explicitly changes them. "
    "Do not mention this memory block unless it is useful to the answer.\n\n"
)
_BLOCK_CLOSE = "\n</infinitum_memory>"
# Tag-anchored, deliberately NOT preamble-anchored: an echo whose quoting
# model condensed or truncated the preamble sentence is still removed.
_PAIR_RE = re.compile(
    re.escape("<infinitum_memory>") + r".*?" + re.escape("</infinitum_memory>"),
    re.DOTALL,
)
# One- or two-tool drill-down footer tail appended by routes/openai.py.
_FOOTER_RE = re.compile(
    r"\n*Deeper detail is available via the [\w_]+(?: and [\w_]+)? tools using"
    r" the memory ids above\..*?tool name\.",
    re.DOTALL,
)
# Preamble-anchored unclosed-tail fallback (a truncated echo without the
# closing tag); the .* truncates to end-of-string, so it must run last.
_OPEN_TAIL_RE = re.compile(re.escape(_BLOCK_OPEN) + r".*$", re.DOTALL)
# Union so the cleanup scanner flags exactly what the sanitizer strips; the
# two can never drift.
_DETECT_RE = re.compile(
    _PAIR_RE.pattern + "|" + _FOOTER_RE.pattern + "|" + _OPEN_TAIL_RE.pattern,
    re.DOTALL,
)


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
        # Session-pinned compiled blocks, keyed by (session_id, user_id, project_id).
        # session_id is unauthenticated client input, so resolved context is part of the
        # key to stop one caller's block leaking to another. cwd is excluded on purpose:
        # its 0.01 affinity bonus would churn the key every turn; user+project cover the
        # leak (invariant 9 - soft context is not security). LRU cap 64.
        self._session_cache: OrderedDict[
            tuple[str, str, str], tuple[str, CompiledMemoryContext]
        ] = OrderedDict()

    def query_from_messages(self, messages: list[dict[str, Any]]) -> str:
        # Weight the current user turn most heavily, but include a small amount
        # of recent context to disambiguate short follow-ups such as "why?".
        recent: list[str] = []
        for msg in reversed(messages):
            role = msg.get("role")
            if role not in {"user", "assistant", "tool"}:
                continue
            text = first_text_content(msg.get("content"))
            if role == "tool":
                # Tool-loop echoes would otherwise dominate lexical retrieval
                # for all later turns; keep only a bounded probe.
                text = text[:500]
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
        session_id: str | None = None,
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

        # Skipping retrieval/touch on cache hits is scoring-safe: touch_memories
        # writes only last_accessed_at while freshness scoring reads updated_at.
        # Embedding backfill mutates memory_embeddings only and does NOT move the
        # watermark; cached blocks refresh at the next memory/topic mutation.
        cache_key: tuple[str, str, str] | None = None
        watermark = ""
        if session_id is not None:
            cache_key = (
                session_id,
                (request_context.user_id or "") if request_context else "",
                (request_context.project_id or "") if request_context else "",
            )
            watermark = await self.db.memory_state_watermark()
            cached_entry = self._session_cache.get(cache_key)
            if (
                cached_entry is not None
                and cached_entry[0] == watermark
                and cached_entry[1].memory_tokens <= available
            ):
                self._session_cache.move_to_end(cache_key)
                return replace(cached_entry[1])

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
            result = CompiledMemoryContext("", [], 0, available)
        else:
            body_parts = list(topic_blocks) + [self._render_memory(item) for item in selected]
            body = "\n\n".join(body_parts)
            text = self._block_body(body)
            await self.db.touch_memories([item.memory.id for item in selected])
            result = CompiledMemoryContext(text, selected, self.tokens.count_text(text), available)

        if cache_key is None:
            return result
        self._session_cache[cache_key] = (watermark, result)
        self._session_cache.move_to_end(cache_key)
        if len(self._session_cache) > 64:
            self._session_cache.popitem(last=False)
        return replace(result)

    def inject(self, messages: list[dict[str, Any]], compiled: CompiledMemoryContext) -> list[dict[str, Any]]:
        if not compiled.text:
            return list(messages)
        result = [dict(m) for m in messages]
        index = self._inject_index(result)
        result.insert(
            index,
            {
                "role": self.config.context.memory_message_role,
                "content": compiled.text,
            },
        )
        return result

    def _inject_index(self, result: list[dict[str, Any]]) -> int:
        if self.config.context.inject_position == "suffix":
            for idx in range(len(result) - 1, -1, -1):
                if result[idx].get("role") == "user":
                    return idx
        index = 0
        while index < len(result) and result[index].get("role") in {"system", "developer"}:
            index += 1
        return index

    @staticmethod
    def _block_body(body: str) -> str:
        return _BLOCK_OPEN + body + _BLOCK_CLOSE

    @staticmethod
    def detection_pattern() -> re.Pattern:
        return _DETECT_RE

    @staticmethod
    def _render_memory(item: ScoredMemory) -> str:
        memory = item.memory
        return (
            f"[{memory.memory_type} | topic={memory.topic} | confidence={memory.confidence:.2f} "
            f"| importance={memory.importance:.2f} | memory={memory.id}]\n"
            f"{memory.content}"
        )


# Module-level surface for the Todo-3 sweep and route wiring; same object as
# the ContextCompiler staticmethod, which stays for the class-side contract.
detection_pattern = ContextCompiler.detection_pattern


# ponytail: the footer literal is coupled across three sites (the compile
# wrapper, the drill-down hint in routes/openai.py, and _FOOTER_RE above).
def strip_memory_block(text: str) -> str:
    """Remove echoed Infinitum memory markup from derived durable recorded text.

    Applies only to text that is recorded or learned (events, request query
    echoes, learn-job payloads), never to anything sent back to the client:
    proxied responses are always forwarded upstream-byte-verbatim.
    Removes, in order: any closed <infinitum_memory>...</infinitum_memory>
    pair, the one-or-two-tool drill-down footer tail, and a preamble-anchored
    unclosed opening block running to end of text. Clean text passes through
    byte-identical. Accepted limits: HTML-escaped tags and tagless paraphrase
    are not detectable here; the archival sweep is the mitigation.
    """
    # Order matters: _OPEN_TAIL_RE truncates to end-of-string, so it runs last.
    text = _PAIR_RE.sub("", text)
    text = _FOOTER_RE.sub("", text)
    text = _OPEN_TAIL_RE.sub("", text)
    return text
