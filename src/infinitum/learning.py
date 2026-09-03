from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING, Any

from .config import AppConfig
from .database import Database
from .embeddings import EmbeddingClient
from .models import Memory, MemoryCandidate, RequestContext, TopicSummary
from .retrieval import MemoryRetriever
from .text import compact_whitespace, lexical_similarity
from .upstream import UpstreamClient, extract_nonstream_assistant

if TYPE_CHECKING:
    from .runtime import ActiveRequestCounter

log = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.S)


class MemoryLearner:
    def __init__(
        self,
        db: Database,
        retriever: MemoryRetriever,
        embeddings: EmbeddingClient,
        upstream: UpstreamClient,
        config: AppConfig,
    ):
        self.db = db
        self.retriever = retriever
        self.embeddings = embeddings
        self.upstream = upstream
        self.config = config

    async def learn(self, payload: dict[str, Any]) -> None:
        """Learn durable memories from one completed interaction.

        This is the "every interaction" timescale. The extraction LLM sees the
        current interaction plus only the nearest existing memories. Topic
        summaries are *not* rebuilt here. Instead changed memories mark a topic
        dirty and schedule a debounced incremental summary job.
        """

        query = payload.get("user_text", "")
        assistant = payload.get("assistant_text", "")
        model = self.config.learning.model or payload.get("model") or ""
        if not model or not (query or assistant):
            return

        context_payload = payload.get("request_context")
        try:
            request_context = RequestContext.model_validate(
                context_payload if isinstance(context_payload, dict) else {}
            )
        except Exception:
            request_context = RequestContext()

        existing = await self.retriever.search(
            f"{query}\n{assistant}", limit=12, request_context=request_context
        )
        existing_lines = [
            f"{x.memory.id} | {x.memory.memory_type} | {x.memory.topic} | {x.memory.content}"
            for x in existing
        ]
        context_lines = []
        if request_context.user_id:
            context_lines.append(f"user_id={request_context.user_id}")
        if request_context.project_id:
            context_lines.append(f"project_id={request_context.project_id}")
        if request_context.cwd:
            context_lines.append(f"cwd={request_context.cwd}")
        context_text = "\n".join(context_lines) or "(none)"

        prompt = f"""Extract durable memories from this interaction. Return JSON only. Do not call tools or functions.
Do not save transient chit-chat, guesses, assistant inventions, or obvious restatements.
Prefer concise current-state facts, decisions, preferences, goals, procedures, lessons, or episodic events.
If the user explicitly corrects or replaces an existing memory, set operation_hint='supersede', explicit_correction=true, and list only relevant existing memory IDs.
If this merely confirms an existing memory, use operation_hint='reinforce', set reinforces_memory_id to that existing memory ID, and copy that memory's memory_type and topic exactly.
The request context below is provenance/affinity metadata. Use it only to disambiguate nearby memories; do not save the user ID, project ID, or CWD as a memory unless the interaction explicitly discusses them.

Request context:\n{context_text}

Existing nearby memories:\n{chr(10).join(existing_lines) or '(none)'}

Interaction:\nUSER: {query}\nASSISTANT: {assistant}

Schema:
{{"memories":[{{"memory_type":"fact|decision|preference|goal|procedure|lesson|episodic","topic":"short-stable-topic","content":"durable statement","importance":0.0,"confidence":0.0,"operation_hint":"new|reinforce|supersede","reinforces_memory_id":null,"supersedes_memory_ids":[],"explicit_correction":false,"reason":"brief"}}]}}
"""
        result = await self.upstream.learning_chat_completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a conservative persistent-memory extraction engine. Output strict JSON directly in assistant content. Do not call tools or functions.",
                },
                {"role": "user", "content": prompt},
            ],
            base_url=self.config.learning.base_url or None,
            api_key=self.config.learning.api_key,
            timeout_seconds=self.config.learning.timeout_seconds,
            max_tokens=self.config.learning.max_tokens,
            extra_body=self.config.learning.extra_body,
        )
        raw, metadata = extract_nonstream_assistant(result)
        if raw.strip():
            obj = self._parse_json(raw)
        else:
            # Some OpenAI-compatible servers/tool parsers return a perfectly
            # usable structured model answer as message.tool_calls with
            # finish_reason="tool_calls", even though Infinitum did not provide
            # any tools. Treat schema-shaped function arguments as another model
            # output transport rather than silently dropping the learning turn.
            obj = self._memory_payload_from_tool_calls(metadata)
            if obj:
                log.info(
                    "memory extraction used schema-shaped tool-call arguments (%s)",
                    self._response_diagnostics(result),
                )
            else:
                log.warning(
                    "memory extraction model returned no usable final content; no candidates extracted (%s). "
                    "If finish_reason is 'tool_calls', consider setting learning.extra_body.tool_choice='none'; "
                    "if this is a reasoning model, also consider disabling thinking or increasing learning.max_tokens",
                    self._response_diagnostics(result),
                )
                return
        candidates: list[MemoryCandidate] = []
        for item in obj.get("memories", []) if isinstance(obj, dict) else []:
            try:
                candidate = MemoryCandidate.model_validate(item)
                candidate.content = compact_whitespace(candidate.content)
                candidate.topic = compact_whitespace(candidate.topic.lower()) or "general"
                candidate.importance = min(1.0, max(0.0, candidate.importance))
                candidate.confidence = min(1.0, max(0.0, candidate.confidence))
                if candidate.content:
                    candidates.append(candidate)
            except Exception:
                continue

        source_ids = [x for x in payload.get("source_event_ids", []) if x]
        allowed_ids = {x.memory.id for x in existing}
        changed_by_topic: dict[str, set[str]] = {}
        for candidate in candidates:
            affected_ids = await self._apply(
                candidate, source_ids, allowed_ids, request_context=request_context
            )
            if affected_ids:
                changed_by_topic.setdefault(candidate.topic, set()).update(affected_ids)

        if self.config.learning.topic_summaries:
            for topic, memory_ids in changed_by_topic.items():
                await self.db.mark_topic_dirty(
                    topic,
                    sorted(memory_ids),
                    model=model,
                    debounce_seconds=self.config.learning.topic_summary_debounce_seconds,
                    update_threshold=self.config.learning.topic_summary_update_threshold,
                )

    def _parse_json(self, text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        try:
            return json.loads(text)
        except Exception:
            match = _JSON_RE.search(text)
            if not match:
                return {}
            try:
                return json.loads(match.group(0))
            except Exception:
                return {}

    def _memory_payload_from_tool_calls(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Recover memory JSON from schema-shaped tool/function arguments.

        A few OpenAI-compatible model servers run an automatic tool-call parser
        even when the caller did not provide tools. Qwen-family models can then
        emit the requested JSON through ``message.tool_calls`` and leave
        ``message.content`` empty. We only accept arguments that already match
        Infinitum's memory-extraction shape; arbitrary tool calls are ignored.
        """

        payloads: list[Any] = []
        tool_calls = metadata.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if isinstance(function, dict):
                    payloads.append(function.get("arguments"))

        function_call = metadata.get("function_call")
        if isinstance(function_call, dict):
            payloads.append(function_call.get("arguments"))

        for payload in payloads:
            parsed: Any = payload
            if isinstance(payload, str):
                parsed = self._parse_json(payload)
            if isinstance(parsed, list):
                parsed = {"memories": parsed}
            if not isinstance(parsed, dict):
                continue

            memories = parsed.get("memories")
            if isinstance(memories, list):
                return parsed

            # Be tolerant of a single candidate object as tool arguments while
            # still requiring the fields that identify an actual memory record.
            if isinstance(parsed.get("content"), str) and parsed.get("memory_type"):
                return {"memories": [parsed]}

        return {}

    @staticmethod
    def _summary_from_tool_calls(metadata: dict[str, Any]) -> str:
        """Recover a plain topic summary from simple tool-call arguments."""

        tool_calls = metadata.get("tool_calls")
        calls = tool_calls if isinstance(tool_calls, list) else []
        function_call = metadata.get("function_call")
        if isinstance(function_call, dict):
            calls = [*calls, {"function": function_call}]

        for call in calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            args = function.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    continue
            if not isinstance(args, dict):
                continue
            for key in ("summary", "text", "content"):
                value = args.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return ""

    async def _apply(
        self,
        candidate: MemoryCandidate,
        source_ids: list[str],
        allowed_ids: set[str],
        request_context: RequestContext | None = None,
    ) -> set[str]:
        """Apply an LLM-proposed candidate and return every affected memory ID."""

        matches = await self.retriever.search(
            candidate.content, limit=12, request_context=request_context
        )
        compatible = [
            m
            for m in matches
            if m.memory.topic == candidate.topic
            and m.memory.memory_type == candidate.memory_type
            and m.memory.status == "active"
        ]

        # Prefer an explicit reinforcement target selected from the bounded set
        # of nearby memories shown to the extraction model. It is only a hint:
        # deterministic compatibility/similarity checks still gate mutation.
        best = None
        if candidate.reinforces_memory_id and candidate.reinforces_memory_id in allowed_ids:
            best = next(
                (m for m in compatible if m.memory.id == candidate.reinforces_memory_id),
                None,
            )
        if best is None and compatible:
            best = compatible[0]

        reinforcement_reason = self._reinforcement_reason(candidate, best)
        if reinforcement_reason and best is not None:
            lexical = lexical_similarity(candidate.content, best.memory.content)
            reinforced = await self.db.reinforce_memory(
                best.memory.id,
                confidence=candidate.confidence,
                importance=candidate.importance,
                source_event_ids=source_ids,
                reinforcement_metadata={
                    "method": reinforcement_reason,
                    "operation_hint": candidate.operation_hint,
                    "lexical_similarity": round(lexical, 4),
                    "semantic_similarity": round(best.semantic_score, 4),
                    "retrieval_score": round(best.score, 4),
                    "request_context": request_context.compact() if request_context else {},
                },
            )
            if reinforced is not None:
                return {best.memory.id}

        new_memory = Memory(
            memory_type=candidate.memory_type,
            topic=candidate.topic,
            content=candidate.content,
            importance=candidate.importance,
            confidence=candidate.confidence,
            source_event_ids=source_ids,
            metadata={
                "extraction_reason": candidate.reason,
                "origin_context": request_context.compact() if request_context else {},
            },
        )
        await self.db.create_memory(new_memory)
        vector = await self.embeddings.embed(new_memory.content)
        if vector is not None:
            await self.db.set_embedding(new_memory.id, self.config.embeddings.model, vector)

        affected = {new_memory.id}
        if candidate.operation_hint == "supersede":
            for old_id in candidate.supersedes_memory_ids:
                if old_id not in allowed_ids:
                    continue
                old = await self.db.get_memory(old_id)
                if not old or old.status != "active" or old.topic != candidate.topic:
                    continue
                related = (
                    lexical_similarity(old.content, candidate.content)
                    >= self.config.memory.supersede_similarity_floor
                )
                if related or candidate.explicit_correction:
                    await self.db.supersede_memory(old.id, new_memory.id)
                    affected.add(old.id)
        return affected

    def _reinforcement_reason(self, candidate: MemoryCandidate, best: Any) -> str | None:
        """Return the deterministic reason a candidate may reinforce a memory.

        Reinforcement is intentionally stricter than retrieval. Exact type and
        topic compatibility are established before this helper is called. A
        supersede/correction proposal can never be converted into reinforcement
        merely because the new wording is similar to the old state.
        """

        if best is None or candidate.operation_hint == "supersede" or candidate.explicit_correction:
            return None

        lexical = lexical_similarity(candidate.content, best.memory.content)
        semantic = best.semantic_score
        cfg = self.config.memory

        if lexical >= cfg.reinforce_similarity:
            return "lexical_equivalence"

        if semantic >= cfg.reinforce_semantic_similarity:
            return "semantic_equivalence"

        if candidate.operation_hint == "reinforce":
            evidence_ok = (
                lexical >= cfg.reinforce_hint_min_lexical
                or semantic >= cfg.reinforce_hint_min_semantic
            )
            if best.score >= cfg.reinforce_hint_min_score and evidence_ok:
                if candidate.reinforces_memory_id == best.memory.id:
                    return "model_targeted_reinforce"
                return "model_reinforce_hint"

        return None

    async def refresh_topic_summary(self, topic: str, model: str) -> bool:
        """Incrementally refresh one dirty topic summary.

        This is the "incremental" timescale. A successful call consumes only a
        bounded batch of changed memory IDs. The prompt contains the current
        summary, those changed records, and a small amount of current topic
        context. It does not resend the entire topic corpus on each turn.
        """

        dirty_updates = await self.db.get_topic_updates(
            topic, limit=self.config.learning.topic_summary_max_changed_memories
        )
        if not dirty_updates:
            return False
        changed_ids = [memory_id for memory_id, _created_at in dirty_updates]

        active_count = await self.db.count_active_topic_memories(topic)
        if active_count < self.config.learning.topic_summary_min_memories:
            # There is not enough evidence to justify a summary yet. Consume the
            # current dirty batch; future memories will mark the topic dirty again.
            await self.db.clear_topic_updates(topic, dirty_updates)
            return False

        current = await self.db.get_topic(topic)
        changed = await self.db.list_memories_by_ids(changed_ids)

        if current is None:
            # Bootstrap once from a bounded recent sample. Subsequent updates are
            # incremental, so topic growth does not make every learning call larger.
            bootstrap = await self.db.list_active_topic_memories(
                topic, limit=self.config.learning.topic_summary_bootstrap_max_memories
            )
            lines = "\n".join(
                f"- [{m.status} | {m.memory_type}] {m.content}" for m in bootstrap
            )
            prompt = f"""Create a compact current-state persistent-memory summary for this topic.
Preserve meaningful uncertainty or disagreement. Do not invent facts. Prefer current active state over obsolete history.

Topic: {topic}
Active memory sample ({len(bootstrap)} of {active_count} active memories):
{lines}
"""
        else:
            changed_set = set(changed_ids)
            context_memories = await self.db.list_active_topic_memories(
                topic,
                limit=self.config.learning.topic_summary_context_memories
                + len(changed_ids),
            )
            context_memories = [
                memory
                for memory in context_memories
                if memory.id not in changed_set
            ][: self.config.learning.topic_summary_context_memories]

            changed_lines = "\n".join(
                f"- {m.id} | status={m.status} | type={m.memory_type} | {m.content}"
                + (f" | superseded_by={m.superseded_by}" if m.superseded_by else "")
                for m in changed
            ) or "(none)"
            context_lines = "\n".join(
                f"- [{m.memory_type}] {m.content}" for m in context_memories
            ) or "(none)"
            prompt = f"""Update an existing persistent-memory topic summary using the changed memory records below.
Preserve still-valid information from the current summary. Incorporate new active information. If a changed record is superseded, remove or qualify outdated statements it previously supported. Preserve meaningful uncertainty or disagreement. Do not invent facts. Return only the updated summary text. Do not call tools or functions.

Topic: {topic}

Current summary:
{current.summary}

Changed memory records:
{changed_lines}

Small current-topic context sample:
{context_lines}
"""

        result = await self.upstream.learning_chat_completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Maintain a compact, accurate current-state topic summary from persistent memory. Return plain text directly in assistant content. Do not call tools or functions.",
                },
                {"role": "user", "content": prompt},
            ],
            base_url=self.config.learning.base_url or None,
            api_key=self.config.learning.api_key,
            timeout_seconds=self.config.learning.timeout_seconds,
            max_tokens=self.config.learning.topic_summary_max_tokens,
            extra_body=self.config.learning.extra_body,
        )
        summary, metadata = extract_nonstream_assistant(result)
        summary = summary.strip()
        if not summary:
            summary = self._summary_from_tool_calls(metadata)
            if summary:
                log.info(
                    "topic summary used tool-call arguments for topic %r (%s)",
                    topic,
                    self._response_diagnostics(result),
                )
        if not summary:
            # Empty final content is common with reasoning-capable local models
            # when the completion budget is consumed by hidden/reasoning tokens,
            # and some tool parsers can also produce unusable tool calls. Replaying
            # the identical durable job several times is both expensive and
            # unlikely to help. Topic summaries are an optimization, while
            # detailed active memories remain authoritative, so degrade safely to
            # a bounded deterministic active-memory summary instead.
            summary = await self._deterministic_topic_fallback(topic, active_count)
            log.warning(
                "topic summary model returned no usable final content for topic %r; "
                "using deterministic fallback instead of retrying (%s)",
                topic,
                self._response_diagnostics(result),
            )

        await self.db.upsert_topic(
            TopicSummary(topic=topic, summary=summary, memory_count=active_count)
        )
        # Clear only the exact dirty IDs captured before the model call. New
        # changes that arrived while the summary was running stay dirty.
        await self.db.clear_topic_updates(topic, dirty_updates)
        return (await self.db.count_topic_updates(topic)) > 0

    async def _deterministic_topic_fallback(self, topic: str, active_count: int) -> str:
        """Build a bounded no-LLM summary when the model returns no final text.

        This deliberately contains only active canonical memory content. It is
        less compact than a model-generated summary but cannot invent state and
        prevents a permanently empty reasoning-model response from retrying the
        same topic job up to ``max_attempts`` times.
        """

        limit = max(1, self.config.learning.topic_summary_fallback_memories)
        memories = await self.db.list_active_topic_memories(topic, limit=limit)
        if not memories:
            return "No active memories currently remain for this topic."
        lines = [
            f"- [{memory.memory_type}] {compact_whitespace(memory.content)[:800]}"
            for memory in memories
        ]
        header = "Current active memory state"
        if active_count > len(memories):
            header += f" (bounded sample of {len(memories)} of {active_count} active memories)"
        return f"{header}:\n" + "\n".join(lines)

    @staticmethod
    def _response_diagnostics(result: dict[str, Any]) -> str:
        """Return non-content diagnostics for an empty learning response."""

        choices = result.get("choices") or []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice, dict) else {}
        message = message if isinstance(message, dict) else {}
        reasoning = message.get("reasoning_content")
        if reasoning is None:
            reasoning = message.get("reasoning")
        reasoning_chars = len(reasoning) if isinstance(reasoning, str) else 0
        tool_calls = message.get("tool_calls")
        tool_calls = tool_calls if isinstance(tool_calls, list) else []
        tool_names: list[str] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                tool_names.append(function["name"][:80])
        if isinstance(message.get("function_call"), dict):
            name = message["function_call"].get("name")
            if isinstance(name, str):
                tool_names.append(name[:80])
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        return (
            f"finish_reason={choice.get('finish_reason')!r}, "
            f"tool_calls={len(tool_calls)}, tool_names={tool_names!r}, "
            f"reasoning_chars={reasoning_chars}, completion_tokens={completion_tokens!r}"
        )


class LearningWorker:
    def __init__(
        self,
        db: Database,
        learner: MemoryLearner,
        config: AppConfig,
        active_requests: ActiveRequestCounter | None = None,
    ):
        self.db = db
        self.learner = learner
        self.config = config
        self._active_requests = active_requests
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self.config.learning.enabled and self._task is None:
            self._task = asyncio.create_task(
                self._run(), name="infinitum-learning-worker"
            )

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task
            self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            if (
                self.config.learning.skip_when_upstream_busy
                and self._active_requests is not None
                and self._active_requests.value > 0
            ):
                # Defer job START only; a check->claim race with a request
                # starting right now is harmless. The queue is durable, so the
                # skipped work runs on a later poll once the upstream is idle.
                log.debug("deferring learning job start; upstream busy")
                try:
                    await asyncio.wait_for(
                        self._stop.wait(),
                        timeout=self.config.learning.poll_interval_seconds,
                    )
                except TimeoutError:
                    pass
                continue
            job = await self.db.claim_job()
            if not job:
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self.config.learning.poll_interval_seconds
                    )
                except asyncio.TimeoutError:
                    pass
                continue
            try:
                followup_topic: str | None = None
                followup_model: str = ""
                if job["job_type"] == "learn_interaction":
                    await self.learner.learn(job["payload"])
                elif job["job_type"] == "refresh_topic_summary":
                    topic = str(job["payload"].get("topic") or "")
                    model = (
                        self.config.learning.model
                        or str(job["payload"].get("model") or "")
                    )
                    if topic and model:
                        if await self.learner.refresh_topic_summary(topic, model):
                            followup_topic = topic
                            followup_model = model
                await self.db.finish_job(job["id"])
                if followup_topic:
                    # The just-finished job is no longer "running", so remaining
                    # dirty rows can safely create one debounced follow-up job.
                    await self.db.ensure_topic_summary_job(
                        followup_topic,
                        model=followup_model,
                        debounce_seconds=self.config.learning.topic_summary_debounce_seconds,
                        update_threshold=self.config.learning.topic_summary_update_threshold,
                    )
            except Exception as exc:
                log.exception("learning job %s failed", job["id"])
                retry = job["attempts"] < self.config.learning.max_attempts
                await self.db.fail_job(
                    job["id"],
                    str(exc),
                    retry=retry,
                    delay_seconds=min(60.0, 2.0 ** job["attempts"]),
                )
