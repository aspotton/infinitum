"""Periodic consolidation: clustering primitives + the one-pass consolidator.

:func:`build_clusters` stays a pure, deterministic function so benchmarks can
reuse it; all thresholds are passed in by the caller (the consolidator wires
``memory.dedup_similarity`` and ``memory.reinforce_semantic_similarity``).
:class:`MemoryConsolidator` below is the runtime "periodic" timescale: it
makes exactly ONE bounded LLM proposal call per topic pass and applies merges
only through the deterministic guarded reinforce/supersede primitives the
learner already trusts (invariant 2: the LLM proposes, code mutates).
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from infinitum.database import _CONSOLIDATION_META_PREFIX
from infinitum.models import Event, Memory, utc_now
from infinitum.text import lexical_similarity
from infinitum.upstream import extract_nonstream_assistant

if TYPE_CHECKING:
    from infinitum.config import AppConfig
    from infinitum.database import Database
    from infinitum.upstream import UpstreamClient

log = logging.getLogger(__name__)

# Churn counting compares against an ISO string; a missing checkpoint means
# "never consolidated", i.e. the epoch.
_EPOCH_ISO = "0001-01-01T00:00:00+00:00"
_SESSION_ID = "consolidation"
_JSON_RE = re.compile(r"\{.*\}", re.S)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors; 0.0 when either is zero-length.

    Vectors of mismatched length are compared over their common prefix.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_clusters(
    memories: list[Memory],
    embeddings: dict[str, list[float]] | None,
    lexical_floor: float,
    semantic_floor: float,
) -> list[list[Memory]]:
    """Group near-duplicate memories via greedy union-find clustering.

    An edge joins a pair iff they share ``memory_type`` AND either
    ``lexical_similarity(a.content, b.content) >= lexical_floor`` or both ids
    have embeddings whose cosine ``>= semantic_floor``. Transitive links merge
    into one cluster. Only multi-member clusters are returned; members are
    sorted by id and clusters are sorted by their first member id, so the
    output is fully deterministic regardless of input order.
    """
    parent = list(range(len(memories)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for i in range(len(memories)):
        a = memories[i]
        for j in range(i + 1, len(memories)):
            b = memories[j]
            if a.memory_type != b.memory_type:
                continue
            if lexical_similarity(a.content, b.content) >= lexical_floor:
                union(i, j)
                continue
            if embeddings:
                va = embeddings.get(a.id)
                vb = embeddings.get(b.id)
                if va is not None and vb is not None and _cosine(va, vb) >= semantic_floor:
                    union(i, j)

    groups: dict[int, list[Memory]] = {}
    for i, mem in enumerate(memories):
        groups.setdefault(find(i), []).append(mem)
    clusters = [sorted(group, key=lambda m: m.id) for group in groups.values() if len(group) > 1]
    clusters.sort(key=lambda c: c[0].id)
    return clusters


ProposalFn = Callable[..., Awaitable[dict[str, Any]]]


class MemoryConsolidator:
    """One bounded LLM proposal per topic pass; deterministic code owns mutation.

    Lease guarantee: exactly ONE upstream call per pass covers ALL clusters, so
    one call bounded by ``learning.timeout_seconds`` plus local DB work stays
    well under the stale-adoption lease of ``timeout_seconds + 60``
    (learning.STALE_LOCK_GRACE_SECONDS). Never loop the upstream per cluster.
    """

    def __init__(
        self,
        db: Database,
        upstream: UpstreamClient,
        config: AppConfig,
        proposal_fn: ProposalFn | None = None,
    ):
        self.db = db
        self.upstream = upstream
        self.config = config
        # Test seam: replaces the LLM round-trip with an async callable that
        # receives the clustered working set and returns the parsed proposal.
        self.proposal_fn = proposal_fn

    async def consolidate_topic(
        self, topic: str, model: str, *, continuation: bool = False
    ) -> dict[str, Any]:
        """Run one periodic consolidation pass over a single topic.

        ``continuation`` bypasses ONLY the churn floor: the checkpoint is
        stamped after this pass's own mutations, so merged rows never count as
        churn and a rotation continuation pass over an oversized topic would
        otherwise silently no-op itself. The working-set gate and every apply
        guard still apply.
        """
        cfg = self.config.learning
        checkpoint = await self.db.get_meta_value(_CONSOLIDATION_META_PREFIX + topic)
        changed = await self.db.count_active_topic_memories_updated_since(
            topic, checkpoint or _EPOCH_ISO
        )
        working = await self.db.list_active_topic_memories_oldest(
            topic, cfg.consolidation_max_memories_per_pass
        )
        stats: dict[str, Any] = {
            "changed": changed,
            "clusters": 0,
            "merged": 0,
            "conflicts": 0,
            "skipped": 0,
            "no_op": False,
        }
        if (not continuation and changed < cfg.consolidation_min_changed_memories) or len(
            working
        ) < 2:
            stats["no_op"] = True
            await self._audit(topic, stats)
            return stats

        embeddings_map: dict[str, Any] | None = None
        if self.config.embeddings.enabled:
            stored = await self.db.get_embeddings([m.id for m in working])
            embeddings_map = {mid: vector for mid, (_model, vector) in stored.items()}
        clusters = build_clusters(
            working,
            embeddings_map,
            self.config.memory.dedup_similarity,
            self.config.memory.reinforce_semantic_similarity,
        )
        stats["clusters"] = len(clusters)
        if clusters:
            if self.proposal_fn is not None:
                proposal = await self.proposal_fn(clusters, topic=topic, model=model)
            else:
                proposal = await self._propose(clusters, topic=topic, model=model)
            entries = proposal.get("clusters", []) if isinstance(proposal, dict) else []
            survivors = await self._apply(topic, clusters, entries, stats)
            if survivors:
                await self.db.mark_topic_dirty(
                    topic,
                    survivors,
                    model=model,
                    debounce_seconds=cfg.topic_summary_debounce_seconds,
                    update_threshold=cfg.topic_summary_update_threshold,
                )
        await self._audit(topic, stats)
        if len(working) == cfg.consolidation_max_memories_per_pass and stats["merged"]:
            await self.db.enqueue_job(
                "consolidate_topic",
                {"topic": topic, "model": model, "continuation": True},
            )
        return stats

    async def _propose(
        self, clusters: list[list[Memory]], *, topic: str, model: str
    ) -> dict[str, Any]:
        """One upstream call for the whole pass; strict-JSON merge proposals."""
        blocks = []
        for index, cluster in enumerate(clusters):
            lines = "\n".join(
                f"- {m.id} | {m.memory_type} | observation_count={m.observation_count} | {m.content}"
                for m in cluster
            )
            blocks.append(f"Cluster {index}:\n{lines}")
        prompt = f"""Consolidate the near-duplicate memories in each cluster below into one canonical durable statement.
For each cluster output exactly one entry. survivor_id must be one of that cluster's member ids; canonical_content must merge the durable detail of the members without inventing facts. If members contradict each other, set conflict=true and leave canonical_content empty; never merge contradictions silently.
Return strict JSON directly in assistant content. Do not call tools or functions.

Topic: {topic}

{chr(10).join(blocks)}

Schema:
{{"clusters":[{{"cluster_id":0,"survivor_id":"<member id>","canonical_content":"<merged durable statement>","conflict":false}}]}}
"""
        result = await self.upstream.learning_chat_completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a conservative memory consolidator. Output strict JSON directly in assistant content. Do not call tools or functions.",
                },
                {"role": "user", "content": prompt},
            ],
            base_url=self.config.learning.base_url or None,
            api_key=self.config.learning.api_key,
            timeout_seconds=self.config.learning.timeout_seconds,
            max_tokens=self.config.learning.consolidation_max_tokens,
            extra_body=self.config.learning.extra_body,
        )
        raw, metadata = extract_nonstream_assistant(result)
        if raw.strip():
            obj = self._parse_json(raw)
        else:
            obj = self._proposal_from_tool_calls(metadata)
        if not isinstance(obj, dict) or not isinstance(obj.get("clusters"), list):
            raise RuntimeError(
                f"consolidation proposal model returned no usable strict-JSON output "
                f"for topic {topic!r} (model={model!r})"
            )
        return obj

    async def _apply(
        self,
        topic: str,
        clusters: list[list[Memory]],
        entries: list[Any],
        stats: dict[str, Any],
    ) -> list[str]:
        """Apply proposals behind deterministic guards; return survivor ids.

        Every member's ``updated_at`` is checked against the snapshot taken
        before the proposal call: the LLM's canonical content was built from
        that snapshot, so any row a foreground learner touched mid-pass is
        skipped instead of merged on stale evidence. Evidence rows of merged
        members are never reassigned; superseded rows keep their observations
        as intact history.
        """
        survivors: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                stats["skipped"] += 1
                continue
            index = entry.get("cluster_id")
            if not isinstance(index, int) or not 0 <= index < len(clusters):
                stats["skipped"] += 1
                continue
            cluster = clusters[index]
            survivor_id = entry.get("survivor_id")
            if survivor_id not in {m.id for m in cluster}:
                log.warning(
                    "consolidation skipped cluster %s in topic %r: survivor %r is not a member",
                    index,
                    topic,
                    survivor_id,
                )
                stats["skipped"] += 1
                continue
            snapshot = {m.id: m for m in cluster}
            survivor = await self.db.get_memory(survivor_id)
            if (
                survivor is None
                or survivor.status != "active"
                or survivor.topic != topic
                or survivor.updated_at != snapshot[survivor_id].updated_at
            ):
                stats["skipped"] += 1
                continue
            if entry.get("conflict") is True:
                await self.db.add_event(
                    Event(
                        session_id=_SESSION_ID,
                        event_type="consolidation.conflict",
                        role="system",
                        content="; ".join(
                            f"{m.id} [{m.memory_type}] {m.content}" for m in cluster
                        ),
                        metadata={"topic": topic, "cluster_ids": [m.id for m in cluster]},
                    )
                )
                stats["conflicts"] += 1
                continue
            source_ids = list(survivor.source_event_ids)
            doomed: list[Memory] = []
            for member in cluster:
                if member.id == survivor_id:
                    continue
                fresh = await self.db.get_memory(member.id)
                if (
                    fresh is None
                    or fresh.status != "active"
                    or fresh.updated_at != member.updated_at
                ):
                    stats["skipped"] += 1
                    continue
                doomed.append(fresh)
                source_ids.extend(member.source_event_ids)
            canonical = entry.get("canonical_content")
            canonical = canonical if isinstance(canonical, str) and canonical.strip() else None
            refined = await self.db.reinforce_memory(
                survivor_id,
                confidence=snapshot[survivor_id].confidence,
                importance=snapshot[survivor_id].importance,
                source_event_ids=sorted(set(source_ids)),
                content=canonical,
                evidence_type="agent_observation",
                reinforcement_metadata={
                    "method": "consolidation",
                    "cluster_ids": [m.id for m in cluster],
                },
            )
            if refined is None:
                stats["skipped"] += 1
                continue
            for member in doomed:
                await self.db.supersede_memory(member.id, refined.id)
            stats["merged"] += 1
            survivors.append(refined.id)
        return survivors

    async def _audit(self, topic: str, stats: dict[str, Any]) -> None:
        """Write the pass event, then stamp the checkpoint AFTER the mutations."""
        await self.db.add_event(
            Event(
                session_id=_SESSION_ID,
                event_type="consolidation.pass",
                role="system",
                content=json.dumps(stats, separators=(",", ":")),
                metadata={"topic": topic, **stats},
            )
        )
        await self.db.set_meta_value(
            _CONSOLIDATION_META_PREFIX + topic, utc_now().isoformat()
        )

    def _parse_json(self, text: str) -> dict[str, Any]:
        """Same parser discipline as learning.py: fences, then greedy brace."""
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

    def _proposal_from_tool_calls(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Recover a consolidation proposal from schema-shaped tool-call arguments."""
        payloads: list[Any] = []
        tool_calls = metadata.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if isinstance(call, dict) and isinstance(call.get("function"), dict):
                    payloads.append(call["function"].get("arguments"))
        function_call = metadata.get("function_call")
        if isinstance(function_call, dict):
            payloads.append(function_call.get("arguments"))
        for payload in payloads:
            parsed: Any = payload
            if isinstance(payload, str):
                parsed = self._parse_json(payload)
            if isinstance(parsed, list):
                return {"clusters": parsed}
            if not isinstance(parsed, dict):
                continue
            if isinstance(parsed.get("clusters"), list):
                return parsed
            if isinstance(parsed.get("survivor_id"), str):
                return {"clusters": [parsed]}
        return {}
