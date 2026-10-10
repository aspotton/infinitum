"""N1 topic canonicalization at apply time (plan todo 4).

Variant topic slugs converge onto the deterministic representative so repeated
facts reinforce instead of duplicating; the supersede/refine gates accept topic
EQUIVALENCE (memory_type stays exact everywhere); the extraction prompt shows a
capped list of existing topics to reuse. The corruption controls (vi)/(vi')
pin that a misnamed target id can never mutate the unrelated memory it names.
"""

import json
import tempfile
from unittest.mock import AsyncMock

from infinitum.config import AppConfig
from infinitum.database import Database
from infinitum.embeddings import EmbeddingClient
from infinitum.learning import MemoryLearner
from infinitum.models import Event, Memory
from infinitum.retrieval import MemoryRetriever
from infinitum.upstream import UpstreamClient

_PROMPT_MARKER = (
    "Existing topics (reuse verbatim when the subject matches; "
    "coin a new slug only if none fits):"
)


async def _learner(tmp: str):
    cfg = AppConfig()
    cfg.memory.database_path = f"{tmp}/runtime.db"
    db = Database(cfg.memory.database_path)
    await db.connect()
    embeddings = EmbeddingClient(cfg.embeddings)
    upstream = UpstreamClient(cfg)
    retriever = MemoryRetriever(db, embeddings, cfg)
    learner = MemoryLearner(db, retriever, embeddings, upstream, cfg)
    return cfg, db, embeddings, upstream, learner


def _envelope(candidates: list[dict]) -> dict:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"memories": candidates}),
                    "tool_calls": None,
                },
            }
        ]
    }


def _candidate(**extra) -> dict:
    candidate = {
        "memory_type": "fact",
        "topic": "web-app-deploy",
        "content": "the web app deploys with a blue-green rollout",
        "importance": 0.7,
        "confidence": 0.8,
        "operation_hint": "new",
    }
    candidate.update(extra)
    return candidate


async def _seed(db: Database, memory_type: str, topic: str, content: str) -> Memory:
    event = Event(session_id="s1", event_type="message.user", role="user", content=content)
    await db.add_event(event)
    return await db.create_memory(
        Memory(
            memory_type=memory_type, topic=topic, content=content, source_event_ids=[event.id]
        )
    )


async def _drive(db: Database, learner: MemoryLearner, envelope: dict, user_text: str) -> None:
    event = Event(session_id="s2", event_type="message.user", role="user", content=user_text)
    await db.add_event(event)
    learner.upstream.learning_chat_completion = AsyncMock(return_value=envelope)
    await learner.learn(
        {
            "request_id": "req_1",
            "session_id": "ses_2",
            "model": "memory-model",
            "user_text": user_text,
            "assistant_text": "Understood, recorded.",
            "source_event_ids": [event.id],
            "request_context": {},
        }
    )


def _tuple(memory: Memory) -> tuple:
    return (
        memory.content,
        memory.memory_type,
        memory.topic,
        memory.observation_count,
        memory.confidence,
        memory.status,
        memory.updated_at,
    )


def _block_topics(prompt: str) -> list[str]:
    if _PROMPT_MARKER not in prompt:
        return []
    rest = prompt.split(_PROMPT_MARKER, 1)[1]
    return [line[2:] for line in rest.split("\n") if line.startswith("- ")]


def _shown_topics(prompt: str) -> set[str]:
    nearby = prompt.split("Existing nearby memories:\n", 1)[1].split("\n\n")[0]
    return {line.split(" | ")[2] for line in nearby.split("\n") if line.count(" | ") >= 2}


# (i) variant-slug duplicate converges: ONE active, reinforced, canonical topic


async def test_variant_slug_candidate_reinforces_canonical_memory():
    with tempfile.TemporaryDirectory() as tmp:
        _cfg, db, embeddings, upstream, learner = await _learner(tmp)
        try:
            seed = await _seed(db, "fact", "web-app-deployment", _candidate()["content"])
            await _drive(db, learner, _envelope([_candidate(topic="web-app-deploy")]), "how do we ship the web app?")

            actives = await db.fetchall("SELECT * FROM memories WHERE status='active'")
            assert len(actives) == 1
            assert actives[0]["id"] == seed.id
            assert int(actives[0]["observation_count"]) == 2
            assert actives[0]["topic"] == "web-app-deployment"
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()


# (ii) refine fires in place across a slug-variant stored target topic


async def test_refine_fires_when_target_stores_an_old_variant_topic():
    with tempfile.TemporaryDirectory() as tmp:
        _cfg, db, embeddings, upstream, learner = await _learner(tmp)
        try:
            await _seed(db, "fact", "nightly-backup-strategy", "our backup strategy favors nightly full dumps")
            await _seed(db, "fact", "nightly-backup-strategy", "the nightly backup strategy is reviewed quarterly")
            target = await _seed(db, "fact", "nightly-backup-strat", "the postgres backup job runs nightly and keeps one week of dumps")
            candidate = _candidate(
                memory_type="fact",
                topic="nightly-backup-strategy",
                content="the postgres backup job runs nightly and keeps compressed pg_dump archives for one week",
                operation_hint="supersede",
                supersedes_memory_ids=[target.id],
                explicit_correction=False,
            )
            await _drive(db, learner, _envelope([candidate]), "we clarified the nightly postgres backup retention")

            refined = await db.get_memory(target.id)
            assert refined is not None and refined.status == "active"
            assert "pg_dump" in refined.content
            assert int(refined.observation_count) == 2
            actives = await db.fetchall("SELECT id FROM memories WHERE status='active'")
            assert len(actives) == 3
            superseded = await db.fetchall("SELECT id FROM memories WHERE status='superseded'")
            assert superseded == []
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()


# (iii) digit veto keeps dated variants apart


async def test_digit_veto_variant_creates_two_memories():
    with tempfile.TemporaryDirectory() as tmp:
        _cfg, db, embeddings, upstream, learner = await _learner(tmp)
        try:
            content = "the release runbook starts with a database snapshot check"
            await _seed(db, "fact", "runbook-start-2026-09-24", content)
            await _drive(db, learner, _envelope([_candidate(topic="runbook-start-2026-10-01", content=content)]), "the runbook for the next release")

            actives = await db.fetchall("SELECT topic, observation_count FROM memories WHERE status='active'")
            assert len(actives) == 2
            assert {a["topic"] for a in actives} == {"runbook-start-2026-09-24", "runbook-start-2026-10-01"}
            assert all(int(a["observation_count"]) == 1 for a in actives)
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()


# (iv) prompt shows the capped, deduped Existing-topics block; cap 0 omits it


async def test_prompt_lists_capped_existing_topics():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, embeddings, upstream, learner = await _learner(tmp)
        try:
            await _seed(db, "fact", "alpha-service", "alpha service routes invoice webhooks")
            await _seed(db, "fact", "alpha-service", "alpha service runs in eu-west")
            await _seed(db, "fact", "billing-cron", "nightly billing cron reconciles ledger rows")
            await _seed(db, "fact", "graphql-layer", "public api exposes a graphql layer")
            await _seed(db, "fact", "image-cache", "product imagery served from a cdn cache")
            await _seed(db, "fact", "image-cache", "image cache invalidates on deploy")
            await _seed(db, "fact", "kafka-topics", "events fan out over kafka topics")
            cfg.memory.topic_canonicalization_prompt_topics = 3

            await _drive(db, learner, _envelope([]), "what does the alpha service do")
            prompt = upstream.learning_chat_completion.await_args.kwargs["messages"][1]["content"]
            assert _PROMPT_MARKER in prompt
            shown = _shown_topics(prompt)
            assert "alpha-service" in shown
            # Deterministic counts order (count DESC, topic ASC) minus shown, capped.
            order = ["alpha-service", "image-cache", "billing-cron", "graphql-layer", "kafka-topics"]
            assert _block_topics(prompt) == [t for t in order if t not in shown][:3]

            cfg.memory.topic_canonicalization_prompt_topics = 0
            await _drive(db, learner, _envelope([]), "anything new today")
            prompt = upstream.learning_chat_completion.await_args.kwargs["messages"][1]["content"]
            assert "Existing topics" not in prompt
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()


# (v) replayed learn job converges without observation inflation


async def test_replayed_learn_job_does_not_inflate_observation_count():
    with tempfile.TemporaryDirectory() as tmp:
        _cfg, db, embeddings, upstream, learner = await _learner(tmp)
        try:
            seed = await _seed(db, "fact", "web-app-deployment", _candidate()["content"])
            event = Event(session_id="s2", event_type="message.user", role="user", content="how do we ship?")
            await db.add_event(event)
            envelope = _envelope([_candidate(topic="web-app-deploy")])
            payload = {
                "request_id": "req_1",
                "session_id": "ses_2",
                "model": "memory-model",
                "user_text": "how do we ship the web app?",
                "assistant_text": "Blue-green rollout.",
                "source_event_ids": [event.id],
                "request_context": {},
            }
            learner.upstream.learning_chat_completion = AsyncMock(return_value=envelope)
            await learner.learn(payload)
            await learner.learn(payload)

            actives = await db.fetchall("SELECT id, observation_count FROM memories WHERE status='active'")
            assert len(actives) == 1
            assert actives[0]["id"] == seed.id
            assert int(actives[0]["observation_count"]) == 2
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()


# (vi) corruption control: a misnamed supersede target is never mutated


async def test_misnamed_supersede_target_is_never_mutated():
    with tempfile.TemporaryDirectory() as tmp:
        _cfg, db, embeddings, upstream, learner = await _learner(tmp)
        try:
            z = await _seed(db, "preference", "coffee-order", "the team prefers fairtrade coffee in the office coffee order")
            before = _tuple(z)
            candidate = _candidate(
                memory_type="fact",
                topic="database-choice",
                content="we chose postgres for the primary database",
                operation_hint="supersede",
                supersedes_memory_ids=[z.id],
                explicit_correction=False,
            )
            await _drive(db, learner, _envelope([candidate]), "our office coffee order is due again")

            after = await db.get_memory(z.id)
            assert after is not None
            assert _tuple(after) == before
            assert after.status == "active"
            actives = await db.fetchall("SELECT id FROM memories WHERE status='active'")
            assert len(actives) == 2
            superseded = await db.fetchall("SELECT id FROM memories WHERE status='superseded'")
            assert superseded == []
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()


# (vi') reinforce-path control: named but unrelated target stays byte-identical


async def test_misnamed_reinforce_target_is_never_mutated():
    with tempfile.TemporaryDirectory() as tmp:
        _cfg, db, embeddings, upstream, learner = await _learner(tmp)
        try:
            z = await _seed(db, "preference", "coffee-order", "the team prefers fairtrade coffee in the office coffee order")
            before = _tuple(z)
            candidate = _candidate(
                memory_type="fact",
                topic="database-choice",
                content="we chose postgres for the primary database",
                operation_hint="reinforce",
                reinforces_memory_id=z.id,
            )
            await _drive(db, learner, _envelope([candidate]), "our office coffee order is due again")

            after = await db.get_memory(z.id)
            assert after is not None
            assert _tuple(after) == before
            actives = await db.fetchall("SELECT topic, memory_type FROM memories WHERE status='active'")
            assert len(actives) == 2
            created = [a for a in actives if a["memory_type"] == "fact"]
            assert len(created) == 1
            # Documented ceiling of the topic-only override: the new memory may
            # be filed under Z's topic; Z itself is never mutated.
            assert created[0]["topic"] == "coffee-order"
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()


# (vii) multi-id supersede over slug-variant targets: both superseded, one canonical active


async def test_multi_id_supersede_across_slug_variants():
    with tempfile.TemporaryDirectory() as tmp:
        _cfg, db, embeddings, upstream, learner = await _learner(tmp)
        try:
            first = await _seed(db, "fact", "web-app-deployment", "the web app deployment uses a blue-green rollout")
            second = await _seed(db, "fact", "web-app-deploy", "the web app deployment runs blue-green rollouts on thursdays")
            assert first.created_at != second.created_at
            canonical = first.topic if first.created_at < second.created_at else second.topic
            candidate = _candidate(
                topic="web-app-deploy",
                content="the web app deployment uses a blue-green rollout on thursdays",
                operation_hint="supersede",
                supersedes_memory_ids=[first.id, second.id],
                explicit_correction=False,
            )
            await _drive(db, learner, _envelope([candidate]), "we consolidated the web app deployment rollout note")

            actives = await db.fetchall("SELECT id, topic FROM memories WHERE status='active'")
            assert len(actives) == 1
            assert actives[0]["id"] not in (first.id, second.id)
            assert actives[0]["topic"] == canonical
            for old_id in (first.id, second.id):
                old = await db.get_memory(old_id)
                assert old is not None and old.status == "superseded"
        finally:
            await upstream.close()
            await embeddings.close()
            await db.close()
