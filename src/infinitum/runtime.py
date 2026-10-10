from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .compiler import ContextCompiler
from .config import AppConfig
from .consolidation import MemoryConsolidator
from .database import Database
from .embeddings import EmbeddingClient
from .learning import LearningWorker, MemoryLearner
from .request_context import RequestContextResolver
from .retrieval import MemoryRetriever
from .text import topic_slug_similarity
from .tokenizer import TokenCounter
from .upstream import UpstreamClient

log = logging.getLogger(__name__)


class ActiveRequestCounter:
    """Count of foreground requests actively forwarded to the upstream.

    ``last_busy`` stamps the most recent foreground activity so the
    idle-grace window can defer learning until the upstream has been quiet
    long enough. Single event loop, no awaits between increment and read, so
    a plain int suffices. Decrement saturates at zero so a double-decrement
    can never permanently starve deferral that reads this value, and a
    no-op decrement at zero does not fake activity.
    """

    def __init__(self) -> None:
        self._n = 0
        self.last_busy = time.monotonic()

    @property
    def value(self) -> int:
        return self._n

    def increment(self) -> None:
        self._n += 1
        self.last_busy = time.monotonic()

    def decrement(self) -> None:
        if self._n:
            self._n -= 1
            self.last_busy = time.monotonic()


@dataclass(slots=True)
class Runtime:
    config: AppConfig
    db: Database
    embeddings: EmbeddingClient
    upstream: UpstreamClient
    retriever: MemoryRetriever
    request_context: RequestContextResolver
    compiler: ContextCompiler
    learner: MemoryLearner
    worker: LearningWorker
    active_requests: ActiveRequestCounter = field(default_factory=ActiveRequestCounter)


async def backfill_topic_canonicalization(db: Database, config: AppConfig) -> None:
    """One-time repair of a historical store's drifted topic slugs.

    Every distinct active topic is clustered transitively via greedy
    union-find (mirroring ``consolidation.build_clusters``) at
    ``memory.topic_canonical_floor`` slug similarity — the digit veto in
    ``topic_slug_similarity`` means date-distinct slugs never share an edge,
    so they land in separate clusters. Each multi-member cluster's
    representative is picked by the same rule as
    ``learning.MemoryLearner._canonical_topic`` (active count DESC, earliest
    created ASC, topic lexicographic ASC) and every other member is mapped
    onto it; singletons contribute nothing. The mapping is applied through
    ``db.rename_topics`` (one transaction keeping FTS, jobs, checkpoints, and
    the topics table consistent), then the done flag is stamped LAST, so a
    crash before the stamp re-runs the whole pass and ``rename_topics`` of an
    already-applied mapping is a verified no-op. Data hygiene, not a learning
    feature: this runs regardless of ``learning.enabled`` — without
    ``learning.model`` the summary enqueue inside ``rename_topics`` degrades
    to dirty-marking, which startup dirty-recovery picks up.
    """
    if await db.get_meta_value("topic_canonicalization_done"):
        return
    counts = await db.list_topic_counts()
    topics = [topic for topic, _, _ in counts]
    floor = config.memory.topic_canonical_floor
    parent = list(range(len(topics)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            if topic_slug_similarity(topics[i], topics[j]) >= floor:
                union(i, j)

    groups: dict[int, list[str]] = {}
    for i, topic in enumerate(topics):
        groups.setdefault(find(i), []).append(topic)

    by_topic = {topic: (count, earliest) for topic, count, earliest in counts}
    mapping: dict[str, str] = {}
    clusters = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        representative = min(members, key=lambda t: (-by_topic[t][0], by_topic[t][1], t))
        clusters += 1
        for member in members:
            if member != representative:
                mapping[member] = representative

    await db.rename_topics(
        mapping,
        model=config.learning.model,
        debounce_seconds=config.learning.topic_summary_debounce_seconds,
        min_memories=config.learning.topic_summary_min_memories,
    )
    await db.set_meta_value("topic_canonicalization_done", "1")
    log.info(
        "topic canonicalization backfill: %d topics remapped from %d clusters",
        len(mapping),
        clusters,
    )


async def build_runtime(config: AppConfig) -> Runtime:
    db = Database(config.memory.database_path)
    await db.connect()
    # Runs BEFORE the recovery blocks below: dirty-recovery re-enqueues from
    # the renamed topic_updates rows, so the store repair must land first.
    await backfill_topic_canonicalization(db, config)
    embeddings = EmbeddingClient(config.embeddings)
    upstream = UpstreamClient(config)
    retriever = MemoryRetriever(db, embeddings, config)
    request_context = RequestContextResolver(config.request_context)
    compiler = ContextCompiler(db, retriever, TokenCounter(), config)
    learner = MemoryLearner(db, retriever, embeddings, upstream, config)
    # Gate differs from the topic-recovery line below on purpose: requeuing
    # jobs interrupted by a crash is needed even without topic summaries.
    if config.learning.enabled:
        recovered = await db.recover_interrupted_jobs()
        if recovered:
            log.info("recovered %d interrupted learning job(s) after restart", recovered)
    if config.learning.enabled and config.learning.topic_summaries:
        await db.recover_dirty_topic_summary_jobs(default_model=config.learning.model)
    active_requests = ActiveRequestCounter()
    consolidator = (
        MemoryConsolidator(db, upstream, config) if config.learning.consolidation else None
    )
    worker = LearningWorker(db, learner, config, active_requests, consolidator=consolidator)
    return Runtime(
        config,
        db,
        embeddings,
        upstream,
        retriever,
        request_context,
        compiler,
        learner,
        worker,
        active_requests,
    )
