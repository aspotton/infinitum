from __future__ import annotations

from dataclasses import dataclass, field

from .compiler import ContextCompiler
from .config import AppConfig
from .database import Database
from .embeddings import EmbeddingClient
from .learning import LearningWorker, MemoryLearner
from .request_context import RequestContextResolver
from .retrieval import MemoryRetriever
from .tokenizer import TokenCounter
from .upstream import UpstreamClient


class ActiveRequestCounter:
    """Count of foreground requests actively forwarded to the upstream.

    Single event loop, no awaits between increment and read, so a plain int
    suffices. Decrement saturates at zero so a double-decrement can never
    permanently starve deferral that reads this value.
    """

    def __init__(self) -> None:
        self._n = 0

    @property
    def value(self) -> int:
        return self._n

    def increment(self) -> None:
        self._n += 1

    def decrement(self) -> None:
        self._n = max(0, self._n - 1)


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


async def build_runtime(config: AppConfig) -> Runtime:
    db = Database(config.memory.database_path)
    await db.connect()
    embeddings = EmbeddingClient(config.embeddings)
    upstream = UpstreamClient(config)
    retriever = MemoryRetriever(db, embeddings, config)
    request_context = RequestContextResolver(config.request_context)
    compiler = ContextCompiler(db, retriever, TokenCounter(), config)
    learner = MemoryLearner(db, retriever, embeddings, upstream, config)
    if config.learning.enabled and config.learning.topic_summaries:
        await db.recover_dirty_topic_summary_jobs(default_model=config.learning.model)
    active_requests = ActiveRequestCounter()
    worker = LearningWorker(db, learner, config, active_requests)
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
