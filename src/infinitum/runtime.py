from __future__ import annotations

from dataclasses import dataclass

from .compiler import ContextCompiler
from .config import AppConfig
from .database import Database
from .embeddings import EmbeddingClient
from .learning import LearningWorker, MemoryLearner
from .request_context import RequestContextResolver
from .retrieval import MemoryRetriever
from .tokenizer import TokenCounter
from .upstream import UpstreamClient


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
    worker = LearningWorker(db, learner, config)
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
    )
