from __future__ import annotations

import logging

import httpx
import numpy as np

from .config import EmbeddingConfig

log = logging.getLogger(__name__)


class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=config.timeout_seconds)

    async def close(self) -> None:
        await self.client.aclose()

    async def embed(self, text: str) -> np.ndarray | None:
        if not self.config.enabled or not text.strip():
            return None
        url = self.config.base_url.rstrip("/") + "/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        try:
            response = await self.client.post(
                url,
                headers=headers,
                json={"model": self.config.model, "input": text},
            )
            response.raise_for_status()
            data = response.json()
            vector = data["data"][0]["embedding"]
            return np.asarray(vector, dtype=np.float32)
        except Exception as exc:  # graceful degradation is intentional here
            log.warning("embedding request failed; continuing without semantic vector: %s", exc)
            return None
