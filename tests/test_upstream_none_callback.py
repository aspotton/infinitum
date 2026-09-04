"""stream_bytes must accept on_complete=None and still yield full bytes.

Batch 2 of the memory deep-retrieval-tools plan: intermediate tool-loop
iterations must not record partial assistant messages, so the completion
callback becomes optional.
"""

import httpx

from infinitum.config import AppConfig
from infinitum.upstream import UpstreamClient

SSE_CHUNKS = [
    b'data: {"choices":[{"delta":{"content":"he"}}]}\n\n',
    b'data: {"choices":[{"delta":{"content":"llo"}}]}\n\n',
    b"data: [DONE]\n\n",
]


def make_client() -> UpstreamClient:
    """UpstreamClient wired to a MockTransport that streams SSE_CHUNKS."""

    async def body():
        for chunk in SSE_CHUNKS:
            yield chunk

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body())

    client = UpstreamClient(AppConfig())
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


async def test_stream_without_callback_yields_all_bytes():
    client = make_client()
    stream = await client.stream_bytes("chat/completions", {"model": "m"}, {}, None)
    chunks = [chunk async for chunk in stream]
    assert b"".join(chunks) == b"".join(SSE_CHUNKS)


async def test_stream_with_callback_invoked_once_with_full_bytes():
    client = make_client()
    calls: list[bytes] = []

    async def on_complete(raw: bytes) -> None:
        calls.append(raw)

    stream = await client.stream_bytes("chat/completions", {"model": "m"}, {}, on_complete)
    chunks = [chunk async for chunk in stream]
    assert b"".join(chunks) == b"".join(SSE_CHUNKS)
    assert calls == [b"".join(SSE_CHUNKS)]
