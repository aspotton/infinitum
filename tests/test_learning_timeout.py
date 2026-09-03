import json

import httpx
import pytest

from infinitum.config import AppConfig
from infinitum.upstream import UpstreamClient


@pytest.mark.asyncio
async def test_learning_call_uses_learning_controls():
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "{}"}}]
            },
        )

    cfg = AppConfig()
    upstream = UpstreamClient(cfg)
    await upstream.client.aclose()
    upstream.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        await upstream.learning_chat_completion(
            model="memory-model",
            messages=[{"role": "user", "content": "extract"}],
            timeout_seconds=123,
            max_tokens=777,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}, "stream": True},
        )
    finally:
        await upstream.close()

    assert seen["body"]["stream"] is False
    assert seen["body"]["max_tokens"] == 777
    assert seen["body"]["chat_template_kwargs"] == {"enable_thinking": False}
