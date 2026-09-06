from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any

import httpx

from .config import AppConfig
from .models import RequestContext

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


class UpstreamClient:
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=config.upstream.timeout_seconds)

    async def close(self) -> None:
        await self.client.aclose()

    def url(self, path: str, *, base_url: str | None = None) -> str:
        base = (base_url or self.config.upstream.base_url).rstrip("/")
        return f"{base}/{path.lstrip('/')}"

    def build_headers(
        self,
        incoming: Mapping[str, str] | None = None,
        request_context: RequestContext | None = None,
    ) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        incoming = incoming or {}
        rcfg = self.config.request_context
        consumed = {
            name.lower()
            for name in (rcfg.user_headers + rcfg.project_headers + rcfg.cwd_headers)
        }
        for key, value in incoming.items():
            lower = key.lower()
            if (
                lower in _HOP_BY_HOP
                or lower.startswith("x-infinitum-")
                or lower.startswith("x-context-")
                or lower in consumed
            ):
                continue
            if lower == "authorization":
                continue
            headers[key] = value

        incoming_auth = incoming.get("authorization") or incoming.get("Authorization")
        if self.config.upstream.passthrough_authorization and incoming_auth:
            headers["Authorization"] = incoming_auth
        elif self.config.upstream.api_key:
            headers["Authorization"] = f"Bearer {self.config.upstream.api_key}"

        # Explicit opt-in for Infinitum -> Headroom deployments. Re-emit
        # canonical trusted-by-configuration headers from the already-resolved
        # context rather than blindly forwarding arbitrary aliases.
        if rcfg.forward_to_headroom and request_context is not None:
            if request_context.user_id:
                headers["x-headroom-user-id"] = request_context.user_id
            if request_context.project_id:
                headers["x-headroom-project-id"] = request_context.project_id
            if request_context.cwd:
                headers["x-headroom-cwd"] = request_context.cwd
        return headers

    async def request(
        self,
        method: str,
        path: str,
        *,
        incoming_headers: Mapping[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        request_context: RequestContext | None = None,
    ) -> httpx.Response:
        return await self.client.request(
            method,
            self.url(path),
            headers=self.build_headers(incoming_headers, request_context),
            json=json_body,
        )

    async def stream_bytes(
        self,
        path: str,
        body: dict[str, Any],
        incoming_headers: Mapping[str, str],
        on_complete: Callable[[bytes], Awaitable[None]] | None,
        request_context: RequestContext | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream the upstream response; ``on_complete=None`` skips the completion callback."""
        request = self.client.build_request(
            "POST",
            self.url(path),
            headers=self.build_headers(incoming_headers, request_context),
            json=body,
        )
        response = await self.client.send(request, stream=True)
        if response.status_code >= 400:
            content = await response.aread()
            await response.aclose()
            # StreamingResponse cannot change its status after yielding. Raise an
            # HTTPStatusError so the route can convert before creating the stream.
            raise httpx.HTTPStatusError(
                f"upstream returned {response.status_code}", request=request, response=httpx.Response(
                    response.status_code,
                    headers=response.headers,
                    content=content,
                    request=request,
                )
            )

        async def iterator() -> AsyncIterator[bytes]:
            captured = bytearray()
            try:
                async for chunk in response.aiter_bytes():
                    captured.extend(chunk)
                    yield chunk
            finally:
                await response.aclose()
                if on_complete is not None:
                    await on_complete(bytes(captured))

        return iterator()

    async def learning_chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        base_url: str | None = None,
        api_key: str = "",
        temperature: float = 0.0,
        timeout_seconds: float | None = None,
        max_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        elif self.config.upstream.api_key:
            headers["Authorization"] = f"Bearer {self.config.upstream.api_key}"
        # Vendor-specific OpenAI-compatible request fields may be supplied for
        # background learning (for example chat-template controls on compatible servers).
        # Core fields below intentionally win so extensions cannot accidentally
        # turn a learning call into streaming or replace its messages/model.
        body: dict[str, Any] = dict(extra_body or {})
        body.update({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        })
        if max_tokens is not None and max_tokens > 0:
            body["max_tokens"] = max_tokens

        try:
            response = await self.client.post(
                self.url("chat/completions", base_url=base_url),
                headers=headers,
                json=body,
                timeout=(timeout_seconds if timeout_seconds is not None else self.config.upstream.timeout_seconds),
            )
        except httpx.ReadTimeout as exc:
            timeout_value = timeout_seconds if timeout_seconds is not None else self.config.upstream.timeout_seconds
            raise RuntimeError(
                f"learning model timed out after {timeout_value:g}s "
                f"(model={model!r}, base_url={(base_url or self.config.upstream.base_url)!r}); "
                "increase learning.timeout_seconds, lower learning.max_tokens, or use a faster learning model"
            ) from exc
        response.raise_for_status()
        return response.json()


def extract_nonstream_assistant(body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    choices = body.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return "", {}
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        text = "\n".join(parts)
    else:
        text = ""
    metadata: dict[str, Any] = {}
    if message.get("tool_calls") is not None:
        metadata["tool_calls"] = message.get("tool_calls")
    if message.get("function_call") is not None:
        metadata["function_call"] = message.get("function_call")
    return text, metadata


def extract_stream_assistant(raw: bytes) -> tuple[str, dict[str, Any]]:
    text_parts: list[str] = []
    tool_chunks: list[Any] = []
    decoded = raw.decode("utf-8", errors="replace")
    for line in decoded.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str):
            text_parts.append(content)
        if delta.get("tool_calls"):
            tool_chunks.extend(delta["tool_calls"])
    metadata = {"tool_call_chunks": tool_chunks} if tool_chunks else {}
    return "".join(text_parts), metadata
