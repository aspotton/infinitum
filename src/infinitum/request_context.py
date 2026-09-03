from __future__ import annotations

import hashlib
import posixpath
import re
from collections.abc import Mapping

from .config import RequestContextConfig
from .models import RequestContext

_PROJECT_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _first_header(headers: Mapping[str, str], names: list[str], max_length: int) -> str | None:
    # Starlette/httpx headers are case-insensitive, but ordinary mappings used
    # by tests or SDKs may not be. Normalize once rather than depending on the
    # concrete Mapping implementation.
    normalized = {str(k).lower(): str(v) for k, v in headers.items()}
    for name in names:
        value = normalized.get(name.lower())
        if value is None:
            continue
        value = value.strip()
        if value:
            return value[:max_length]
    return None


def normalize_cwd(value: str) -> str:
    # CWD is an identity hint, not a path that Infinitum opens. Normalize
    # separators and dot components without resolving symlinks or touching disk.
    value = value.strip().replace("\\", "/")
    if not value:
        return ""
    normalized = posixpath.normpath(value)
    return normalized if normalized != "." else ""


def project_id_from_cwd(cwd: str) -> str:
    normalized = normalize_cwd(cwd)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    basename = posixpath.basename(normalized.rstrip("/")) or "root"
    basename = _PROJECT_NAME_RE.sub("-", basename).strip("-")[:48] or "project"
    return f"cwd:{basename}:{digest}"


class RequestContextResolver:
    """Resolve soft request-context hints from OpenAI/OpenCode HTTP headers.

    V0.2.0 deliberately does *not* treat these values as authenticated identity.
    They are persisted for provenance and may softly influence retrieval order,
    but they never exclude globally visible memory. Hard authorization belongs
    to the later scoped-memory/authenticated-edge architecture.
    """

    def __init__(self, config: RequestContextConfig):
        self.config = config

    @property
    def consumed_header_names(self) -> set[str]:
        names = (
            self.config.user_headers
            + self.config.project_headers
            + self.config.cwd_headers
        )
        return {name.lower() for name in names}

    def resolve(self, headers: Mapping[str, str] | None) -> RequestContext:
        if not self.config.enabled or not headers:
            return RequestContext()

        user_id = _first_header(headers, self.config.user_headers, 256)
        project_id = _first_header(headers, self.config.project_headers, 256)
        cwd_raw = _first_header(headers, self.config.cwd_headers, 2048)
        cwd = normalize_cwd(cwd_raw) if cwd_raw else None
        derived = False

        if not project_id and cwd and self.config.derive_project_from_cwd:
            project_id = project_id_from_cwd(cwd)
            derived = True

        return RequestContext(
            user_id=user_id,
            project_id=project_id,
            cwd=cwd,
            project_derived_from_cwd=derived,
        )
