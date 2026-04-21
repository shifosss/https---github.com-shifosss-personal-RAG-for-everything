"""Claude.ai export ingestion adapter.

Parses the JSON export format produced by Claude.ai (conversations array).
Each conversation becomes one SourceCandidate; each non-empty message becomes
one ChunkDraft.  Bidirectional sequential edges (conversation_next /
conversation_prev) link adjacent turns.

Tokeniser is lazy-loaded via cached_property — same deviation from spec as
PDFAdapter to avoid ~0.4 s cold-start during adapter registry construction.
"""

from __future__ import annotations

import functools
import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tokenizers import Tokenizer  # type: ignore[import-untyped]

from contextd.ingest.protocol import ChunkDraft, EdgeDraft, SourceCandidate

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from contextd.storage.models import SourceType


class ClaudeExportAdapter:
    source_type: SourceType = "claude_export"

    # ------------------------------------------------------------------
    # Tokenizer: lazy-loaded once per instance via cached_property.
    # ------------------------------------------------------------------
    @functools.cached_property
    def _tok(self) -> Tokenizer:
        return Tokenizer.from_pretrained("BAAI/bge-m3")

    # ------------------------------------------------------------------
    # Protocol: can_handle
    # ------------------------------------------------------------------

    def can_handle(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() == ".json"

    # ------------------------------------------------------------------
    # Protocol: sources
    # ------------------------------------------------------------------

    def sources(self, path: Path) -> Iterable[SourceCandidate]:
        if not path.is_file():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        conversations = data if isinstance(data, list) else data.get("conversations", [])
        for conv in conversations:
            uuid = conv.get("uuid") or conv.get("id")
            if not uuid:
                continue
            title = conv.get("name") or conv.get("title") or ""
            canonical = f"{path}#conversations/{uuid}"
            canon = json.dumps(conv, sort_keys=True, ensure_ascii=False)
            h = "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()
            mtime = None
            if ts := conv.get("updated_at") or conv.get("created_at"):
                mtime = _parse_iso(ts)
            yield SourceCandidate(
                path=path,
                source_type="claude_export",
                canonical_id=canonical,
                content_hash=h,
                title=title or None,
                source_mtime=mtime,
                metadata={"uuid": uuid},
            )

    # ------------------------------------------------------------------
    # Protocol: parse
    # ------------------------------------------------------------------

    def parse(self, source: SourceCandidate) -> Iterable[ChunkDraft]:
        file_path_str, _, frag = source.canonical_id.partition("#conversations/")
        data = json.loads(_read_text(file_path_str))
        conversations = data if isinstance(data, list) else data.get("conversations", [])
        conv = next(
            (c for c in conversations if (c.get("uuid") or c.get("id")) == frag),
            None,
        )
        if conv is None:
            return
        messages = conv.get("chat_messages", []) or conv.get("messages", [])
        ordinal = 0
        for i, msg in enumerate(messages):
            text = (msg.get("text") or msg.get("content") or "").strip()
            if not text:
                continue
            sender = msg.get("sender") or "user"
            role: str = "assistant" if sender in ("assistant", "claude") else "user"
            ts_raw = msg.get("created_at")
            yield ChunkDraft(
                ordinal=ordinal,
                content=text,
                token_count=len(self._tok.encode(text, add_special_tokens=False).ids),
                role=role,  # type: ignore[arg-type]
                chunk_timestamp=_parse_iso(ts_raw) if ts_raw else None,
                metadata={"message_id": msg.get("uuid") or msg.get("id") or str(i)},
            )
            ordinal += 1

    # ------------------------------------------------------------------
    # Protocol: metadata
    # ------------------------------------------------------------------

    def metadata(self, source: SourceCandidate) -> dict[str, str]:
        file_path_str, _, frag = source.canonical_id.partition("#conversations/")
        data = json.loads(_read_text(file_path_str))
        conversations = data if isinstance(data, list) else data.get("conversations", [])
        conv = next(
            (c for c in conversations if (c.get("uuid") or c.get("id")) == frag),
            None,
        )
        if conv is None:
            return {}
        messages = conv.get("chat_messages", []) or conv.get("messages", [])
        non_empty = [m for m in messages if (m.get("text") or m.get("content"))]
        meta: dict[str, str] = {"message_count": str(len(non_empty))}
        if ts := conv.get("created_at"):
            meta["created_at"] = ts
        if ts := conv.get("updated_at"):
            meta["updated_at"] = ts
        if url := conv.get("url"):
            meta["conversation_url"] = url
        return meta

    # ------------------------------------------------------------------
    # Protocol: edges
    # ------------------------------------------------------------------

    def edges(self, chunks: list[ChunkDraft]) -> Iterable[EdgeDraft]:
        for i in range(len(chunks) - 1):
            yield EdgeDraft(
                source_ordinal=i,
                target_ordinal=i + 1,
                edge_type="conversation_next",
            )
            yield EdgeDraft(
                source_ordinal=i + 1,
                target_ordinal=i,
                edge_type="conversation_prev",
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_text(path_str: str) -> str:
    from pathlib import Path

    return Path(path_str).read_text(encoding="utf-8")


def _parse_iso(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC)
    except Exception:
        return None
