from __future__ import annotations

from typing import TYPE_CHECKING

from contextd.ingest.adapters.claude_export import ClaudeExportAdapter
from contextd.ingest.adapters.pdf import PDFAdapter

if TYPE_CHECKING:
    from contextd.ingest.protocol import Adapter


def load_default_adapters() -> list[Adapter]:
    adapters: list[Adapter] = [PDFAdapter(), ClaudeExportAdapter()]
    try:
        from contextd.ingest.adapters.git_repo import GitRepoAdapter  # type: ignore[import-untyped]

        adapters.append(GitRepoAdapter())
    except ImportError:
        pass
    return adapters
