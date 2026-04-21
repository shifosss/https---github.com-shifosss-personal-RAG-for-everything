from __future__ import annotations

from typing import TYPE_CHECKING

from contextd.ingest.adapters.pdf import PDFAdapter

if TYPE_CHECKING:
    from contextd.ingest.protocol import Adapter


def load_default_adapters() -> list[Adapter]:
    # ClaudeExportAdapter and GitRepoAdapter ship in Tasks 5 and 6 respectively;
    # imported lazily so the ingest subpackage stays importable until those
    # modules exist.  Deviation from plan lines 702-712 which shows unconditional
    # inner imports — guarded here because T5/T6 are not yet implemented.
    adapters: list[Adapter] = [PDFAdapter()]
    try:
        from contextd.ingest.adapters.claude_export import ClaudeExportAdapter  # type: ignore[import-untyped]

        adapters.append(ClaudeExportAdapter())
    except ImportError:
        pass
    try:
        from contextd.ingest.adapters.git_repo import GitRepoAdapter  # type: ignore[import-untyped]

        adapters.append(GitRepoAdapter())
    except ImportError:
        pass
    return adapters
