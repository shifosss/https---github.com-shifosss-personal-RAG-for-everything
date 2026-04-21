from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from contextd.ingest.protocol import Adapter


def load_default_adapters() -> Iterable[Adapter]:
    # Tasks 4-6 will populate this with the PDF, Claude export, and git adapters.
    return []
