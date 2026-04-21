from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from contextd.ingest.protocol import Adapter

_REGISTRY: dict[str, Adapter] = {}


def register(adapter: Adapter) -> None:
    _REGISTRY[adapter.source_type] = adapter


def get(source_type: str) -> Adapter:
    if source_type not in _REGISTRY:
        raise KeyError(f"no adapter registered for source_type={source_type!r}")
    return _REGISTRY[source_type]


def all_adapters() -> Iterable[Adapter]:
    return _REGISTRY.values()
