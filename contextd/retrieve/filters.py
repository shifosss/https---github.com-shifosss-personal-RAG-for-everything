"""Post-hydration filter application for ``QueryFilter``.

PRD ref: §15.5 (query filters). Applied after ``hydrate_results`` so the
filter logic sees the joined Source + Chunk + metadata view in one place,
and so the same predicate can later be ported to a SQL WHERE clause for
the high-throughput path if v0.2 needs it.

v0.1 is post-hoc: this may drop results below ``req.limit`` when the
filter excludes candidates. That is explicit — users asking for
``--source-type pdf`` over a corpus with few PDFs get fewer than N
results, not silently different results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextd.retrieve.preprocess import QueryFilter
    from contextd.storage.models import ChunkResult


_REFERENCE_SECTION_LABELS = frozenset({"references", "bibliography", "works cited", "citations"})


def _matches(r: ChunkResult, f: QueryFilter) -> bool:
    # source_types: keep result only if its source type is in the allow-list
    # (empty tuple means "no restriction").
    if f.source_types and r.source.source_type not in f.source_types:
        return False

    # source_path_prefix: string prefix match on the stored path (which is
    # canonical_id — for claude_export that includes the #conversations/uuid
    # fragment, so a prefix of just the .json path still matches all
    # conversations from that file).
    if f.source_path_prefix and not r.source.path.startswith(f.source_path_prefix):
        return False

    # date_from / date_to: check against source.ingested_at. chunk_timestamp
    # would be more precise for claude exports but is nullable for PDFs/git,
    # so ingested_at is the portable choice.
    if f.date_from is not None and r.source.ingested_at < f.date_from:
        return False
    if f.date_to is not None and r.source.ingested_at > f.date_to:
        return False

    # metadata: every key/value pair in the filter must appear in the
    # result's metadata dict. Empty dict = no restriction.
    for key, want in f.metadata.items():
        if r.metadata.get(key) != want:
            return False

    # exclude_reference_sections: default True per QueryFilter dataclass.
    # Reference lists dilute retrieval quality because "Author et al."
    # entries keyword-match everything.
    return not (
        f.exclude_reference_sections
        and r.chunk.section_label
        and r.chunk.section_label.strip().lower() in _REFERENCE_SECTION_LABELS
    )


def apply_filter(results: list[ChunkResult], f: QueryFilter) -> list[ChunkResult]:
    """Return a new list with entries that satisfy every filter predicate.

    Preserves input order so upstream rank / score semantics survive.
    """
    return [r for r in results if _matches(r, f)]
