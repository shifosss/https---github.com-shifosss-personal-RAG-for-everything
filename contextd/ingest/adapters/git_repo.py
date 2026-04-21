"""Git repository ingestion adapter.

Walks the HEAD tree via pygit2, skipping binaries and files > 1 MB.
Tree-sitter grammars provide per-declaration chunking for 6 languages;
YAML/TOML/JSON get whole-file treatment; everything else falls through to a
512/64-overlap sliding-window tokeniser split.

Deviations from plan spec §02-phase2-ingestion.md lines 938-1164:
  1. Lazy tokenizer + parsers via ``functools.cached_property`` (avoids
     ~0.4 s BGE-M3 tokenizer load on every adapter construction, matching
     the T4/T5 pattern used by the PDF adapter).
  2. ``_MAX_TOKENS = 1024`` replaces ``_MAX_FILE_BYTES`` in the per-
     declaration token guard (spec bug: compared token int against byte int).
  3. Generator materialised before ``ordinal +=`` to avoid double-parse
     (spec bug: spec re-ran the generator a second time just to count).
  4. ``datetime.UTC`` (PEP 693 / ruff UP017) instead of ``timezone.utc``.
  5. TypeScript handled via ``language_typescript`` entry-point (the wheel
     exposes ``language_typescript`` / ``language_tsx``, not ``language``).
"""

from __future__ import annotations

import functools
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pygit2
from tokenizers import Tokenizer  # type: ignore[import-untyped]
from tree_sitter import Language, Parser

from contextd.ingest.protocol import ChunkDraft, EdgeDraft, SourceCandidate

if TYPE_CHECKING:
    from collections.abc import Iterable

    from contextd.storage.models import SourceType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TARGET_TOKENS = 512
_OVERLAP_TOKENS = 64
_MAX_FILE_BYTES = 1 * 1024 * 1024  # 1 MB — skip very large blobs
_MAX_TOKENS = 1024  # token cap for a single scoped chunk

_LANG_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
}

_DECL_QUERY_BY_LANG: dict[str, str] = {
    "python": """
        (function_definition name: (identifier) @name) @decl
        (class_definition name: (identifier) @name) @decl
    """,
    "typescript": """
        (function_declaration name: (identifier) @name) @decl
        (class_declaration name: (type_identifier) @name) @decl
        (interface_declaration name: (type_identifier) @name) @decl
        (method_definition name: (property_identifier) @name) @decl
    """,
    "javascript": """
        (function_declaration name: (identifier) @name) @decl
        (class_declaration name: (identifier) @name) @decl
        (method_definition name: (property_identifier) @name) @decl
    """,
    "rust": """
        (function_item name: (identifier) @name) @decl
        (struct_item name: (type_identifier) @name) @decl
        (enum_item name: (type_identifier) @name) @decl
        (trait_item name: (type_identifier) @name) @decl
        (impl_item type: (type_identifier) @name) @decl
    """,
    "go": """
        (function_declaration name: (identifier) @name) @decl
        (method_declaration name: (field_identifier) @name) @decl
        (type_declaration (type_spec name: (type_identifier) @name)) @decl
    """,
    "java": """
        (method_declaration name: (identifier) @name) @decl
        (class_declaration name: (identifier) @name) @decl
        (interface_declaration name: (identifier) @name) @decl
    """,
}

# Languages whose module entry-point is not simply ``mod.language()``.
# tree-sitter-typescript exposes language_typescript / language_tsx.
_LANG_ENTRY_OVERRIDE: dict[str, str] = {
    "typescript": "language_typescript",
}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _load_lang(name: str) -> Language | None:
    """Import the grammar wheel and return a ``Language``, or ``None`` on failure."""
    try:
        mod_name = f"tree_sitter_{name}"
        mod = __import__(mod_name)
        entry = _LANG_ENTRY_OVERRIDE.get(name, "language")
        fn = getattr(mod, entry)
        return Language(fn())
    except Exception:  # noqa: BLE001
        return None


def _walk_tree(
    tree: pygit2.Tree, repo: pygit2.Repository, prefix: str
) -> Iterable[tuple[str, bytes]]:
    """Yield (repo-relative path, raw bytes) for every blob in ``tree``."""
    for entry in tree:
        name = entry.name
        full = f"{prefix}{name}"
        if entry.type_str == "tree":
            yield from _walk_tree(repo[entry.id], repo, prefix=f"{full}/")  # type: ignore[index]
        elif entry.type_str == "blob":
            yield full, repo[entry.id].data  # type: ignore[index]


def _is_binary(b: bytes) -> bool:
    """Heuristic: a null byte in the first 8 KiB → treat as binary."""
    return b"\x00" in b[:8192]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class GitRepoAdapter:
    """One SOURCE per git repository; per-declaration chunks via tree-sitter."""

    source_type: SourceType = "git_repo"

    # ------------------------------------------------------------------
    # Lazy-loaded heavy resources (avoids model load at construction time)
    # ------------------------------------------------------------------

    @functools.cached_property
    def _tok(self) -> Tokenizer:
        return Tokenizer.from_pretrained("BAAI/bge-m3")

    @functools.cached_property
    def _parsers(self) -> dict[str, Parser]:
        parsers: dict[str, Parser] = {}
        for lang in _DECL_QUERY_BY_LANG:
            lang_obj = _load_lang(lang)
            if lang_obj is not None:
                parsers[lang] = Parser(lang_obj)
        return parsers

    # ------------------------------------------------------------------
    # Adapter protocol
    # ------------------------------------------------------------------

    def can_handle(self, path: Path) -> bool:
        return path.is_dir() and (path / ".git").exists()

    def sources(self, path: Path) -> Iterable[SourceCandidate]:
        if not self.can_handle(path):
            return
        repo = pygit2.Repository(str(path))
        head_commit = str(repo.head.target) if not repo.head_is_unborn else ""
        try:
            branch = repo.head.shorthand
        except pygit2.GitError:
            branch = "(detached)"
        h = hashlib.sha256((str(path) + head_commit).encode()).hexdigest()
        yield SourceCandidate(
            path=path,
            source_type="git_repo",
            canonical_id=str(path),
            content_hash="sha256:" + h,
            title=path.name,
            source_mtime=datetime.now(UTC),
            metadata={"repo_head_commit": head_commit, "repo_branch": branch},
        )

    def parse(self, source: SourceCandidate) -> Iterable[ChunkDraft]:
        repo = pygit2.Repository(str(source.path))
        if repo.head_is_unborn:
            return
        head_commit = str(repo.head.target)
        ordinal = 0
        tree = repo.head.peel(pygit2.Commit).tree
        for file_path, blob_bytes in _walk_tree(tree, repo, prefix=""):
            if len(blob_bytes) > _MAX_FILE_BYTES:
                continue
            if _is_binary(blob_bytes):
                continue
            ext = Path(file_path).suffix.lower()
            lang = _LANG_BY_EXT.get(ext, "text")
            text = blob_bytes.decode("utf-8", errors="replace")
            base_meta: dict[str, str] = {
                "file_path": file_path,
                "language": lang,
                "commit_hash": head_commit,
            }
            if lang in self._parsers:
                # Materialise the generator once — avoids double-parse (spec bug fix #1).
                parsed = list(
                    self._parse_with_tree_sitter(text, lang, base_meta, ordinal_start=ordinal)
                )
                yield from parsed
                ordinal += len(parsed)
            else:
                for piece in self._split_text(text):
                    yield ChunkDraft(
                        ordinal=ordinal,
                        content=piece,
                        token_count=self._count(piece),
                        scope="",
                        metadata=base_meta,
                    )
                    ordinal += 1

    def metadata(self, source: SourceCandidate) -> dict[str, str]:
        return dict(source.metadata)

    def edges(self, chunks: list[ChunkDraft]) -> Iterable[EdgeDraft]:
        return iter(())  # v0.1: code_imports deferred to v0.2

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def _split_text(self, text: str) -> Iterable[str]:
        """Sliding-window split with _TARGET_TOKENS / _OVERLAP_TOKENS stride."""
        tokens = self._tok.encode(text, add_special_tokens=False).ids
        if len(tokens) <= _MAX_TOKENS:
            yield text
            return
        step = _TARGET_TOKENS - _OVERLAP_TOKENS
        for i in range(0, len(tokens), step):
            window = tokens[i : i + _TARGET_TOKENS]
            yield self._tok.decode(window)

    def _parse_with_tree_sitter(
        self,
        text: str,
        lang: str,
        base_meta: dict[str, str],
        ordinal_start: int,
    ) -> Iterable[ChunkDraft]:
        """Yield per-declaration chunks using the tree-sitter grammar for ``lang``."""
        parser = self._parsers[lang]
        source_bytes = text.encode("utf-8")
        tree = parser.parse(source_bytes)
        language = parser.language
        assert language is not None  # noqa: S101 — parser is always initialised with a language
        query = language.query(_DECL_QUERY_BY_LANG[lang])

        captures = query.captures(tree.root_node)
        # tree-sitter 0.23: captures = {capture_name: [node, ...]}
        decl_nodes = captures.get("decl", [])
        name_nodes = captures.get("name", [])

        decls: list[tuple[int, int, str]] = []
        for decl in decl_nodes:
            name_text = ""
            for n in name_nodes:
                if n.start_byte >= decl.start_byte and n.end_byte <= decl.end_byte:
                    name_text = source_bytes[n.start_byte : n.end_byte].decode("utf-8", "replace")
                    break
            decls.append((decl.start_byte, decl.end_byte, name_text))

        if not decls:
            # Whole-file fallback when no declarations matched.
            if self._count(text) <= _MAX_TOKENS:
                yield ChunkDraft(
                    ordinal=ordinal_start,
                    content=text,
                    token_count=self._count(text),
                    scope="",
                    offset_start=0,
                    offset_end=len(source_bytes),
                    metadata=base_meta,
                )
            else:
                for i, piece in enumerate(self._split_text(text)):
                    yield ChunkDraft(
                        ordinal=ordinal_start + i,
                        content=piece,
                        token_count=self._count(piece),
                        scope="",
                        metadata=base_meta,
                    )
            return

        decls.sort()
        # Module-top: bytes before the first declaration
        module_top = source_bytes[: decls[0][0]].decode("utf-8", "replace").rstrip()
        idx = ordinal_start
        if module_top.strip() and self._count(module_top) <= _TARGET_TOKENS:
            yield ChunkDraft(
                ordinal=idx,
                content=module_top,
                token_count=self._count(module_top),
                scope="",
                offset_start=0,
                offset_end=decls[0][0],
                metadata=base_meta,
            )
            idx += 1

        for start, end, name in decls:
            body = source_bytes[start:end].decode("utf-8", "replace")
            # Spec bug fix #2: compare token count against _MAX_TOKENS (not _MAX_FILE_BYTES bytes).
            if self._count(body) <= _MAX_TOKENS:
                yield ChunkDraft(
                    ordinal=idx,
                    content=body,
                    token_count=self._count(body),
                    scope=name,
                    offset_start=start,
                    offset_end=end,
                    metadata=base_meta,
                )
                idx += 1
            else:
                for j, piece in enumerate(self._split_text(body)):
                    yield ChunkDraft(
                        ordinal=idx,
                        content=piece,
                        token_count=self._count(piece),
                        scope=name,
                        metadata={**base_meta, "split_of": f"{name}#{j}"},
                    )
                    idx += 1
