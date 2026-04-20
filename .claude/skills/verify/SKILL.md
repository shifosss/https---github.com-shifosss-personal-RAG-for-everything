---
name: verify
description: Run the full verification suite for contextd (lint, format-check, typecheck, tests) before claiming work is complete. Runs Python (ruff + pytest) and/or TypeScript (biome + vitest) depending on which subpackages exist. Safe to run at any stage — skips layers that aren't scaffolded yet.
---

# /verify — full verification gate

Before saying "done", run this. It adapts to whatever subpackages currently exist, so it works in Phase 1 (only `contextd/` exists) and in Phase 5 (both Python and TS packages).

## Flow

1. Detect which subpackages are present (Read `pyproject.toml` and `mcp-server/package.json` or equivalent; silently skip layers that don't exist yet).
2. Run the layers that apply, in order. Stop on the first failure and report it — don't swallow errors.

### Python layer (if `pyproject.toml` or `contextd/` exists)

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest -x --cov=contextd --cov-report=term-missing
```

Optional (if `mypy` is configured): `uv run mypy contextd`.

### TypeScript layer (if `mcp-server/package.json` exists)

```bash
cd mcp-server
pnpm biome format --check .
pnpm biome check .
pnpm vitest run
```

Optional (if `tsc` is configured): `pnpm tsc --noEmit`.

### Privacy + non-mutation smoke (Phase 5+ only)

Run the dedicated privacy and non-mutation tests if they exist (see PRD §16.7):

```bash
uv run pytest tests/privacy tests/non_mutation -x
```

## Output format

Report one line per layer run: `✓ ruff format`, `✓ ruff check`, `✗ pytest (3 failures in tests/retrieve/test_rrf.py)`. On failure, show the first failing assertion and the file:line — don't dump the entire traceback.

## Guardrails

- **No auto-fix.** Don't run `ruff format .` or `biome format --write` — those are mutations. Use `--check` / `--format-check` only. Auto-format happens via the PostToolUse hook during editing, not during verification.
- **Coverage targets per PRD §13.8.3:** storage 70%, adapters 60%, overall 50% for v0.1. Don't fail on missing coverage below these; do flag regressions.
- **Don't skip layers silently to make it green.** If a layer is broken (e.g., `pyproject.toml` exists but `pytest` errors on collection), that's a failure — not "skip and move on."
