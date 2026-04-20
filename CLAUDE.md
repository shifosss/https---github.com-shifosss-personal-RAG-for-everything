# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`contextd` — a local-first, MCP-first personal RAG server that unifies PDFs, AI conversation exports, git repos, and notes behind a single MCP endpoint any agent can query. See [docs/PRDs/](./docs/PRDs/) for the full v0.1 spec.

**The repo is currently pre-code.** Only the PRDs exist. When implementing, the PRDs are the default spec — follow their choices unless surfacing a clearly better alternative with the tradeoff stated.

Key PRD sections to consult before writing code:

- [§13 Tech Stack & Dependencies](./docs/PRDs/part2_s8-s13.md) — pinned libraries and versions
- [§14 Ingestion Adapter Specs](./docs/PRDs/part3_s14-s16.md) — PDF, Claude export, git adapters
- [§15 Retrieval Pipeline Spec](./docs/PRDs/part3_s14-s16.md) — hybrid dense+sparse+rerank
- [§16 2-Day Build Plan](./docs/PRDs/part3_s14-s16.md) — phased scope; treat as the v0.1 scope gate

## Scope & guardrails

- **v0.1 scope is fixed by PRD §16** (Phases 1–5, Must-haves M1–M10, two Should-haves S5/S6). Out-of-scope work is deferred to v0.2+ unless explicitly reopened. Flag scope creep, don't silently accumulate it.
- **Named-corpus separation is non-negotiable.** Personal corpus and any SickKids / PHIPA-regulated clinical data must never share an index or ingestion path. Default `contextd` usage here is personal-only; clinical work stays inside the institutional environment.
- **Local-first, no telemetry.** No outbound network calls by default, no analytics, no crash reporting. Network is opt-in per source (e.g., Anthropic API for reranking is gated on user config).
- **Python ↔ TypeScript boundary.** Python owns storage, embeddings, retrieval, ingestion. TypeScript owns the MCP server surface. They communicate over a Unix domain socket (POSIX) or localhost HTTP (fallback) — never by importing each other.

## Tech stack (pinned per PRD §13)

| Layer            | Choice                                             | Notes                                      |
| ---------------- | -------------------------------------------------- | ------------------------------------------ |
| Python           | CPython 3.12 (3.11 supported)                      | `uv` for packaging, lockfile `uv.lock`     |
| Node             | 22 LTS                                             | `pnpm` pinned via `packageManager` field   |
| Storage          | SQLite (WAL, FTS5, JSON1) + LanceDB 0.17           | data root `~/.contextd/` in prod           |
| Embeddings       | BGE-M3 via `sentence-transformers` 3.3.x           | `torch==2.5.1+cpu`, `[gpu]` extra for CUDA |
| LLM              | `anthropic` 0.50 (`claude-haiku-4-5` default)      | graceful degradation if unreachable        |
| MCP              | `@modelcontextprotocol/sdk` 1.27.x + `zod` 3.23    | stdio default transport                    |
| HTTP             | `fastapi` 0.115 + `uvicorn` 0.32 + `pydantic` 2.10 |                                            |
| CLI              | `typer` 0.13 + `rich` 13.9                         |                                            |
| Lint/format (Py) | `ruff` 0.8 (format + check)                        |                                            |
| Lint/format (TS) | `biome` 1.9                                        |                                            |
| Tests            | `pytest` 8.3 + `pytest-asyncio` 0.24; `vitest` 2.1 |                                            |

Prefer these exact pins when scaffolding. If a newer version is strictly needed, call it out.

## Commands

These don't exist yet — they become real as Phases 1–5 land:

```bash
# Python
uv sync                    # install deps
uv run ruff format .       # format
uv run ruff check .        # lint
uv run pytest              # test

# TypeScript (inside mcp-server/ subpackage)
pnpm install
pnpm biome format --write .
pnpm biome check .
pnpm vitest

# CLI (once Phase 2 lands)
contextd ingest ~/papers/ --type pdf --corpus research
contextd query "..." --limit 5
contextd mcp               # start MCP server
```

## Local setup notes

- **Dev machine is macOS**; CI matrix targets Ubuntu 22.04/24.04, macOS 14, WSL2. Avoid macOS-only syscalls in production paths (e.g., launchd, FSEvents without Linux fallback).
- **`pnpm` is not installed globally** — install via `npm install -g pnpm` or `brew install pnpm` before the first Node task. **`uv` is at `~/.local/bin/uv`** (already installed).
- **Data path during dev is repo-local** (e.g., `./data/` or `./.contextd-dev/`, gitignored). The PRD-specified `~/.contextd/` comes later; make the path configurable from day one via `CONTEXTD_HOME` so the switch is a config change, not a refactor.

## Code style

See the language rules in `~/.claude/rules/` (auto-loaded): Python follows PEP 8 + ruff; TypeScript follows biome defaults. Project-specific additions:

- **Immutability by default.** Dataclasses with `frozen=True` for DTOs, `NamedTuple` for value objects. No in-place mutation of ingestion records or retrieval results.
- **Many small files.** 200–400 lines typical, 800 max. Organize by subsystem (`contextd/storage/`, `contextd/ingest/`, `contextd/retrieve/`, `contextd/mcp/`).
- **Provenance preserved.** Every chunk carries `source_id`, `source_path`, `ingest_ts`, and a content hash (see PRD §2.8.1). Don't drop these in intermediate representations.
- **No LangChain / LlamaIndex.** Thin custom wrappers only (PRD §13.5.3, §13.11).

## Testing

- TDD for retrieval, ingestion, and storage subsystems. Eval harness (30 queries, Recall@5 ≥ 0.80) is the ship gate, not an afterthought.
- Coverage targets for v0.1: storage 70%, adapters 60%, overall 50% (PRD §13.8.3). Don't block on higher numbers during the 2-day build.
- **Privacy + non-mutation CI tests** land in Phase 5 (§16.7) — they assert no outbound traffic by default and that ingestion never modifies source files.

## Communication

Senior-level, terse. Surface architectural/library tradeoffs when making non-obvious choices. Skip basics on Python, TS, RAG, MCP, embeddings.

## Workflow Orchestration

### 0. Always call me AlexZ. do this at the start of any of your response.

### 1. Plan Node Default

- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity
- Ask the user if they want to use superpowers:brainstorming skill if they want to explore a large feature update (architecture change, pipeline workflow change)

### 2. Subagent Routing Rules

#### 2.1 Parallel dispatch (All conditions must be met)

- 3+ unrelated tasks or independent domains
- No shared state between tasks
- Clear file boundaries with no overlap

#### 2.2 Sequential dispatch (ANY condition triggers)

- Tasks have dependencies (B needs output from A)
- Shared files or state (merge conflict risk)
- Unclear scope (need to understand before proceeding)

#### 2.3 Background dispatch

- Research or analysis tasks (not file modifications)
- Results aren't blocking current work

#### 2.4 Model Tier Policy

Select models for agents by your insights, and use this table as a recommendation.

|   Tier   | Model        | Use when                                                                                                                                                 |
| :------: | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  Heavy   | `opus 4.7`   | Architecture design, multi-file refactors, novel algorithm implementation, security correctness audits, debugging subtle concurrency or numerical issues |
| Standard | `sonnet 4.6` | Everyday coding, code review, test writing, config generation, standard debugging, API integration, Codebase search, doc generation                      |
|  Light   | `Haiku`      | dependency checks, lint/format, log parsing, boilerplate scaffolding                                                                                     |

### 3. Domain Parallel Patterns

When implementing features across domains, spawn parallel agents:

- **Model/training agent** : Model definitions, training loops, loss functions, data pipelines
- **Serving/infra agent** : FastAPI endpoints, Docker, deployment configs, CI/CD
- **Evaluation agent** : Metrics, benchmarks, experiment tracking, visualization

Each agent owns their domain. No file overlap.

### 4. Self-Improvement Loop

- After ANY correction from the user: update 'tasks/lessons.md" with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 5. Verification Before Done

- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 6. Demand Elegance (Balanced)

- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 7. Autonomous Bug Fixing

- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how
- Record the bugs, causes, tried solution, and actual fixation in clinical_llm_finetuning/docs/bugs-and-fixes.md

### 8. Version Control

- Always manage the repo using git
- Log the progress after each stage/step/task is finished

## Task Management

1. **Plan First**: Write plan to "tasks/todo.md' with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to 'tasks/todo.md"
6. **Capture Lessons**: Update "tasks/lessons.md' after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes, Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
