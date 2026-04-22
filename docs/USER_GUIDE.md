# contextd — User Guide

A complete walkthrough from install to troubleshooting. For the product pitch and design principles, see the [README](../README.md); for the full spec, see [`docs/PRDs/`](./PRDs/).

---

## Quick start — 5 commands to working RAG

```bash
# 1. Install Python deps (slow first time: BGE-M3 is ~2.2 GB)
uv sync

# 2. (optional) Install Node MCP server deps
cd mcp-server && pnpm install && cd ..

# 3. (optional) Anthropic key for rerank + query rewriting + judge
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env

# 4. Ingest → query
uv run contextd ingest ~/papers/ --corpus research
uv run contextd query "transformer attention mechanism" --corpus research --limit 5

# 5. Expose to any MCP agent (Claude Code, Codex CLI, Cursor, Continue)
uv run contextd serve --corpus research
```

---

## 1. Prerequisites

| Tool | Required? | Install |
|---|---|---|
| Python 3.11–3.13 | yes | system package manager |
| `uv` | yes | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `pnpm` | only for MCP server | `brew install pnpm` or `npm i -g pnpm` |
| Anthropic API key | optional | https://console.anthropic.com/settings/keys → `.env` as `ANTHROPIC_API_KEY=...` |
| GPU | optional | `uv sync --extra gpu` and set `CONTEXTD_EMBEDDING_DEVICE=cuda` |

---

## 2. Core concepts

- **Corpus** — a named index. `personal`, `research`, `work`, whatever. Corpora are **hard-isolated**: a query against `research` cannot see `work`. This is the mandated separation boundary (personal vs. any regulated data).
- **Source** — one file, repo, or conversation you ingested. Addressable by `source_id` (int) or `path`.
- **Chunk** — the atomic retrieval unit; carries provenance (`source_id`, `source_path`, `ingest_ts`, `content_hash`).
- **Data root** — where everything lives on disk. Default `~/.contextd/`. Override with `CONTEXTD_HOME=/some/path`.
- **Pipeline** — dense (BGE-M3 over LanceDB) + sparse (SQLite FTS5) → RRF fusion → optional Anthropic rerank → optional `QueryFilter` → top-N.

---

## 3. Installation

```bash
git clone https://github.com/shifosss/personal-RAG-for-everything.git
cd personal-RAG-for-everything

# Python (first run downloads BGE-M3 ≈ 2.2 GB to ~/.cache/huggingface/)
uv sync

# Node MCP server (skip if you won't expose MCP)
cd mcp-server && pnpm install && cd ..

# Sanity check
uv run contextd version            # → 0.1.0.dev0
uv run contextd status             # → data root, api_key_present, reranker config
uv run pytest                      # → all tests pass in ~7s
```

### Smoke the API key (optional)

```bash
uv run python -c "
from dotenv import load_dotenv; load_dotenv()
from anthropic import Anthropic
import os
r = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY']).messages.create(
    model='claude-haiku-4-5', max_tokens=5,
    messages=[{'role':'user','content':'OK'}])
print(r.content[0].text)
"
```

---

## 4. Ingestion

### Syntax
```
contextd ingest PATH [--corpus NAME] [--type TYPE] [--force]
```

### By source type

| Source | Command | Notes |
|---|---|---|
| Single PDF | `contextd ingest paper.pdf --corpus research` | PyMuPDF4LLM extracts semantic chunks |
| Directory of PDFs | `contextd ingest ~/papers/ --corpus research` | recurses into subdirs |
| Claude Code / Codex export | `contextd ingest export.json --corpus research` | multi-conversation exports split into one source per conversation |
| Git repo (tree-sitter chunking) | `contextd ingest ~/code/foo --type git_repo --corpus research` | **extracted dir**, not a `.tar.gz`. Parsers: Python, TS/JS, Rust, Go, Java |
| Single source file | `contextd ingest foo.py --type git_repo --corpus research` | tree-sitter chunks by file extension |

### Re-ingest behavior
- **Same content hash** → ingest is a no-op by design. Safe to run on a cron.
- **Updated file but same path** → hash mismatch triggers re-embed and replaces old chunks atomically.
- **Force re-embed** → `--force` ignores the hash.

### Inspect what landed
```bash
contextd list --corpus research
contextd list --corpus research --json    # machine-readable
```

### Remove a source
```bash
contextd forget ~/papers/retracted.pdf --corpus research --dry-run   # preview cascade
contextd forget ~/papers/retracted.pdf --corpus research --yes       # execute
```
Cascades source → chunks → vectors → FTS5 tokens. No orphans left behind.

---

## 5. Query

### Syntax
```
contextd query "free-text query" [--corpus NAME] [--limit 10] [--rerank|--no-rerank] [--rewrite|--no-rewrite] [--json]
```

### Flag reference

| Flag | Default | When to flip |
|---|---|---|
| `--corpus` | `personal` | always set explicitly once you have >1 corpus |
| `--limit` | 10 | raise to 20 for big-context agents; lower to 3 for fast interactive use |
| `--rerank` | on | turn off if you have no API key or want pure local retrieval |
| `--rewrite` | off | turn on for short, ambiguous queries — Haiku generates 3-5 paraphrases to broaden recall |
| `--json` | off | machine-readable envelope for scripts |

### Patterns
```bash
# Fast, offline, zero API calls
contextd query "bge-m3 tokenization" --corpus research --no-rerank --limit 5

# Best quality (rerank + rewrite). Requires ANTHROPIC_API_KEY.
contextd query "how do I configure the retrieval threshold" --rerank --rewrite

# Pipe into an LLM as context
contextd query "auth middleware design" --json | jq '.chunks[].content'
```

---

## 6. MCP — exposing contextd to any agent

`contextd` is MCP-first. `contextd serve` starts both the Python HTTP backend and the Node MCP stdio server — MCP clients connect over stdio.

### Start the server
```bash
contextd serve --corpus research                # HTTP (127.0.0.1:8787) + MCP stdio
contextd serve --http-only                      # just the HTTP backend (useful with curl)
contextd serve --mcp-only                       # just the MCP server; HTTP must be running separately
```

### Wire into Claude Code

`~/.claude.json` (global) or `.mcp.json` (project):
```json
{
  "mcpServers": {
    "contextd": {
      "command": "contextd",
      "args": ["serve", "--corpus", "research"]
    }
  }
}
```

### Wire into Codex CLI

`~/.codex/config.toml`:
```toml
[mcp_servers.contextd]
command = "contextd"
args = ["serve", "--corpus", "research"]
```

### Available MCP tools
7 tools: `search-corpus`, `fetch-chunk`, `expand-context`, `get-edges`, `list-sources`, `get-source`, `list-corpora`.

---

## 7. Maintenance & introspection

```bash
contextd version                 # semver
contextd status                  # data root, corpus count, api_key_present, reranker model
contextd config show             # all effective settings
contextd config path             # prints data root (respects $CONTEXTD_HOME)
contextd list --corpus X         # sources + chunk counts
```

### Evaluate retrieval quality
```bash
contextd eval contextd/eval/seed_queries.json --corpus eval --rerank --judge
```
Runs the 30-query ship-gate harness. Gates: Recall@5 ≥ 0.80, Recall@10 ≥ 0.90, MRR ≥ 0.60, judge ≥ 6.5. The `eval` corpus must be populated first with the fixtures under `tests/fixtures/`.

---

## 8. Configuration reference

Every setting is also an env var with prefix `CONTEXTD_`. A `.env` file in the repo root is auto-loaded on CLI start.

| Var | Default | Meaning |
|---|---|---|
| `CONTEXTD_HOME` | `~/.contextd` | Data root (SQLite + LanceDB + FTS5) |
| `CONTEXTD_DEFAULT_CORPUS` | `personal` | Used when `--corpus` is omitted |
| `CONTEXTD_LOG_LEVEL` | `INFO` | structlog level. Chunk content is never logged at INFO — a privacy test enforces this. |
| `CONTEXTD_EMBEDDING_MODEL` | `BAAI/bge-m3` | Don't change unless you also change `embedding_dim` |
| `CONTEXTD_EMBEDDING_DEVICE` | `cpu` | `cuda` if you `uv sync --extra gpu` |
| `CONTEXTD_EMBEDDING_BATCH_SIZE` | `16` | Raise to 32+ on GPU |
| `CONTEXTD_RETRIEVAL_DEFAULT_LIMIT` | `10` | Default `--limit` |
| `CONTEXTD_RETRIEVAL_RERANK_ENABLED` | `true` | Global kill-switch |
| `CONTEXTD_RETRIEVAL_RERANK_TIMEOUT_MS` | `5000` | Bump to 15000 if you see `rerank.unavailable: TimeoutError` |
| `CONTEXTD_RETRIEVAL_REWRITE_ENABLED` | `false` | Off by default |
| `CONTEXTD_RERANKER_MODEL` | `claude-haiku-4-5` | Any Anthropic model ID |
| `CONTEXTD_MCP_HOST` / `CONTEXTD_MCP_PORT` | `127.0.0.1` / `8787` | HTTP backend bind |
| `ANTHROPIC_API_KEY` | — | In `.env` or shell; needed only for rerank, rewrite, judge |

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `api_key_present: false` in `contextd status` | `.env` not loaded, or key name wrong. Must be `ANTHROPIC_API_KEY` (underscores, not hyphens). |
| Ingest of a `.tar.gz` with `--type git_repo` returns `0 sources` | Extract first: `tar -xzf repo.tar.gz && contextd ingest ./repo --type git_repo` |
| `rerank.unavailable: TimeoutError` in logs | Bump `CONTEXTD_RETRIEVAL_RERANK_TIMEOUT_MS=15000`. 5s is tight for 50 candidates. |
| `rerank.unavailable: invalid rerank JSON` | Should be gone as of the markdown-fence fix. If it recurs, check `contextd/llm_json.py` handles whatever new fence variant the model emitted. |
| Query returns nothing | `contextd list --corpus X` — did anything actually get ingested? Verify `--corpus` matches between ingest and query. |
| Privacy test fails in CI | Some new dep opened a socket on import. Run `uv run pytest tests/privacy/` locally; the traceback names the offender. |
| BGE-M3 download is slow | First-run only. ~2.2 GB to `~/.cache/huggingface/`. Cached thereafter. |
| Claude-export ingest picks up only 1 conversation | Known bug, fixed. Upgrade; verify `canonical_id` carries `#conversations/<uuid>` via `contextd list --corpus X --json`. |

---

## 10. What to read next

- [README](../README.md) — project pitch + scope table
- [`docs/PRDs/`](./PRDs/) — full v0.1 spec (§13 tech stack, §14 adapters, §15 retrieval, §16 build plan)
- [`docs/plans/`](./plans/) — per-phase implementation plans
- [`docs/demo/v0.1-script.md`](./demo/v0.1-script.md) — 90-second demo walkthrough
- [`CLAUDE.md`](../CLAUDE.md) — agent/Claude Code project instructions
