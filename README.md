# contextd

**Local-first, MCP-first personal RAG.** Ingest PDFs, Claude conversation exports, and git repos; query them from any MCP-capable agent (Claude Code, Codex CLI, Cursor, Continue, anything that speaks MCP stdio).

No telemetry. No outbound network by default. Named-corpus isolation so personal and regulated clinical data never share an index.

**New here?** Start with the [**User Guide**](./docs/USER_GUIDE.md) for a complete walkthrough from install to troubleshooting.

---

## 90-second demo

See [`docs/demo/v0.1-script.md`](./docs/demo/v0.1-script.md) for the recorded walkthrough: ingest a paper corpus, query it from the CLI, then call the same corpus from Claude Code and Codex CLI back-to-back. Same data, any agent.

---

## Install

```bash
pipx install contextd          # or: uv tool install contextd
contextd --help
```

Python 3.12 (3.11 works) and Node 22 LTS are required for the MCP server. `pipx` installs the CLI into its own venv; the bundled TypeScript MCP server is launched by `contextd serve` when an agent connects.

---

## Ingest

```bash
# PDFs (single file or directory; PDF adapter auto-detects from extension)
contextd ingest ~/papers/ --corpus research

# Claude Code / claude.ai conversation export (single JSON file)
contextd ingest ~/claude-exports/2026-03.json --corpus research

# Git repository (local clone; HEAD commit only, no history walk)
contextd ingest ~/code/my-project --corpus research
```

Each command is idempotent on content hash: re-running on an unchanged source is a no-op. Pass `--force` to re-ingest.

---

## Query

```bash
contextd query "how does Fu handle negation" --corpus research --limit 3
```

Flags: `--rerank/--no-rerank` (default on, requires `ANTHROPIC_API_KEY`), `--rewrite/--no-rewrite` (default off), `--json` (machine-readable output).

---

## Use from any MCP client

Start the combined stdio MCP + local HTTP backend:

```bash
contextd serve
```

The server is multi-corpus — every MCP tool takes `corpus` as a parameter, so one server instance handles all of your corpora. Use `list-corpora` from the client to see what's available.

Claude Code (`.mcp.json` in your project or `~/.claude.json`):

```json
{
  "mcpServers": {
    "contextd": {
      "command": "contextd",
      "args": ["serve"]
    }
  }
}
```

Codex CLI (`~/.codex/config.toml`):

```toml
[mcp_servers.contextd]
command = "contextd"
args = ["serve"]
```

The MCP server exposes 7 tools: `search-corpus`, `fetch-chunk`, `expand-context`, `get-edges`, `list-sources`, `get-source`, `list-corpora`.

---

## Why contextd vs. existing RAG products

Most RAG products are **"bring your data to the retriever"** — a cloud index, or an indexer welded into a single product. `contextd` is **"bring the retriever to the data, and let any agent call it."**

| | SaaS RAG<br/>(Glean, Mem, Pinecone+Assistants) | Vendor-native<br/>(Claude Projects, Cursor index, ChatGPT Knowledge) | DIY frameworks<br/>(LangChain, LlamaIndex) | **`contextd`** |
|---|---|---|---|---|
| Data location | Their cloud | Their cloud, one vendor | Wherever you wire it | **Local only; CI-enforced zero outbound** |
| Agents that can query it | One web UI + API | Exactly one product | Whatever you build yourself | **Any MCP client — same index** |
| Personal ↔ regulated boundary | Org pool + ACLs | Not a concept | Your problem | **Hard corpus-per-DB on disk** |
| Ingestion model | Auto-polling connectors | UI drag-drop | You write the pipeline | **Explicit CLI, idempotent on content hash** |
| Source-file mutation | N/A (they own the data) | Unclear | Your problem | **CI sha256s source tree before/after every ingest** |
| Retrieval underneath | Proprietary | Proprietary | You pick + tune | **Pinned stack: BGE-M3 + FTS5 + RRF + optional Haiku rerank, 30-query ship-gate** |

**The one thing nobody else does:** ingest once, then query the same local index from Claude Code, Codex CLI, Cursor, and Continue back-to-back — no re-upload, no vendor lock, no cloud detour.

**Where it's weaker, honestly:** no web UI, no auto-sync connectors (Notion/Gmail/Drive land in v0.2+), no team sharing, no cross-corpus search. If you want a dashboard or a multi-user knowledge base, this is the wrong tool.

---

## Design principles

- **Local-first.** SQLite + LanceDB under `~/.contextd/` (override with `CONTEXTD_HOME`). No data leaves the machine unless you explicitly enable reranking or query rewriting, which call Anthropic.
- **Zero telemetry.** No analytics, no crash reporting. Enforced by a privacy test suite that blocks every non-loopback socket and replays a full ingest + query cycle.
- **Named-corpus isolation.** Each corpus has its own SQLite DB and LanceDB directory on disk. Personal and clinical data cannot collide.
- **Source non-mutation.** Ingestion reads sources and writes only into the contextd data home. A CI test sha256-hashes the source tree before and after ingest on every push.

---

## Scope (v0.1)

| Feature | Status |
| --- | --- |
| PDF, Claude export, git-repo adapters | Shipped |
| Hybrid retrieval: dense (BGE-M3) + sparse (FTS5 BM25) + RRF | Shipped |
| Optional Claude Haiku reranker | Shipped (opt-in) |
| MCP stdio server (7 tools) | Shipped |
| FastAPI local HTTP backend (`/v1/*`) | Shipped |
| CLI: `ingest`, `query`, `serve`, `list`, `forget`, `status`, `config`, `version`, `eval` | Shipped |
| Named-corpus isolation (S5) | Shipped |
| Notion / Gmail / web adapters | v0.2+ |
| Cross-corpus federation | v0.2+ |
| Web UI | Not planned |

---

## Development

```bash
uv sync --dev
uv run ruff format .
uv run ruff check .
uv run mypy contextd/
uv run pytest                      # unit + integration
uv run pytest -m privacy           # privacy invariants

# MCP server (TypeScript)
cd mcp-server
pnpm install
pnpm biome check .
pnpm test
pnpm build
```

Eval harness against the 30-query seed set:

```bash
contextd eval contextd/eval/seed_queries.json --corpus eval
# Gate: Recall@5 ≥ 0.80, Recall@10 ≥ 0.90, MRR ≥ 0.60, judge ≥ 6.5
```

---

## Documentation

- [`docs/USER_GUIDE.md`](./docs/USER_GUIDE.md) — complete user guide (install, ingestion, querying, MCP, config reference, troubleshooting)
- [`docs/PRDs/`](./docs/PRDs/) — full v0.1 product spec (architecture, ingestion, retrieval, MCP, CI gates)
- [`docs/plans/`](./docs/plans/) — per-phase implementation plans (bootstrap → polish)
- [`CLAUDE.md`](./CLAUDE.md) — project instructions for Claude Code

---

## License

MIT. See [LICENSE](./LICENSE).
