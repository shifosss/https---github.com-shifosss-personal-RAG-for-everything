# Phase 4 Cross-AI Smoke Protocol

Manual verification that the contextd MCP server works with Claude Code and Codex.
This is the final Phase 4 exit gate check — run before marking the phase complete.

## Prerequisites

1. Python + Node deps installed:
   ```bash
   uv sync
   cd mcp-server && pnpm install && pnpm run build && cd ..
   ```
2. A corpus with real content ingested (ideally `research`):
   ```bash
   uv run contextd ingest ~/papers/ --corpus research --type pdf
   ```
3. `ANTHROPIC_API_KEY` exported (optional but recommended for rerank).

## Smoke 1 — Claude Code

1. Start the combined server:
   ```bash
   uv run contextd serve
   ```
   Confirm HTTP listens on `127.0.0.1:8787` and the stdio MCP child process starts.

2. Add to your `.mcp.json` (at project root, or `~/.claude.json` for user-level):
   ```json
   {
     "mcpServers": {
       "contextd": {
         "command": "node",
         "args": ["/ABSOLUTE/PATH/TO/personal-RAG-for-everything/mcp-server/dist/index.js"],
         "env": { "CONTEXTD_HTTP_HOST": "127.0.0.1", "CONTEXTD_HTTP_PORT": "8787" }
       }
     }
   }
   ```

3. In Claude Code, open the tool palette (or type `/mcp`). Verify all 7 tools appear:
   - `search_corpus`
   - `fetch_chunk`
   - `expand_context`
   - `get_edges`
   - `list_sources`
   - `get_source`
   - `list_corpora`

4. Prompt: _"Use contextd to search the research corpus for 'transformer architecture' and summarise the top 3 chunks."_

5. Verify: Claude calls `search_corpus`, receives ranked chunks with source paths + metadata, and summarises. Screenshot for demo.

6. Follow-up: _"Fetch chunk with id N and expand context by 2 chunks before and after."_
   Verify: `fetch_chunk` + `expand_context` work end-to-end.

## Smoke 2 — Codex CLI

1. Add to Codex CLI MCP config (path depends on Codex version; typically `~/.codex/config.toml` or similar). Pointed at the same `mcp-server/dist/index.js`.

2. Start a Codex session and ask: _"Search my contextd corpus for 'X' and fetch chunk Y."_

3. Verify at least `search_corpus` and `fetch_chunk` return valid data. Other tools are bonus.

## Smoke 3 — Error paths

1. Stop `contextd serve` (kill the HTTP backend). Ask Claude Code to search again.
2. Verify: the MCP tool returns a JSON-RPC error with `code: "INTERNAL"` or similar (via the `HttpError` path in TS client).
3. Restart `contextd serve`. Verify recovery: Claude Code can search again without restarting the MCP child process.

## Recording

- Screenshot the 7-tool palette from Claude Code.
- Screenshot a successful `search_corpus` response in both Claude Code and Codex.
- Save under `docs/demo/` for the Phase 5 demo video script.

## Exit gate

- [ ] All 7 tools listed in Claude Code
- [ ] `search_corpus` + `fetch_chunk` + `expand_context` work end-to-end in Claude Code
- [ ] `search_corpus` + `fetch_chunk` work in Codex CLI
- [ ] Graceful error when `contextd serve` is down
- [ ] Screenshots captured
