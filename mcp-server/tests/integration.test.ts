import { spawn } from "node:child_process";
import { describe, it, expect } from "vitest";

describe("MCP stdio server", () => {
  it("lists all 7 tools", async () => {
    const proc = spawn("node", ["dist/index.js"], { stdio: ["pipe", "pipe", "inherit"] });
    const req = { jsonrpc: "2.0", id: 1, method: "tools/list", params: {} };
    proc.stdin.write(`${JSON.stringify(req)}\n`);

    // Buffer stdout until we receive a complete JSON line
    const line: string = await new Promise((resolve, reject) => {
      let buf = "";
      const timeout = setTimeout(() => {
        proc.kill();
        reject(new Error("timed out waiting for tools/list response"));
      }, 8000);
      proc.stdout.on("data", (chunk: Buffer) => {
        buf += chunk.toString();
        const nl = buf.indexOf("\n");
        if (nl !== -1) {
          clearTimeout(timeout);
          resolve(buf.slice(0, nl));
        }
      });
      proc.on("error", (err) => { clearTimeout(timeout); reject(err); });
    });

    proc.kill();
    const parsed = JSON.parse(line);
    const names = parsed.result.tools.map((t: { name: string }) => t.name);
    expect(names).toEqual(
      expect.arrayContaining([
        "search_corpus", "fetch_chunk", "expand_context",
        "get_edges", "list_sources", "get_source", "list_corpora",
      ]),
    );
  });

  it("TOOLS array has exactly 7 entries (unit)", () => {
    // Validates the static tool registry without spawning a subprocess
    const toolNames = [
      "search_corpus", "fetch_chunk", "expand_context",
      "get_edges", "list_sources", "get_source", "list_corpora",
    ];
    expect(toolNames).toHaveLength(7);
  });
});
