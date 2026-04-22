#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import { zodToJsonSchema } from "zod-to-json-schema";
import { HttpError } from "./http-client.js";
import { EXPAND_CONTEXT } from "./tools/expand-context.js";
import { FETCH_CHUNK } from "./tools/fetch-chunk.js";
import { GET_EDGES } from "./tools/get-edges.js";
import { GET_SOURCE } from "./tools/get-source.js";
import { LIST_CORPORA } from "./tools/list-corpora.js";
import { LIST_SOURCES } from "./tools/list-sources.js";
import { SEARCH_CORPUS } from "./tools/search-corpus.js";

const TOOLS = [
  SEARCH_CORPUS, FETCH_CHUNK, EXPAND_CONTEXT, GET_EDGES,
  LIST_SOURCES, GET_SOURCE, LIST_CORPORA,
] as const;
const TOOLS_BY_NAME = Object.fromEntries(TOOLS.map((t) => [t.name, t]));

const server = new Server({ name: "contextd", version: "0.1.0" }, { capabilities: { tools: {} } });

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: TOOLS.map((t) => ({
    name: t.name,
    description: t.description,
    inputSchema: zodToJsonSchema(t.inputSchema as never) as Record<string, unknown>,
  })),
}));

server.setRequestHandler(CallToolRequestSchema, async (req) => {
  const tool = TOOLS_BY_NAME[req.params.name];
  if (!tool) throw new Error(`unknown tool: ${req.params.name}`);
  const parsed = tool.inputSchema.safeParse(req.params.arguments ?? {});
  if (!parsed.success) throw new Error(`invalid args: ${parsed.error.message}`);
  try {
    const result = await tool.handler(parsed.data as never);
    return { content: [{ type: "text", text: JSON.stringify(result) }] };
  } catch (e) {
    if (e instanceof HttpError) {
      return {
        isError: true,
        content: [{ type: "text", text: JSON.stringify({ code: e.code, message: e.message, status: e.status }) }],
      };
    }
    throw e;
  }
});

await server.connect(new StdioServerTransport());
