import { z } from "zod";
import { get } from "../http-client.js";
import { ListSourcesInput } from "../schemas.js";

export const LIST_SOURCES = {
  name: "list_sources",
  description: "Enumerate ingested sources in a corpus, newest first. Supports type and since filters.",
  inputSchema: ListSourcesInput,
  async handler(input: z.infer<typeof ListSourcesInput>): Promise<unknown> {
    const { corpus, source_types, ingested_since, limit, offset } = input;
    const query: Record<string, string | number | boolean> = { corpus, limit, offset };
    if (ingested_since) query.ingested_since = ingested_since;
    if (source_types && source_types.length > 0) {
      const url = new URL("http://placeholder/v1/sources");
      for (const [k, v] of Object.entries(query)) url.searchParams.set(k, String(v));
      for (const t of source_types) url.searchParams.append("source_types", t);
      return await get(url.pathname + url.search);
    }
    return await get("/v1/sources", query);
  },
} as const;
