import { z } from "zod";
import { get } from "../http-client.js";
import { GetEdgesInput } from "../schemas.js";

export const GET_EDGES = {
  name: "get_edges",
  description:
    "Traverse typed relationships (wikilinks, conversation threads, code imports, citations) from a chunk.",
  inputSchema: GetEdgesInput,
  async handler(input: z.infer<typeof GetEdgesInput>): Promise<unknown> {
    const { chunk_id, direction, edge_types, include_target_chunks, limit, corpus } = input;
    const query: Record<string, string | number | boolean> = {
      direction, include_target_chunks, limit, corpus,
    };
    if (edge_types && edge_types.length > 0) {
      const url = new URL(`http://placeholder/v1/chunks/${chunk_id}/edges`);
      for (const [k, v] of Object.entries(query)) url.searchParams.set(k, String(v));
      for (const t of edge_types) url.searchParams.append("edge_types", t);
      return await get(url.pathname + url.search);
    }
    return await get(`/v1/chunks/${chunk_id}/edges`, query);
  },
} as const;
