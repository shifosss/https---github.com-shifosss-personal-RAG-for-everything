import { z } from "zod";
import { get } from "../http-client.js";
import { FetchChunkInput } from "../schemas.js";

export const FETCH_CHUNK = {
  name: "fetch_chunk",
  description: "Return the full ChunkResult for a chunk_id, with source metadata and (optional) edges.",
  inputSchema: FetchChunkInput,
  async handler(input: z.infer<typeof FetchChunkInput>): Promise<unknown> {
    const { chunk_id, corpus, include_edges, include_metadata } = input;
    return await get(`/v1/chunks/${chunk_id}`, { corpus, include_edges, include_metadata });
  },
} as const;
