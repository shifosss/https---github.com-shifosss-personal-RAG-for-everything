import { z } from "zod";
import { post } from "../http-client.js";
import { SearchInput } from "../schemas.js";

export const SEARCH_CORPUS = {
  name: "search_corpus",
  description:
    "Hybrid retrieval across the given corpus (dense + sparse + RRF, optional rerank). Returns ranked chunks with full provenance.",
  inputSchema: SearchInput,
  async handler(input: z.infer<typeof SearchInput>): Promise<unknown> {
    return await post("/v1/search", input);
  },
} as const;
