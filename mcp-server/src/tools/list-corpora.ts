import { z } from "zod";
import { get } from "../http-client.js";
import { ListCorporaInput } from "../schemas.js";

export const LIST_CORPORA = {
  name: "list_corpora",
  description: "List all named corpora on this machine with source_count, chunk_count, embed_model.",
  inputSchema: ListCorporaInput,
  async handler(_input: z.infer<typeof ListCorporaInput>): Promise<unknown> {
    return await get("/v1/corpora");
  },
} as const;
