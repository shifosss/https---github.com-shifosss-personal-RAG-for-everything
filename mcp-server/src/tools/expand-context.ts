import { z } from "zod";
import { get } from "../http-client.js";
import { ExpandContextInput } from "../schemas.js";

export const EXPAND_CONTEXT = {
  name: "expand_context",
  description: "Return N chunks before and N chunks after the anchor chunk, in source order.",
  inputSchema: ExpandContextInput,
  async handler(input: z.infer<typeof ExpandContextInput>): Promise<unknown> {
    const { chunk_id, before, after, corpus } = input;
    return await get(`/v1/chunks/${chunk_id}/context`, { before, after, corpus });
  },
} as const;
