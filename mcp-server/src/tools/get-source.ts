import { z } from "zod";
import { get } from "../http-client.js";
import { GetSourceInput } from "../schemas.js";

export const GET_SOURCE = {
  name: "get_source",
  description: "Return a source's registry entry and all source_meta keys.",
  inputSchema: GetSourceInput,
  async handler(input: z.infer<typeof GetSourceInput>): Promise<unknown> {
    const { source_id, corpus } = input;
    return await get(`/v1/sources/${source_id}`, { corpus });
  },
} as const;
