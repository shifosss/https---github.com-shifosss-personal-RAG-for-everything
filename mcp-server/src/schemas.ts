import { z } from "zod";

export const SourceType = z.enum([
  "pdf",
  "claude_export",
  "git_repo",
  "markdown",
  "notion",
  "gmail",
  "arxiv_bookmark",
  "web_page",
]);
export type SourceType = z.infer<typeof SourceType>;

export const EdgeType = z.enum([
  "wikilink",
  "conversation_next",
  "conversation_prev",
  "code_imports",
  "pdf_cites",
  "email_reply_to",
  "email_thread",
]);

export const SearchFilters = z.object({
  source_types: z.array(SourceType).optional().default([]),
  date_from: z.string().optional(),
  date_to: z.string().optional(),
  source_path_prefix: z.string().optional(),
  metadata: z.record(z.string()).optional().default({}),
});

export const SearchInput = z.object({
  query: z.string().min(1),
  corpus: z.string().default("personal"),
  limit: z.number().int().min(1).max(100).default(10),
  rewrite: z.boolean().default(false),
  rerank: z.boolean().default(true),
  filters: SearchFilters.optional(),
});
export type SearchInput = z.infer<typeof SearchInput>;

export const FetchChunkInput = z.object({
  chunk_id: z.number().int(),
  corpus: z.string().default("personal"),
  include_edges: z.boolean().default(true),
  include_metadata: z.boolean().default(true),
});

export const ExpandContextInput = z.object({
  chunk_id: z.number().int(),
  before: z.number().int().min(0).max(20).default(2),
  after: z.number().int().min(0).max(20).default(2),
  corpus: z.string().default("personal"),
});

export const GetEdgesInput = z.object({
  chunk_id: z.number().int(),
  direction: z.enum(["inbound", "outbound", "both"]).default("both"),
  edge_types: z.array(EdgeType).optional(),
  include_target_chunks: z.boolean().default(false),
  limit: z.number().int().min(1).max(500).default(50),
  corpus: z.string().default("personal"),
});

export const ListSourcesInput = z.object({
  corpus: z.string().default("personal"),
  source_types: z.array(SourceType).optional(),
  ingested_since: z.string().optional(),
  limit: z.number().int().min(1).max(500).default(50),
  offset: z.number().int().min(0).default(0),
});

export const GetSourceInput = z.object({
  source_id: z.number().int(),
  corpus: z.string().default("personal"),
});

export const ListCorporaInput = z.object({});
