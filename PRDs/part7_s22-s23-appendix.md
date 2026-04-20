# Section 22 — Roadmap Beyond v0.1

## 22.1 Framing

Three release milestones with meaningful distance between them:

- **v0.2** — ~3 months post-v0.1 (target August 2026). The "deferred Should-haves" release. Closes gaps in v0.1 that the dogfood period surfaces. Still single-user, still local-first, still open source.
- **v0.3** — ~6 months post-v0.1 (target November 2026). The "tier-2 ICP" release. Makes the tool usable by non-Alex researchers who share the ICP without requiring Alex's specific tooling familiarity.
- **v1.0** — ~12 months post-v0.1 (target May 2027). The maturity release. Enough stability, adoption, and reliability to carry the 1.0 name honestly.

These dates are aspirational, not committed. Alex's availability is the binding constraint; any of the three dates can slip by a quarter without invalidating the project. The ordering of features within each phase is more durable than the dates.

Each phase lists features under: ingestion, retrieval, surfaces, extensibility, ops. Unincluded areas mean "no planned change this phase" — stability is a feature.

## 22.2 v0.2 — the gap-closure release (target: August 2026)

The goal: convert deferred v0.1 Should-haves into shipped features, fix whatever dogfood surfaced, expand source coverage to the "obvious next three."

### 22.2.1 Ingestion

- **Obsidian / markdown adapter (full implementation).** S1 from v0.1; summary-specced in §14.5. Full wikilink graph, frontmatter parsing, incremental watch.
- **Watch-mode daemon.** S4 from v0.1. `contextd watch` runs as a background process, re-ingests changed files within 30 seconds.
- **Notion export adapter.** Lowest-friction path to Notion users: they export to HTML/Markdown, `contextd` ingests. Direct Notion API integration deferred to v0.3.
- **arXiv bookmark ingestion.** For the research ICP specifically — ingests a list of arXiv IDs, fetches PDFs (with user consent), ingests them with arXiv-specific metadata.
- **Automatic source type inference.** v0.1 requires `--type` for ambiguous paths; v0.2 infers from extension, file signature, and content sniffing.

### 22.2.2 Retrieval

- **Query rewriting enabled by default for MCP calls.** The full §15.3 specification activates. Specific numbers validate the directional expectations in §15.4.7.
- **Cross-encoder reranker.** `BAAI/bge-reranker-v2-m3` as a local alternative to LLM reranking. Closes the "fully-offline quality" gap — retrieval quality with no cloud dependency. Per §13.5.2.
- **Biomedical-specialized embeddings as a supported option.** Users with biomedical corpora can select `MedCPT` or `SPECTER2`. Per-corpus embed model selection (already in schema §11.3) makes this a config change, not a migration.
- **Metadata-filter improvements.** v0.1 supports equality filters; v0.2 supports range queries on numeric metadata (e.g., `pdf_page >= 5`) and `IN` filters on lists.
- **Retrieval quality target bump:** Recall@5 ≥ 0.85, MRR ≥ 0.70, LLM-judge ≥ 7.2 (per §18.3).

### 22.2.3 Surfaces

- **Minimal Web UI.** S2 from v0.1. Single query box, streaming results with citations, click-through to source chunks. Per §8.5.4. Runs on `localhost:8787` with zero authentication — still local-only.
- **`contextd doctor` diagnostic command.** C6 from v0.1. Checks dependencies, model cache, database integrity, MCP connectivity. Self-diagnosable setup.
- **`contextd stats` corpus statistics.** C5 from v0.1.
- **Systemd user-service template.** Alex-side polish; install script creates a user-level systemd unit so `contextd serve` can be auto-started on boot without root.

### 22.2.4 Extensibility

- **Adapter tutorial and template.** A `create-contextd-adapter` scaffolding command plus a written walkthrough. Lowers the barrier for community-contributed adapters.
- **First community adapters merged.** Goal: 2 adapters from non-Alex contributors (plausible candidates based on community interest: Kindle highlights, Logseq, Roam JSON export).

### 22.2.5 Ops

- **Native Windows support without WSL.** D10 from v0.1. SQLite-FTS5 paths, tree-sitter binaries, and tokenizer downloads all work on native Windows.
- **Offline install path.** Pre-download all models, ship a tarball, document air-gapped installation. Makes the tool usable in secure-network environments.
- **Eval harness as a first-class CLI command.** `contextd eval` executes the harness against a configurable query set. Published baseline for the default eval corpus.
- **Migrations framework.** First schema migration (if any) happens via the framework described in §11.12.

### 22.2.6 What v0.2 explicitly does not include

- Team or multi-user support. Still permanent non-goal N1/N4.
- Any cloud-hosted offering.
- GraphRAG — still deferred to v0.3.
- Fine-tuning embeddings on user corpus.

## 22.3 v0.3 — the tier-2-user release (target: November 2026)

The goal: make `contextd` usable by a research student who is not Alex, with their own ecosystem, their own corpus composition, and their own hardware. v0.3 is the "generalize beyond the author" release.

### 22.3.1 Ingestion

- **Gmail via MCP integration.** With user-configured Gmail MCP server, emails in a specified label become ingestible. Attachments handled as child sources.
- **Native Notion API integration.** Direct API pull (not just export ingestion). Requires user to provide a Notion integration token.
- **Web-page ingestion.** Given a URL, fetch, extract main content (via Readability), chunk. For bookmarks and occasional web references.
- **Code-imports edges.** Tree-sitter-level import graph extraction, producing `code_imports` edges in the EDGE table (§11.6). Enables "find all callers of this function" queries.
- **Better PDF table extraction.** v0.1 treats tables as opaque text; v0.3 produces structured table metadata for chart/figure-heavy papers.
- **Jupyter notebook adapter.** Ingest `.ipynb` files with cell-level chunking, preserving outputs as separate chunks flagged with `is_output` metadata.

### 22.3.2 Retrieval

- **GraphRAG-style retrieval.** With edges populated across more adapters, queries can traverse graph relationships. `get_edges` + `expand_context` + `search_corpus` chaining becomes a first-class pattern, aided by automatic query-to-graph-walk inference.
- **Clustering and topic summaries.** Embedding-based clustering over chunks produces topic labels visible in `list_sources` and `stats` output. Makes corpus introspection qualitatively different.
- **Per-corpus retrieval tuning.** User can set per-corpus defaults (reranker on/off, top_k, rewrite strategy) reflecting different corpus shapes.
- **Cross-corpus retrieval.** Opt-in `corpora: ["*"]` queries that span multiple corpora, with per-chunk corpus attribution. Previously a permanent v0.1 boundary; relaxed at user request.
- **Recency weighting.** Time-aware ranking for queries where recency matters ("what did I decide last month").

### 22.3.3 Surfaces

- **Packaged desktop app.** Lightweight Tauri or Electron wrapper around the Web UI + bundled MCP server + bundled Python runtime. Target: one-click install on macOS, Windows, Linux. Per D8 from v0.1 deferral.
- **Web UI polish.** Beyond the minimal v0.2 UI: corpus browser, source-level drill-downs, chunk-level annotation (user marks a chunk as "important" or "outdated").
- **MCP elicitation support.** When a query is ambiguous, the MCP server can ask the calling agent a follow-up question. Per §8.5.2 stretch.
- **Alternative MCP transports.** HTTP+SSE for remote-configured agents, WebSocket for future agent frameworks that prefer it.

### 22.3.4 Extensibility

- **Plugin system for retrieval hooks.** Users can insert custom logic at specific pipeline stages — e.g., a pre-embedding hook that augments PDFs with author lookup. Makes `contextd` a platform rather than a tool.
- **Community adapter registry.** A lightweight registry (just a JSON manifest in-repo) where community-contributed adapters are listed. Discovery without centralization.

### 22.3.5 Ops

- **Automatic backup integration.** Optional config for `contextd` to export its storage to a user-specified backup location on a schedule.
- **Telemetry infrastructure (opt-in only).** For users who want to contribute anonymized usage patterns to guide development, a strictly opt-in telemetry framework. Default remains off. Per non-goal N2, the posture is explicit and auditable.
- **Logging levels and rotation polish.** v0.1's minimal logging becomes a production-grade logging story.

### 22.3.6 What v0.3 explicitly does not include

- Hosted cloud version. Still permanent non-goal N1.
- Multi-user shared corpora. Still deferred.
- Mobile apps. Out of scope.
- Fine-tuning on user data. Deferred to v1.0 or later.

## 22.4 v1.0 — the maturity release (target: May 2027)

The goal: enough stability, adoption, and reliability to call the project 1.0 without embarrassment. Not "feature complete" — feature complete is never true — but "mature enough to rely on."

### 22.4.1 Ingestion

- **Source type coverage stabilizes.** At least 12 first-party adapters plus a healthy community-contributed set. New adapters are mechanical rather than research-level contributions.
- **Incremental ingestion for large sources.** Very large git repositories and large email archives ingest incrementally rather than all-at-once.
- **Content enrichment at ingest.** Optional LLM-based summarization, key-fact extraction, and entity linking during ingestion. Produces rich metadata without requiring query-time work.
- **Source-quality scoring.** Per-source quality scores based on usage patterns; high-query-hit sources get prioritized in ingestion batches.

### 22.4.2 Retrieval

- **Learned reranker.** A small reranker model fine-tuned on the user's own click and selection patterns (where available). Opt-in; no data leaves the machine for training.
- **Retrieval explanation mode.** `contextd query --explain` or the equivalent UI mode shows why each result was ranked where it was — which sub-queries matched, which dimensions contributed to the RRF score, what the reranker's reasoning was.
- **Structured query language.** Beyond natural-language queries, a user can construct structured queries (`source.type=pdf AND chunk.section=methods AND text MATCH "negation"`). Power-user only.
- **Retrieval quality target:** Recall@5 ≥ 0.90, MRR ≥ 0.75, LLM-judge ≥ 7.8.

### 22.4.3 Surfaces

- **Stability of MCP schemas.** v1.0 promises a 12-month stability commitment on the public MCP tool schemas — breaking changes only with 6-month deprecation notices.
- **Installer polish.** Native installers for each major OS that handle all dependencies (Python, models, tree-sitter grammars) in a single package.
- **Keyboard-first Web UI.** All operations accessible via keyboard shortcuts. Caters to the power-user ICP.
- **CLI UX refinements.** Years of shell-use feedback baked in; consistent flag conventions across all subcommands.

### 22.4.4 Extensibility

- **Stable plugin API.** The v0.3 plugin system graduates to a stable API with semver guarantees.
- **Non-Python adapters.** Adapters written in TypeScript, Go, or Rust can plug into `contextd` via a well-defined subprocess protocol. Lowers the contribution barrier for non-Python communities.
- **Mature community.** Target: 10+ non-Alex maintainers with merge rights or area ownership. Alex is a BDFL-with-advisors, not a single-point-of-failure.

### 22.4.5 Ops

- **Observability for power users.** Trace IDs exposed to end users; any retrieval can be dissected down to the millisecond per stage.
- **Upgrade guarantees.** Upgrading from any v0.x to v1.0 works via the migrations framework without manual intervention. Upgrading from v1.0 to any v1.x is trivial.
- **Public benchmarks.** Quarterly published eval runs against a canonical corpus, on reference hardware. External verifiability.

### 22.4.6 What v1.0 explicitly does not include

- Cloud offering. If a cloud version makes sense, it's a separate product built on `contextd`, not v1.0 itself.
- Fine-tuning of embedding models. Training infrastructure is a different engineering investment.
- Native mobile apps. Mobile agent ecosystem isn't mature enough to justify the effort.

## 22.5 Cross-cutting themes across phases

Not tied to a specific version; evolve continuously.

### 22.5.1 Retrieval quality

Every phase has a Recall@5 target (0.80 → 0.85 → 0.90 → 0.90+). Each target requires specific investments from the phase: query rewriting in v0.2, GraphRAG in v0.3, learned reranker in v1.0. Quality is not a side effect; it is a continuously-funded line.

### 22.5.2 Privacy posture

Privacy is non-negotiable in every phase. New features must not weaken the "zero outbound by default" contract. When they do require optional cloud features (reranker LLM, Gmail API), they are opt-in, documented, and auditable.

### 22.5.3 Documentation

Documentation grows proportionally with the codebase. v0.1: README only. v0.2: README plus `docs/` walkthroughs. v0.3: dedicated documentation site. v1.0: searchable docs, versioned, with examples for every MCP tool and every adapter.

### 22.5.4 Maintenance cadence

- v0.1 to v0.2: releases roughly monthly, bug fixes as needed.
- v0.2 to v0.3: releases roughly monthly, with a feature release every 2–3 months.
- v0.3 to v1.0: stabilization period; bugfix releases weekly, feature releases quarterly.

### 22.5.5 Testing and eval

- v0.1: 30-query eval set.
- v0.2: 50-query eval set with added real-world queries from dogfood.
- v0.3: 100-query eval set with coverage for all source types.
- v1.0: multi-corpus eval — quality measured across 3 canonical corpora (general research, pure engineering, mixed knowledge work).

## 22.6 What's deliberately absent from the roadmap

Features that are *not* on the path, to prevent scope creep:

- **Conversational agent inside `contextd`.** Permanent non-goal N4.
- **Hosted / cloud version.** Permanent non-goal N1.
- **Team or enterprise collaboration.** Would be a fundamentally different product.
- **Screen capture / OCR.** Screenpipe and Rewind handle this space well; no differentiation advantage in replicating.
- **Fine-tuning models on user corpus.** Possibly valuable but outside the "retrieval layer" thesis; a different project.
- **Email reply / drafting features.** `contextd` provides context; composing belongs to the calling agent.
- **Voice input, mobile clients.** Out of scope for the tier-1 and tier-2 ICPs; reconsider if tier-3 demand justifies the investment.
- **Multi-modal (image, audio, video) retrieval.** Retrieval over embedded text captions would be the first step, but full multi-modal is a v2.0+ conversation.

## 22.7 Roadmap governance

- Features move from "speculative" to "planned" only when:
  - Alex has used `contextd` daily for at least 4 weeks.
  - The feature's value is evidenced by either Alex's own friction or a recurring pattern of user requests.
  - An implementation path exists that fits within the architecture's existing abstractions (or requires a documented extension of them).
- Features move from "planned" to "in progress" only when:
  - The prior release has stabilized.
  - An initial spec (akin to the sections of this PRD) has been written.
- Users and contributors can propose roadmap changes via GitHub Discussions. Alex decides inclusion; decisions are logged in a `ROADMAP.md` in-repo with rationale.

## 22.8 Dates are intentions, not promises

Every date in this section is an internal target, not a commitment. Students' time is constrained, unpredictable, and non-fungible. The roadmap is a set of ordered priorities, not a Gantt chart.

If v0.2 slips from August to November 2026, the priorities don't change — the shipping date does. If a feature turns out to be materially harder than expected, it slides to a later phase rather than crowding out higher-priority work in its own phase.

The most important guarantee: v0.1 is a self-contained useful thing. If no v0.2 ever ships, users who install v0.1 still have a working tool, not a half-built promise.

---

# Section 23 — Open Questions & Decisions Log

## 23.1 Framing

Two parallel ledgers:

- **Decisions log (§23.2):** every substantive choice made during this PRD's drafting, with the alternatives considered and why this option won. Serves as the authoritative "why does the project look this way?" reference.
- **Open questions (§23.3):** unresolved design questions the drafting surfaced but did not settle. Each has an ADR-style entry with ID, description, options on the table, and status. The log is alive — entries move from `open` to `decided` or `deferred` as the project evolves.

Format for both: compact, scannable, hyperlinkable to the relevant PRD section.

## 23.2 Decisions log

All decisions below are recorded as `D-NN` identifiers. Each has: the decision, the alternatives considered, why the chosen option won, and a reference to the PRD section where it is authoritative.

**D-01: Project name is `contextd`.**
*Alternatives:* `corpora`, `trove`, `ledger`, placeholder.
*Rationale:* daemon suffix signals always-on MCP service (matching MCP-first architecture); unix-flavored; easy to type; available on GitHub and PyPI.
*Ref:* §1.

**D-02: Scope ambition for v0.1 is personal-use-first.**
*Alternatives:* shippable-product-first, both equally.
*Rationale:* YC demo is a side benefit, not the primary goal; avoids scope inflation during the 2-day build.
*Ref:* §1.

**D-03: Open source (MIT) from day one.**
*Alternatives:* open-core, closed-source.
*Rationale:* moat is community and adoption; closed-source would face permanent trust skepticism in the local-first privacy space.
*Ref:* §1.5, §2.6.

**D-04: Local-only deployment for v0.1.**
*Alternatives:* local-with-optional-cloud-sync, cloud-SaaS, hybrid.
*Rationale:* aligns with privacy posture; simplifies v0.1 substantially (no auth, no multi-tenancy); the user's machine has sufficient compute.
*Ref:* §1, §2.5, §9.3.

**D-05: Primary interface is MCP-first.**
*Alternatives:* CLI-first, chat-first, equal-weight.
*Rationale:* the wedge (§5.5) requires agents to be the primary consumer; CLI and UI are lenses over the same API.
*Ref:* §1, §5, §8.5.

**D-06: ICP for public messaging is "technical knowledge workers who use 3+ AI tools daily."**
*Alternatives:* ultra-narrow (ML researchers only), very narrow (technical research + engineering), broad (any AI user).
*Rationale:* wide enough for tier-3 ambition, narrow enough to produce focused messaging. Tiered structure (§4.1) preserves the Alex-first primary user without sacrificing breadth later.
*Ref:* §4.

**D-07: Non-goals included in the executive summary, not deferred.**
*Alternatives:* defer to Section 6 only.
*Rationale:* non-goals at the top of the PRD prevent misreadings and signal discipline early.
*Ref:* §1.10.

**D-08: Humble long-term vision framing.**
*Alternatives:* aggressive "open standard" framing, pragmatic middle.
*Rationale:* humility matches Alex's actual position (third-year student, side project); aggressive claims would read as unearned.
*Ref:* §1.9.

**D-09: Quantitative market data included in background section.**
*Alternatives:* qualitative only, mixed.
*Rationale:* concrete numbers on agent adoption, MCP uptake, and RAG market size strengthen the "why now" argument for evaluators (§2.2). Sourced via web search with citations.
*Ref:* §2.2.

**D-10: Legal/regulatory context addressed in the background section (not deferred to Risks).**
*Alternatives:* defer to Section 19.
*Rationale:* copyright, PII/GDPR, and HIPAA overlap with the design posture directly; stating them alongside the design rationale keeps the two linked.
*Ref:* §2.8.

**D-11: Competitive teardown kept light in Section 2, detailed in Section 20.**
*Alternatives:* full matrix in Section 2.
*Rationale:* background section should stay scan-friendly; detailed competitive analysis lives in its own section where it can breathe.
*Ref:* §2.3, §20.

**D-12: Problem statement is qualitative; quantitative measures live in Section 18.**
*Alternatives:* quantified Section 3.
*Rationale:* separation of concerns — Section 3 tells the story, Section 18 measures it.
*Ref:* §3, §18.

**D-13: "Day-in-the-life" narrative included in Section 3.**
*Alternatives:* defer narratives to Section 7 only.
*Rationale:* makes the abstract problem concrete before the PRD gets to specs; a reader who skips to Section 3 still gets the pain story.
*Ref:* §3.5, §3.6.

**D-14: ICP tiered Alex → research students → heavy AI users broadly.**
*Alternatives:* collective ML-researchers primary; single "heavy AI user" tier.
*Rationale:* Alex-first matches the build-it-for-yourself ethos; tiers preserve expansion optionality.
*Ref:* §4.

**D-15: Section 5 depth is one-sentence pitch + extended pitch + positioning + three differentiators + explicit wedge.**
*Alternatives:* shorter, longer with matrix.
*Rationale:* standard depth for a PRD positioning section; the explicit wedge (§5.5) is the load-bearing addition because the "why now" story determines the project's timing bet.
*Ref:* §5.

**D-16: No tagline in v0.1.**
*Alternatives:* commit to one now, offer candidates.
*Rationale:* tagline is downstream of product reality; choosing one before v0.1 ships risks anchoring on the wrong framing.
*Ref:* §5 (absence).

**D-17: MoSCoW structure for goals with explicit success criteria and three-way non-goal split.**
*Alternatives:* goals/non-goals binary, four-list only without criteria, criteria in Section 18 only.
*Rationale:* success criteria attached to goals make "done" checkable without flipping between sections; three-way non-goal split (never / deferred / v0.1-only) preserves roadmap integrity.
*Ref:* §6.

**D-18: Functional requirements grouped by subsystem, prose-only, with MCP schemas deferred to Section 12.**
*Alternatives:* behavior-grouped, schemas inline.
*Rationale:* subsystem grouping reads naturally for engineers implementing the system; schemas live in the interface spec where they belong.
*Ref:* §8, §12.

**D-19: Non-functional requirements grouped by NFR category with realistic-but-ambitious targets and threat model deferred.**
*Alternatives:* subsystem-mirrored, at-scale-vs-at-rest, flat list.
*Rationale:* NFR categories map to how reviewers think about quality dimensions; targets are achievable in a 2-day build while remaining meaningful.
*Ref:* §9.

**D-20: Multiple diagram types in Section 10 plus worked retrieval lifecycle.**
*Alternatives:* single diagram, prose only.
*Rationale:* logical architecture, data flow, and deployment topology answer different questions; one diagram per question is clearer than a single omnibus diagram.
*Ref:* §10.

**D-21: Deployment topology is process-per-component (MCP server / watcher / CLI as separate processes).**
*Alternatives:* single-process, hybrid, deferred.
*Rationale:* crash isolation and clean mental model outweigh the IPC cost; coordination via SQLite WAL keeps IPC complexity minimal.
*Ref:* §10.4.

**D-22: Data model is ER-style with forward compatibility for v0.2+ sources and first-class edge table.**
*Alternatives:* v0.1-only schema, edges-as-metadata-fields, partial edges.
*Rationale:* forward-compatibility cost is low; first-class edges unlock GraphRAG in v0.3 without migration.
*Ref:* §11.

**D-23: 7 MCP tools in v0.1 (search_corpus, fetch_chunk, expand_context, get_edges, list_sources, get_source, list_corpora), designed for chaining.**
*Alternatives:* minimal 3-tool set, rich 8+ set, one-shot-only tools.
*Rationale:* chaining primitives (fetch_chunk, expand_context, get_edges) multiply the expressive power without adding many tools; 7 is enough for the chaining patterns in §12.10.
*Ref:* §12.

**D-24: Full JSON Schema for every MCP tool plus full CLI spec plus full HTTP API spec in Section 12.**
*Alternatives:* MCP only, schemas simplified, HTTP deferred.
*Rationale:* the stability contract (§12.14) requires precise schemas; HTTP API is needed for the Web UI anyway.
*Ref:* §12.

**D-25: Tech stack is mixed Python (retrieval + ingestion core) + TypeScript (MCP server) + localhost-HTTP bridge.**
*Alternatives:* Python-only, TypeScript-only, Rust-core.
*Rationale:* each language plays to its ecosystem strengths (Python for ML, TS for MCP SDK); bridge complexity is bounded by using the same HTTP API both surfaces use.
*Ref:* §13.7.3.

**D-26: Locked dependency versions (no ranges).**
*Alternatives:* minimum versions, compatible ranges.
*Rationale:* reproducibility and supply-chain safety outweigh the maintenance cost of explicit bumps; `uv` makes updates trivial.
*Ref:* §13.9.

**D-27: Three Must-have adapters fully specified (PDF, Claude export, git); markdown adapter summary only.**
*Alternatives:* all four full, all four plus v0.2 adapters, one-per-page condensed.
*Rationale:* concentrating depth on the three adapters that ship first avoids bikeshedding on the one that can wait.
*Ref:* §14.

**D-28: Concrete input→output examples per adapter plus measurable quality bars.**
*Alternatives:* spec-only, examples-in-tests-only, partial quality bars.
*Rationale:* examples ground the spec; quality bars make adapter correctness checkable without subjective review.
*Ref:* §14.2.9, §14.3.9, §14.4.8.

**D-29: Retrieval pipeline spec includes stages, pseudocode for RRF and prompts, plus directional A/B expectations.**
*Alternatives:* prose-only, inline-tests, hardcoded numerical targets.
*Rationale:* pseudocode for non-obvious parts (RRF, prompts) prevents reimplementation drift; directional expectations in the spec guide the eval.
*Ref:* §15.

**D-30: Query rewriting spec'd but disabled by default in v0.1 production config.**
*Alternatives:* on-by-default for MCP, off everywhere, wait to ship at all.
*Rationale:* the complexity is written; the production switch is conservatively off until the first week of dogfood confirms baseline retrieval quality without it. Documented in §16.2.
*Ref:* §15.3, §16.2.

**D-31: Phase-based 2-day build plan (bootstrap / ingestion / retrieval / MCP / polish) with conservative scope and explicit cut ladder.**
*Alternatives:* hour-by-hour, half-day blocks, by-deliverable, aggressive scope.
*Rationale:* phase discipline matches the real failure mode ("spent 12h on ingestion, no time for MCP"); the cut ladder prevents silent scope creep at 3am on day 2.
*Ref:* §16.

**D-32: Evaluation is comprehensive — eval set + Recall@k + MRR + LLM-as-judge + per-adapter tests + manual QA.**
*Alternatives:* retrieval-focused, harness-first, lightweight.
*Rationale:* quality is the product's defensibility; under-investing in eval would be false economy. The 30-query set is itemized in §17.2.3.
*Ref:* §17.

**D-33: Success metrics three-tiered (personal / technical / community) with specific numeric targets and kill criteria.**
*Alternatives:* by audience, by phase, north-star plus supporting, directional targets, no kill criteria.
*Rationale:* three-tiered matches the three audiences the project serves; concrete numbers force honesty; kill criteria prevent zombie-project mode.
*Ref:* §18.

**D-34: Four-category risk matrix (technical / legal-privacy / competitive / adoption) with likelihood-impact scoring and a dedicated prompt-injection threat model.**
*Alternatives:* risk-ordered list, full threat model for everything, lightweight.
*Rationale:* category grouping matches the actual kinds of risk the project faces; prompt injection is novel enough to merit dedicated treatment.
*Ref:* §19.

**D-35: Competitive analysis is a 12×10 matrix plus prose teardown of top 6, with explicit naming and "what we can learn from each."**
*Alternatives:* quadrant framing, archetype-based, feature-by-feature.
*Rationale:* the matrix lets readers compare at a glance; the prose teardown acknowledges strengths honestly, which is tactically better than dismissal.
*Ref:* §20.

**D-36: GTM split into YC application track + broader open-source GTM, structural talking points only (no verbatim drafts), no named-person outreach list.**
*Alternatives:* timeline-based, channel-based, verbatim drafts, named-person list.
*Rationale:* the two motions have different timelines and success criteria; verbatim drafts age quickly; named outreach lists belong in tactical execution docs, not the PRD.
*Ref:* §21.

**D-37: Three-phase roadmap (v0.2 / v0.3 / v1.0) without formal "open research questions" subsection and without commercial planning.**
*Alternatives:* quarterly, feature-tracks, research questions section, sustainability planning.
*Rationale:* phase-based matches how releases actually happen; research questions are implicit in feature descriptions; commercial/sustainability is a separate conversation not essential for v0.1.
*Ref:* §22.

**D-38: Dates in the roadmap are explicitly intentions, not commitments.**
*Alternatives:* hard commitments, no dates at all.
*Rationale:* student-builder reality requires flexibility; stating this explicitly is more honest than either alternative.
*Ref:* §22.8.

**D-39: ADR-style open questions log plus comprehensive decisions log plus unnumbered Appendix (glossary, version history, MCP quick reference).**
*Alternatives:* topical open-questions log, partial decisions log, minimal appendix, numbered Section 24.
*Rationale:* every decision made in this PRD is recorded once in its authoritative section and again here for cross-reference; the Appendix is reference material, not a section — unnumbered reflects its status.
*Ref:* §23, Appendix.

## 23.3 Open questions (ADR-style)

Each entry: ID, description, options, status, next-step trigger.

**OQ-01 — Does v0.1 ship with the TypeScript MCP server, or fall back to Python-only?**
- *Description:* the bridge complexity described in §13.7.3 may not fit in the 2-day budget. §16.9 Cut 3 exists for this exact scenario.
- *Options:*
  - **(a)** Keep TypeScript MCP server; pay the setup cost.
  - **(b)** Fall back to Python MCP SDK; accept slightly less feature-parity.
- *Status:* open; decision deferred to Day 2 morning based on Day 1 progress.
- *Trigger:* end of Phase 3 (retrieval working) — if remaining time < 4h, take option (b).

**OQ-02 — Should query rewriting ship enabled by default in v0.1 or v0.2?**
- *Description:* spec'd in §15.3; decision to disable in v0.1 production was driven by conservative dogfooding.
- *Options:*
  - **(a)** v0.1 enabled for MCP calls, disabled for CLI.
  - **(b)** v0.1 disabled everywhere; enable in v0.1.1 after dogfood week.
  - **(c)** Never on by default; users opt in.
- *Status:* deferred to post-v0.1 (decision D-30 is provisional).
- *Trigger:* first week of dogfood retrieval-quality observations.

**OQ-03 — What is the canonical "default corpus" name?**
- *Description:* when a user runs `contextd query` without `--corpus`, it hits the default corpus. The default's name matters because it will appear in logs, help text, and example commands.
- *Options:* `default`, `main`, `personal`, user's username, blank (use "").
- *Status:* open.
- *Trigger:* Phase 1 implementation; can also be decided by user convention during v0.1.x.

**OQ-04 — How does `contextd` handle cross-corpus deduplication?**
- *Description:* if a user ingests the same PDF into two different corpora, the embedding is computed twice, and the chunk content is stored twice.
- *Options:*
  - **(a)** Accept duplication; corpora are independent partitions.
  - **(b)** Content-hash-based deduplication at the shared-cache level; chunks referenced across corpora via a pointer.
  - **(c)** Warn the user at ingest time and let them choose.
- *Status:* deferred to v0.2.
- *Trigger:* first user who ingests overlapping content.

**OQ-05 — What is the canonical Claude export format across Anthropic product changes?**
- *Description:* Claude's export format has changed over time (conversation vs. chat_messages naming; single-JSON vs. ZIP bundle). The adapter in §14.3 documents an assumed shape.
- *Options:*
  - **(a)** Pin to the format as of v0.1 ship; break loudly if it changes.
  - **(b)** Support multiple format versions; adapt at ingest time.
  - **(c)** Maintain a community-sourced format-spec doc.
- *Status:* open; currently (a) is implicit.
- *Trigger:* first time Anthropic's export format changes.

**OQ-06 — Should metadata filtering support fuzzy or regex matching in v0.1?**
- *Description:* §12.3.1 specifies equality-only filters. A user filtering `metadata: {arxiv_id: "2412.*"}` cannot match a range of IDs.
- *Options:*
  - **(a)** Equality-only in v0.1 (current spec).
  - **(b)** Glob/prefix matching in v0.1.
  - **(c)** Full regex in v0.1.
- *Status:* decided (a) for v0.1; reconsider in v0.2 (per §22.2.2).
- *Trigger:* user request surfacing in the first month.

**OQ-07 — How is the eval query set maintained over time?**
- *Description:* §17.9 describes a monthly refresh; mechanics are under-specified.
- *Options:*
  - **(a)** Monthly snapshot; old queries retained, new queries added.
  - **(b)** Rolling window of last 100 queries; older ones deprecated.
  - **(c)** Never retire queries; the set only grows.
- *Status:* decided (a) implicitly in §17.9; formalization deferred.
- *Trigger:* first monthly refresh event.

**OQ-08 — What happens when the user edits an ingested source manually while `contextd` is not watching?**
- *Description:* if `contextd watch` isn't running and the user modifies a PDF, the stored chunks become stale.
- *Options:*
  - **(a)** Staleness accepted; user re-ingests manually.
  - **(b)** Startup sweep compares content hashes on every run.
  - **(c)** CLI `contextd refresh` command performs the sweep on demand.
- *Status:* likely (c) for v0.1.1, deferred.
- *Trigger:* first user report of stale results.

**OQ-09 — Should `contextd` expose a write-back tool over MCP (e.g., "mark this chunk as important")?**
- *Description:* the §5.4.2 "corpus gets better with use" argument implies some feedback loop, but §10.6 explicitly scopes out any write-back operations in v0.1.
- *Options:*
  - **(a)** No write-back in v0.1, add in v0.3+ with user annotations.
  - **(b)** Minimal feedback tool (`mark_chunk`) in v0.2.
  - **(c)** No write-back ever; annotation belongs in the calling agent's memory.
- *Status:* decided (a); see §22.3.3 reference.
- *Trigger:* user demand.

**OQ-10 — What is the long-term plan for storing configured LLM API keys?**
- *Description:* v0.1 assumes keys are in env vars (`ANTHROPIC_API_KEY`). For a packaged desktop app (v0.3), env-var setup is user-hostile.
- *Options:*
  - **(a)** Continue env vars only; document clearly.
  - **(b)** Store keys in OS keychain (macOS Keychain, Windows Credential Manager, Secret Service on Linux).
  - **(c)** Store encrypted in `~/.contextd/config.enc`.
- *Status:* deferred to v0.3 packaging discussion.
- *Trigger:* v0.3 packaged-app design phase.

**OQ-11 — Does `contextd` support multiple MCP servers for the same corpus simultaneously?**
- *Description:* a user running both Claude Code and Codex CLI may spawn two `contextd` MCP server processes. SQLite WAL supports concurrent readers, but the MCP server itself holds some locks.
- *Options:*
  - **(a)** Single-server model; subsequent spawns detect and connect to the existing server.
  - **(b)** Multiple independent server instances; each serves one agent.
  - **(c)** One server, many client connections via stdio multiplexing.
- *Status:* open; default behavior is (b) in Phase 4 implementation; (a) is the long-term direction.
- *Trigger:* observed contention during multi-agent use.

**OQ-12 — How should `contextd` handle corpus rotation (e.g., "archive my research corpus from 2024")?**
- *Description:* a user accumulating corpora indefinitely will eventually want to export an older corpus to cold storage.
- *Options:*
  - **(a)** Manual: the user moves `~/.contextd/corpora/<name>/` elsewhere.
  - **(b)** `contextd archive <corpus>` command with a defined archive format.
  - **(c)** Automatic rotation based on configurable rules.
- *Status:* deferred to v0.3+; (a) works for v0.1.
- *Trigger:* a user asking.

---

# Appendix

## A.1 Glossary

Terms specific to `contextd` or frequently used in the PRD. Definitions oriented toward a reader who isn't deeply familiar with the retrieval or MCP ecosystems.

**Adapter** — A source-type-specific ingestion module. One adapter per source type (PDF, Claude export, git repo, etc.). Implements parsing, chunking, metadata extraction, and edge production. See §14.1.

**AGPL** — Affero GNU Public License. A strong copyleft license used by pymupdf. Relevant here because it's a transitive dependency of the default PDF adapter; its implications for users are documented. See §19.3.3.

**BGE-M3** — A multilingual embedding model from Beijing Academy of AI (BAAI). Default embedder in `contextd`. Outputs 1024-dimensional vectors.

**BM25** — A lexical (keyword-based) ranking function used in text search. SQLite's FTS5 extension implements it. `contextd` uses BM25 as the sparse half of hybrid retrieval.

**Chunk** — The atomic unit of retrieval in `contextd`. A section of source content (one PDF section, one conversation turn, one function) that's embedded and indexed. See §11.5.

**Content hash** — SHA-256 of a source's canonical content. Used for idempotent ingestion: if the hash hasn't changed, re-ingestion is a no-op. See §14.2.

**Corpus** — A named partition of ingested content. Users can maintain multiple corpora (`research`, `personal`, `coursework`) with independent retrieval. See §11.3.

**Dense retrieval** — Semantic search based on vector similarity between the query embedding and chunk embeddings. Good for paraphrase matching; weaker for exact identifiers.

**Edge** — A typed relationship between chunks (`conversation_next`, `wikilink`, `pdf_cites`, etc.). First-class entity in the data model. See §11.6.

**FTS5** — SQLite's full-text search extension, version 5. Provides BM25 ranking out of the box. Used for sparse retrieval.

**HIPAA / PHIPA** — Health Insurance Portability and Accountability Act (US) / Personal Health Information Protection Act (Ontario). Regulate clinical data handling. `contextd`'s design explicitly separates personal corpora from regulated data to avoid accidental commingling. See §2.8.3.

**Household exemption** — Article 2(2)(c) of the GDPR: the Regulation does not apply to personal data processed by a natural person "in the course of a purely personal or household activity." `contextd`'s single-user local-only posture fits this exemption. See §2.8.2.

**Hybrid retrieval** — Combining dense (semantic) and sparse (BM25) retrieval to capture queries that neither method alone handles well. Results merged via RRF. See §15.4.

**Idempotent ingestion** — Re-ingesting the same source produces no duplicate chunks. A design requirement for `contextd`'s ingestion pipeline. See §8.2.5.

**LanceDB** — An embedded vector database using the Lance columnar format. `contextd`'s chosen store for embeddings. Competes with Chroma, Pinecone, Qdrant.

**LLM-as-judge** — Using a language model to score the quality of retrieval results. An evaluation technique complementary to purely-numeric metrics like Recall@k. See §17.3.3.

**MCP (Model Context Protocol)** — Open standard introduced by Anthropic in November 2024 for connecting AI agents to external tools and data. `contextd` is an MCP server. See §2.2.

**MoSCoW** — Must-have / Should-have / Could-have / Won't-have. A prioritization framework used for v0.1 scope. See §6.

**MRR (Mean Reciprocal Rank)** — An IR metric measuring how highly the first correct result is ranked across a query set. Complements Recall@k. See §17.3.2.

**Non-mutation guarantee** — `contextd` never modifies source files during ingestion. Enforced by test. See §8.2.8.

**Provenance** — The traceable origin of a chunk: source path, ingestion timestamp, original content hash, chunk offset. Every retrieval result includes full provenance.

**Recall@k** — Fraction of queries where at least one expected-correct result appears in the top-k ranked results. Primary retrieval quality metric.

**RRF (Reciprocal Rank Fusion)** — A score-free method for merging multiple ranked lists into one. Used in `contextd`'s hybrid retrieval. See §15.5.

**Section-aware chunking** — Chunking that respects the structural boundaries of a source (PDF sections, conversation turns, code functions) rather than using fixed token windows. See §14.

**Source** — A single ingested unit (one PDF, one conversation, one repo). Tracked in the `SOURCE` table. One `SOURCE` row produces multiple `CHUNK` rows.

**Sparse retrieval** — Keyword-based search (BM25). Complements dense retrieval in hybrid setups.

**Stdio transport** — MCP's default way of talking between a client (e.g., Claude Code) and a server (e.g., `contextd`) via standard input/output streams of a spawned subprocess.

**Tree-sitter** — A parser generator that produces syntax trees for source code. `contextd` uses it for function-aware chunking of code sources.

**Trace ID** — A ULID assigned to each retrieval call, propagated through every stage, and logged. Enables post-hoc debugging.

**Wikilink** — The `[[target]]` syntax in Obsidian/Markdown that creates a named reference to another note. `contextd`'s markdown adapter parses these into edges.

## A.2 Version history of this PRD

Maintained in reverse chronological order.

| Version | Date | Author | Changes |
|---|---|---|---|
| 0.1.0 | 2026-04-19 | Alex + Claude (co-authored) | Initial complete draft. Sections 1–23 plus Appendix. Decisions D-01 through D-39. Open questions OQ-01 through OQ-12. Consistency pass applied fixes for latency targets (§8.3.7 → §9), query-rewriting default (§15.3.2), markdown-adapter scope (§14.5). |

Subsequent versions track: every decision status change, every open question resolution, every section revision. Format: `major.minor.patch` where:
- `major` — structural change to the PRD (section added/removed).
- `minor` — substantive change to an existing section.
- `patch` — typo fixes, citation updates, link refreshes.

## A.3 MCP tool quick reference

A single-page reference for the seven MCP tools exposed by `contextd`. Matches the full specs in §12.

### A.3.1 `search_corpus` — primary retrieval

- **Input:** `query` (required), plus optional `corpus`, `limit`, `source_types`, `date_range`, `source_path_prefix`, `metadata_filters`, `rewrite`, `rerank`.
- **Output:** ranked `results[]` (each: chunk content, score, source reference, metadata, edges), plus `query` echo, plus `trace`.
- **Typical use:** entry point for any natural-language query.

### A.3.2 `fetch_chunk` — retrieve by ID

- **Input:** `chunk_id`, optional `include_edges`, `include_metadata`.
- **Output:** the chunk with its source reference and metadata.
- **Typical use:** after `search_corpus`, agent wants full detail on a specific result.

### A.3.3 `expand_context` — zoom out

- **Input:** `chunk_id`, optional `before` (N chunks), `after` (N chunks).
- **Output:** chunks in source order from anchor−before to anchor+after.
- **Typical use:** agent wants surrounding context for a retrieved chunk.

### A.3.4 `get_edges` — traverse relationships

- **Input:** `chunk_id`, optional `direction` (inbound/outbound/both), `edge_types`, `include_target_chunks`, `limit`.
- **Output:** edges from the anchor chunk, optionally with target chunks embedded.
- **Typical use:** follow wikilinks, walk a conversation's turns, trace code imports (v0.2+).

### A.3.5 `list_sources` — enumerate

- **Input:** optional `corpus`, `source_types`, `ingested_since`, `limit`, `offset`.
- **Output:** source registry entries with chunk counts and status.
- **Typical use:** agent or user wants to know what's in the corpus.

### A.3.6 `get_source` — full source entry

- **Input:** `source_id` OR `path` (exactly one), optional `corpus`.
- **Output:** the source's full registry entry including metadata.
- **Typical use:** drill down from a chunk to see the full source's metadata.

### A.3.7 `list_corpora` — enumerate corpora

- **Input:** none.
- **Output:** available corpora with name, embed model, source count, chunk count.
- **Typical use:** agent discovering what corpora are available before querying.

### A.3.8 Chaining patterns

- **Search → expand:** `search_corpus` then `expand_context` on a chosen result.
- **Search → follow:** `search_corpus` then `get_edges(type=wikilink)`.
- **Search → traverse conversation:** `search_corpus` then `get_edges(type=conversation_next)` repeatedly.
- **Enumerate → query-within-source:** `list_sources` then `search_corpus(source_path_prefix=...)`.

Full input/output JSON Schemas in §12. This quick reference is the summary card.

---

*End of PRD.*
