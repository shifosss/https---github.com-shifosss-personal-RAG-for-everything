# Section 17 — Evaluation Methodology

## 17.1 Framing

Evaluation in `contextd` serves three distinct audiences:

- **Alex as developer:** "did this change break something?" — regression tests, run on every commit.
- **Alex as user:** "is the tool actually helping me?" — longitudinal quality tracking, run weekly.
- **External evaluators (YC, contributors, users):** "does this thing work?" — reproducible benchmarks, documented in the README.

A single eval harness (`contextd eval`) serves all three. The harness is itself a first-class CLI command (Could-have C4 in §6.3; promoted to Must-have of evaluation work here because regression testing is non-negotiable).

The methodology covers four evaluation targets:

1. **Retrieval quality** — the hardest and most important; §17.2–§17.5.
2. **Adapter quality** — per-adapter correctness bars from §14; §17.6.
3. **System-level integration** — end-to-end happy paths and failure modes; §17.7.
4. **Non-functional verification** — privacy, determinism, non-mutation, latency; §17.8.

## 17.2 Retrieval eval set: construction

### 17.2.1 Corpus

A fixed evaluation corpus ships with the repo under `eval/corpus/`. It contains:

- **30 PDFs** from arXiv (CC-BY / CC-BY-SA / permissive preprint licenses only). Mix of clinical NLP, ML theory, and systems papers — matching Alex's actual reading distribution.
- **1 synthetic Claude conversation export** with 40 conversations covering ML topics, engineering decisions, and coursework discussions. Hand-crafted to resemble real patterns without exposing real data.
- **1 small git repository** with ~5k LOC of Python and TypeScript, intentionally containing known functions with known names (so queries like "the `embed_query` function" have an unambiguous answer).

Corpus composition is fixed by a manifest file (`eval/corpus/manifest.yaml`) with SHA-256 hashes. Changes to the corpus require a version bump of the eval harness.

### 17.2.2 Query set design

30 queries total, distributed across realistic workflow patterns. Each query has:

- **id**: stable identifier (`Q01`..`Q30`).
- **text**: the query as a user would write it.
- **category**: one of {`prior_art`, `decision_recall`, `code_recall`, `synthesis`, `meeting_prep`, `retrospective`, `edge_case`}.
- **expected_sources**: list of source paths (in the eval corpus) that must contribute to a correct answer.
- **expected_topics**: optional topical keywords that must appear in at least one top-5 result.
- **difficulty**: {`easy`, `medium`, `hard`}.
- **notes**: why this query is in the set.

### 17.2.3 The 30 queries

*Query IDs reference fixture filenames in the eval corpus. The full JSONL is shipped as `eval/queries.jsonl`; this is the documentation.*

**Prior-art queries (6):** testing paper-synthesis narrative (§7.2).

- **Q01** `easy` — "negation handling in clinical NLP"
  - Expected: `fu_2024_negation.pdf`, `kaster_2024_zeroshot.pdf`
  - Expected topics: negation, clinical, tagged sequences
  - Notes: baseline semantic query; both papers are topical.

- **Q02** `medium` — "compare Fu and Kaster negation approaches"
  - Expected: `fu_2024_negation.pdf:methods`, `kaster_2024_zeroshot.pdf:methods`
  - Expected topics: Fu, Kaster, comparison
  - Notes: compound query; tests whether rewriting (when enabled) or RRF surfaces both methods sections.

- **Q03** `hard` — "which paper uses tagged sequences?"
  - Expected: `fu_2024_negation.pdf:methods`
  - Expected topics: tagged, sequence, insertion
  - Notes: one-source answer; exact-phrase query tests sparse retrieval.

- **Q04** `medium` — "what did I read about MoE quantization incompatibility?"
  - Expected: `claude-export.json#conv:baichuan-moe-quant`
  - Expected topics: MoE, BitsAndBytes, fused
  - Notes: past-tense first-person; tests conversation retrieval over PDFs.

- **Q05** `hard` — "LoRA for small clinical corpora"
  - Expected: `liu_2024_lora_clinical.pdf`, `claude-export.json#conv:lora-small-corpora`
  - Expected topics: LoRA, low-rank, clinical, fine-tuning
  - Notes: cross-source — a paper AND a conversation both contribute.

- **Q06** `medium` — "BGE-M3 vs nomic embed comparison"
  - Expected: `reimers_2024_multilingual_eval.pdf`, `claude-export.json#conv:embed-model-choice`
  - Expected topics: BGE-M3, nomic, embedding, MTEB
  - Notes: tests specific-model-name retrieval.

**Decision recall queries (5):** testing narrative 2 and 4 (§7.3, §7.5).

- **Q07** `easy` — "why did we choose MedGemma over Qwen"
  - Expected: `claude-export.json#conv:medgemma-vs-qwen`
  - Expected topics: MedGemma, Qwen, architecture
  - Notes: a real decision from Alex's clinical NLP work.

- **Q08** `medium` — "last time I hit a CUDA arch mismatch"
  - Expected: `claude-export.json#conv:h100-cuda-arch`, `eval_repo:commits`
  - Expected topics: CUDA, sm_90, H100, cuda-toolkit
  - Notes: tests cross-source retrieval (conversation + commit).

- **Q09** `medium` — "what did I conclude about fused 3D expert tensors?"
  - Expected: `claude-export.json#conv:baichuan-moe-quant`
  - Expected topics: fused, expert tensors, BitsAndBytes, incompatible
  - Notes: tests retrieval of user's own conclusion vs. the assistant's.

- **Q10** `hard` — "the approach I rejected three weeks ago for ingestion"
  - Expected: `claude-export.json#conv:ingestion-pipeline-design`
  - Expected topics: rejected, ingestion
  - Notes: intentionally vague to test recency-weighted retrieval and inference.

- **Q11** `easy` — "continued pretraining strategy for ~1400 records"
  - Expected: `claude-export.json#conv:continued-pretrain-small`
  - Expected topics: continued pretraining, small corpus
  - Notes: specific numeric constraint.

**Code recall queries (6):** testing git adapter and function-scope chunking.

- **Q12** `easy` — "where is the PDF adapter defined"
  - Expected: `eval_repo:src/adapters/pdf.py:PDFAdapter`
  - Expected topics: PDFAdapter, class
  - Notes: structure-aware retrieval.

- **Q13** `medium` — "how does the chunker handle oversized sections?"
  - Expected: `eval_repo:src/adapters/pdf.py:_chunk`
  - Expected topics: split, paragraph, token
  - Notes: intent query against implementation.

- **Q14** `easy` — "the `embed_query` function"
  - Expected: `eval_repo:src/retrieval/dense.py:embed_query`
  - Expected topics: embed_query, FlagEmbedding
  - Notes: exact identifier query (sparse-friendly).

- **Q15** `medium` — "what embedding model does the codebase use?"
  - Expected: `eval_repo:src/retrieval/dense.py`, `eval_repo:pyproject.toml`
  - Expected topics: BGE-M3, sentence-transformers, FlagEmbedding
  - Notes: cross-file code knowledge.

- **Q16** `hard` — "function that builds the RRF score"
  - Expected: `eval_repo:src/retrieval/fusion.py:reciprocal_rank_fusion`
  - Expected topics: RRF, score, rank
  - Notes: intent-style query against code.

- **Q17** `medium` — "TypeScript MCP server entrypoint"
  - Expected: `eval_repo:packages/mcp-server/src/index.ts`
  - Expected topics: MCP, server, tool
  - Notes: tests cross-language retrieval.

**Synthesis queries (4):** testing narrative 1 (§7.2).

- **Q18** `medium` — "summarize the negation approaches across Fu and Kaster"
  - Expected: `fu_2024_negation.pdf:methods`, `kaster_2024_zeroshot.pdf:methods`
  - Expected topics: negation, tagged, zero-shot
  - Notes: synthesis over multiple papers; agent performs the synthesis, `contextd` provides the chunks.

- **Q19** `hard` — "how do recent papers handle clinical note de-identification?"
  - Expected: 2+ of {`bannett_2024.pdf`, `fu_2024_negation.pdf`, `liu_2024_lora_clinical.pdf`}
  - Expected topics: de-identification, PHI
  - Notes: topic that appears tangentially in multiple papers.

- **Q20** `hard` — "methods I could adopt for my pipeline"
  - Expected: 3+ of the clinical NLP papers
  - Expected topics: method, pipeline, clinical
  - Notes: intentionally broad; tests whether retrieval surfaces a breadth of candidates.

- **Q21** `medium` — "what hyperparameters do the comparison papers use?"
  - Expected: `fu_2024_negation.pdf:methods`, `kaster_2024_zeroshot.pdf:methods`
  - Expected topics: learning rate, batch size, epochs, LoRA rank
  - Notes: tests structured numeric retrieval.

**Meeting-prep queries (3):** testing narrative 3 (§7.4).

- **Q22** `medium` — "relevant papers I've read for a meeting about game AI"
  - Expected: `silver_2018_alphazero.pdf`, `weber_2020_stellaris_rl.pdf`
  - Expected topics: game AI, RL, strategy
  - Notes: topical filter for meeting context.

- **Q23** `easy` — "the Stellaris advisor PRD"
  - Expected: `eval_corpus:stellaris_advisor_prd.md`
  - Expected topics: Stellaris, PRD, advisor
  - Notes: name-based retrieval of a known artifact.

- **Q24** `hard` — "my prior correspondence about Prof. Gao's research"
  - Expected: `claude-export.json#conv:gao-outreach-draft`
  - Expected topics: Gao, outreach, collaboration
  - Notes: name-entity retrieval; tests whether retrieval handles proper nouns.

**Retrospective queries (3):** testing narrative 5 (§7.5).

- **Q25** `medium` — "what did I work on this week"
  - Expected: any sources with `ingested_at >= now - 7 days` in the eval corpus (simulated via fixture timestamps)
  - Expected topics: (open)
  - Notes: tests date-range filtering — not pure retrieval quality.

- **Q26** `hard` — "decisions I made in March about embeddings"
  - Expected: conversations in `claude-export.json` with timestamps in March
  - Expected topics: embedding, BGE-M3, nomic
  - Notes: tests date + topic filter combination.

- **Q27** `medium` — "commits I made to the retrieval engine"
  - Expected: `eval_repo:src/retrieval/*` with commits
  - Expected topics: retrieval, commit
  - Notes: tests git metadata in retrieval.

**Edge case queries (3):** testing failure modes and robustness.

- **Q28** `edge` — "" (empty query)
  - Expected: INVALID_ARGUMENT error
  - Notes: verifies preprocessing rejects empty queries per §15.2.2.

- **Q29** `edge` — "asdfjkl qwerty nonexistent_term_xyz"
  - Expected: zero or near-zero-score results
  - Notes: tests graceful behavior on no-match queries; results should exist but LLM-as-judge should score them low.

- **Q30** `edge` — "the" (stopword-only)
  - Expected: near-zero-quality results with a `trace.note` flagging low-information query
  - Notes: tests whether retrieval degrades gracefully on information-poor queries.

### 17.2.4 Query set discipline

- **No overlap with training corpus of the embedder.** BGE-M3 was trained before March 2026; the eval queries use specific project names (`contextd`, `MedGemma LoRA pipeline`, `stellaris-advisor`) that won't be memorized.
- **Each query tests one thing cleanly.** Edge cases isolated; multi-aspect queries labeled as synthesis.
- **Balanced difficulty.** 10 easy, 12 medium, 5 hard, 3 edge.
- **Balanced category.** Matches workflow distribution from §7.

## 17.3 Retrieval metrics

Three complementary metrics. No single one captures quality; together they do.

### 17.3.1 Recall@k

For each query, does the expected source appear in the top-k results?

- **k=5**: the primary target. Agents typically consume top-3 to top-5; anything below k=5 isn't an agent answer.
- **k=10**: secondary; tests whether the correct chunk is anywhere in the working set.

Computation: for each query with at least one expected source, recall@k = 1 if any expected source appears in the top-k, else 0. Averaged across the query set.

Target (v0.1): Recall@5 ≥ 0.80 on the 30-query set. Recall@10 ≥ 0.90.

This is the §6.1 M4 success criterion, measured on this exact set.

### 17.3.2 Mean Reciprocal Rank (MRR)

For each query, the reciprocal of the rank of the first expected-source hit. Captures *how highly* the right answer is ranked, not just whether it appears.

- For query *i*, if the first correct result is at rank *r*, MRR contribution = 1/*r*; if no correct result in top-10, contribution = 0.
- Averaged across queries.

Target (v0.1): MRR ≥ 0.60. This is a conservative target; strong RAG systems score MRR > 0.70.

### 17.3.3 LLM-as-judge quality score

For each query, an LLM evaluates the top-5 results against the query on a 0–10 scale and returns an average. This captures quality dimensions metrics miss: relevance of partially-correct matches, quality of false-positive chunks, whether retrieval returned *the useful content* vs. *a technically on-topic chunk*.

#### 17.3.3.1 The judge prompt

```
You are a strict evaluator of a personal retrieval system.

Given a user query and a top-5 ranked list of retrieved chunks, score
each chunk's usefulness on a 0-10 scale:

- 10: chunk directly answers the query with specific information
- 7-9: chunk is strongly on-topic and contributes to answering
- 4-6: chunk is tangentially related
- 1-3: chunk shares vocabulary but is off-topic
- 0: chunk is irrelevant or nonsensical

Also compute an AGGREGATE score for the whole top-5 result set:
- 10: the top-5 comprehensively answer the query
- 7-9: the top-5 answer with minor gaps
- 4-6: the top-5 partially answer; a user would need more searches
- 1-3: the top-5 mostly miss the point
- 0: total failure

Respond in strict JSON:
{
  "per_chunk": [
    {"id": 123, "score": 8, "reason": "..."},
    ...
  ],
  "aggregate": 7,
  "aggregate_reason": "..."
}

Scoring discipline:
- Be skeptical. A chunk mentioning the query's topic but not answering
  is a 4-6, not 7+.
- For compound queries, a chunk that answers ONE sub-question is 6-7;
  a chunk that answers ALL is 9-10.
- For decision-recall queries ("what did I conclude..."), the chunk
  must state the conclusion, not merely discuss the topic.
- For code queries, the chunk must contain the actual code referenced.
```

Judge model: `claude-sonnet-4-6` or higher (stronger than the reranker to provide an independent judgment). Temperature 0. Evaluated once per eval run.

Target (v0.1): average aggregate score ≥ 6.5 across the query set.

### 17.3.4 Metric combination

The three metrics are reported together, not collapsed into one number:

```
v0.1 eval run summary:
  Recall@5:          0.83  (target: 0.80 ✓)
  Recall@10:         0.93  (target: 0.90 ✓)
  MRR:               0.67  (target: 0.60 ✓)
  LLM aggregate avg: 7.2   (target: 6.5 ✓)
  Queries passing all targets: 28/30
```

A failing metric on a single query is expected; a failing metric across the set blocks a release.

## 17.4 A/B methodology for retrieval configurations

The eval harness supports running the same query set under different retrieval configurations and comparing.

### 17.4.1 Fixed configurations to benchmark

- **`dense_only`**: dense retrieval only, no sparse, no RRF, no rerank.
- **`sparse_only`**: BM25 only, no dense.
- **`hybrid_rrf`**: dense + sparse + RRF, no rerank (the v0.1 default when rerank off).
- **`hybrid_rerank`**: dense + sparse + RRF + Haiku rerank (the v0.1 full pipeline).
- **`hybrid_rewrite_rerank`**: + query rewriting (v0.1.1+).

### 17.4.2 Expected directional wins

Per §15.4.7:

- `hybrid_rrf` > `dense_only` on the full set, with the largest wins on code-identifier queries (Q14, Q17).
- `hybrid_rrf` > `sparse_only` on the full set, with the largest wins on paraphrase queries (Q02, Q18).
- `hybrid_rerank` > `hybrid_rrf` on Recall@5 and MRR, with some regression possible on Recall@10 (reranker can over-filter).
- `hybrid_rewrite_rerank` > `hybrid_rerank` on compound queries (Q02, Q18, Q21) and no worse on simple queries.

Any reversal of these expectations is a signal to investigate before shipping.

### 17.4.3 Running an A/B

```bash
contextd eval --config-a hybrid_rrf --config-b hybrid_rerank --queries eval/queries.jsonl
```

Output: paired metrics per config, delta per query, and a Wilcoxon signed-rank test on per-query LLM-aggregate scores (n=30 gives meaningful significance).

## 17.5 Retrieval regression protocol

### 17.5.1 Pre-commit baseline

Before any retrieval-layer change, run the eval and save metrics as the baseline:

```bash
contextd eval --save-baseline baselines/v0.1.0.json
```

### 17.5.2 CI regression test

On every PR, CI runs:

```bash
contextd eval --compare-baseline baselines/v0.1.0.json --fail-on-regression
```

Regression thresholds:
- Recall@5 drop > 0.05 → fail.
- MRR drop > 0.05 → fail.
- LLM aggregate drop > 0.5 → fail.
- Any metric drop on > 3 individual queries → warn but don't fail.

Soft failures (warnings) are reviewed by the author; hard failures block merge.

### 17.5.3 Baseline updates

Baselines are updated deliberately, not automatically. Updating a baseline requires:

1. A commit message stating why the change is an intended improvement or acceptable tradeoff.
2. A note in the release notes for the affected version.

## 17.6 Adapter-level tests

Each adapter carries the quality bars from §14. The test harness encodes them.

### 17.6.1 PDF adapter tests (§14.2.11)

```python
# tests/adapters/test_pdf.py

def test_section_label_accuracy():
    """On 10 fixture PDFs, ≥90% of random-sampled chunks have correct labels."""
    corpus = load_fixture_corpus("pdf_10_papers")
    chunks = sample(all_chunks(corpus), 20)
    correct = sum(1 for c in chunks if c.section_label == EXPECTED_LABELS[c.chunk_id])
    assert correct / len(chunks) >= 0.90

def test_title_extraction():
    """On 10 fixture PDFs, ≥8 titles match expected."""
    ...

def test_password_protected_pdf():
    """Encrypted PDFs produce a recorded failure without aborting the batch."""
    ...

def test_scanned_pdf():
    """Scanned PDFs produce a 'likely scanned' error, not a crash."""
    ...

def test_re_ingestion_idempotent():
    """Re-ingesting an unchanged PDF produces zero new chunks."""
    ...
```

Expected labels are curated by hand for the fixture set; a separate `EXPECTED_LABELS.yaml` file documents them for reproducibility.

### 17.6.2 Claude export adapter tests (§14.3.11)

```python
# tests/adapters/test_claude_export.py

def test_turn_preservation():
    """A 50-message conversation produces 50 chunks with correct role/timestamps."""
    ...

def test_edge_completeness():
    """conversation_next/prev edges form a complete linked list."""
    ...

def test_idempotent_reingestion():
    """Re-ingesting the same export produces no new chunks."""
    ...
```

### 17.6.3 Git adapter tests (§14.4.10)

```python
# tests/adapters/test_git.py

def test_function_isolation():
    """On a fixture repo with 60 top-level declarations, ≥54 appear as distinct chunks."""
    ...

def test_gitignore_honored():
    """Ignored files produce zero chunks."""
    ...

def test_commit_metadata():
    """10 random chunks have commit_hash matching `git log` output."""
    ...
```

### 17.6.4 Cross-adapter invariants

```python
# tests/adapters/test_invariants.py

def test_no_source_mutation():
    """For each adapter, hash all sources before and after ingestion."""
    # This satisfies §9.3.4 at the adapter level.

def test_determinism():
    """For each adapter, ingestion twice produces identical DB state."""
    # Diffs SQLite dumps post-ingest.

def test_offset_validity():
    """For every chunk, offset_start < offset_end <= source_length."""
```

## 17.7 System-level integration tests

End-to-end tests that exercise multiple components. Slower; run pre-merge and pre-release, not on every commit.

### 17.7.1 Happy-path lifecycle

```python
def test_full_lifecycle():
    # 1. Install contextd (checked by separate install smoke test).
    # 2. Initialize a corpus.
    # 3. Ingest one PDF, one Claude export file, one small repo.
    # 4. Run `contextd query` with 5 specific queries.
    # 5. Assert top-1 result matches expected source for each.
    # 6. Start MCP server.
    # 7. Make a JSON-RPC call to each of the 7 tools.
    # 8. Verify responses match schemas.
    # 9. Forget one source.
    # 10. Verify cascade deletion (no chunks remain, no FTS entries, no vectors).
    # 11. Clean up.
```

This is the "sleep easy" test. If it passes, the system works end-to-end.

### 17.7.2 Failure-mode tests

```python
def test_mcp_server_survives_malformed_request():
    """Invalid JSON-RPC doesn't crash the server."""

def test_retrieval_degrades_without_reranker():
    """With reranker unreachable, retrieval returns RRF results with a trace note."""

def test_corrupted_storage_detected():
    """A corrupted SQLite file produces a clear startup error."""

def test_partial_ingestion_failure():
    """One corrupt PDF in a batch of 10 doesn't abort the batch; 9 succeed."""
```

### 17.7.3 Multi-agent integration test (manual)

A manual protocol, not automated:

1. Configure `contextd` in Claude Code and Codex CLI.
2. Ask Claude Code: "Compare Fu and Kaster negation handling from my research corpus."
3. Verify Claude Code calls `search_corpus` with appropriate parameters.
4. Verify the response contains chunks from both fixture papers.
5. In Codex CLI, ask the same question.
6. Verify Codex gets structurally-identical chunk content (same source paths, same content_hashes).
7. Log any differences in formatting or result order.

Run this protocol once per release and once per major MCP SDK version bump.

## 17.8 Non-functional verification

### 17.8.1 Privacy verification

```python
def test_zero_outbound_network_on_default_config():
    """Full lifecycle with default config produces zero outbound HTTP calls."""
    # Uses a pytest fixture that monkeypatches socket.connect to raise on
    # any non-loopback address. Allowlist: loopback, Unix sockets, AF_LOCAL.
    # Reranker calls are disabled via config override in this test.
```

On CI, an additional environment-level check runs `strace -e trace=network` and greps for non-loopback connections. Zero matches required.

### 17.8.2 Non-mutation verification

```python
def test_sources_unchanged_after_ingestion():
    """SHA-256 hashes of all source files are identical before and after."""
    # Runs against all fixture sources.
```

### 17.8.3 Determinism verification

```python
def test_retrieval_deterministic():
    """Identical query on identical corpus returns identical results across runs."""
    # Seed-controlled; reranker temperature 0.
```

### 17.8.4 Latency verification

```python
def test_retrieval_latency_budget():
    """p95 retrieval latency on the 30-query set meets §9.2 baseline."""
    # Runs the query set 3 times; records p50/p95/p99.
    # Asserts p95 <= 800ms (no rerank) or <= 3000ms (with rerank).
```

Latency tests are sensitive to hardware; CI reports them as informational, not gating.

## 17.9 Longitudinal dogfood protocol

Beyond the 30-query eval, Alex's own use produces the richest signal. The protocol:

### 17.9.1 Weekly query log review

Every Sunday:
1. Review the last week's `AUDIT_LOG` entries for retrievals (excluding DEBUG-level query text, which is off by default).
2. For 5 random retrievals, open the result set and rate each on the LLM-judge scale by hand.
3. Record in `DOGFOOD.md` in a private dotfile (not in the public repo).

### 17.9.2 Monthly eval set refresh

Every month, add 5 new queries to the eval set drawn from the prior month's actual use. Retire or rewrite queries that have become trivially easy (a signal that the underlying retrieval improved).

The eval set grows by roughly 60 queries/year under this protocol. By v0.5 the set is meaningfully robust.

### 17.9.3 Regression alerts

If weekly review reveals a subjective quality drop that the automated eval didn't catch, add the offending query to `eval/queries.jsonl` immediately. This is how the eval set learns from real use.

## 17.10 Manual QA protocol (pre-release)

A release checklist run before tagging a new version.

1. ✅ CI green on `main` for the release commit.
2. ✅ Eval on dogfood corpus: Recall@5 ≥ 0.80, MRR ≥ 0.60, LLM aggregate ≥ 6.5.
3. ✅ Adapter tests pass on all fixture sources.
4. ✅ System lifecycle test passes.
5. ✅ Privacy test passes (zero outbound calls).
6. ✅ Non-mutation test passes (no source files touched).
7. ✅ `pipx install` on a clean VM or WSL instance (fresh, not author's workstation).
8. ✅ `contextd version` returns the expected version.
9. ✅ `contextd --help` output reviewed.
10. ✅ Manual multi-agent integration protocol (§17.7.3).
11. ✅ README install instructions followed end-to-end.
12. ✅ Demo video still reflects current UX.
13. ✅ Release notes drafted; known issues honestly listed.

One failure blocks release. No exceptions, no "it's close enough."

## 17.11 What the evaluation methodology does not include

Honest boundaries:

- **No formal benchmark comparison (BEIR, MTEB).** Those benchmarks are for generic retrieval over well-studied corpora; `contextd`'s distinguishing quality comes from source-specific chunking on a personal corpus. Generic benchmarks would understate its usefulness and overstate a generic RAG baseline's.
- **No user study with external users (v0.1).** The single-user scope makes external users not representative. Post-v0.1 a small (n=5–10) invited-user study is appropriate; not in scope now.
- **No adversarial or red-team evaluation.** Prompt injection via ingested content (a real risk — a malicious PDF could instruct a downstream LLM to do something) is deferred to Section 19 (Risks). A systematic evaluation is v0.2+ work.
- **No long-term drift tracking.** BGE-M3 doesn't self-update, so drift is zero by default. If the embedder ever changes, the whole corpus re-embeds and the baseline is re-established.

---