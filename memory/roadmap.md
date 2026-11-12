# Roadmap - engineering pass 2 (November 2026)

The engineering log of this pass. One item per session; each item ends in one pushed commit.
Statuses: NOT STARTED, IN PROGRESS, BLOCKED, DONE (tested, committed, pushed), CI VERIFIED,
SKIPPED (with the reason), REVERTED (with the numbers).

## A. Overview

Deep Learning Tutor is a Streamlit workbench: datasets, Keras models, training, evaluation,
Grad-CAM, script export, and a Gemini tutor with a knowledge base, an agent and an MCP server.
This pass finishes the held-out test set work, makes API cost visible and bounded, hunts bugs,
and tries a few measured optimisations. Small steps, measured, resumable.

## B. Starting condition (checked 2026-11-12)

- main at 979f56f, 5 commits, clean tree, CI green (GitHub Actions "tests": Python 3.13,
  pyflakes, pytest, ubuntu-latest, about 2.5 minutes). 450 tests pass locally (figure from the
  first pass; re-counted at the first full run of this pass).
- Unfinished from the first pass: the exported script still makes the old 80/20 split; the AI
  review prompt, the tutor context, the agent tools and the MCP server/client speak only of
  validation accuracy; the CNN speed experiment was never run. CNN training measured about twice
  as slow as the September figure (MNIST 3 epochs: 39.6 s against 19 s), cause unknown.
- The Gemini layer already has (do not rebuild): SHA-256 cache keys over (kind, model, json mode,
  thinking budget, system prompt, prompt); in-memory LRU response cache (256) and embedding cache
  (5,000) behind a lock; only the training review, the fact-check and embeddings are cached, the
  tutor chat never; one client per API key, reused; key rotation with a 60 s cooldown; bounded
  retries with exponential backoff and jitter, server retry delay respected up to 20 s; errors
  classified quota / transient / fatal; embeddings in batches of 100; documents de-duplicated by
  content hash; a per-knowledge-base retrieval index; agent loop capped at 5 tool rounds with
  repeated identical tool calls answered from memory; a cache of identical agent experiments.
- The Gemini layer does not have (from reading the code, each to be confirmed in its item):
  a request timeout; any record of calls / tokens / retries / cache hits; streaming of the tutor
  answer; use of system_instruction in the chat; compact embedding storage (3,072-number Python
  lists, stored in the cache and again in each chunk; about 200 MB for 2,000 chunks, an estimate
  from the code, not a measurement).

## C. Stack

Python 3.13, Streamlit 1.64.0, TensorFlow 2.20.0 (CPU), Keras 3.15.1, NumPy 2.5.3, pandas 3.0.6,
scikit-learn 1.9.1, google-genai 2.25.0, langchain-text-splitters 1.1.2, pypdf 6.19.0,
pytest 9.1.1, pyflakes 4.0.0. Versions are pinned in requirements.txt; nothing is upgraded in
this pass. Development machine: a CPU-only laptop (Intel Core Ultra 5 125H, 15.6 GB RAM).

## D. Baseline measurements

Measured 2026-11-12 at commit 8025149 with `benchmarks/run_benchmarks.py` (hermetic, fake LLM),
on mains power, CPU only. Full results are kept as baseline.json in the run folder. One run;
medians are over the repeats the script makes (3 to 20, as noted).

| What | Baseline |
|------|----------|
| Cold first run (fresh process: all app imports + first page) | 5,347 ms (median of 3) |
| of which `import tensorflow` | 4,361 ms (median of 3) |
| No-op rerun, nothing trained | 23.8 ms (median of 5) |
| No-op rerun, Iris model trained | 29.8 ms (median of 5) |
| Train click on Iris, 50 epochs (whole rerun) | 3,139 ms |
| Chat message round trip, instant fake LLM (app overhead) | 31.3 ms (median of 3) |
| RAG keyword search, 500 chunks, index built per query | 22.1 ms (median of 5) |
| RAG keyword search, 500 chunks, cached index | 0.27 ms (median of 5) |
| RAG cosine search, 500 x 3,072, built per query | 38.5 ms (median of 5) |
| RAG cosine search, 500 x 3,072, cached index | 0.46 ms (median of 5) |
| One MNIST epoch, 6,000 images, Train click in the app | 2,159 ms |
| One MNIST epoch, 6,000 images, training only (app's tf.data pipeline) | 1,300 ms |
| Response cache: 1 miss + 51 identical requests | 1 SDK call |
| MCP: Iris 30 epochs round trip | 6,255 ms |

### API calls per user action (read from the code at 8025149, not measured)

"Attempts" are what `call_with_retry` allows per call: max(3, min(keys + 2, 6)), so 3 to 6
depending on the number of keys. No request has a timeout yet (item 3.2).

| User action | Gemini calls when all goes well | Worst case | Cached? |
|-------------|---------------------------------|------------|---------|
| Tutor question, no knowledge base | 1 generate | 1 call x 3-6 attempts | never |
| Tutor question, knowledge base with Gemini embeddings | 1 embed (the question) + 1 generate | each x 3-6 attempts | the question's embedding only |
| Tutor question, knowledge base without embeddings | 1 generate (keyword search is local) | 1 call x 3-6 attempts | never |
| The above with the fact-checker on | + 1 generate (fast model) | + 2 generates (fast model fails, selected model retried), each x 3-6 attempts | yes, same answer + same sources |
| Agent question | 1 generate per tool round + the answer; typically 2-3 | 5 tool rounds + 1 forced final answer = 6 generates, each x 3-6 attempts | no (identical tool calls and identical experiments are answered locally) |
| AI review of a training run | 1 generate | 1 call x 2 attempts | yes, identical prompt = 0 calls |
| Learning a document of N chunks | ceil(N / 100) embed calls | each x 3-6 attempts | yes, per chunk text; a duplicate document = 0 calls |

## E. Storage (measured 2026-11-12)

- Tracked repo files: about 2 MB (largest: handwritten_seven.png 690 KB, five screenshots 0.9 MB).
- Project environment site-packages: 2,063 MB, of which tensorflow 1,303, pyarrow 85,
  matplotlib 30 (used for one colormap import), langchain_core + langsmith 14 (pulled in by
  langchain-text-splitters, used for one splitter class).
- pip cache in the run folder: 522 MB.

## F. Bug list

Empty until item 4.1.

## G. Items

| Item | Priority | Size | What | Status | Commit | CI |
|------|----------|------|------|--------|--------|----|
| 0.1 | MUST | S | Set up CLAUDE.md and this roadmap | CI VERIFIED | 8025149 | green |
| 1.1 | MUST | S | Baseline benchmark; table "API calls per user action" | CI VERIFIED | f4f2c03 | green |
| 2.1 | MUST | M | Exported script uses the app's three-way split | CI VERIFIED | 3fecf64 | green |
| 2.2 | MUST | M | Test accuracy in AI review, tutor status, agent tools, MCP | CI VERIFIED | d866512 | green |
| 3.1 | MUST | M | API usage meter (backend/usage.py + sidebar expander) | DONE | e6f67b0 | running when checked |
| 3.2 | MUST | S | Request timeout on the Gemini client | DONE | see git log | pending |
| 4.1 | MUST | M | Bug hunt, time-boxed; at most three fixes (4.1a-c) | NOT STARTED | | |
| 3.3 | SHOULD | M | Embedding memory: float32 arrays, maybe smaller dimension | NOT STARTED | | |
| 5.1 | SHOULD | M | CNN speed: explain the gap, try cheap levers | NOT STARTED | | |
| 3.4 | OPTIONAL | M | Prompt structure: system_instruction, stable-first order | NOT STARTED | | |
| 3.5 | OPTIONAL | M | Streaming for the plain English tutor answer | NOT STARTED | | |
| 6.1 | OPTIONAL | S | Dependency check (langchain-text-splitters) | NOT STARTED | | |
| 6.2 | OPTIONAL | S | Cleanup audit of unreferenced files and dead functions | NOT STARTED | | |
| 7.1 | MUST | M | Final proof: tests, benchmarks, docs, README, clone check | NOT STARTED | | |
| 7.2 | MUST | S | Private guide and run summary (outside the repo) | NOT STARTED | | |

Item details (short form of the brief; the acceptance rules are binding):

- 1.1 No code changes. On mains power run `python benchmarks/run_benchmarks.py --out
  <run folder>/baseline.json` and copy about ten headline numbers into section D: no-op rerun,
  Train click on Iris, chat message overhead, RAG search with and without the index, cold first
  run, one MNIST epoch. From the code, fill a table "API calls per user action": tutor question,
  with a knowledge base, with the fact-checker on, agent question (worst case), AI review,
  learning a document of N chunks.
- 2.1 core/codegen.py and tests/test_codegen.py. The script reproduces the app's rows for the
  same seed and prints validation and test accuracy. Remove the "older two-way split" captions
  and README lines.
- 2.2 The AI review prompt (tts_engine.py), the tutor's project status (app.py project_status),
  the agent tools (keep ranking by validation, also report test), mcp_server.py, mcp_client.py.
  New UI text in English and natural Malayalam, technical terms in English. Remove the matching
  README limitations. If too large, split into 2.2a (review + tutor + agent) and 2.2b (MCP).
- 3.1 backend/usage.py, called by llm_api after every Gemini request: calls, retries, cache hits
  and misses, prompt / output / cached tokens from response.usage_metadata (confirm field names
  in the google-genai docs), seconds per call. Per Streamlit session, same fallback as
  llm_api._state for tests. Sidebar expander "API usage (this session)", both languages. Tests
  with a fake client.
- 3.2 Confirm in the google-genai docs how to set a per-request timeout (HttpOptions), its unit,
  and whether the SDK retries on its own. One timeout for generation, a shorter one for
  embeddings, as named constants. A timeout is classified "transient"; SDK retries and
  call_with_retry must not multiply. Test with a fake that raises the timeout error.
- 4.1 Read backend/llm_api.py, backend/agent.py, agent_tools.py, core/rag.py,
  components/chat_helpers.py and the ZIP loader in core/data.py. At most eight HYPOTHESES in
  section F with severity P0-P3. Failing test first; fix only what a test reproduces; three fixes
  at most, one commit each. Unconfirmed leads: fact_check retries with the selected model after
  ANY LLMError (even "all keys rate-limited"); embed_texts trusts result.embeddings only by
  count; rag.fingerprint looks at the first and last chunk id only; app.py answer() leaves the
  user message in history with no reply when Gemini fails.
- 3.3 Measure first with tracemalloc (500-chunk knowledge base, embedding cache). Then float32
  NumPy arrays shared by cache and chunks. If the docs confirm output_dimensionality for
  gemini-embedding-001, try 768 on 20 fixed questions over deep_learning_study_guide.txt (at
  most 3 real embedding calls). Keep only if memory falls at least 4x and top-3 is unchanged for
  float32, or at least 18 of 20 questions keep the same first result for the smaller dimension.
- 5.1 Explain the 39.6 s against 19 s gap first: three repeats of 3 MNIST epochs on mains power,
  old split sizes against new, training and evaluation timed separately. Then one lever at a
  time (steps_per_execution, larger evaluation batch, dataset cache/prefetch). Keep only if at
  least 15% faster with test accuracy within 0.3 points; otherwise undo and record the numbers.
- 3.4 Fixed instructions into system_instruction; user message stable-first (instructions,
  knowledge base, history, status, question). Read the docs on implicit caching and its minimum
  prompt size for the 2.5 models. At most 4 real calls: prompt and cached tokens before/after.
  Keep only if all chat tests pass and prompt tokens do not rise.
- 3.5 Plain English tutor answer only (not agent mode, not Malayalam JSON mode). Confirm
  generate_content_stream in the docs. Retry only before the first chunk. At most 3 real calls:
  time to first text and total time. Streaming shortens the wait for the first words, not the
  total time: say exactly that.
- 6.1 Measure the import time of langchain_text_splitters and count the packages only it needs
  (`python -m pip show`, no installs). Replace with a short recursive splitter of our own only
  if that removes at least 8 packages or 0.3 s from the first document upload and chunk-level
  tests still pass. Otherwise "kept, with numbers". Leave matplotlib alone unless the same test
  shows a gain of that size.
- 6.2 For every tracked file that is not code or a test, find what references it. List dead
  functions with search. Remove only what has zero references, in one commit; prove install +
  tests + app start afterwards. In the run folder delete only the pip cache, the clone-check
  folder and temp contents.
- 7.1 Full tests and pyflakes. Re-run the 1.1 benchmark and benchmarks/verify_models.py. Add
  "Second pass, November 2026" to docs/PERFORMANCE.md with a before/after table including
  everything tried and undone. Update README (features, results, limitations still true, link
  to this roadmap). Run scripts/demo_walk.py once only if the UI changed; refresh only the
  screenshots that differ. Fresh clone in the run folder, tests there. Push. CI green.
- 7.2 Private guide, its HTML and the run summary, outside the repo. Nothing is committed.

## H. Measurements before and after

- 2.1 (exported script): not a speed item. Before: the script made an 80/20 split and printed a
  validation accuracy on rows the app never used that way. After: tests execute the generated
  script and check that its train / validation / test row numbers and arrays equal the app's for
  the same seed (Iris, Wine, Breast Cancer, Digits, MNIST and Fashion-MNIST with a fake loader,
  a custom CSV, a custom ZIP, and a tiny CSV that falls back to 80/20 in both). Files:
  core/codegen.py, app.py (meta gains "has_test"), config/translations.py (caption, both
  languages), README.md, tests/test_codegen.py. Tests: 450 -> 451.
- 2.2 (test accuracy everywhere else): not a speed item. The AI review prompt now carries a
  "Test Acc" line (or says there is no test set) and a rule never to call a validation number a
  test number; the tutor's project status quotes the test accuracy; compare_experiments still
  ranks by validation accuracy and now reports each run's test accuracy and says why it does not
  rank by it; the agent's experiment box and the MCP job log show both numbers. Done in one
  item (no 2.2a / 2.2b split was needed). Files: tts_engine.py, app.py, agent_tools.py,
  mcp_server.py, config/translations.py, README.md, tests (narrative, agent tools, MCP, app
  smoke). Tests: 451 -> 453.
- 3.1 (API usage meter): the instrument, not an optimisation. New backend/usage.py; llm_api
  reports every attempt (calls, failed, retries), every cache lookup (hits, misses; per text for
  embeddings), tokens from response.usage_metadata (prompt_token_count, candidates_token_count,
  thoughts_token_count, cached_content_token_count: names confirmed in the google-genai docs)
  and seconds per call. Sidebar expander "API usage (this session)" in both languages, filled
  at the end of each run so it includes that run's calls. Files: backend/usage.py (new),
  backend/llm_api.py, app.py, config/translations.py, tests (llm_api, app smoke, conftest).
  Tests: 453 -> 458. No real Gemini call was used.
- 3.2 (request timeout). Before: no time limit; a hung connection could block a click without
  limit. After: 120 s per generation request and 30 s per embedding request
  (types.HttpOptions(timeout=<milliseconds>), confirmed in the google-genai docs; client-level
  for generation, per-request config for embeddings). A request that times out is retried once
  at most, so the worst case for one hung generation call is about 2 x 120 s plus one back-off
  instead of unbounded. The SDK's own retry loop is set to one attempt
  (HttpRetryOptions(attempts=1)) so it cannot multiply with call_with_retry. Not measured
  against a real hung connection: verified with fakes that raise httpx.ReadTimeout. Files:
  backend/llm_api.py, tests/test_llm_api.py. Tests: 458 -> 462.

## I. Ideas evaluated and rejected (no code written)

- Explicit Gemini context caching: prompts are a few thousand tokens and differ per question;
  storage is billed and a minimum size applies. Not worth it for a single-user tutor.
- Batch API: for background jobs. Every call here is a person waiting for an answer.
- Moving to another Gemini API surface (Interactions): novelty, no measured need.
- In-flight request coalescing: one Streamlit session runs one script at a time; a second
  identical request cannot be in flight.
- Circuit breaker: the per-key 60 s cooldown and the bounded retries already stop hammering.
- Async rewrite and connection pooling: the SDK client is already reused per key; Streamlit runs
  the script synchronously.
- Persistent (on-disk) response cache: answers depend on the session's model and data; a stale
  review of an old model is worse than one extra call.
- Database, code splitting, bundle size: there is no database and no JavaScript build.

## J. Decisions taken for me

- 0.1: the local main branch had no upstream configured, so `git status -sb` could not show
  whether it is level with origin/main. Set the upstream to origin/main (local git config only;
  pushes still use exactly `git push origin main`).
- 0.1: the privacy check is run twice before each push, once against the repo's `.env` and once
  against the old private project's `.env`, because both hold keys.
- 0.1: the full test suite was not run for this item (documentation only, no code touched).
  It was run at item 1.1: 450 passed in 122 s, pyflakes silent.

- 0.1: the 0.1 commit was amended once before it was pushed, to bring CLAUDE.md down to its
  40-line limit. Nothing published was rewritten.
- 1.1: the benchmark was run once (it repeats each timing itself); it took about 5 minutes.
  The agent "typically 2-3" call count is a reading of the loop, not a measurement; item 3.1's
  usage meter will give real numbers.
- 2.1: when a custom dataset is too small for a test set the script follows the app's 80/20
  fallback and prints validation accuracy only (the app passes "has_test" in the export meta).
- 2.1: the generated script keeps plain model.evaluate; it reproduces the app's rows and split
  sizes, not bit-identical weights (the app trains through its own tf.data pipeline).
- From 2.1 on the user asked for all items to be done in one sitting instead of one per
  session; the one item = one commit = one push rule is unchanged.
- 2.2: mcp_client.py needed no change: it never mentions an accuracy; it passes the server's
  result through, which already carried final_test_accuracy.
- 2.2: the "below / within / above the typical range" verdict still compares the VALIDATION
  accuracy with the reference ranges, because those ranges were set for validation accuracy.
  The README limitation now says exactly that.
- 2.2: the only new UI string (agent experiment box with a test accuracy) keeps its three
  labels in English in the Malayalam UI, as they are technical terms.
- 3.1: "calls" counts every attempt sent to Gemini, failed ones included, because each
  attempt uses quota. "seconds" is time inside the calls and excludes back-off sleeps.
- 3.1: the meter is per browser session (st.session_state), with a module-level fallback
  outside Streamlit, the same arrangement as llm_api._state. Ollama requests are not counted.
- 3.1: no reset button was added to the sidebar: one more control for no user need.
- 3.2: the docs list the SDK's retry defaults (5 attempts; 408, 429, 5xx) but do not say
  clearly whether they apply when no retry options are given. Cautious choice: set
  attempts=1 explicitly, which is documented as "total attempts including the initial call".
- 3.2: 120 s for generation because an answer with an extended thinking budget can be slow;
  30 s for embeddings. These are judgement values, not measured ones.
- 3.2: added MAX_TIMEOUT_RETRIES = 1 (not in the brief): without it a hung service would cost
  attempts x 120 s. Other transient errors (5xx) keep all their attempts.

## K. Real Gemini calls used

0 of 12.

## L. Known issues

- The typical-range verdict compares validation accuracy, not test accuracy (by design, see J).
- CNN training about twice as slow as the September figure, unexplained (item 5.1).
- The local-Ollama tutor path is not verified on the development laptop; do not claim it works.

## M. Exact next action

Start item 4.1: write the 3.2 commit hash and its CI result into table G. Read
backend/llm_api.py, backend/agent.py, agent_tools.py, core/rag.py, components/chat_helpers.py
and the ZIP loader in core/data.py; list at most eight hypotheses in section F; failing test
first; at most three fixes, one commit each (4.1a, 4.1b, 4.1c).
