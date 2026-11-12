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

Hypotheses from reading backend/llm_api.py, backend/agent.py, agent_tools.py, core/rag.py,
components/chat_helpers.py and the ZIP loader in core/data.py (item 4.1). "Reproduced" means a
test or a probe script showed the behaviour; only three are fixed, as the item allows.

| # | Severity | Hypothesis | Result |
|---|----------|------------|--------|
| 1 | P2 | chat_helpers.fact_check asks the selected model again after ANY LLMError, also after "all keys rate-limited" / "no key", which repeats a full round of retries and waits | Reproduced by a failing test. FIXED in 4.1a |
| 2 | P2 | app.py answer(): when the reply fails, the question stays in the chat history with no answer; after the next rerun the error is gone, and the unanswered question is sent to the model as history | Reproduced by a failing app test. FIXED in 4.1b |
| 3 | P2 | core/data.py ZIP loader: a password-protected ZIP passes the summary and then fails at Train with a raw RuntimeError instead of a friendly DataError | Reproduced by a failing test. FIXED in 4.1c |
| 4 | P3 | ZIP loader: a ZIP with the same member name stored several times loads that image several times (15 images from 12 files in the probe), so copies can land in both training and test | Reproduced by a probe. NOT FIXED (fix budget used) |
| 5 | P3 | llm_api.embed_texts checks only the number of vectors: an embedding whose values are None raises a raw TypeError instead of an LLMError. The order of the vectors is trusted | Reproduced with a fake (TypeError). Not seen from the real API. NOT FIXED |
| 6 | P3 | llm_api.classify_error falls back to words in the message: a non-API error that merely contains "500" (e.g. "1500 tokens") is classed transient and retried | Reproduced by a probe. NOT FIXED |
| 7 | P3 | rag.build_chunks: a document cut off by the capacity limit is remembered as known, so learning it again says "already in the knowledge base"; the message also always says "2000 chunks" whatever the remaining capacity was | Reproduced by a probe. NOT FIXED |
| 8 | P3 | rag.fingerprint looks at the first and last chunk id only | Two different knowledge bases with equal fingerprints need the same document hashes, count and tag counts; the app cannot produce that (documents are only added whole or all cleared). UNCONFIRMED, not a defect in practice |

## G. Items

| Item | Priority | Size | What | Status | Commit | CI |
|------|----------|------|------|--------|--------|----|
| 0.1 | MUST | S | Set up CLAUDE.md and this roadmap | CI VERIFIED | 8025149 | green |
| 1.1 | MUST | S | Baseline benchmark; table "API calls per user action" | CI VERIFIED | f4f2c03 | green |
| 2.1 | MUST | M | Exported script uses the app's three-way split | CI VERIFIED | 3fecf64 | green |
| 2.2 | MUST | M | Test accuracy in AI review, tutor status, agent tools, MCP | CI VERIFIED | d866512 | green |
| 3.1 | MUST | M | API usage meter (backend/usage.py + sidebar expander) | CI VERIFIED | e6f67b0 | green |
| 3.2 | MUST | S | Request timeout on the Gemini client | CI VERIFIED | d2bc89e | green |
| 4.1 | MUST | M | Bug hunt, time-boxed; at most three fixes (4.1a-c) | CI VERIFIED | 4.1a 0caebba; 4.1b d77ecda; 4.1c 5cd2d77 | green (all three) |
| 3.3 | SHOULD | M | Embedding memory: float32 arrays, maybe smaller dimension | CI VERIFIED | 0d9fd7f | green |
| 5.1 | SHOULD | M | CNN speed: explain the gap, try cheap levers | CI VERIFIED | 9e7fdb8 | green |
| 3.4 | OPTIONAL | M | Prompt structure: system_instruction, stable-first order | REVERTED | 1659a77 | green |
| 3.5 | OPTIONAL | M | Streaming for the plain English tutor answer | CI VERIFIED | 191899a | green |
| 6.1 | OPTIONAL | S | Dependency check (langchain-text-splitters) | DONE | see git log | pending |
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
- 4.1a (fact-check fallback). Before: with every key rate-limited the fact-checker made two
  generate_text calls (fast model, then selected model), each with its own retries and waits.
  After: one. Shown by tests/test_chat_helpers.py with a fake; no real call. Files:
  components/chat_helpers.py, tests/test_chat_helpers.py. Tests: 462 -> 463.
- 4.1b (unanswered question). Before: a failed reply left the question in the chat history;
  after the next rerun the error box was gone and the question was sent to the model as
  history. After: the question is marked "unanswered", shows a caption saying so (both
  languages) and is left out of the history sent to the model. Shown by a failing-then-passing
  app test with a fake Gemini; no real call. Files: app.py, config/translations.py,
  tests/test_app_smoke.py. Tests: 463 -> 464.
- 4.1c (password-protected ZIP). Before: the summary showed the classes, then Train failed
  with a raw RuntimeError from zipfile. After: both the summary and the loader raise a
  friendly DataError ("The ZIP is password-protected"). Shown by a failing-then-passing test
  that sets the encryption flag on a test ZIP. Files: core/data.py, tests/test_data.py.
  Tests: 464 -> 465.
- 4.1 total: 8 hypotheses, 7 reproduced, 3 fixed (one commit each), 4 reproduced and left
  open as P3 (section F, numbers 4-7), 1 unconfirmed.
- 3.3 (embedding memory), measured with tracemalloc and a fake client that returns Python
  lists as the SDK does, 500 chunks:

  | Storage | Vectors held by chunks + cache | Per chunk | Search matrix | Total |
  |---------|-------------------------------|-----------|---------------|-------|
  | before: Python float lists, 3,072 numbers | 62.4 MB | 124.8 KB | 6.2 MB | 68.6 MB |
  | float32 arrays shared by cache and chunk, 3,072 | 6.4 MB | 12.8 KB | 6.2 MB | 12.6 MB |
  | float32 arrays, 768 numbers (kept) | 1.8 MB | 3.6 KB | 1.5 MB | 3.3 MB |

  float32 step: 9.8x less (rule: at least 4x) and the top-3 passages are identical (a test
  compares 20 random questions against the old list storage; the search matrix was float32
  before as well). KEPT. With 2,000 chunks the vectors now take 25.4 MB at 3,072 numbers
  (measured); the old 200 MB figure for 2,000 chunks was an estimate and was not re-measured.
  768 step, 3 real embedding calls: 15 passages of the bundled study guide (chunk size 200 for
  this test, to get more than 3 passages) and 20 fixed questions. The API's 768-number vector
  equals the first 768 numbers of the 3,072 one (cosine 0.9999998 at worst). Same first result
  for 20 of 20 questions (rule: at least 18); same top-3 in the same order for 16 of 20, same
  top-3 as a set for 17 of 20. KEPT. Caveat: one short document, 15 passages.
  Files: backend/llm_api.py, core/rag.py, tests (llm_api, rag). Tests: 465 -> 467.
- 5.1 (CNN speed), MNIST, CNN 2 conv x 16 filters, Adam lr 0.001, batch 32, 3 epochs, seed 42,
  mains power, CPU only, one process, variants interleaved, median of the repeats:

  | Variant | Total | of which fit | final evaluation | per epoch | Val acc | Test acc | Verdict |
  |---------|-------|--------------|------------------|-----------|---------|----------|---------|
  | old split sizes (60,000 train / 10,000 val), 3 repeats | 33.3 s | 31.5 s | 1.8 s | 10.4 s | 98.73 % | - | reference |
  | new split (54,000 / 6,000 / 10,000 test), 3 repeats | 29.1 s (22.5 - 31.0) | 27.4 s | 1.8 s | 9.1 s | 98.62 % | 98.84 % | baseline |
  | larger evaluation batch (4096 / 2048), 3 repeats | 29.9 s | 28.5 s | 1.4 s | 9.1 s | 98.62 % | 98.84 % | no gain: NOT ADOPTED |
  | plain model.fit on NumPy arrays, 3 repeats | 30.3 s | 28.5 s | 1.6 s | 9.4 s | 97.90 % | 98.20 % | no gain: NOT ADOPTED |
  | steps_per_execution=32, endless stream, 5 repeats | 25.5 s vs 30.5 s | 23.7 s | 1.8 s | 7.6 s | 98.50 % | 98.67 % | 16 % faster but WRONG, see below: REPLACED |
  | steps_per_execution=32, finite stream (kept), 5 repeats | 24.8 s vs 30.4 s | 22.9 s | 1.9 s | 7.3 s | 98.42 % | 98.71 % | 18.5 % faster, test -0.13 points: KEPT |

  The gap (39.6 s against 19 s). The split is not the cause: the new split is faster than
  the old sizes (29.1 s against 33.3 s), and the final evaluation of all three parts costs
  1.8 s. The same code gave 22.5 s to 31.0 s within one session, so the machine's state
  (heat, background load) moves the number by a third. September's 6.3 - 7.0 s per epoch on
  60,000 images was not reproduced (10.4 s today); the 39.6 s was not reproduced either
  (29 - 30 s today). The cause is the machine, not a code change this pass could find.
  A defect found on the way: with the app's endless data stream, steps_per_execution=32
  rounds every epoch UP to a multiple of 32 batches (a new test showed 32 optimizer steps
  where 28 were due), so a few batches were trained on twice. Long epochs now use a stream
  that ends after one pass; the test checks that every batch runs exactly once.
  dataset cache: not tried, because caching the batched data would freeze the shuffle order;
  prefetch was already in the pipeline. Only epochs of 320 batches or more use the lever
  (image models); tables keep one batch per call. Files: core/training.py,
  tests/test_training.py. Tests: 467 -> 469.
- 3.4 (prompt structure), 4 real calls on gemini-2.5-flash, Intermediate level, no model
  trained, two short questions per structure:

  | Structure | Prompt tokens | Cached tokens | Seconds |
  |-----------|---------------|---------------|---------|
  | old: everything in one user message | 513, 514 | 0, 0 | 2.3, 2.5 |
  | new: fixed instructions as system_instruction | 514, 515 | 0, 0 | 1.8, 1.9 |

  The fixed instructions are 1,996 characters, about 500 tokens. The Gemini docs give 2,048
  tokens as the minimum prompt size for implicit caching on Gemini 2.5 Flash and 2.5 Pro, so
  these prompts are four times too small to be cached, and no cached tokens were reported.
  The new structure cost one prompt token MORE per call. Rule: keep only if prompt tokens do
  not rise. TRIED, NO GAIN: the app code was not changed (the experiment lived in a script in
  the run folder). The seconds differ by less than normal network variation and prove nothing.
  The user message was already ordered stable-first (instructions, knowledge base, history,
  status, question).
- 3.5 (streaming the plain English tutor answer), 3 real calls on gemini-2.5-flash, the
  tutor's Intermediate prompt, no model trained:

  | Question | First text after | Whole answer after | Answer | Output + thinking tokens |
  |----------|------------------|--------------------|--------|--------------------------|
  | overfitting and three ways to reduce it | 6.7 s | 13.2 s | 6,080 characters | 1,454 + 1,025 |
  | how a convolutional layer works | 3.1 s | 9.6 s | 5,310 characters | 1,473 + 385 |
  | validation against test sets | 2.0 s | 5.6 s | 3,440 characters | 708 + 206 |

  What streaming changes: the reader sees the first words after 2.0 - 6.7 s instead of
  waiting 5.6 - 13.2 s behind a spinner. What it does not change: the total time. The whole
  answer takes as long as before (no blocking call was made for comparison; streaming has no
  mechanism that would shorten it). The wait before the first text is mostly the model
  thinking (the slowest start had the most thinking tokens). Retries and key rotation apply
  only until the first piece arrives; an error after that is shown, not retried. Not streamed:
  agent mode, the Malayalam answer (one JSON object), Ollama, the AI review, the fact-check.
  Files: backend/llm_api.py (generate_stream), backend/usage.py, app.py,
  benchmarks/run_benchmarks.py (fake), tests (llm_api, app smoke). Tests: 469 -> 473.
- 6.1 (dependency check), measured in the project environment, fresh process, median of 5:

  | What | Before | After |
  |------|--------|-------|
  | Lines in requirements.txt | 17 | 16 |
  | Packages a fresh install pulls in (dependency closure) | 109 | 94 |
  | Packages needed only by langchain-text-splitters | 15 | 0 |
  | import langchain_text_splitters | 447 ms | not imported |
  | First chunk_text call (import + split 250 KB) | 534 ms | 141 ms |
  | Splitting 250 KB, warm | 32.5 ms (library) | 22.5 ms (ours) |

  Both rules were met (at least 8 packages, at least 0.3 s on the first document upload), so
  the one class used from the library was replaced by about 50 lines in core/rag.py (the same
  recursive scheme: paragraphs, lines, words, characters; chunk 1,000, overlap 100). Before
  the dependency was dropped, our splitter was compared with the library on 1,232 texts (the
  study guide, edge cases and 300 random texts, four size/overlap settings): identical chunk
  lists for all 1,232 (2.3 million chunks). One recorded library output stays in the tests.
  The 15 packages: httpcore2, httpx2, jsonpatch, jsonpointer, langchain-core,
  langchain-protocol, langchain-text-splitters, langsmith, orjson, pyyaml,
  requests-toolbelt, truststore, uuid-utils, xxhash, zstandard.
  matplotlib, same test: 6 packages only it needs and about 125 ms to import its colormaps.
  Below both thresholds: KEPT, untouched.
  Files: core/rag.py, requirements.txt, tests/test_rag.py. Tests: 473 -> 478.

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
- 5.1: the first measured steps_per_execution variant met the 15 % rule but trained on some
  batches twice per epoch, so it was replaced by a correct variant and timed again (5 repeats)
  before being kept.
- 5.1: the lever is used only when an epoch has at least 320 batches (10 calls of 32), so
  small tables behave exactly as before and their live charts update after every batch.
- 5.1: MNIST results for a given seed change slightly (test 98.84 % -> 98.71 % in this run)
  because the batches are grouped differently. The README numbers are re-measured in 7.1.
- 6.1: the dependency was removed from requirements.txt only. The packages were not
  uninstalled from the local project environment (nothing imports them any more; CI installs
  from requirements.txt, so CI is the proof that the app works without them).

## K. Real Gemini calls used

10 of 12.

- 3.3: 3 embedding calls (study-guide passages at 3,072; 20 questions at 3,072; 20 questions at
  768). All succeeded at the first attempt.
- 3.4: 4 generate calls (two questions in the old prompt structure, two in the new). All
  succeeded at the first attempt.
- 3.5: 3 streamed generate calls. All succeeded at the first attempt.

## L. Known issues

- The typical-range verdict compares validation accuracy, not test accuracy (by design, see J).
- Four small defects found in the bug hunt are reproduced but not fixed (section F, 4-7).
- Embeddings are 768 numbers instead of 3,072: checked on one short document only (item 3.3).
- CNN timings vary by about a third from run to run on the development laptop; the September
  figure (19 s for 3 MNIST epochs) could not be reproduced (item 5.1).
- While an answer is streaming it is shown as plain markdown; the final rendering (code
  blocks with a copy button) replaces it when the answer is complete (item 3.5).
- The exported script trains with plain model.fit, without steps_per_execution.
- The local-Ollama tutor path is not verified on the development laptop; do not claim it works.

## M. Exact next action

Start item 6.2: write the 6.1 commit hash and its CI result into table G. For every tracked
file that is not code or a test, find what references it; list dead functions with search;
remove only what has zero references, in one commit; prove tests + app start afterwards. In
the run folder delete only the pip cache, the clone-check folder and temp contents.
