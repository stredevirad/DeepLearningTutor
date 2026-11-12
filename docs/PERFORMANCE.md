# Performance Report

This compares the app before its performance pass with the app after it. These are earlier measurements, taken on 2026-09-30 and 2026-10-01 in the private working repository, before this public release. Both versions were measured with the same scripts on the same laptop:

- **Machine.** Intel Core Ultra 5 125H, 18 logical CPUs, 15.6 GB RAM, Windows 11. CPU only.
- **Software.** Python 3.13.0, TensorFlow 2.20.0, Streamlit 1.64.0.

> **Important: power state.** A first baseline was recorded while the laptop ran on **battery**, with the CPU throttled. Identical code ran about 2.5× slower then. For a fair comparison, the "before" code was measured **again on mains power**, from a separate checkout, straight before the "after" run. The tables below compare only these two mains-power runs. The "before" version and the raw JSON files are not part of this repository, so the "before" column cannot be reproduced from here; the "after" column can.

Reproduce with:

```bash
python benchmarks/run_benchmarks.py --out results.json            # AppTest + component timings (hermetic)
python benchmarks/browser_timings.py http://localhost:8501 --out browser.json --train
python benchmarks/verify_models.py --out models.json              # accuracy + Grad-CAM quality
```

---

## 1. What a user feels (real Chrome, warm server)

Measured with `benchmarks/browser_timings.py`: time from the click until the page stops changing and Streamlit has finished the script run. Both servers ran hermetically, without API keys.

| Interaction | Before | After | Change |
|---|---:|---:|---:|
| New browser session, server already warm | 2.68 s | **0.61 s** | 4.4× faster |
| Toggle a sidebar switch, no model | 855 ms | **183 ms** | 4.7× |
| Toggle a sidebar switch, Iris model trained | 1,157 ms | **184 ms** | 6.3× |
| Switch language | 705 ms | **193 ms** | 3.7× |
| Click **Train** with the default Iris settings | 10.8 s (30 epochs) | **3.9 s (50 epochs)** | more epochs, 2.7× less time |
| Switch tab, no model | 37 ms | 185 ms | slower: lazy tabs run on the server |
| Switch tab, Iris model trained (Visualizations draws 3 charts) | 57 ms | 286 ms | slower (see §6) |

Before the change, every click paid for Streamlit's module scan and for re-drawing every tab. That is why a one-checkbox change took about a second once a model existed.

## 2. Streamlit reruns (AppTest, script time only)

| Scenario | Before | After | Change |
|---|---:|---:|---:|
| No-op rerun, no model | 283 ms | 26 ms | **10.7× faster** |
| Slider change, no model | 285 ms | 27 ms | 10.7× |
| Language switch, no model | 284 ms | 34 ms | 8.3× |
| New session, warm server | 476 ms | 206 ms | 2.3× |
| **Click Train: Iris, 50 epochs** (data + training + evaluation + result) | 14.0 s | **2.9 s** | **4.8×** |
| No-op rerun, Iris model trained | 767 ms | 27 ms | **28×** |
| Slider change, Iris model trained | 790 ms | 29 ms | 27× |
| Language switch, Iris model trained | 807 ms | 27 ms | 30× |
| Dataset change, Iris model trained | 805 ms | 35 ms | 23× |
| Chat message (instant fake LLM, so this is the app's own overhead) | 291 ms | 52 ms | 5.6× |
| Click Train: CNN on a 6,000-image MNIST subset, 1 epoch | 3.57 s | 2.17 s | 1.6× |
| Upload an image in the Test tab (preprocess + predict + draw) | 1,266 ms | 115 ms | **11×** |
| No-op rerun with an uploaded image | 1,268 ms | 52 ms | **24×** |
| Slider change with an uploaded image | 1,285 ms | 53 ms | 24× |
| Click **Explain with Grad-CAM** | ran on every rerun | 106 ms, only when clicked | on demand |
| Switch tab | 0 ms (client-side; every tab had already been computed) | 63–81 ms | **trade-off, see §6** |

## 3. Start-up

| Step | Before | After |
|---|---:|---:|
| `import tensorflow` (fresh process) | 3.28 s | 3.22 s (unchanged, library cost) |
| First script run, cold (all imports + first page) | 4.83 s | 4.28 s |
| Cold start in the browser: time until the page title is visible (fresh server, 2 runs each) | 5.3–6.7 s (blank page until the first run ends) | **0.44–0.51 s**, with a "Starting up: loading TensorFlow" spinner; the first run ends at about 5.0 s |
| `streamlit run` → health check OK | 1.40 s | 1.43 s |

## 4. Building blocks

| Operation | Before | After | Why |
|---|---:|---:|---|
| Iris epoch (8 batches of 16) | 79 ms | **21 ms** | One repeating `tf.data` pipeline; the iterator is built once per `fit`, not once per epoch. |
| Iris 2×16, 100 epochs, including final evaluation | 9.6 s | **2.9 s** | Same pipeline, plus one compiled evaluation pass. |
| Agent tool `run_experiment` (Iris, 30 epochs) | 3.96 s | 1.52 s | Same code path. An identical repeat is now answered from memory. |
| MCP remote round trip, Iris 30 epochs | 7.77 s | 5.66 s | Faster training on the server. |
| Predict one image (the app's path) | 56 ms (`model.predict`) | **1.2 ms** (`predict_on_batch`) | Skips building a data pipeline for one sample. |
| Loss/accuracy chart | 232 ms (matplotlib → 200-dpi PNG) | 35 ms (Altair spec) | The browser draws the chart; nothing is rendered on the server. |
| Confusion matrix, 10 classes | 308 ms | 30 ms | Same. |
| Probability chart, 10 classes | 93 ms | 10 ms | Same. |
| Grad-CAM overlay image | 45 ms | 0.4 ms | NumPy composition, with no matplotlib figure. |
| Grad-CAM heatmap (warm) | 10 ms | 9 ms | Unchanged: it was already fast, so the only change is *when* it runs. |
| Image preprocessing, 690 KB photo | 19 ms | 18 ms | Unchanged; the result is now cached per image. |
| Ollama status with the server **down** | 2,043 ms on **every rerun** | 0.31 s for the first check, then < 0.01 ms for 20 s | 0.3 s TCP probe plus a 20 s per-host cache. |
| RAG keyword search, 500 chunks | 24 ms | 28 ms uncached · **0.30 ms** with the per-KB index | Statistics are built once per knowledge base. |
| RAG cosine search, 500 × 3072 | 39 ms | 41 ms uncached · **0.52 ms** with the index | The normalised matrix is cached per knowledge base. |
| RAG ingest of a 250 KB text | 20 ms | 18 ms | Unchanged. Re-learning the same file is now skipped (content hash). |
| Gemini response-cache hit | 0.002 ms | 0.002 ms | Unchanged: 1 real call for 52 identical requests. |

**Image models.** For CNNs the computation dominates, so the new training pipeline makes no measurable difference. A full MNIST epoch (60,000 images, 2 conv layers with 16 filters, batch 32) took 6.3–7.0 s with plain `model.fit` and 6.8–7.3 s with the pipeline. The improvement is for small tables, where Keras' fixed per-epoch and per-call overhead used to dominate. The earlier estimate of "37 s per epoch" came from the battery-throttled baseline and should not be used.

### Per-run work inside Streamlit (in-process, about 5,000 modules loaded, mains power)

| Work Streamlit did after **every** script run | Cost | Now |
|---|---:|---|
| File watcher: `update_watched_modules()` scans every module's path | 0.78 s (2.2 s on battery) | Off (`server.fileWatcherType = "none"`) |
| `keras.backend.clear_session()` | 0.25 s | Skipped. The app clears Keras itself before building a model. |
| `gc.collect()` (`runner.postScriptGC`) | 0.24 s | Off. Python's normal GC still runs. |

## 5. What changed, in order of impact

1. **Lazy tabs.** Only the open tab's code runs. Before, every click redrew the charts of the Visualizations tab and repeated the Test tab's prediction and Grad-CAM.
2. **Results are kept, not recomputed.** Predictions, Grad-CAM, the AI review and the audio are stored per model version and image hash. A new training run clears them together with the model, so a model can never be shown with another model's results.
3. **Streamlit configuration.** The file watcher, the post-run `gc.collect()` and the per-run Keras reset are off. Together they cost more than the app's own code on every click.
4. **Keras overhead.** A repeating `tf.data` pipeline, `predict_on_batch` for single inputs, and a compiled forward pass for the final evaluation. The loss is identical to Keras' own (it is computed from the logits).
5. **On-demand AI.** After training, the review and gTTS no longer delay the results. Both are network calls and they cost API quota, so they now run only when requested.
6. **Charts.** Altair specs drawn by the browser replace 200-dpi matplotlib PNGs rendered on the server.
7. **Cheap status checks.** The Ollama status is cached. Custom uploads are hashed once per upload, and every parse, summary and decode is cached by that hash.
8. **Live training display.** The charts redraw at most every 0.3 s, instead of after every Iris epoch (a few ms each).

## 6. Trade-offs and what was not measured

- **Tab switches now reach the server** (60–80 ms of script time) because tabs are lazy. Before, they were instant because every tab had already been computed on *every* click. That was the reason every other interaction cost 0.8–1.3 s.
- **Charts are drawn by the browser.** Server time dropped, but the browser needs a few milliseconds per chart; this is included in the real-browser numbers in §1.
- **Not measured:**
  - the real latency of Gemini, Ollama and gTTS, because the benchmarks are hermetic;
  - GPU training.

  The accuracy verification (`benchmarks/verify_models.py`) trained on the full datasets, with these results (details in `README.md`):
  - Iris: 96.7 % in 1.6 s;
  - MNIST: 98.7 % in 19 s for 3 epochs;
  - Grad-CAM: a median 2.8× chance on digit pixels.

## 7. Second pass, November 2026

A second engineering pass on 2026-11-12, in small measured steps. The full log, with every
decision and every number, is [memory/roadmap.md](../memory/roadmap.md). Everything below was
measured in that pass on the same laptop (Intel Core Ultra 5 125H, CPU only, mains power).
A change was kept only when it met a rule fixed in advance; what did not is listed too.

### Kept

| Change | Before | After | How it was measured |
|---|---|---|---|
| Embeddings stored as shared float32 arrays (500 passages) | 62.4 MB | 6.4 MB | `tracemalloc`, fake client returning Python lists as the SDK does; top-3 passages identical |
| Embeddings with 768 numbers instead of 3,072 (500 passages) | 6.4 MB | 1.8 MB | same; on 20 fixed questions over the study guide the best passage was the same for 20 of 20, the top-3 in the same order for 16 of 20 (3 real embedding calls) |
| Search matrix for 500 passages | 6.2 MB | 1.5 MB | same |
| CNN training, MNIST, 3 epochs: several batches per TensorFlow call | 30.4 s | 24.8 s (18.5 % faster) | 5 interleaved repeats in one process, median; test accuracy 98.84 % → 98.71 % |
| Wait before the first words of a tutor answer (streaming) | 5.6 – 13.2 s (the whole answer) | 2.0 – 6.7 s | 3 real calls, gemini-2.5-flash. **The total time is unchanged**; streaming only shows the words as they arrive |
| First document upload: our own text splitter instead of a library | 534 ms | 141 ms | fresh process, median of 5; identical chunks on 1,232 test texts |
| Packages pulled in by a fresh install | 109 | 94 | dependency closure of `requirements.txt` |
| A Gemini request that hangs | no limit | 120 s (30 s for embeddings), retried once at most | verified with fakes that raise the timeout error; not measured against a real hung connection |
| Fact-check when every key is rate-limited | 2 rounds of retries | 1 | test with a fake |

New instrument: the sidebar panel **API usage (this session)** counts requests, failed
attempts, retries, cache hits and misses, tokens (as reported by Gemini) and seconds.

### Tried and not kept

| Idea | Result | Decision |
|---|---|---|
| Fixed tutor instructions as `system_instruction` | 513 / 514 prompt tokens before, 514 / 515 after; 0 cached tokens either way. The prompt is about 500 tokens; implicit caching needs 2,048 | not adopted: one token more, no gain |
| Larger evaluation batch for the CNN | 29.1 s → 29.9 s (3 repeats) | no gain |
| Plain `model.fit` on NumPy arrays instead of the `tf.data` pipeline | 29.1 s → 30.3 s (3 repeats) | no gain |
| `steps_per_execution` on the endless data stream | 16 % faster, but a new test showed that every epoch then trains on a few batches twice | replaced by the variant above, which ends the stream after one pass |
| Removing matplotlib (used for one colormap) | 6 packages and about 125 ms | below the threshold, kept |

### The slow-CNN question

The first pass measured 39.6 s for 3 MNIST epochs where September had measured 19 s. The
split is not the cause: with the old sizes (60,000 training images) the same code took
33.3 s, with the new ones (54,000) 29.1 s, and the final evaluation of all three parts costs
1.8 s. What moves the number is the state of the laptop. Within one session the same code
took between 22.5 s and 31.0 s, and the first training of a fresh process is the fastest:
`benchmarks/verify_models.py`, which trains MNIST once at the start of a process, took
12.9 s at the end of this pass. So a single timing of CNN training on this laptop is good
to about a factor of two; comparisons are only meaningful when the variants are interleaved
in one process, as in the table above.

### Unchanged user-interface timings (before → after this pass, `run_benchmarks.py`)

| What | Start of the pass | End of the pass |
|---|---|---|
| Cold first run | 5,347 ms | 5,264 ms |
| No-op rerun, nothing trained | 23.8 ms | 26.7 ms |
| No-op rerun, Iris model trained | 29.8 ms | 28.7 ms |
| Train click on Iris, 50 epochs | 3,139 ms | 2,984 ms |
| Chat message round trip, instant fake LLM | 31.3 ms | 32.1 ms |
| RAG keyword search, 500 passages, cached index | 0.27 ms | 0.28 ms |
| Train click, CNN on a 6,000-image subset, 1 epoch | 2,159 ms | 2,042 ms |

One run each; the differences are within run-to-run variation. Nothing in this table was
a target of the pass, and nothing got measurably slower.
