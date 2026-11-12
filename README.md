# 🧠 Deep Learning Tutor

**Design, train, understand, and test neural networks, without writing code.**

Deep Learning Tutor is a Streamlit workbench for learning deep learning by doing:

1. Pick a dataset and configure a network.
2. Train it while live loss and accuracy curves update.
3. Inspect the results.
4. Test the model on your own inputs.

An AI tutor explains the results in English or Malayalam. It runs on Google Gemini (online) or Ollama (offline).

It began as an internship project. It is an educational project, not a production service.

```
Configure  →  Train (live curves)  →  Result  →  Understand (charts, AI review, tutor)  →  Test & export
```

---

## Features

### Train
- **Two network types:**
  - A Simple NN (dense layers) for tables.
  - A small CNN (1–3 conv + max-pool blocks) for 28×28 images.
- **Built-in datasets:**
  - Simple NN: Iris, Wine, Breast Cancer, Digits.
  - CNN: MNIST, Fashion-MNIST.
- **Your own data.** A summary appears in the sidebar *before* you train.
  - **CSV:** you choose the target column.
    - Numeric columns are standardised.
    - Low-cardinality text columns are one-hot encoded.
    - ID and free-text columns are ignored and listed.
    - Rows with missing values are removed and counted.
    - Regression-like targets are rejected with an explanation.
  - **ZIP of class folders** (`cats/…`, `dogs/…`):
    - Checked for path traversal, file count and size, and zip bombs.
    - Empty classes and class imbalance are reported.
    - Every image goes through the same 28×28 preprocessing that the Test tab uses.
- **Hyperparameters.** Layers, nodes or filters, epochs, learning rate, batch size, optimizer (Adam, SGD or RMSprop), dropout, early stopping and a random seed.
  - Defaults depend on the network type.
  - Simple NN: lr 0.01, 50 epochs, batch 16.
  - CNN: lr 0.001, 5 epochs, batch 32.
- **Correct ML practice:**
  - The data is split first (stratified), and the scaler and encoder are fitted on the training split only.
  - The final metrics are evaluated on the weights the model actually ends up with. That matters when early stopping restores the best epoch.
- **Live training display.** One progress area with live loss and accuracy curves, followed by a result card: validation accuracy, validation loss, epochs trained, parameter count, and how the result compares with typical accuracy for the dataset.

### Understand
- **Interactive charts.** Loss and accuracy curves mark the "best epoch" when early stopping restored it. The confusion matrix names the most frequent mistake. The model architecture and parameter counts are shown too.
- **AI Professor's review (on request).** A short critique of the run, calibrated against typical results, so an under-trained 83 % Iris model is called under-performing rather than "brilliant". **Listen** reads it aloud with gTTS in English or Malayalam.
- **Grad-CAM (on request).** For an image prediction, it shows which pixels drove the decision.
  - Keras-3-compatible, computed from the pre-softmax score.
  - On correctly classified MNIST digits, the heat sits on the digit at 2.8× chance.

### Ask
- **AI Tutor chat.** Three explanation levels:
  - **Beginner:** plain language and analogies, no maths.
  - **Intermediate:** technical, with code.
  - **Advanced:** maths and implementation detail.

  Code in answers is rendered as proper code blocks, and you can attach images, PDFs and text files. In the Malayalam interface, Gemini answers bilingually.
- **Agent mode (Gemini function calling).** The tutor can use five tools, and shows each call live:
  - `run_experiment`: trains real models;
  - `diagnose_training`: diagnoses training curves;
  - `compare_experiments`: compares runs;
  - `generate_code`: generates code;
  - `explain_concept`: explains concepts.

  Every argument is validated.
- **Knowledge base (RAG).** Upload PDF or TXT study material. It is chunked (with LangChain's text splitter) and embedded with Gemini `gemini-embedding-001` or Ollama `nomic-embed-text`.
  - Questions retrieve the closest passages. Vectors from different backends are never mixed, and BM25 keyword search is the fallback.
  - The same file is never learned twice.
- **Document check (optional).** A second LLM call (an LLM acting as judge) estimates how much of an answer the retrieved passages support. Treat it as a helpful signal, not a proof.

### Test & export
- **Predictions:**
  - Upload an image, or try the bundled samples: a handwritten 7, a sneaker and an ankle boot.
  - For Simple NNs, fill in feature values or pick a random validation example.
  - Probabilities are labelled as the network's softmax output, not as "confidence".
- **Downloads:** the trained model as `.keras`, and an **equivalent Python script** that reproduces the run: same split, preprocessing, seed, architecture and hyperparameters, including custom CSV and ZIP data.

### Remote training (advanced)
An **MCP training server** (`mcp_server.py`) can train built-in datasets on another machine, for example one with a GPU. The app shows its progress live and downloads the trained model. See [MCP remote training](#mcp-remote-training-optional).

---

## Verified results

Measured on a laptop CPU (Intel Core Ultra 5 125H) with seed 42, using `python benchmarks/verify_models.py`.

| Configuration | Validation accuracy | Training time |
|---|---:|---:|
| Iris, 2 × 16 dense, Adam lr 0.01, batch 16, 50 epochs (the defaults) | **96.7 %** | 1.6 s |
| Iris, same settings, 100 epochs | 93.3 % | 2.5 s |
| MNIST, CNN 2 conv × 16 filters, Adam lr 0.001, batch 32, 3 epochs | **98.7 %** | 19 s |
| Fashion-MNIST, same CNN, 3 epochs | 89.4 % | 24 s |

The bundled `handwritten_seven.png` is predicted as 7. Grad-CAM heat on 300 correctly classified MNIST digits is a median 2.8× chance on the digit's pixels; 2 of the 300 maps were empty.

---

## Setup

Requires **Python 3.11–3.13**. Verified on Windows 11 with Python 3.13.0, TensorFlow 2.20.0, Keras 3.15.1 and Streamlit 1.64.0. A clean install from the pinned `requirements-dev.txt` was verified, and `pip check` passed.

### Windows (PowerShell)

```powershell
git clone https://github.com/stredevirad/DeepLearningTutor.git
cd DeepLearningTutor
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt      # or requirements-dev.txt to also run the tests
```

In `cmd.exe`, activate with `venv\Scripts\activate.bat` instead.

### macOS / Linux

```bash
git clone https://github.com/stredevirad/DeepLearningTutor.git
cd DeepLearningTutor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### API keys (`.env`)

1. Copy the example file: `copy .env.example .env` on Windows, or `cp .env.example .env` elsewhere.
2. Put one or more Gemini keys in `.env`, separated by commas. The app rotates to another key when one hits its rate limit. You can get keys at <https://aistudio.google.com/apikey>.

   ```
   GEMINI_API_KEYS=your-first-key,your-second-key
   ```

3. `.env` is git-ignored, so **never commit it**. If a key is ever pushed anywhere public, revoke it in AI Studio.

Without a key, training, charts, Grad-CAM and export still work. Only the AI features need Gemini or Ollama.

### Ollama (optional, offline AI)

1. Install Ollama from <https://ollama.com>. On Windows the installer adds `ollama` to your PATH.
2. Pull a chat model. You can also pull the embedding model used for semantic knowledge-base search:

   ```
   ollama pull llama3.2:1b
   ollama pull nomic-embed-text
   ```

3. In the sidebar, under **AI tutor**, choose **Ollama (offline)**. The **AI model settings** section shows the server status.
   - If the server is not running, it offers **Start Ollama server**.
   - It can download a missing model for you.
4. If Ollama runs on a different address, set `OLLAMA_HOST` in `.env`.

Small local models answer in English only, and Agent mode needs Gemini (it relies on function calling).

---

## Run the app

```bash
streamlit run app.py
```

Streamlit opens <http://localhost:8501>. Work through the tabs: **1. Configure & Train → 2. Visualizations → 3. AI Tutor → 4. Test & Export**.

`.streamlit/config.toml` turns off Streamlit's file watcher, for speed. While you are editing the code, run it with auto-reload instead:

```bash
streamlit run app.py --server.fileWatcherType auto --server.runOnSave true
```

### Custom datasets
In the sidebar, open **Use your own data**. The expander shows how every column or class folder will be used before you train.

The CSV needs a header row, and the target column must hold categories (such as a species or yes/no), not a continuous number. The ZIP needs one folder of images per class. Limits: 25 MB for a CSV, 100 MB for a ZIP, and 10,000 images.

### MCP remote training (optional)

On the machine that should do the training:

```bash
python mcp_server.py --port 8765                 # listens on 127.0.0.1 only (safe default)
```

To reach it from another computer:

1. Set the same `MCP_AUTH_TOKEN` in both machines' `.env` files.
2. Run `python mcp_server.py --host 0.0.0.0`, or use a tunnel such as `ngrok http 8765`.

In the app, open **Advanced: remote training (MCP)**, turn it on, paste the URL and click **Connect**. Built-in datasets then train on the server as background jobs; custom uploads always train locally.

The server exposes these endpoints:

- `GET /health`
- `GET /tools`
- `POST /call`, with the tools `train_model_remote` and `get_training_status`
- `GET /logs/{job_id}`

If the optional `mcp` package is installed, it also serves an MCP SSE endpoint at `/sse`. Arguments are validated against fixed ranges, the job queue is bounded, and no user input is ever executed.

---

## Tests & benchmarks

```bash
pip install -r requirements-dev.txt
pytest                      # 432 tests, about 4 minutes on a laptop CPU
pytest -m "not slow"        # quick subset (no model training)
```

- **Hermetic.** `tests/conftest.py` blocks every network connection, including the local Ollama and MCP ports. `.env` is never loaded, and every Gemini and Ollama call is faked.
- **Coverage includes:**
  - data loading, including malicious ZIPs;
  - the CSV column analysis;
  - leakage checks and reproducibility;
  - Grad-CAM and image preprocessing;
  - chat parsing and rendering;
  - LLM retry and key rotation, and caching;
  - Ollama helpers and the status cache;
  - RAG for both backends, and deduplication;
  - the hallucination checker;
  - the agent tools and the agent loop;
  - the MCP routes and client;
  - exported code;
  - translations;
  - end-to-end AppTest runs of every tab.

Performance and accuracy can be reproduced:

```bash
python benchmarks/run_benchmarks.py --out results.json        # start-up, rerun and component timings
python benchmarks/verify_models.py --out models.json          # accuracy + Grad-CAM checks (downloads MNIST)
python benchmarks/browser_timings.py http://localhost:8501 --out browser.json   # needs: pip install playwright
```

Results and method: [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

---

## Architecture

A diagram of the modules, and the path of one training run and one tutor question, are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```
app.py                     Streamlit UI: sidebar + four lazy tabs (only the open tab runs)
├── core/                  UI-independent logic (unit-tested)
│   ├── data.py            datasets, leak-free split → fit-on-train preprocessing, CSV analysis, safe ZIP loader
│   ├── models.py          Simple NN / CNN builders, optimizers, .keras (de)serialisation
│   ├── training.py        config validation + limits, seeding, fast tf.data training loop, evaluation
│   ├── image_preprocessing.py   one preprocessing for training ZIPs and Test-tab uploads
│   ├── gradcam.py         Keras-3-safe Grad-CAM
│   ├── codegen.py         "equivalent Python code" generator
│   └── rag.py             PDF/TXT extraction, chunking, dedup, cosine (per backend) + BM25 retrieval index
├── backend/
│   ├── llm_api.py         Gemini (google-genai): key rotation, backoff, caching, embeddings
│   ├── agent.py           function-calling agent loop
│   └── ollama_api.py      Ollama detection/start, cached status, models, embeddings
├── components/
│   ├── chat_helpers.py    bilingual JSON parsing, markdown/code rendering, Ollama chat, document check
│   ├── training_ui.py     live training callback (throttled redraws)
│   └── visualizations.py  Altair charts, Grad-CAM overlay
├── config/                translations (English + Malayalam), prompts, session-state defaults
├── agent_tools.py         tools the Gemini agent can call (validated arguments)
├── tts_engine.py          calibrated training review + gTTS narration
├── mcp_server.py / mcp_client.py   remote training server (REST + optional MCP SSE) and client
├── benchmarks/            performance, browser-timing and accuracy scripts
└── tests/                 pytest suite
```

`app.py`, `agent_tools.py` and `mcp_server.py` all train through `core/`, so the preprocessing, seeding and evaluation are identical everywhere. A new training run replaces the model together with everything derived from it: the scaler, class names, predictions, Grad-CAM results and the review. The model is never paired with a stale preprocessor.

The sample files `handwritten_seven.png`, `fashion_sneaker.png`, `fashion_ankle_boot.png`, `iris_dataset.csv`, `wine_dataset.csv` and `deep_learning_study_guide.txt` are for trying out the Test tab, the custom uploads and the knowledge base.

---

## Limitations

- **The same validation set is used twice.** It is used for early stopping and for the reported metrics, so the numbers are slightly optimistic. MNIST and Fashion-MNIST use their official test split as the validation set. A separate test split would be the next step.
- **Deliberately small models.** The models are for teaching, with CPU training in seconds, not state-of-the-art accuracy.
- **The document check is an estimate.** An LLM judging another LLM can be wrong.
- **gTTS needs internet access.** Ollama models answer in English only.
- **One user per server.** Training runs inside that user's session. The app has no user accounts; run it locally or behind your own access control.

---

## Troubleshooting

- **"No Gemini API key found".** Create `.env` from `.env.example` and restart Streamlit.
- **"All Gemini API keys have hit their rate limit".** Wait about a minute, or add more keys to `GEMINI_API_KEYS`.
- **"The selected Gemini model is not available for this API key".** Pick another model under **AI model settings**.
- **Ollama "server is not running".** Click **Start Ollama server**, or run `ollama serve` in a terminal and then click **Check Ollama again**.
- **No audio for the review.** gTTS needs an internet connection.
- **`pip install` fails with "No such file or directory … tensorflow\include\…" on Windows.** The folder path exceeds Windows' 260-character limit. Keep the project in a short path (for example `C:\dlt`), or enable long paths: <https://pip.pypa.io/warnings/enable-long-paths>.
- **Code changes don't show up.** The file watcher is off (see *Run the app*). Restart Streamlit, or use `--server.fileWatcherType auto`.

---

## License

[MIT](LICENSE)
