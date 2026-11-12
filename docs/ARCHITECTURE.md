# Architecture

Deep Learning Tutor is one Streamlit app (`app.py`) on top of three small packages.
The rule that keeps it tidy: **`app.py` only draws the user interface; everything that
can be tested without a browser lives in `core/`, `backend/` or `components/`.**

## Modules

```mermaid
flowchart TD
    user([Browser])

    subgraph UI["User interface"]
        app["app.py<br/>sidebar + four lazy tabs"]
        comp["components/<br/>training_ui · visualizations · chat_helpers"]
        cfg["config/<br/>translations (en + ml) · prompts · session_defaults"]
    end

    subgraph CORE["core/ — machine learning, no UI code"]
        data["data.py<br/>datasets, split, preprocessing,<br/>CSV analysis, safe ZIP loader"]
        models["models.py<br/>Simple NN + CNN builders"]
        training["training.py<br/>validate → seed → fit → evaluate"]
        imgprep["image_preprocessing.py<br/>any image → 28×28"]
        gradcam["gradcam.py<br/>Grad-CAM heatmap"]
        codegen["codegen.py<br/>equivalent Python script"]
        rag["rag.py<br/>chunk, embed, retrieve"]
    end

    subgraph AI["AI tutor"]
        llm["backend/llm_api.py<br/>Gemini: key rotation, retry, cache"]
        agent["backend/agent.py<br/>function-calling loop"]
        tools["agent_tools.py<br/>five validated tools"]
        ollama["backend/ollama_api.py<br/>local Ollama server"]
        tts["tts_engine.py<br/>calibrated review + speech"]
    end

    subgraph REMOTE["Optional remote training"]
        mcpc["mcp_client.py"]
        mcps["mcp_server.py<br/>REST (+ optional MCP SSE)"]
    end

    gemini[(Google Gemini API)]
    ollamasrv[(Ollama server)]

    user --> app
    app --> comp
    app --> cfg
    app --> data
    app --> models
    app --> training
    app --> imgprep
    app --> gradcam
    app --> codegen
    app --> rag
    app --> llm
    app --> agent
    app --> ollama
    app --> tts
    app --> mcpc
    training --> data
    training --> models
    data --> imgprep
    agent --> llm
    agent --> tools
    tools --> training
    tts --> llm
    comp --> llm
    mcpc -. HTTP .-> mcps
    mcps --> training
    llm -. HTTPS .-> gemini
    ollama -. HTTP .-> ollamasrv
```

Three callers train models: the **Train button** (`app.py`), the **AI agent**
(`agent_tools.py`) and the **remote server** (`mcp_server.py`). All three go through
`core/training.py`, so the split, the preprocessing, the seeding and the evaluation are the
same code everywhere.

| Folder / file | What it holds |
|---|---|
| `app.py` | The whole user interface: the sidebar and four tabs. Only the open tab runs. |
| `core/` | Data loading and splitting, model builders, training and evaluation, image preprocessing, Grad-CAM, the code exporter, the document search (RAG). No Streamlit code. |
| `backend/` | Talking to language models: Gemini (`llm_api.py`), the agent loop (`agent.py`), Ollama (`ollama_api.py`). |
| `components/` | UI pieces that are worth testing alone: the live training callback, the Altair charts, chat parsing and rendering. |
| `config/` | Every UI string in English and Malayalam, the tutor prompts, and the session-state defaults. |
| `agent_tools.py` | The five tools the Gemini agent may call. Every argument is validated. |
| `tts_engine.py` | Builds the "AI Professor's review" prompt (the verdict is computed in Python, not by the LLM) and turns text into speech with gTTS. |
| `mcp_server.py`, `mcp_client.py` | Optional: train built-in datasets on another machine. |
| `benchmarks/` | Scripts that measure speed and accuracy. |
| `tests/` | The pytest suite. It is hermetic: no network, no `.env`. |

## The path of one training run

What happens between a click on **Train model** and the result card
(function names are in the files shown):

```mermaid
sequenceDiagram
    participant U as User
    participant A as app.py
    participant D as core/data.py
    participant M as core/models.py
    participant T as core/training.py
    participant L as components/training_ui.py

    U->>A: click "Train model"
    A->>T: validate_config(ui_config)
    Note over T: fills defaults, checks every range,<br/>rejects unknown keys
    A->>D: prepare_builtin_tabular("Iris", seed)
    Note over D: 1. split (stratified)<br/>2. fit scaler on TRAIN rows only<br/>3. transform the other rows with it
    D-->>A: PreparedData
    A->>T: set_global_seed(seed)
    A->>M: build_model(cfg, input_shape, classes)
    A->>T: train_model(model, data, cfg, callbacks=[live])
    loop every epoch
        T->>L: on_epoch_end(loss, accuracy)
        L-->>U: live curves (redrawn at most every 0.3 s)
    end
    Note over T: evaluate the weights the model<br/>ends up with (early stopping may<br/>have restored an earlier epoch)
    T-->>A: history, final metrics, confusion matrix
    A->>A: store_training_results(): replaces the model<br/>AND everything derived from it
    A-->>U: result card
```

Details worth knowing:

1. **Split before fit.** `prepare_tabular` splits the rows first and fits the
   `StandardScaler` / one-hot encoder on the training rows only. Fitting on all rows would
   leak information about the evaluation rows into training ("data leakage").
2. **One data pipeline per fit.** `batch_pipeline` builds one repeating `tf.data` stream, so
   Keras does not rebuild an iterator every epoch. On Iris that fixed cost was larger than
   the maths.
3. **Metrics describe the final weights.** With early stopping, Keras restores the best
   epoch's weights. `train_model` therefore re-evaluates the model after `fit` instead of
   reporting the last line of the history.
4. **No stale state.** `store_training_results` calls `reset_trained_model()` first, so a
   new model can never be shown with the previous model's scaler, class names, predictions
   or Grad-CAM images.

## The path of one tutor question

```mermaid
sequenceDiagram
    participant U as User
    participant A as app.py (answer)
    participant R as core/rag.py
    participant G as backend/llm_api.py
    participant AG as backend/agent.py
    participant TO as agent_tools.py
    participant C as components/chat_helpers.py

    U->>A: types a question
    A->>A: build context: level instructions,<br/>chat history, current model's results
    opt documents were uploaded
        A->>R: retrieve(question, knowledge_base)
        Note over R: cosine similarity on embeddings,<br/>or BM25 keyword search as fallback
        R-->>A: top 3 passages
    end
    alt Agent mode (Gemini only)
        A->>AG: run_agent(prompt)
        loop until the model answers in text (bounded)
            AG->>G: generate(contents, tools)
            G-->>AG: function call, e.g. run_experiment(...)
            AG->>TO: execute_tool(name, args)
            Note over TO: validates every argument,<br/>trains through core/training.py
            TO-->>AG: result (slimmed before it goes back)
        end
        AG-->>A: final text + experiments
    else Normal chat
        A->>G: generate(contents)
        Note over G: rotates API keys on rate limits,<br/>retries with backoff
        G-->>A: reply text
    end
    A->>C: parse_bilingual(reply) → render_response()
    opt Document check is on
        A->>C: run_hallucination_check(answer, passages)
        Note over C: a second LLM call judges how much<br/>of the answer the passages support
    end
    C-->>U: answer (code in code blocks)
```

With the **Ollama** backend the same function builds a shorter prompt and calls the local
server through `components/chat_helpers.call_ollama`; agent mode is not available there
because it relies on Gemini's function calling.

## Design decisions in one line each

- **Lazy tabs**: a click reruns only the open tab, so a checkbox change does not redraw
  charts or repeat a prediction. See [PERFORMANCE.md](PERFORMANCE.md).
- **Results are stored, not recomputed**: predictions and Grad-CAM maps are kept per model
  version and image hash.
- **AI calls happen on request**: the review, the speech and Grad-CAM run when their button
  is clicked, so training results appear immediately and no API quota is spent silently.
- **Uploads are identified by content hash** (SHA-256), so an unchanged file is never
  parsed, decoded or embedded twice.
- **Every outside input is validated**: uploaded files (size limits, ZIP path checks),
  LLM tool arguments and remote requests all pass through the same `validate_config`.
