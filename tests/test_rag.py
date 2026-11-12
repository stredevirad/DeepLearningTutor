"""Knowledge-base extraction and retrieval with fake embeddings for both backends."""

import io

import numpy as np
import pytest

from core import rag as R

DOCS = [
    "Dropout randomly disables neurons during training to reduce overfitting.",
    "The Adam optimizer adapts the learning rate for every parameter.",
    "Convolutional layers slide small filters over an image to detect edges and textures.",
    "Early stopping ends training when the validation loss stops improving.",
]
VOCAB = ["dropout", "adam", "convolution", "stopping"]


def fake_embed(texts):
    """Deterministic 'embedding': one axis per topic (NOT unit length, to test cosine)."""
    out = []
    for t in texts:
        t = t.lower()
        v = [5.0 if "dropout" in t or "overfit" in t else 0.1,
             3.0 if "adam" in t or "optimizer" in t else 0.1,
             9.0 if "convolution" in t or "filter" in t else 0.1,
             2.0 if "stopping" in t or "validation loss" in t else 0.1]
        out.append(v)
    return out


def chunks():
    return [{"id": i, "text": t, "source": "notes.txt", "embeddings": {}} for i, t in enumerate(DOCS)]


@pytest.mark.parametrize("tag", [R.GEMINI_EMBED_TAG, R.ollama_tag("nomic-embed-text")])
def test_cosine_retrieval_for_each_backend(tag):
    kb = chunks()
    assert R.add_embeddings(kb, tag, fake_embed) == 4
    results, method = R.retrieve("How does dropout prevent overfitting?", kb, tag, lambda q: fake_embed([q])[0], k=2)
    assert method == "cosine"
    assert results[0][1]["text"].startswith("Dropout")


def test_cosine_not_fooled_by_vector_length():
    """With raw dot products the long 'convolution' vector would win every query."""
    kb = chunks()
    R.add_embeddings(kb, R.GEMINI_EMBED_TAG, fake_embed)
    results, _ = R.retrieve("tell me about the adam optimizer", kb, R.GEMINI_EMBED_TAG, lambda q: fake_embed([q])[0])
    assert "Adam" in results[0][1]["text"]


def test_backends_are_never_mixed():
    kb = chunks()
    R.add_embeddings(kb, R.GEMINI_EMBED_TAG, fake_embed)
    ollama_tag = R.ollama_tag("nomic-embed-text")
    called = []
    # Chunks only have Gemini vectors -> an Ollama query vector must not be compared with them.
    results, method = R.retrieve("dropout", kb, ollama_tag, lambda q: called.append(q) or [1, 0, 0, 0])
    assert method == "keyword" and not called
    assert results[0][1]["text"].startswith("Dropout")
    # Partial coverage (one chunk missing) also falls back to keywords.
    kb2 = chunks()
    R.add_embeddings(kb2[:3], ollama_tag, fake_embed)
    assert R.retrieve("dropout", kb2, ollama_tag, lambda q: [1, 0, 0, 0])[1] == "keyword"


def test_both_backends_can_coexist_on_a_chunk():
    kb = chunks()
    R.add_embeddings(kb, R.GEMINI_EMBED_TAG, fake_embed)
    R.add_embeddings(kb, R.ollama_tag("nomic-embed-text"), lambda ts: [[1.0, 0.0] for _ in ts])
    assert set(kb[0]["embeddings"]) == {R.GEMINI_EMBED_TAG, "ollama:nomic-embed-text"}
    assert R.add_embeddings(kb, R.GEMINI_EMBED_TAG, fake_embed) == 0      # nothing re-embedded


def test_query_embedding_failure_falls_back_to_keywords():
    kb = chunks()
    R.add_embeddings(kb, R.GEMINI_EMBED_TAG, fake_embed)

    def boom(q):
        raise RuntimeError("quota")
    results, method = R.retrieve("early stopping validation", kb, R.GEMINI_EMBED_TAG, boom)
    assert method == "keyword" and results[0][1]["text"].startswith("Early stopping")


def test_bm25_ranking():
    scores = R.bm25_scores("convolutional filters image", DOCS)
    assert int(np.argmax(scores)) == 2
    assert R.bm25_scores("the and of", DOCS).sum() == 0                   # stop-words only
    results, method = R.retrieve("zebra giraffe", chunks())
    assert method == "keyword" and results == []                          # nothing relevant -> nothing


def test_format_context_includes_sources():
    kb = chunks()
    ctx = R.format_context([(1.0, kb[0]), (0.5, kb[1])])
    assert ctx.count("[Source: notes.txt]") == 2


# ─────────────────────────── extraction ───────────────────────────

def _pdf_with_text(text):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4, 2))
    fig.text(0.1, 0.5, text)
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    plt.close(fig)
    return buf.getvalue()


def _pdf_image_only():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(np.random.default_rng(0).random((10, 10)))
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    plt.close(fig)
    return buf.getvalue()


def test_extract_pdf_text():
    text, warnings = R.extract_text("notes.pdf", _pdf_with_text("Backpropagation computes gradients"))
    assert "Backpropagation" in text and warnings == []


def test_scanned_pdf_gives_helpful_error():
    with pytest.raises(R.DocumentError, match="scanned"):
        R.extract_text("scan.pdf", _pdf_image_only())


def test_encrypted_pdf():
    from pypdf import PdfReader, PdfWriter
    writer = PdfWriter()
    for page in PdfReader(io.BytesIO(_pdf_with_text("secret"))).pages:
        writer.add_page(page)
    writer.encrypt("password")
    buf = io.BytesIO()
    writer.write(buf)
    with pytest.raises(R.DocumentError, match="password"):
        R.extract_text("locked.pdf", buf.getvalue())


@pytest.mark.parametrize("name,data,match", [
    ("empty.txt", b"", "empty"),
    ("bad.pdf", b"%PDF-1.4 garbage", "not a readable PDF"),
    ("blank.txt", b"   \n  ", "no text"),
    ("binary.txt", b"\x00\x01\x02" * 50, "does not look like a text"),
])
def test_bad_documents(name, data, match):
    with pytest.raises(R.DocumentError, match=match):
        R.extract_text(name, data)


def test_too_large_document():
    with pytest.raises(R.DocumentError, match="too large"):
        R.extract_text("big.txt", b"a" * 100, max_bytes=10)


def test_txt_latin1_and_chunking():
    text, _ = R.extract_text("notes.txt", "Café résumé ".encode("latin-1") * 300)
    assert "Café" in text
    parts = R.chunk_text(text)
    assert len(parts) > 1 and all(len(p) <= R.CHUNK_SIZE for p in parts)


def test_build_chunks_reports_bad_files_and_keeps_good_ones():
    good = ("guide.txt", ("Neural networks learn weights. " * 100).encode())
    bad = ("scan.pdf", _pdf_image_only())
    kb, messages = R.build_chunks([good, bad])
    assert kb and all(c["source"] == "guide.txt" and c["embeddings"] == {} for c in kb)
    assert any(level == "error" and "scan.pdf" in msg for level, msg in messages)


# ─────────────────────────── de-duplication & retrieval index ───────────────────────────

def test_learning_the_same_file_twice_adds_nothing():
    data = b"Dropout randomly disables neurons during training. " * 40
    first, _ = R.build_chunks([("notes.txt", data)])
    again, messages = R.build_chunks([("renamed.txt", data), ("notes.txt", data)], start_id=len(first),
                                       known_hashes=R.document_hashes(first))
    assert first and all(c["doc_sha"] == R.content_hash(data) for c in first)
    assert again == [] and all(level == "info" for level, _ in messages)


def test_capacity_limits_new_chunks():
    data = ("word " * 400 + "\n\n") * 20
    chunks, messages = R.build_chunks([("big.txt", data.encode())], capacity=3)
    assert len(chunks) == 3 and any("full" in m for _, m in messages)


def test_index_is_reused_until_the_knowledge_base_changes():
    chunks, _ = R.build_chunks([("a.txt", b"gradient descent updates weights. " * 60)])
    cache = {}
    idx = R.get_index(chunks, cache)
    assert R.get_index(chunks, cache) is idx
    tag = R.ollama_tag("nomic-embed-text")
    R.add_embeddings(chunks, tag, lambda texts: [[1.0, 0.0]] * len(texts))
    assert R.get_index(chunks, cache) is not idx           # new backend vectors -> rebuilt
    more, _ = R.build_chunks([("b.txt", b"pooling layers downsample feature maps. " * 60)],
                               start_id=len(chunks), known_hashes=R.document_hashes(chunks))
    idx2 = R.get_index(chunks, cache)
    assert R.get_index(chunks + more, cache) is not idx2    # new document -> rebuilt


def test_index_results_match_uncached_retrieval():
    texts = [b"Dropout reduces overfitting by disabling neurons. " * 30,
             b"Convolution kernels detect edges in images. " * 30,
             b"Learning rate controls the optimizer step size. " * 30]
    chunks, _ = R.build_chunks([(f"{i}.txt", t) for i, t in enumerate(texts)])
    index = R.get_index(chunks, {})
    for q in ("why does dropout help", "what does a convolution kernel detect", "learning rate"):
        a, _ = R.retrieve(q, chunks, k=2)
        b, _ = R.retrieve(q, chunks, k=2, index=index)
        assert [c["id"] for _, c in a] == [c["id"] for _, c in b]


def test_cosine_through_index_never_mixes_backends():
    chunks, _ = R.build_chunks([("a.txt", b"alpha " * 300), ("b.txt", b"beta " * 300)])
    g, o = R.GEMINI_EMBED_TAG, R.ollama_tag("nomic-embed-text")
    R.add_embeddings(chunks, g, lambda texts: [[1.0, 0.0, 0.0] if "alpha" in t else [0.0, 1.0, 0.0] for t in texts])
    R.add_embeddings(chunks, o, lambda texts: [[0.0, 1.0] if "alpha" in t else [1.0, 0.0] for t in texts])
    index = R.get_index(chunks, {})
    res_g, method_g = R.retrieve("q", chunks, g, lambda q: [1.0, 0.0, 0.0], k=1, index=index)
    res_o, method_o = R.retrieve("q", chunks, o, lambda q: [1.0, 0.0], k=1, index=index)
    assert method_g == method_o == "cosine"
    assert "alpha" in res_g[0][1]["text"] and "beta" in res_o[0][1]["text"]


def test_duplicate_passages_are_returned_once():
    chunks = [{"id": i, "text": t, "source": s, "embeddings": {}}
              for i, (t, s) in enumerate([("dropout helps regularise", "a.txt"), ("dropout helps regularise", "b.txt"),
                                          ("dropout rate of 0.2 is common", "c.txt")])]
    results, _ = R.retrieve("dropout", chunks, k=3)
    assert len(results) == 2
