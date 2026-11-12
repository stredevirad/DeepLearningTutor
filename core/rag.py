"""
Knowledge Base (RAG) utilities
==============================
Text extraction (PDF/TXT), chunking, and retrieval for the AI tutor.

Every chunk remembers which embedding backend produced its vector
(`chunk["embeddings"] = {"gemini:gemini-embedding-001": array, "ollama:nomic-embed-text": array}`,
each a float32 NumPy array), and a similarity search only ever compares vectors from the SAME backend.
When no matching embeddings exist (e.g. the Ollama embedding model is not
pulled, or Gemini has no key) retrieval falls back to BM25 keyword ranking,
so retrieval is always real - never "the first N chunks".
"""

import hashlib
import io
import math
import re
from collections import Counter

import numpy as np

MAX_DOC_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 300
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MAX_CHUNKS = 2000

GEMINI_EMBED_TAG = "gemini:gemini-embedding-001"
OLLAMA_EMBED_TAG_PREFIX = "ollama:"

_STOPWORDS = set("""a an and are as at be by for from has have how i in is it its of on or that the this to
was what when where which who why will with you your do does can me my we our""".split())


class DocumentError(ValueError):
    """A document could not be read (message is user-friendly)."""


def ollama_tag(model_name):
    return f"{OLLAMA_EMBED_TAG_PREFIX}{model_name}"


# ─────────────────────────── extraction & chunking ───────────────────────────

def extract_text(name, data, max_pages=MAX_PDF_PAGES, max_bytes=MAX_DOC_BYTES):
    """Return (text, warnings) for a PDF or TXT file given as bytes."""
    if not data:
        raise DocumentError(f"'{name}' is empty.")
    if len(data) > max_bytes:
        raise DocumentError(f"'{name}' is too large ({len(data) / 1e6:.1f} MB); the limit is {max_bytes / 1e6:.0f} MB.")
    warnings = []
    if name.lower().endswith(".pdf") or data[:5] == b"%PDF-":
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                try:
                    if not reader.decrypt(""):
                        raise DocumentError(f"'{name}' is password-protected.")
                except DocumentError:
                    raise
                except Exception as e:
                    raise DocumentError(f"'{name}' is password-protected.") from e
            pages = reader.pages
            n_pages = len(pages)
        except DocumentError:
            raise
        except (PdfReadError, ValueError, KeyError, TypeError, OSError) as e:
            raise DocumentError(f"'{name}' is not a readable PDF file.") from e
        if n_pages > max_pages:
            warnings.append(f"'{name}' has {n_pages} pages; only the first {max_pages} were read.")
        parts, failed = [], 0
        for page in list(pages)[:max_pages]:
            try:
                parts.append(page.extract_text() or "")
            except Exception:  # one bad page should not lose the whole document
                failed += 1
        if failed:
            warnings.append(f"{failed} page(s) of '{name}' could not be read.")
        text = "\n".join(parts)
        if not text.strip():
            raise DocumentError(f"No text could be extracted from '{name}'. It may be a scanned PDF "
                                "(images of pages); run it through OCR first or upload a text version.")
        return text, warnings

    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if "\x00" in text[:1000]:
        raise DocumentError(f"'{name}' does not look like a text file.")
    if not text.strip():
        raise DocumentError(f"'{name}' contains no text.")
    return text, warnings


# Where a text may be cut, best place first: between paragraphs, between lines, between
# words and, if a single "word" is still too long, anywhere.
SEPARATORS = ("\n\n", "\n", " ", "")


def _pieces(text, separator):
    """Cut `text` at every `separator`; each piece after the first keeps the separator in front."""
    if separator == "":
        return list(text)
    parts = text.split(separator)
    return [p for p in [parts[0]] + [separator + part for part in parts[1:]] if p]


def _merge(pieces, chunk_size, overlap):
    """Glue small pieces together into chunks of at most `chunk_size` characters. A new chunk
    starts with the tail of the previous one (up to `overlap` characters), so a sentence
    cut at a chunk border can still be found whole in one of the two."""
    chunks, current, total = [], [], 0
    for piece in pieces:
        if total + len(piece) > chunk_size and current:
            chunk = "".join(current).strip()
            if chunk:
                chunks.append(chunk)
            # drop pieces from the front until only the overlap is left and the new piece fits
            while total > overlap or (total + len(piece) > chunk_size and total > 0):
                total -= len(current[0])
                current = current[1:]
        current.append(piece)
        total += len(piece)
    chunk = "".join(current).strip()
    if chunk:
        chunks.append(chunk)
    return chunks


def _split(text, separators, chunk_size, overlap):
    """Recursive character splitting: cut at the first separator that occurs in the text,
    merge the pieces that fit, and cut the pieces that are still too long with the next separator."""
    separator, finer = separators[-1], ()
    for i, candidate in enumerate(separators):
        if candidate == "" or candidate in text:
            separator, finer = candidate, separators[i + 1:]
            break
    chunks, small = [], []
    for piece in _pieces(text, separator):
        if len(piece) < chunk_size:
            small.append(piece)
            continue
        if small:
            chunks.extend(_merge(small, chunk_size, overlap))
            small = []
        chunks.extend(_split(piece, finer, chunk_size, overlap) if finer else [piece])
    if small:
        chunks.extend(_merge(small, chunk_size, overlap))
    return chunks


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split a document into overlapping chunks of at most `chunk_size` characters,
    cutting between paragraphs where possible, then lines, then words."""
    return [c for c in _split(text, SEPARATORS, chunk_size, overlap) if c.strip()]


def content_hash(data):
    return hashlib.sha256(data).hexdigest()


def build_chunks(files, start_id=0, known_hashes=(), capacity=MAX_CHUNKS):
    """files: iterable of (name, bytes). Returns (chunks, messages) where messages are (level, text).

    Files whose content hash is in `known_hashes` (already in the knowledge base) are
    skipped, so learning the same document twice never duplicates its chunks.
    `capacity` is how many more chunks the knowledge base can take.
    """
    chunks, messages = [], []
    next_id = start_id
    seen = set(known_hashes)
    for name, data in files:
        digest = content_hash(data or b"")
        if digest in seen:
            messages.append(("info", f"'{name}' is already in the knowledge base."))
            continue
        seen.add(digest)
        try:
            text, warns = extract_text(name, data)
        except DocumentError as e:
            messages.append(("error", str(e)))
            continue
        messages.extend(("warning", w) for w in warns)
        for piece in chunk_text(text):
            if len(chunks) >= capacity:
                messages.append(("warning", f"Stopped after {MAX_CHUNKS} chunks; the knowledge base is full."))
                return chunks, messages
            chunks.append({"id": next_id, "text": piece, "source": name, "doc_sha": digest, "embeddings": {}})
            next_id += 1
    return chunks, messages


def document_hashes(chunks):
    return {c["doc_sha"] for c in chunks if c.get("doc_sha")}


def add_embeddings(chunks, backend_tag, embed_fn):
    """Embed chunks that do not yet have a vector for `backend_tag`. Returns how many were embedded.
    `embed_fn(list_of_texts) -> list_of_vectors`. Errors propagate to the caller."""
    todo = [c for c in chunks if backend_tag not in c.setdefault("embeddings", {})]
    if not todo:
        return 0
    vectors = embed_fn([c["text"] for c in todo])
    for chunk, vec in zip(todo, vectors):
        if vec is not None and len(vec):
            # float32: 4 bytes per number. No copy when the backend already returned a float32 array.
            chunk["embeddings"][backend_tag] = np.asarray(vec, dtype=np.float32)
    return sum(1 for c in todo if backend_tag in c["embeddings"])


def embedding_coverage(chunks, backend_tag):
    """Fraction of chunks that have an embedding from this backend."""
    if not chunks:
        return 0.0
    return sum(1 for c in chunks if backend_tag in c.get("embeddings", {})) / len(chunks)


# ─────────────────────────── ranking ───────────────────────────

def tokenize(text):
    return [t for t in re.findall(r"\w+", text.lower()) if t not in _STOPWORDS and len(t) > 1]


class RetrievalIndex:
    """Retrieval data prepared once per knowledge base: BM25 term statistics and, per
    embedding backend, a row-normalised embedding matrix. A question then costs one
    pass over the precomputed statistics or one matrix-vector product, instead of
    re-tokenising every chunk and rebuilding the matrix from Python lists."""

    def __init__(self, chunks):
        self.fingerprint = fingerprint(chunks)
        self._texts = [c["text"] for c in chunks]
        self.term_counts = None          # BM25 statistics, built on the first keyword search
        self._matrices = {}

    def _build_bm25(self):
        self.term_counts = [Counter(tokenize(t)) for t in self._texts]
        self.lengths = [sum(tc.values()) or 1 for tc in self.term_counts]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 1.0
        self.doc_freq = Counter(term for tc in self.term_counts for term in tc)

    def bm25(self, query, k1=1.5, b=0.75):
        if self.term_counts is None:
            self._build_bm25()
        n = len(self.term_counts)
        scores = np.zeros(n)
        q_terms = tokenize(query)
        if not n or not q_terms:
            return scores
        for term in q_terms:
            df = self.doc_freq.get(term)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, tc in enumerate(self.term_counts):
                f = tc.get(term)
                if f:
                    scores[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * self.lengths[i] / self.avg_len))
        return scores

    def matrix(self, chunks, backend_tag):
        """Row-normalised float32 matrix of this backend's vectors (None if sizes differ)."""
        if backend_tag not in self._matrices:
            vectors = [c["embeddings"][backend_tag] for c in chunks]
            if len({len(v) for v in vectors}) != 1:
                self._matrices[backend_tag] = None
            else:
                m = np.asarray(vectors, dtype=np.float32)
                self._matrices[backend_tag] = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
        return self._matrices[backend_tag]


def fingerprint(chunks):
    """Changes whenever chunks are added or removed, or gain embeddings from another backend."""
    tags = Counter(tag for c in chunks for tag in c.get("embeddings", {}))
    return (len(chunks), tuple(sorted(document_hashes(chunks))), tuple(c["id"] for c in chunks[:1] + chunks[-1:]),
            tuple(sorted(tags.items())))


def get_index(chunks, cache):
    """The RetrievalIndex for `chunks`, reused from `cache` (a dict, e.g. the user's session
    state) while the knowledge base is unchanged."""
    fp = fingerprint(chunks)
    index = cache.get("index")
    if index is None or index.fingerprint != fp:
        index = RetrievalIndex(chunks)
        cache["index"] = index
    return index


def bm25_scores(query, texts, k1=1.5, b=0.75):
    """Okapi BM25 relevance of each text to the query (0 = no shared terms)."""
    return RetrievalIndex([{"id": i, "text": t} for i, t in enumerate(texts)]).bm25(query, k1, b)


def _top_k(scores, chunks, k, keep=lambda s: True):
    """Best-scoring chunks, skipping exact duplicate texts (e.g. the same passage in two files)."""
    out, seen = [], set()
    for i in np.argsort(-scores, kind="stable"):
        if not keep(scores[i]):
            break
        text = chunks[i]["text"].strip()
        if text in seen:
            continue
        seen.add(text)
        out.append((float(scores[i]), chunks[i]))
        if len(out) == k:
            break
    return out


def retrieve(query, chunks, backend_tag=None, query_embed_fn=None, k=3, index=None):
    """Return (results, method) where results is [(score, chunk), ...] best first.

    method is "cosine" when every chunk has an embedding from `backend_tag` and
    the query could be embedded with the same backend, otherwise "keyword" (BM25).
    Vectors from different backends are never compared with each other.
    Pass `index` (from get_index) to reuse precomputed statistics between questions.
    """
    if not chunks or not query.strip():
        return [], "none"
    if index is None or index.fingerprint != fingerprint(chunks):
        index = RetrievalIndex(chunks)
    if backend_tag and query_embed_fn and embedding_coverage(chunks, backend_tag) == 1.0:
        try:
            q_vec = query_embed_fn(query)
        except Exception:  # fall back to keywords rather than failing the chat
            q_vec = None
        matrix = index.matrix(chunks, backend_tag) if q_vec is not None and len(q_vec) else None
        if matrix is not None and len(q_vec) == matrix.shape[1]:
            q = np.asarray(q_vec, dtype=np.float32)
            scores = (matrix @ (q / (np.linalg.norm(q) + 1e-12))).astype(np.float64)
            return _top_k(scores, chunks, k), "cosine"

    scores = index.bm25(query)
    return _top_k(scores, chunks, k, keep=lambda s: s > 0), "keyword"


def format_context(results, max_chars=6000):
    """Retrieved chunks as a prompt section (with source names)."""
    out, used = [], 0
    for _, chunk in results:
        piece = f"[Source: {chunk.get('source', 'document')}]\n{chunk['text']}"
        if used + len(piece) > max_chars:
            break
        out.append(piece)
        used += len(piece)
    return "\n\n".join(out)
