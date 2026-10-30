import os
import re
import logging
from typing import List, Tuple, Optional
import numpy as np
from .document import Document

logger = logging.getLogger(__name__)

# Search stopwords
STOPWORDS = {
    'what', 'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at', 'of', 'for', 'to', 'with', 'about', 
    'how', 'why', 'where', 'when', 'who', 'does', 'do', 'did', 'can', 'could', 'would', 'should', 
    'my', 'name', 'me', 'you', 'your', 'please', 'tell', 'as', 'per', 'from', 'by', 'under', 'above', 
    'here', 'there', 'their', 'them', 'they', 'it', 'its', 'this', 'that', 'these', 'those', 'also', 
    'and', 'or', 'but', 'so', 'if', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'doing', 'done', 
    'i', 'he', 'she', 'we', 'us', 'him', 'her', 'his', 'our', 'ours', 'theirs', 'myself', 'himself', 
    'herself', 'itself', 'yourself', 'yourselves', 'themselves', 
    'day', 'today', 'tomorrow', 'yesterday', 'now', 'will', 'was', 'were', 'get', 'got', 'go', 'going', 
    'went', 'make', 'making', 'need', 'needs', 'needed', 'know', 'knows', 'knowing', 'knew', 'known', 
    'find', 'finding', 'take', 'taking', 'give', 'giving', 'like', 'likes', 'think', 'thinking', 
    'look', 'looking', 'come', 'coming', 'use', 'using', 'want', 'wants', 'wanted', 'tell', 'telling', 
    'say', 'saying', 'said', 'ask', 'asking', 'write', 'writing', 'read', 'reading', 'some', 'any', 
    'all', 'none', 'one', 'two', 'new', 'old', 'good', 'bad', 'hows', 'calculate', 'solve', 'answer', 
    'explain', 'info', 'information', 'details', 'detail', 'mean', 'meaning', 'much', 'many', 'just'
}


class VectorIndex:
    """
    High-performance vector retrieval index supporting:
      1. Contiguous BLAS / Accelerate matrix multiplication (np.dot(matrix, query_vector)).
      2. Optional FAISS IndexFlatIP backend for massive collections (10,000+ chunks).
      3. O(1) Pre-cached document word token sets to eliminate redundant regex overhead.
    """

    def __init__(self, dimension: int = 768, use_faiss: bool = False):
        self.dimension = dimension
        self.documents: List[Document] = []
        self.matrix: Optional[np.ndarray] = None
        self._doc_count = -1
        self.use_faiss = use_faiss
        self.faiss_index = None

        if use_faiss:
            try:
                import faiss
                self._faiss = faiss
            except ImportError:
                self._faiss = None
                self.use_faiss = False
        else:
            self._faiss = None

    def build(self, documents: List[Document], force_rebuild: bool = False) -> None:
        """
        Builds or synchronizes the vector matrix and token caches from the document list.
        Skips rebuilding if document count is unchanged and the matrix is populated.
        """
        if not force_rebuild and len(documents) == self._doc_count and self.matrix is not None:
            return

        valid_docs = []
        vectors = []
        for doc in documents:
            # Pre-cache word tokens on document for O(1) keyword overlap checks
            if not hasattr(doc, "_words") or doc._words is None:
                doc._words = set(re.findall(r'\w+', doc.text.lower()))

            emb = doc.metadata.get("embedding")
            if emb is not None and len(emb) == self.dimension:
                valid_docs.append(doc)
                vectors.append(emb)

        self.documents = valid_docs
        self._doc_count = len(documents)

        if not vectors:
            self.matrix = None
            self.faiss_index = None
            return

        # Stack into contiguous float32 matrix
        mat = np.ascontiguousarray(vectors, dtype=np.float32)
        # L2-normalize vectors for exact cosine similarity via inner product
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self.matrix = mat / np.maximum(norms, 1e-12)

        if self.use_faiss and self._faiss is not None:
            try:
                self.faiss_index = self._faiss.IndexFlatIP(self.dimension)
                self.faiss_index.add(self.matrix)
            except Exception as e:
                logger.warning(f"Failed to initialize FAISS index: {e}. Using BLAS matrix backend.")
                self.faiss_index = None
        else:
            self.faiss_index = None

    def compute_similarities(self, query_emb: np.ndarray) -> np.ndarray:
        """
        Computes cosine similarities for all indexed documents in a single vectorized BLAS operation.
        Returns a 1D float32 array of shape (N,) matching self.documents.
        """
        if self.matrix is None or len(self.documents) == 0:
            return np.empty(0, dtype=np.float32)

        q = query_emb.astype(np.float32).reshape(1, -1)
        q_norm = float(np.linalg.norm(q))
        if q_norm <= 0:
            return np.zeros(len(self.documents), dtype=np.float32)
        q_unit = q / q_norm

        # Single vectorized BLAS matrix multiplication: O(N * D) with SIMD / Accelerate
        return np.dot(self.matrix, q_unit.T).squeeze(-1)

    def search_top_k(self, query_emb: np.ndarray, k: int = 10) -> Tuple[np.ndarray, List[Document]]:
        """
        Returns top-k most similar documents and their cosine similarity scores.
        """
        if self.matrix is None or len(self.documents) == 0:
            return np.empty(0, dtype=np.float32), []

        if self.faiss_index is not None:
            q = query_emb.astype(np.float32).reshape(1, -1)
            q_norm = float(np.linalg.norm(q))
            if q_norm <= 0:
                return np.zeros(0, dtype=np.float32), []
            q_unit = q / q_norm
            k_clamped = min(k, len(self.documents))
            scores, indices = self.faiss_index.search(q_unit, k_clamped)
            matched_docs = [self.documents[idx] for idx in indices[0] if 0 <= idx < len(self.documents)]
            return scores[0], matched_docs
        else:
            sims = self.compute_similarities(query_emb)
            top_indices = np.argsort(sims)[::-1][:k]
            return sims[top_indices], [self.documents[idx] for idx in top_indices]


# Backward-compatible alias
FAISSVectorIndex = VectorIndex

# Module-level cached vector index instance
_GLOBAL_VECTOR_INDEX = VectorIndex(dimension=768)


def hybrid_search(
    query: str, 
    documents: List[Document], 
    encoder, 
    k: int = 3,
    vector_index: Optional[VectorIndex] = None,
) -> Tuple[List[Tuple[Document, float]], bool]:
    """
    Search for relevant documents using accelerated hybrid search
    (40% Keyword Overlap + 60% Vector Cosine Similarity via Vectorized Matrix Multiplication / FAISS).
    Returns matching (Document, score) tuples, and a boolean indicating if on-the-fly embeddings 
    were generated (which warrants saving the database).
    """
    if not documents:
        return [], False

    # 1. Calculate Keyword overlap score (ignoring stopwords and isolated numbers)
    raw_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-]+\b', query.lower()))
    # Casual greetings / conversational fillers that should not trigger knowledge retrieval alone
    casual_tokens = {
        'hi', 'hello', 'hey', 'greetings', 'morning', 'afternoon', 'evening', 'night',
        'thanks', 'thank', 'bye', 'goodbye', 'nice', 'meet', 'pleased', 'feeling', 'feel',
        'doing', 'fine', 'great', 'well', 'okay', 'ok', 'sithija', 'friend', 'talk', 'chat'
    }
    keywords = {
        w for w in (raw_tokens - STOPWORDS - casual_tokens)
        if not (w.isdigit() and len(w) <= 3) and len(w) > 1
    }
    if not keywords:
        return [], False

    # 2. Get vector embedding for the query
    try:
        query_emb = np.array(encoder.encode(query, show_progress_bar=False))
        if query_emb.ndim > 1:
            query_emb = query_emb.squeeze(0)
        query_norm = float(np.linalg.norm(query_emb))
    except Exception as e:
        logger.error(f"Failed to encode query: {e}")
        query_emb = None
        query_norm = 0.0

    needs_save = False
    if query_emb is not None and query_norm > 0:
        # Check if documents need batch embedding (missing or dimension mismatch)
        missing_docs = [
            doc for doc in documents 
            if doc.metadata.get('embedding') is None or len(doc.metadata.get('embedding', [])) != len(query_emb)
        ]
        if missing_docs:
            try:
                batch_texts = [d.text for d in missing_docs]
                embs = encoder.encode(batch_texts, batch_size=32)
                for d, emb in zip(missing_docs, embs):
                    d.metadata['embedding'] = emb.tolist()
                needs_save = True
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")

    # 3. Synchronize Vector Index
    idx = vector_index if vector_index is not None else _GLOBAL_VECTOR_INDEX
    if query_emb is not None and len(query_emb) != idx.dimension:
        idx.dimension = len(query_emb)
    idx.build(documents, force_rebuild=needs_save)

    # 4. Compute Vector Cosine Similarities in a Single Vectorized Operation
    if query_emb is not None and query_norm > 0 and len(idx.documents) > 0:
        vec_scores = idx.compute_similarities(query_emb)
    else:
        vec_scores = np.zeros(len(idx.documents), dtype=np.float32)

    # 5. Fast Hybrid Scoring (O(1) cached keyword overlap + precomputed vector scores)
    scored = []
    num_keywords = len(keywords)

    for i, doc in enumerate(idx.documents):
        # 5.1 O(1) Pre-cached Keyword Overlap
        doc_words = getattr(doc, '_words', None)
        if doc_words is None:
            doc_words = set(re.findall(r'\w+', doc.text.lower()))
            doc._words = doc_words

        if num_keywords > 0:
            overlap = len(keywords & doc_words)
            kw_score = overlap / num_keywords
        else:
            kw_score = 0.0

        # 5.2 Vector Similarity
        raw_vec = float(vec_scores[i]) if i < len(vec_scores) else 0.0
        normalized_vec_score = max(0.0, min(1.0, raw_vec))

        # 5.3 Robust Hybrid Score (require lexical grounding or high dense similarity)
        if kw_score > 0:
            hybrid_score = 0.50 * kw_score + 0.50 * normalized_vec_score
        else:
            # Causal decoder embeddings have high baseline anisotropy; require strong match if 0 keyword overlap
            hybrid_score = 0.70 * normalized_vec_score if normalized_vec_score >= 0.88 else 0.0

        if hybrid_score >= 0.40:
            scored.append((doc, hybrid_score))

    # Also handle any documents without embeddings (keyword only)
    if len(idx.documents) < len(documents):
        indexed_set = set(id(d) for d in idx.documents)
        for doc in documents:
            if id(doc) not in indexed_set:
                doc_words = getattr(doc, '_words', None)
                if doc_words is None:
                    doc_words = set(re.findall(r'\w+', doc.text.lower()))
                    doc._words = doc_words
                overlap = len(keywords & doc_words) if num_keywords > 0 else 0
                kw_score = (overlap / num_keywords) if num_keywords > 0 else 0.0
                h_score = 0.40 * kw_score
                if h_score >= 0.30:
                    scored.append((doc, h_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k], needs_save
