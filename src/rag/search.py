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
    'went', 'make', 'making', 'need', 'needs', 'find', 'finding', 'take', 'taking', 'give', 'giving',
    'like', 'likes', 'think', 'thinking', 'look', 'looking', 'come', 'coming', 'use', 'using', 'want',
    'wants', 'tell', 'telling', 'say', 'saying', 'said', 'ask', 'asking', 'write', 'writing', 'read',
    'reading', 'some', 'any', 'all', 'none', 'one', 'two', 'new', 'old', 'good', 'bad', 'hows'
}

def hybrid_search(
    query: str, 
    documents: List[Document], 
    encoder, 
    k: int = 3
) -> Tuple[List[Tuple[Document, float]], bool]:
    """
    Search for relevant documents using hybrid search (60% Vector Cosine Similarity + 40% Keyword Overlap).
    Returns the list of matching (Document, score) tuples, and a boolean indicating if on-the-fly embeddings 
    were generated (which warrants saving the database).
    """
    if not documents:
        return [], False

    # 1. Calculate Keyword overlap score
    query_words = set(re.findall(r'\w+', query.lower()))
    keywords = query_words - STOPWORDS
    if not keywords:
        keywords = query_words

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

    scored = []
    for doc in documents:
        # 2.1 Calculate Keyword overlap score
        doc_words = set(re.findall(r'\w+', doc.text.lower()))
        overlap = len(keywords & doc_words)
        kw_score = (overlap / len(keywords)) if keywords else 0.0
        
        # 2.2 Calculate Vector semantic similarity score
        vec_score = 0.0
        if query_emb is not None and query_norm > 0:
            doc_emb_list = doc.metadata.get('embedding')
            if doc_emb_list is not None and len(doc_emb_list) == len(query_emb):
                doc_emb = np.array(doc_emb_list)
                doc_norm = float(np.linalg.norm(doc_emb))
                if doc_norm > 0:
                    vec_score = float(np.dot(query_emb, doc_emb) / (query_norm * doc_norm))
        
        # 2.3 Compute Robust Hybrid Score
        normalized_vec_score = max(0.0, min(1.0, vec_score))
        if kw_score > 0:
            hybrid_score = 0.65 * kw_score + 0.35 * normalized_vec_score
        else:
            # If no key lexical terms match, penalize vector baseline noise
            hybrid_score = 0.30 * max(0.0, normalized_vec_score - 0.5)

        if hybrid_score >= 0.30:
            scored.append((doc, hybrid_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k], needs_save
