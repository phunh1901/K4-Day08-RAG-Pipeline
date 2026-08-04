"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import re
from pathlib import Path

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .task4_chunking_indexing import chunk_documents, load_documents


PROJECT_ROOT = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"

# List of {'content': str, 'metadata': dict}
CORPUS: list[dict] = []
_BM25_INDEX: BM25Okapi | None = None
_TOKENIZED_CORPUS: list[list[str]] = []
_TFIDF_VECTORIZER: TfidfVectorizer | None = None
_TFIDF_MATRIX = None


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return tokens


def _load_markdown_corpus() -> list[dict]:
    """Reuse Task 4's 800/100 chunks so dense and sparse retrieval align.

    Indexing entire statutes as one BM25 document produces huge prompts and
    coarse citations. Sharing chunk boundaries keeps every result focused,
    comparable and within the LLM/RAGAS context limit.
    """
    if not STANDARDIZED_DIR.exists():
        return []
    return [
        {"content": chunk["content"], "metadata": dict(chunk.get("metadata") or {})}
        for chunk in chunk_documents(load_documents())
    ]


def _ensure_corpus_loaded() -> None:
    global CORPUS
    if not CORPUS:
        CORPUS = _load_markdown_corpus()


def _ensure_bm25_index() -> BM25Okapi:
    global _BM25_INDEX, _TOKENIZED_CORPUS

    _ensure_corpus_loaded()

    if _BM25_INDEX is None:
        _BM25_INDEX = build_bm25_index(CORPUS)
    return _BM25_INDEX


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    global _TOKENIZED_CORPUS

    _TOKENIZED_CORPUS = [_tokenize(doc.get("content", "")) for doc in corpus]
    return BM25Okapi(_TOKENIZED_CORPUS)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    _ensure_corpus_loaded()

    if not CORPUS:
        return []

    bm25 = _ensure_bm25_index()
    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            continue
        results.append(
            {
                "content": CORPUS[idx]["content"],
                "score": score,
                "metadata": CORPUS[idx].get("metadata", {}),
            }
        )

    return results


def _ensure_tfidf_index() -> tuple[TfidfVectorizer, object]:
    """Build a word/bi-gram TF-IDF matrix once for the standardized corpus."""
    global _TFIDF_VECTORIZER, _TFIDF_MATRIX
    _ensure_corpus_loaded()
    if _TFIDF_VECTORIZER is None or _TFIDF_MATRIX is None:
        _TFIDF_VECTORIZER = TfidfVectorizer(
            tokenizer=_tokenize,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        _TFIDF_MATRIX = _TFIDF_VECTORIZER.fit_transform(
            [doc["content"] for doc in CORPUS]
        )
    return _TFIDF_VECTORIZER, _TFIDF_MATRIX


def tfidf_search(query: str, top_k: int = 10) -> list[dict]:
    """Alternative lexical retriever used for the +5 bonus comparison.

    TF-IDF represents every document as a sparse weighted vector and ranks by
    cosine similarity. Unlike BM25, it has no explicit document-length
    saturation parameters (``k1``/``b``); bigrams help preserve Vietnamese
    legal phrases such as "thử việc" and "lương tối thiểu".
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    _ensure_corpus_loaded()
    if not CORPUS:
        return []

    vectorizer, matrix = _ensure_tfidf_index()
    query_vector = vectorizer.transform([query.strip()])
    scores = cosine_similarity(query_vector, matrix).ravel()
    top_indices = sorted(
        range(len(scores)), key=lambda idx: float(scores[idx]), reverse=True
    )[:top_k]
    return [
        {
            "content": CORPUS[idx]["content"],
            "score": float(scores[idx]),
            "metadata": {**CORPUS[idx].get("metadata", {}), "lexical_method": "tfidf"},
        }
        for idx in top_indices
        if float(scores[idx]) > 0
    ]


if __name__ == "__main__":
    # Test
    results = lexical_search("phương thức thanh toán shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
