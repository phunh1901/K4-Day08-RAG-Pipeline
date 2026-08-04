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


PROJECT_ROOT = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"

# List of {'content': str, 'metadata': dict}
CORPUS: list[dict] = []
_BM25_INDEX: BM25Okapi | None = None
_TOKENIZED_CORPUS: list[list[str]] = []


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return tokens


def _load_markdown_corpus() -> list[dict]:
    corpus: list[dict] = []

    if not STANDARDIZED_DIR.exists():
        return corpus

    for filepath in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if not filepath.is_file():
            continue

        content = filepath.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = filepath.relative_to(STANDARDIZED_DIR)
        metadata = {
            "source_path": str(relative_path).replace("\\", "/"),
            "category": relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown",
            "filename": filepath.name,
        }
        corpus.append({"content": content, "metadata": metadata})

    return corpus


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


if __name__ == "__main__":
    # Test
    results = lexical_search("phương thức thanh toán shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
