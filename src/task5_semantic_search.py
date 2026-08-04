"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

from __future__ import annotations

from .task4_chunking_indexing import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    embed_texts,
    get_collection,
)


QUERY_EXPANSIONS = {
    "tăng ca": "làm thêm giờ thời giờ làm việc",
    "ot": "làm thêm giờ tiền lương làm thêm",
    "nghỉ việc": "chấm dứt hợp đồng lao động trợ cấp thôi việc",
    "sa thải": "xử lý kỷ luật lao động hình thức sa thải",
    "phép năm": "nghỉ hằng năm ngày nghỉ có hưởng lương",
    "thử việc": "thời gian thử việc tiền lương thử việc",
    "lương tối thiểu": "mức lương tối thiểu vùng theo tháng theo giờ",
    "bhxh": "bảo hiểm xã hội bắt buộc",
    "bảo hiểm xã hội": "BHXH mức đóng chế độ bảo hiểm xã hội",
}


def expand_query(query: str) -> list[str]:
    """Create deterministic Vietnamese legal query variants.

    This is a low-latency multi-query expansion method: colloquial terms are
    mapped to statutory terminology found in the corpus. It requires no extra
    LLM call and therefore remains available in offline/demo environments.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    normalized = query.strip()
    lowered = normalized.lower()
    variants = [normalized]
    for phrase, legal_terms in QUERY_EXPANSIONS.items():
        if phrase in lowered:
            variants.append(f"{normalized} {legal_terms}")
    return list(dict.fromkeys(variants))


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    collection = get_collection()
    collection_size = collection.count()
    if collection_size == 0:
        return []

    # Fail fast when Task 4's persisted index was built with another model.
    collection_metadata = collection.metadata or {}
    indexed_model = collection_metadata.get("embedding_model")
    indexed_dimension = collection_metadata.get("embedding_dimension")
    if indexed_model and indexed_model != EMBEDDING_MODEL:
        raise RuntimeError(
            f"Chroma collection uses {indexed_model!r}, but Task 5 uses "
            f"{EMBEDDING_MODEL!r}. Rebuild Task 4's index."
        )
    if indexed_dimension and int(indexed_dimension) != EMBEDDING_DIM:
        raise RuntimeError(
            f"Chroma collection dimension is {indexed_dimension}, but Task 5 "
            f"expects {EMBEDDING_DIM}. Rebuild Task 4's index."
        )

    # Reuse Task 4's provider/model so document and query vectors always share
    # the same embedding space. For this project, this calls OpenAI
    # text-embedding-3-small and returns a 1536-dimensional vector.
    query_vector = embed_texts([query.strip()])[0]
    if len(query_vector) != EMBEDDING_DIM:
        raise ValueError(
            f"Query embedding has {len(query_vector)} dimensions; "
            f"expected {EMBEDDING_DIM}"
        )

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection_size),
        include=["documents", "metadatas", "distances"],
    )

    documents = (results.get("documents") or [[]])[0] or []
    metadatas = (results.get("metadatas") or [[]])[0] or []
    distances = (results.get("distances") or [[]])[0] or []

    output: list[dict] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        # The collection is configured with hnsw:space="cosine". Chroma
        # returns cosine distance, so cosine similarity is 1 - distance.
        similarity = 1.0 - float(distance)
        similarity = max(-1.0, min(1.0, similarity))
        output.append(
            {
                "content": document or "",
                "score": round(similarity, 6),
                "metadata": metadata or {},
            }
        )

    output.sort(key=lambda item: item["score"], reverse=True)
    return output[:top_k]


def semantic_search_with_query_expansion(query: str, top_k: int = 10) -> list[dict]:
    """Search all expanded variants and fuse their rankings with RRF."""
    from .task7_reranking import rerank_rrf

    variants = expand_query(query)
    ranked_lists = [semantic_search(variant, top_k=top_k * 2) for variant in variants]
    results = rerank_rrf(ranked_lists, top_k=top_k)
    for result in results:
        metadata = dict(result.get("metadata") or {})
        metadata["query_expansion"] = len(variants) > 1
        metadata["query_variants"] = len(variants)
        result["metadata"] = metadata
    return results


if __name__ == "__main__":
    results = semantic_search(
        "Thời gian thử việc tối đa và mức lương thử việc là bao nhiêu?",
        top_k=5,
    )
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
