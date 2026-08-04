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


if __name__ == "__main__":
    results = semantic_search(
        "Thời gian thử việc tối đa và mức lương thử việc là bao nhiêu?",
        top_k=5,
    )
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
