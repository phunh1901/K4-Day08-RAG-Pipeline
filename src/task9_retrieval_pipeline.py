"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results

⚠️ BẪY THƯỜNG GẶP — đọc kỹ trước khi code:
    Nếu bạn dùng điểm RRF đã fuse (Task 7) để so với score_threshold, bạn sẽ gặp bug
    thật: RRF max score luôn ≈ 1/(k+1) ≈ 0.0164 (k=60) BẤT KỂ nội dung có liên quan
    hay không. Nếu đặt threshold thấp (như 0.005) để "hợp" với thang điểm RRF, thực
    chất KHÔNG câu hỏi nào đủ thấp để trigger fallback nữa — kể cả query hoàn toàn vô
    nghĩa vẫn trả về kết quả "hybrid" (rác) thay vì fallback đúng như thiết kế.

    Cách sửa đúng: giữ điểm cosine similarity GỐC của semantic_search (trước khi qua
    RRF) làm căn cứ quyết định fallback, tách biệt khỏi điểm RRF dùng để sắp xếp kết
    quả cuối cùng. Calibrate threshold bằng cách tự đo: chạy vài câu hỏi chắc chắn
    liên quan và vài câu chắc chắn lạc đề/rác qua semantic_search, xem khoảng cách
    điểm số giữa hai nhóm rồi chọn ngưỡng nằm giữa.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# Task 4/5 use OpenAI text-embedding-3-small with cosine similarity. Relevant
# Labor Law queries tested against this corpus score comfortably above 0.48;
# the lab uses this boundary to route weak dense matches to PageIndex.
SCORE_THRESHOLD = 0.48
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # "cross_encoder" | "mmr" | "rrf"
FETCH_MULTIPLIER = 3


def _validate_retrieve_arguments(
    query: str,
    top_k: int,
    score_threshold: float,
    use_reranking: bool,
) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if isinstance(score_threshold, bool) or not isinstance(score_threshold, (int, float)):
        raise ValueError("score_threshold must be numeric")
    if not -1.0 <= float(score_threshold) <= 1.0:
        raise ValueError("score_threshold must be between -1 and 1")
    if not isinstance(use_reranking, bool):
        raise ValueError("use_reranking must be a boolean")


def _run_retrievers(query: str, fetch_k: int) -> tuple[list[dict], list[dict]]:
    """Run independent dense and sparse retrieval concurrently."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(semantic_search, query, top_k=fetch_k)
        sparse_future = executor.submit(lexical_search, query, top_k=fetch_k)
        dense_results = dense_future.result()
        sparse_results = sparse_future.result()
    return dense_results or [], sparse_results or []


def _merge_without_reranking(
    dense_results: list[dict], sparse_results: list[dict], top_k: int
) -> list[dict]:
    """Deduplicate results while preserving dense-first retrieval order."""
    merged: list[dict] = []
    seen_contents: set[str] = set()
    for item in [*dense_results, *sparse_results]:
        content = item.get("content")
        if not isinstance(content, str) or not content.strip() or content in seen_contents:
            continue
        seen_contents.add(content)
        merged.append(item.copy())
        if len(merged) == top_k:
            break
    return merged


def _mark_source(results: list[dict], source: str, top_k: int) -> list[dict]:
    """Copy valid results into the common Task 9 output schema."""
    normalized: list[dict] = []
    for rank, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        result = item.copy()
        result["content"] = content
        result["metadata"] = dict(item.get("metadata") or {})
        # PageIndex may not provide a relevance score. A decreasing rank score
        # preserves ordering without pretending it is cosine similarity.
        result["score"] = float(item.get("score", 1.0 / rank))
        result["source"] = source
        normalized.append(result)
        if len(normalized) == top_k:
            break
    return normalized


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → dense_results (giữ điểm cosine gốc)
          ├→ Lexical Search  → sparse_results
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If dense_results[0]["score"] < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm cosine gốc tối thiểu (KHÔNG phải điểm RRF)
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    _validate_retrieve_arguments(query, top_k, score_threshold, use_reranking)
    query = query.strip()
    fetch_k = top_k * FETCH_MULTIPLIER

    # Dense and sparse scores use incompatible scales, so never compare or add
    # those raw scores. RRF combines their ranks instead.
    dense_results, sparse_results = _run_retrievers(query, fetch_k)
    if use_reranking:
        hybrid_results = rerank_rrf(
            [dense_results, sparse_results],
            top_k=top_k,
        )
    else:
        hybrid_results = _merge_without_reranking(
            dense_results,
            sparse_results,
            top_k,
        )
    hybrid_results = _mark_source(hybrid_results, "hybrid", top_k)

    # Fallback relevance is based only on the original cosine score. RRF scores
    # (~0.016 per first-place contribution with k=60) are ranking signals and
    # are not calibrated relevance probabilities.
    best_dense_score = max(
        (float(item.get("score", -1.0)) for item in dense_results),
        default=-1.0,
    )
    if best_dense_score < float(score_threshold):
        try:
            fallback_results = pageindex_search(query, top_k=top_k)
        except Exception:
            # PageIndex is an optional external fallback. Preserve useful local
            # results if its key, upload, SDK, or service is unavailable.
            fallback_results = []
        normalized_fallback = _mark_source(fallback_results or [], "pageindex", top_k)
        if normalized_fallback:
            return normalized_fallback

    return hybrid_results


if __name__ == "__main__":
    test_queries = [
        "Thời gian thử việc tối đa là bao lâu?",
        "Người lao động được nghỉ phép năm bao nhiêu ngày?",
        "Mức phạt khi vi phạm quy định về tiền lương là bao nhiêu?",
        "xyzabc123nonsense",  # Query không có kết quả → test fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
