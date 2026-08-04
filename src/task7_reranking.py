"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement — khuyến nghị vì không cần API key

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

Lưu ý quan trọng về RRF (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — xem ghi chú ở đó.
"""

def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    # TODO: Implement cross-encoder reranking
    #
    # Option A: Jina Reranker API
    # import requests
    # response = requests.post(
    #     "https://api.jina.ai/v1/rerank",
    #     headers={"Authorization": f"Bearer {JINA_API_KEY}"},
    #     json={
    #         "model": "jina-reranker-v2-base-multilingual",
    #         "query": query,
    #         "documents": [c["content"] for c in candidates],
    #         "top_n": top_k
    #     }
    # )
    # reranked = response.json()["results"]
    # return [
    #     {**candidates[r["index"]], "score": r["relevance_score"]}
    #     for r in reranked
    # ]
    #
    # Option B: Local model (Qwen3-Reranker)
    # from transformers import AutoModelForSequenceClassification, AutoTokenizer
    # ...
    raise NotImplementedError("Implement rerank_cross_encoder")


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    # TODO: Implement MMR
    #
    # selected = []
    # remaining = list(range(len(candidates)))
    #
    # for _ in range(min(top_k, len(candidates))):
    #     best_idx = None
    #     best_score = float('-inf')
    #
    #     for idx in remaining:
    #         # Relevance to query
    #         relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])
    #
    #         # Max similarity to already selected
    #         max_sim_to_selected = 0
    #         for sel_idx in selected:
    #             sim = cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
    #             max_sim_to_selected = max(max_sim_to_selected, sim)
    #
    #         # MMR score
    #         mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
    #
    #         if mmr_score > best_score:
    #             best_score = mmr_score
    #             best_idx = idx
    #
    #     selected.append(best_idx)
    #     remaining.remove(best_idx)
    #
    # return [candidates[i] for i in selected]
    raise NotImplementedError("Implement rerank_mmr")


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    if top_k <= 0 or not ranked_lists:
        return []
    if k < 0:
        raise ValueError("RRF smoothing constant k must be >= 0")

    rrf_scores: dict[str, float] = {}
    candidate_map: dict[str, dict] = {}
    ranks_by_candidate: dict[str, list[int]] = {}
    best_rank: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    seen_counter = 0

    for ranked_list in ranked_lists:
        if not isinstance(ranked_list, list):
            raise TypeError("Each ranked result set must be a list")

        # A ranker must contribute at most once to a document. This prevents an
        # accidental duplicate in one result list from inflating its fused score.
        seen_in_ranker: set[str] = set()
        for rank, item in enumerate(ranked_list, start=1):
            if not isinstance(item, dict):
                raise TypeError("Every RRF candidate must be a dictionary")
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Every RRF candidate must contain non-empty 'content'")
            if content in seen_in_ranker:
                continue
            seen_in_ranker.add(content)

            if content not in candidate_map:
                candidate_map[content] = item.copy()
                first_seen[content] = seen_counter
                seen_counter += 1

            rrf_scores[content] = rrf_scores.get(content, 0.0) + 1.0 / (k + rank)
            ranks_by_candidate.setdefault(content, []).append(rank)
            best_rank[content] = min(best_rank.get(content, rank), rank)

    ordered_contents = sorted(
        rrf_scores,
        key=lambda content: (
            -rrf_scores[content],
            best_rank[content],
            first_seen[content],
        ),
    )

    results: list[dict] = []
    for content in ordered_contents[:top_k]:
        result = candidate_map[content].copy()
        result["retrieval_score"] = result.get("score")
        result["score"] = rrf_scores[content]
        result["rrf_score"] = rrf_scores[content]
        result["rrf_ranks"] = ranks_by_candidate[content]
        results.append(result)
    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict] | list[list[dict]],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Một danh sách candidates, hoặc nhiều ranked lists cho RRF
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # Cần query_embedding - embed query trước
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    elif method == "rrf":
        # The public interface accepts a single list for the grading contract,
        # while Task 9 can pass [dense_results, sparse_results] for real fusion.
        if not candidates:
            return []
        if all(isinstance(candidate, list) for candidate in candidates):
            ranked_lists = candidates
        elif any(isinstance(candidate, list) for candidate in candidates):
            raise TypeError("RRF candidates must be either one flat list or only ranked lists")
        else:
            ranked_lists = [candidates]
        return rerank_rrf(ranked_lists, top_k=top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dense_results = [
        {"content": "Người lao động được nghỉ hằng năm theo Bộ luật Lao động.", "score": 0.82, "metadata": {}},
        {"content": "Thời giờ làm việc bình thường không quá giới hạn luật định.", "score": 0.76, "metadata": {}},
    ]
    sparse_results = [
        {"content": "Thời giờ làm việc bình thường không quá giới hạn luật định.", "score": 9.1, "metadata": {}},
        {"content": "Người lao động được nghỉ hằng năm theo Bộ luật Lao động.", "score": 7.4, "metadata": {}},
    ]
    results = rerank_rrf([dense_results, sparse_results], top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
