"""Fast, offline regression tests for bonus and cross-task behavior."""

from unittest.mock import patch

from src.task5_semantic_search import expand_query
from src.task6_lexical_search import tfidf_search
from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import _build_retrieval_query, _normalize_conversation_history


def test_query_expansion_maps_colloquial_legal_terms():
    variants = expand_query("Công ty bắt tăng ca thì sao?")
    assert len(variants) >= 2
    assert any("làm thêm giờ" in variant for variant in variants)


def test_tfidf_search_returns_ranked_chunks():
    results = tfidf_search("thời gian thử việc", top_k=3)
    assert 0 < len(results) <= 3
    assert all(len(item["content"]) <= 880 for item in results)
    assert [item["score"] for item in results] == sorted(
        [item["score"] for item in results], reverse=True
    )


def test_follow_up_history_augments_retrieval_query():
    history = _normalize_conversation_history(
        [{"role": "user", "content": "Thời gian thử việc tối đa là bao lâu?"}]
    )
    expanded = _build_retrieval_query("Còn công việc khác thì sao?", history)
    assert "Thời gian thử việc" in expanded
    assert "Câu hỏi nối tiếp" in expanded


def test_fallback_uses_original_cosine_after_query_expansion():
    dense = [{"content": "legal", "score": 0.016, "retrieval_score": 0.9, "metadata": {}}]
    sparse = [{"content": "legal", "score": 5.0, "metadata": {}}]
    with (
        patch("src.task9_retrieval_pipeline._run_retrievers", return_value=(dense, sparse)),
        patch("src.task9_retrieval_pipeline.pageindex_search") as pageindex,
    ):
        results = retrieve("test query", top_k=1, use_query_expansion=True)
    pageindex.assert_not_called()
    assert results[0]["source"] == "hybrid"
