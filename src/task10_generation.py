"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"

Gợi ý LLM: OpenRouter có nhiều model gắn hậu tố ":free" không tính phí — xem
https://openrouter.ai/models?max_price=0 — phù hợp nếu chưa có credit trả phí.
Base URL: "https://openrouter.ai/api/v1", dùng chung interface với OpenAI SDK.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

# OpenRouter uses provider/model IDs; direct OpenAI uses the model name only.
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
MAX_OUTPUT_TOKENS = 900


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý tra cứu pháp luật lao động Việt Nam dành cho người trẻ.

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ context được cung cấp — KHÔNG bịa đặt
2. Mỗi khẳng định pháp lý phải có trích dẫn ngay sau bằng ĐÚNG nhãn nguồn xuất hiện
   trong context, ví dụ: [Nguồn 1: bo-luat-lao-dong.pdf]
3. Nếu context không đủ thông tin → trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có"
4. Trả lời bằng tiếng Việt, ngắn gọn, có cấu trúc rõ ràng
5. Khi có nhiều nguồn, ưu tiên văn bản pháp luật gốc và nêu rõ nếu các nguồn khác nhau
6. Không suy luận hay mở rộng ngoài những gì được nêu trong context
7. Không tự tạo tên nguồn, số điều, mức tiền hoặc thời hạn không có trong context"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if not isinstance(chunks, list):
        raise TypeError("chunks must be a list")

    # Slicing creates a new list and does not mutate retrieval/reranking output.
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    if not isinstance(chunks, list):
        raise TypeError("chunks must be a list")

    context_parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            continue
        content = chunk.get("content")
        if not isinstance(content, str) or not content.strip():
            continue

        metadata = chunk.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        source = (
            metadata.get("source")
            or metadata.get("title")
            or metadata.get("section")
            or metadata.get("node_title")
            or metadata.get("doc_id")
            or f"tai-lieu-{index}"
        )
        doc_type = metadata.get("type") or chunk.get("source") or "unknown"
        section = metadata.get("section") or metadata.get("node_title")
        location = f" | Mục: {section}" if section else ""
        citation_label = f"Nguồn {index}: {source}"

        context_parts.append(
            f"[{citation_label} | Loại: {doc_type}{location}]\n{content.strip()}"
        )
    return "\n\n---\n\n".join(context_parts)


def _generation_backends():
    """Yield configured OpenAI-compatible clients in fallback order."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Missing openai SDK. Install Task 10 dependencies from requirements.txt."
        ) from exc

    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openrouter_key:
        yield (
            "openrouter",
            OpenAI(
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://github.com/",
                    "X-Title": "Vietnam Labor Law RAG Lab",
                },
            ),
            LLM_MODEL,
        )
    if openai_key:
        yield "openai", OpenAI(api_key=openai_key), OPENAI_LLM_MODEL


def _call_llm(user_message: str) -> str:
    errors: list[str] = []
    attempted = False
    for provider, client, model in _generation_backends():
        attempted = True
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            answer = response.choices[0].message.content
            if isinstance(answer, str) and answer.strip():
                return answer.strip()
            errors.append(f"{provider}: model returned an empty answer")
        except Exception as exc:
            errors.append(f"{provider}: {type(exc).__name__}: {exc}")

    if not attempted:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env for Task 10 generation"
        )
    raise RuntimeError("All generation backends failed: " + " | ".join(errors))


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    query = query.strip()
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    if not context:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    user_message = (
        "CONTEXT ĐƯỢC PHÉP SỬ DỤNG:\n"
        f"{context}\n\n"
        "---\n"
        f"CÂU HỎI: {query}\n\n"
        "Hãy trả lời chỉ từ context và đặt nhãn trích dẫn ngay sau từng "
        "khẳng định pháp lý."
    )
    answer = _call_llm(user_message)
    retrieval_sources = {
        str(chunk.get("source", "hybrid")) for chunk in chunks if isinstance(chunk, dict)
    }
    retrieval_source = (
        next(iter(retrieval_sources)) if len(retrieval_sources) == 1 else "mixed"
    )
    return {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    test_queries = [
        "Thời gian thử việc tối đa là bao lâu và lương thử việc tối thiểu bao nhiêu?",
        "Người lao động được nghỉ phép năm bao nhiêu ngày?",
        "Doanh nghiệp chậm trả lương thì bị xử lý thế nào?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
