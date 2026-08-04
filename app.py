"""Streamlit chatbot for the Vietnamese Labour Law RAG group project.

Run from the project root::

    streamlit run app.py
"""

from __future__ import annotations

import html
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task4_chunking_indexing import EMBEDDING_DIM, EMBEDDING_MODEL
from src.task10_generation import generate_with_citation


st.set_page_config(
    page_title="Trợ lý Luật Lao động Việt Nam",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1120px; padding-top: 1.8rem; padding-bottom: 2rem;}
      [data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.22);}
      .app-kicker {font-size: .76rem; letter-spacing: .13em; text-transform: uppercase;
                   color: #d69e2e; font-weight: 750; margin-bottom: .35rem;}
      .app-subtitle {color: #87909f; margin: -.35rem 0 1.35rem;}
      mark {background: rgba(245, 190, 63, .32); color: inherit; padding: 0 .08rem;
            border-radius: .15rem;}
      .source-preview {font-size: .9rem; line-height: 1.55; overflow-wrap: anywhere;}
      .retrieval-badge {display: inline-block; border: 1px solid rgba(214,158,46,.55);
                        border-radius: 999px; padding: .12rem .55rem; font-size: .78rem;
                        margin-bottom: .55rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


SUGGESTED_QUESTIONS = [
    "Thời gian thử việc tối đa và lương thử việc tối thiểu là bao nhiêu?",
    "Người lao động được nghỉ phép năm bao nhiêu ngày?",
    "Công ty đơn phương chấm dứt hợp đồng phải báo trước bao lâu?",
    "Tiền lương làm thêm giờ được tính như thế nào?",
    "Khi nghỉ việc, người lao động được nhận những khoản tiền nào?",
]

CHANNEL_LABELS = {
    "hybrid": "🟢 Hybrid Search",
    "pageindex": "🟠 PageIndex Fallback",
    "mixed": "🔵 Mixed Retrieval",
    "none": "⚪ Không có nguồn",
    "error": "🔴 Pipeline Error",
}


def _clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.pending_query = None


def _source_name(source: dict, index: int) -> str:
    metadata = source.get("metadata") or {}
    return str(
        metadata.get("source")
        or metadata.get("filename")
        or metadata.get("title")
        or metadata.get("section")
        or metadata.get("node_title")
        or metadata.get("doc_id")
        or f"Tài liệu {index}"
    )


def _highlight_preview(content: str, query: str, limit: int = 900) -> str:
    """Return an escaped source preview with useful query terms highlighted."""
    preview = content.strip()[:limit]
    escaped = html.escape(preview)
    terms = {
        token
        for token in re.findall(r"\w+", query, flags=re.UNICODE)
        if len(token) >= 4
    }
    for term in sorted(terms, key=len, reverse=True)[:10]:
        escaped = re.sub(
            re.escape(html.escape(term)),
            lambda match: f"<mark>{match.group(0)}</mark>",
            escaped,
            flags=re.IGNORECASE,
        )
    suffix = "…" if len(content.strip()) > limit else ""
    return (
        '<div class="source-preview">'
        + escaped.replace("\n", "<br>")
        + suffix
        + "</div>"
    )


def _render_sources(
    sources: list[dict],
    retrieval_source: str,
    query: str,
) -> None:
    if not sources:
        return

    channel = CHANNEL_LABELS.get(retrieval_source, retrieval_source.upper())
    with st.expander(f"📚 Tài liệu tham khảo ({len(sources)} đoạn)", expanded=False):
        st.markdown(
            f'<span class="retrieval-badge">{html.escape(channel)}</span>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Score dùng để xếp hạng trong từng phương pháp; không phải xác suất "
            "câu trả lời đúng."
        )

        for index, source in enumerate(sources, start=1):
            metadata = source.get("metadata") or {}
            source_name = _source_name(source, index)
            doc_type = metadata.get("type") or source.get("source") or "unknown"
            section = metadata.get("section") or metadata.get("node_title")
            try:
                score = float(source.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0

            with st.container(border=True):
                st.markdown(f"**[Nguồn {index}] {source_name}**")
                details = f"Loại: `{doc_type}` · score: `{score:.4f}`"
                if section:
                    details += f" · Mục: `{section}`"
                st.caption(details)
                content = str(source.get("content") or "")
                if content.strip():
                    st.markdown(
                        _highlight_preview(content, query),
                        unsafe_allow_html=True,
                    )


def _conversation_markdown(messages: list[dict]) -> str:
    lines = ["# Lịch sử hỏi đáp Luật Lao động", ""]
    for message in messages:
        heading = "Câu hỏi" if message.get("role") == "user" else "Trả lời"
        lines.extend([f"## {heading}", str(message.get("content") or ""), ""])
        if message.get("role") == "assistant":
            for index, source in enumerate(message.get("sources") or [], start=1):
                lines.append(f"- Nguồn {index}: {_source_name(source, index)}")
            lines.append("")
    return "\n".join(lines)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


with st.sidebar:
    st.title("⚖️ Luật Lao động RAG")
    st.caption("Trợ lý tra cứu quyền và nghĩa vụ lao động từ nguồn pháp luật Việt Nam.")

    st.divider()
    st.subheader("💡 Câu hỏi demo")
    for index, suggestion in enumerate(SUGGESTED_QUESTIONS):
        if st.button(suggestion, key=f"suggestion_{index}", use_container_width=True):
            st.session_state.pending_query = suggestion

    st.divider()
    st.subheader("⚙️ Cấu hình truy xuất")
    top_k = st.slider(
        "Số đoạn tài liệu (top_k)",
        min_value=3,
        max_value=10,
        value=5,
        help="Nhiều đoạn hơn tăng độ bao phủ nhưng làm context dài hơn.",
    )
    use_memory = st.toggle(
        "Conversation memory",
        value=True,
        help="Dùng tối đa 4 tin nhắn gần nhất để hiểu câu hỏi nối tiếp.",
    )
    use_query_expansion = st.toggle(
        "Query expansion",
        value=True,
        help="Mở rộng từ ngữ đời thường sang thuật ngữ pháp lý trước khi tìm kiếm.",
    )

    left, right = st.columns(2)
    with left:
        st.button("🗑️ Xóa chat", on_click=_clear_chat, use_container_width=True)
    with right:
        st.download_button(
            "⬇️ Tải chat",
            data=_conversation_markdown(st.session_state.messages),
            file_name="labor_law_chat.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.divider()
    st.subheader("🧩 Pipeline")
    st.caption(f"Embedding: `{EMBEDDING_MODEL}` ({EMBEDDING_DIM}d)")
    st.caption("Vector store: `ChromaDB` · Rerank: `RRF (k=60)`")
    st.caption("Fallback: `PageIndex` khi cosine `< 0.48`")
    if os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"):
        st.success("LLM generation đã cấu hình", icon="✅")
    else:
        st.warning("Thiếu API key cho LLM", icon="⚠️")
    if os.getenv("PAGEINDEX_API_KEY"):
        st.success("PageIndex đã cấu hình", icon="✅")
    else:
        st.info("PageIndex chưa cấu hình; vẫn dùng Hybrid Search", icon="ℹ️")


st.markdown('<div class="app-kicker">Vietnamese Labour Law Assistant</div>', unsafe_allow_html=True)
st.title("⚖️ Trợ lý Luật Lao động Việt Nam")
st.markdown(
    '<div class="app-subtitle">Hybrid Retrieval · RRF Reranking · PageIndex '
    "Fallback · Citation Generation</div>",
    unsafe_allow_html=True,
)
st.info(
    "Câu trả lời được tổng hợp từ tài liệu trong hệ thống và có trích dẫn. "
    "Nội dung chỉ nhằm mục đích tham khảo, không thay thế tư vấn pháp lý chuyên môn.",
    icon="ℹ️",
)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_sources(
                message.get("sources") or [],
                str(message.get("retrieval_source") or "none"),
                str(message.get("query") or ""),
            )
            if message.get("elapsed_seconds") is not None:
                st.caption(f"Hoàn thành trong {message['elapsed_seconds']:.1f} giây")


user_input = st.chat_input("Nhập câu hỏi về thử việc, hợp đồng, lương, nghỉ phép…")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None
    query = str(query).strip()
    previous_messages = list(st.session_state.messages[-4:]) if use_memory else []

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        started_at = time.perf_counter()
        error_detail = None
        try:
            with st.spinner("Đang tìm nguồn và tổng hợp câu trả lời có trích dẫn…"):
                response: dict[str, Any] = generate_with_citation(
                    query,
                    top_k=top_k,
                    conversation_history=previous_messages,
                    use_query_expansion=use_query_expansion,
                )
            answer = str(response.get("answer") or "").strip()
            sources = response.get("sources") or []
            retrieval_source = str(response.get("retrieval_source") or "none")
            if not answer:
                answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
        except Exception as exc:
            answer = (
                "Không thể hoàn tất truy vấn lúc này. Vui lòng kiểm tra API key, "
                "kết nối mạng và trạng thái dịch vụ rồi thử lại."
            )
            sources = []
            retrieval_source = "error"
            error_detail = f"{type(exc).__name__}: {exc}"

        elapsed_seconds = time.perf_counter() - started_at
        st.markdown(answer)
        if error_detail:
            with st.expander("Chi tiết kỹ thuật"):
                st.code(error_detail)
        _render_sources(sources, retrieval_source, query)
        st.caption(f"Hoàn thành trong {elapsed_seconds:.1f} giây")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "retrieval_source": retrieval_source,
            "elapsed_seconds": elapsed_seconds,
            "query": query,
        }
    )
