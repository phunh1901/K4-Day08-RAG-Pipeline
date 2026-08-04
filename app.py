"""
RAG Chatbot — E-commerce Support (K4 Variant)
Streamlit app kết nối RAG Retrieval (Task 9) và Generation (Task 10).

Chạy:
    streamlit run app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Thêm project root vào sys.path để import các task từ src/
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# PAGE CONFIG & CUSTOM STYLING (Role 3 - UI/UX Specialist)
# =============================================================================

st.set_page_config(
    page_title="E-commerce Support RAG Chatbot",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ff4b4b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .source-card {
        background-color: #f8f9fa;
        border-left: 4px solid #ff4b4b;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 4px;
    }
    .badge-hybrid {
        background-color: #28a745;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
    }
    .badge-fallback {
        background-color: #fd7e14;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR — INFO & SETTINGS (Role 3 - UI Controls)
# =============================================================================

with st.sidebar:
    st.title("🛒 E-commerce Support RAG")
    st.caption("Trợ lý hỏi đáp chính sách thương mại điện tử (đổi trả, thanh toán, bảo mật, người bán)")

    st.divider()

    st.subheader("💡 Câu hỏi gợi ý")
    suggestions = [
        "Thời hạn yêu cầu trả hàng/hoàn tiền là bao lâu?",
        "Shopee hỗ trợ những phương thức thanh toán nào?",
        "Làm sao để đổi phương thức thanh toán đơn hàng?",
        "Quy định về đăng bán sản phẩm cho người bán?",
        "Cách mua hàng trên Shopee của quốc gia khác?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{s[:20]}"):
            st.session_state["pending_query"] = s

    st.divider()
    st.subheader("⚙️ Cấu Hình RAG Pipeline")
    top_k = st.slider("Số lượng chunks retrieval (top_k)", min_value=1, max_value=10, value=5)
    customer_role_select = st.selectbox(
        "Đối tượng áp dụng (customer_role)",
        ["Tất cả (both)", "Người mua (buyer)", "Người bán (seller)"]
    )
    
    st.divider()
    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("**Chỉ số hệ thống & Kiến trúc:**")
    st.caption("• Embedding: `BAAI/bge-m3` (1024d)")
    st.caption("• Vector Store: `ChromaDB` (Local)")
    st.caption("• Reranker: `RRF (k=60)`")
    st.caption("• Vectorless Fallback: `PageIndex` (Cosine < 0.48)")

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# MAIN CHAT AREA
# =============================================================================

st.markdown('<div class="main-header">🛒 E-commerce Support RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hỏi đáp chính sách Shopee Vietnam — Kết hợp Hybrid Retrieval, RRF Rerank & PageIndex Fallback (K4 Variant)</div>', unsafe_allow_html=True)

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            ret_src = msg.get("retrieval_source", "hybrid")
            badge_class = "badge-fallback" if ret_src == "pageindex" else "badge-hybrid"
            with st.expander(f"📚 Nguồn tham khảo ({len(msg['sources'])} chunks) | Nguồn truy vấn: {ret_src.upper()}"):
                for i, src in enumerate(msg["sources"], 1):
                    meta = src.get("metadata", {})
                    source_name = meta.get("source", "Unknown")
                    doc_type = meta.get("type", "unknown")
                    score = src.get("score", 0.0)
                    st.markdown(f"**[{i}] {source_name}** (`{doc_type}`) | score: `{score:.4f}`")
                    st.text(src.get("content", "")[:350] + "...")
                    st.divider()

# =============================================================================
# QUERY HANDLING (Role 2 - Pipeline Integration)
# =============================================================================

user_input = st.chat_input("Nhập câu hỏi của bạn về chính sách/hỗ trợ e-commerce...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    # Hiển thị câu hỏi của user
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Sinh câu trả lời từ RAG Pipeline (Role 2 + Task 10 Integration)
    with st.chat_message("assistant"):
        with st.spinner("Đang truy vấn tài liệu chính sách và tổng hợp câu trả lời..."):
            ret_source = "hybrid"
            try:
                from src.task10_generation import generate_with_citation
                
                # Gọi hàm Task 10 sinh câu trả lời
                response = generate_with_citation(query, top_k=top_k)
                answer = response.get("answer", "Chưa thể trả lời.")
                sources = response.get("sources", [])
                ret_source = response.get("retrieval_source", "hybrid")

            except NotImplementedError:
                answer = "⚠️ **Task 10 chưa được hoàn thiện.** Hãy hoàn thành `src/task10_generation.py` để chạy pipeline!"
                sources = []
            except Exception as e:
                answer = f"❌ **Lỗi khi chạy RAG Pipeline:** {e}"
                sources = []

            # Hiển thị câu trả lời từ LLM
            st.markdown(answer)

            # Hiển thị các chunks tài liệu trích dẫn
            if sources:
                with st.expander(f"📚 Nguồn tham khảo ({len(sources)} chunks) | Nguồn truy vấn: {ret_source.upper()}"):
                    for i, src in enumerate(sources, 1):
                        meta = src.get("metadata", {})
                        source_name = meta.get("source", "Unknown")
                        doc_type = meta.get("type", "unknown")
                        score = src.get("score", 0.0)
                        st.markdown(f"**[{i}] {source_name}** (`{doc_type}`) | score: `{score:.4f}`")
                        st.text(src.get("content", "")[:350] + "...")
                        st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "retrieval_source": ret_source,
    })
