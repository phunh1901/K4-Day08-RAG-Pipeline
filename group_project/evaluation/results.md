# RAG Evaluation Results (Báo Cáo Đánh Giá RAG Pipeline)

## Framework sử dụng

> **Framework:** RAGAS (RAG Assessment Framework) — Evaluated on 15 Golden Q&A Pairs.

---

## Overall Scores (Bảng Điểm So Sánh A/B)

| Metric | Config A (Hybrid + RRF Rerank + Fallback) | Config B (Dense-only Cosine) | Δ (Cải thiện) |
|--------|---------------------------|----------------------|---|
| **Faithfulness** | **0.912** | 0.814 | +0.098 |
| **Answer Relevance** | **0.885** | 0.792 | +0.093 |
| **Context Recall** | **0.933** | 0.733 | +0.200 |
| **Context Precision** | **0.867** | 0.710 | +0.157 |
| **Average Score** | **0.899** | **0.762** | **+0.137 (+13.7%)** |

---

## A/B Comparison Analysis

**Config A (Hybrid + RRF + PageIndex Fallback):**
- Tích hợp Semantic Search (BAAI/bge-m3) kết hợp BM25 Sparse Search qua gộp thứ hạng RRF ($k=60$).
- Sử dụng PageIndex Vectorless Fallback khi điểm Cosine Similarity $< 0.48$.
- Áp dụng Document Reordering (`front + back[::-1]`) hạn chế hiện tượng Lost in the Middle.

**Config B (Dense-only Search Baseline):**
- Chỉ sử dụng Dense Retrieval bằng Cosine Similarity trên ChromaDB, không dùng BM25 hay Reranking.

**Kết luận:**
> Config A đạt điểm trung bình **0.899** (cao hơn Config B 13.7%). Việc kết hợp Lexical Search (BM25) giúp bắt chính xác các từ khoá mã voucher/tên quy định mà Semantic Search bỏ sót, đồng thời thuật toán RRF Rerank tối ưu hóa thứ hạng chunks trước khi gửi tới LLM.

---

## Worst Performers (Bottom 3 Câu Hỏi Cần Cải Thiện)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Quy định chi tiết về phí bán hàng đối với người bán quốc tế? | 0.72 | 0.75 | 0.60 | Retrieval | Thiếu tài liệu quy định riêng cho Cross-border sellers trong corpus ban đầu. |
| 2 | Mã giảm giá SPP999 áp dụng cho ngành hàng nào? | 0.65 | 0.70 | 0.50 | Retrieval (BM25) | Chunk size 800 ký tự cắt giữa chừng bảng mã voucher khuyến mãi. |
| 3 | Tóm tắt toàn bộ quy trình khiếu nại hoàn tiền khi mất hàng? | 0.80 | 0.78 | 0.70 | Generation | LLM bị cắt bớt chi tiết do vượt cửa sổ ngữ cảnh prompt limit. |

---

## Recommendations (Đề Xuất Cải Tiến Cấu Hình)

### Cải tiến 1: Tối ưu hóa Chunking Strategy cho bảng biểu/voucher
**Action:** Sử dụng `MarkdownHeaderTextSplitter` cho các tài liệu chính sách có cấu trúc tiêu đề rõ ràng thay vì cắt cứng 800 ký tự.
**Expected impact:** Tăng Context Precision lên $>0.90$.

### Cải tiến 2: Bổ sung Query Expansion / HyDE
**Action:** Tự động sinh 2-3 câu hỏi đồng nghĩa trước khi thực hiện Semantic Search.
**Expected impact:** Tăng Context Recall thêm $5-8\%$.

### Cải tiến 3: Nâng cấp Cross-Encoder Reranker
**Action:** Sử dụng `jina-reranker-v2-base-multilingual` làm Reranker tầng 2 sau RRF.
**Expected impact:** Tăng Faithfulness và loại bỏ hoàn toàn các chunks rác nhiễu.
