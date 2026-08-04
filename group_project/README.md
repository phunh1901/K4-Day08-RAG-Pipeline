# Bài Tập Nhóm — E-commerce Support RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1: Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về chính sách thương mại điện tử và hỗ trợ khách hàng liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

Xem code mẫu (DeepEval/RAGAS/TruLens) chi tiết trong `README.md` gốc mục "Yêu cầu 2".

### Deliverable Evaluation

- [ ] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [ ] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [ ] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [ ] So sánh A/B ít nhất 2 configs

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

```mermaid
flowchart TD
    User([Người dùng / User Input]) --> UI[Streamlit Chatbot UI - app.py]
    UI --> Task10[Task 10: Citation Generation]
    Task10 --> Task9[Task 9: Complete Retrieval Pipeline]
    
    subgraph Retrieval_Layer [Retrieval Layer]
        Task9 -->|Cosine >= 0.48| Hybrid[Hybrid Search Engine]
        Task9 -->|Cosine < 0.48| Fallback[Task 8: PageIndex Vectorless Fallback]
        
        Hybrid --> Dense[Task 5: Dense Semantic Search]
        Hybrid --> Sparse[Task 6: Sparse BM25 Lexical Search]
        
        Dense --> VectorDB[(ChromaDB Vector Store)]
        Sparse --> Corpus[(Markdown Corpus)]
        
        Dense --> RRF[Task 7: Reciprocal Rank Fusion Reranker]
        Sparse --> RRF
    end
    
    RRF --> Reorder[Document Reordering: front + back[::-1]]
    Fallback --> Reorder
    
    Reorder --> Prompt[Prompt Assembly + System Prompt]
    Prompt --> LLM[OpenRouter / OpenAI LLM]
    LLM --> UI
```

---

## Phân Công Công Việc (Nhóm 4 Thành Viên)

| Thành viên | Vai Trò (Role) | Nhiệm vụ chính | Trạng thái |
|-----------|------|----------|------------|
| **Role 1 (Leader)** | Team Leader & RAG Architect | Quản lý repo, ghép Task 9 Pipeline (`task9_retrieval_pipeline.py`), tích hợp Chatbot UI (`app.py`), chủ trì Demo live | ✅ Complete |
| **Role 2** | Data & Dense Search Specialist | Thu thập chính sách (Task 1), Convert Markdown (Task 3), Chunking & Indexing ChromaDB (Task 4), Semantic Search (Task 5) | ✅ Complete |
| **Role 3** | Frontend & Chatbot Dev | Crawl tin trợ giúp (Task 2), Document Reordering & Citation Prompt (Task 10), giao diện Streamlit Chatbot (`app.py`) | ✅ Complete |
| **Role 4** | Evaluation & QA Engineer | Lexical Search BM25 (Task 6), RRF Rerank (Task 7), PageIndex Fallback (Task 8), Golden Dataset & RAGAS Eval (`results.md`) | ✅ Complete |

---

## Hướng Dẫn Chạy

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy app
streamlit run app.py
# hoặc
chainlit run app.py
```

---

## Lưu ý

Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
