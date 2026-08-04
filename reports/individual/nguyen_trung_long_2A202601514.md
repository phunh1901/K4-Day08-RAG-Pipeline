# Báo cáo cá nhân — Nguyễn Trung Long

- MSSV: **2A202601514**
- Role: **2 — Data & Retrieval Specialist**
- Đề tài: **Trợ lý Luật Lao động Việt Nam**

## Trách nhiệm và kết quả

- Thu thập/chuẩn hóa corpus: 15 legal files, 6 news files, 11 Markdown.
- Chunk 800 ký tự, overlap 100; embedding OpenAI 1536 chiều và ChromaDB.
- Dense retrieval và kết nối Task 10 vào pipeline ứng dụng.
- Hỗ trợ Query Expansion cho thuật ngữ pháp lý tiếng Việt.

## Deliverable chính

- `src/task1_collect_legal_docs.py` đến `src/task5_semantic_search.py`.
- `chroma_db/` và dữ liệu trong `data/`.
- Luồng Task 9 → Task 10 → `app.py`.

## Bằng chứng nghiệm thu

- Tasks 1–5 vượt toàn bộ test tương ứng.
- Semantic results đúng schema, cosine giảm dần và cùng embedding space với index.
- Query Expansion được kiểm tra offline trong `tests/test_bonus_and_integration.py`.

## Tự đánh giá

Hoàn thành data/dense retrieval và tích hợp pipeline; dữ liệu nguồn đáp ứng vượt
mức tối thiểu của rubric.
