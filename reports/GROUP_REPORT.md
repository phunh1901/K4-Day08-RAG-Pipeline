# Báo cáo nhóm — Vietnamese Labour Law RAG

## Thành viên

| Thành viên | MSSV | Role |
|---|---|---:|
| Ngô Hoàng Phú | 2A202601244 | 1 — Team Leader & RAG Architect |
| Nguyễn Trung Long | 2A202601514 | 2 — Data & Retrieval Specialist |
| Đinh Quốc Việt | 2A202601891 | 3 — Frontend & Chatbot Developer |
| Nguyễn Xuân Kiên | 2A202601398 | 4 — Evaluation & QA Engineer |

Nhóm thực hiện chung toàn bộ pipeline nên 50 điểm Tasks 1–10 và 30 điểm bài nhóm
được trình bày như một sản phẩm tích hợp 80 điểm, nhưng vẫn giữ trách nhiệm chính
theo Role để truy vết deliverable.

## Sản phẩm

Chatbot RAG tra cứu Luật Lao động Việt Nam: dữ liệu pháp luật và bài viết chính
thống được chuẩn hóa, chia chunk 800/overlap 100, embedding bằng
`text-embedding-3-small`, lưu ChromaDB, kết hợp Dense + BM25 bằng RRF, fallback
PageIndex khi cosine gốc `<0.48`, sau đó generation có citation.

## Nghiệm thu checkpoint

| CP | Tiêu chí | Bằng chứng | Trạng thái |
|---:|---|---|---|
| 0 | Python 3.11, dependencies, API | `.venv`, `requirements.txt`, `.env.example` | Đạt |
| 1 | ≥3 legal, ≥5 news, Markdown | 15 legal files, 6 news files, 11 standardized Markdown | Đạt |
| 2 | Chunk/index, Dense, BM25 | Tasks 4–6; ChromaDB; BM25 dùng chung chunks Task 4 | Đạt |
| 3 | RRF và PageIndex | Tasks 7–8; manifest PageIndex; fallback test | Đạt |
| 4 | Tasks 1–10 | `tests/test_individual.py`: mục tiêu 35/35 pass | Đạt |
| 5 | UI + RAGAS A/B | Streamlit; 15 golden Q&A; 4 metrics; 2 configs | Đạt |
| 6 | Demo/nộp | `reports/DEMO_SCRIPT.md`, Docker/Render blueprint | Sẵn sàng thao tác live |

## Kết quả RAGAS chính thức

Benchmark chạy trên đủ 15 câu, `top_k=5`, cùng LLM judge:

| Metric | Hybrid + RRF | Dense-only | Delta |
|---|---:|---:|---:|
| Faithfulness | 0.7622 | 0.5881 | +0.1741 |
| Answer Relevancy | 0.4166 | 0.3662 | +0.0505 |
| Context Recall | 0.8778 | 0.8111 | +0.0667 |
| Context Precision | 0.9740 | 0.9317 | +0.0423 |
| **Average** | **0.7576** | **0.6743** | **+0.0834** |

Chi tiết từng câu, context và nguồn nằm trong
`group_project/evaluation/evaluation_artifacts.json`; bottom-three và đề xuất cải
tiến nằm trong `group_project/evaluation/results.md`.

## Đối chiếu rubric

| Nhóm điểm | Hạng mục | Bằng chứng |
|---|---|---|
| Core 50 | Tasks 1–10 | 35 automated tests, source modules và corpus |
| Group 30 | Chatbot + pipeline + README + quality + evaluation | `app.py`, `group_project/README.md`, RAGAS artifacts |
| Bonus 5 | Lexical khác BM25 | `tfidf_search()` và tài liệu so sánh TF-IDF/BM25 |
| Bonus 5 | Semantic nâng cao | Query Expansion tích hợp Task 5 → Task 9 → UI |
| Bonus 3 | Conversation memory | Follow-up history, giới hạn 4 messages, evidence isolation |
| Bonus 3 | UI/UX | Sources, score, channel, highlight, suggestions, download |
| Bonus 4 | Deploy online | Docker + Render blueprint hoàn tất; URL công khai cần thao tác tài khoản Render |

## Kết luận

Toàn bộ phần có thể kiểm chứng trong local repository đã hoàn thành. Hai thao tác
ngoài repository trước khi demo là tạo deployment công khai bằng tài khoản của
nhóm và bảo đảm tài khoản PageIndex còn credits; không đưa API key vào source control.
