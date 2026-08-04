# Bài nhóm — Trợ lý Luật Lao động Việt Nam

Chatbot RAG hỗ trợ người trẻ tra cứu các vấn đề phổ biến như thử việc, hợp đồng,
tiền lương, làm thêm giờ, nghỉ phép và quyền lợi khi nghỉ việc. Câu trả lời chỉ
dựa trên corpus của nhóm và phải kèm trích dẫn nguồn.

## Tính năng CP5

- Giao diện chat Streamlit, lịch sử hội thoại và câu hỏi demo.
- Điều chỉnh `top_k` từ 3–10 ngay trên sidebar.
- Hybrid Retrieval: OpenAI embedding + ChromaDB kết hợp BM25.
- RRF hợp nhất thứ hạng; PageIndex fallback khi cosine `< 0.48`.
- Reorder context chống “lost in the middle” trước khi gọi LLM.
- Citation theo đúng nhãn `[Nguồn N: tên-tài-liệu]`.
- Hiển thị nguồn, score, loại tài liệu, mục liên quan và highlight từ khóa.
- Conversation memory tối đa 4 tin nhắn, có thể bật/tắt.
- Query Expansion tiếng Việt ánh xạ từ ngữ đời thường sang thuật ngữ pháp lý.
- Có cả BM25 và TF-IDF để so sánh cơ chế lexical retrieval.
- Tải lịch sử chat dưới dạng Markdown và xóa session khi demo lại.
- Xử lý lỗi API thân thiện, không hiển thị hoặc ghi log API key.

## Kiến trúc

```mermaid
flowchart TD
    U[Người dùng] --> UI[Streamlit app.py]
    UI --> G[Task 10: Generation + Citation]
    G --> P[Task 9: Retrieval Pipeline]

    P --> D[Task 5: Dense Semantic Search]
    P --> S[Task 6: BM25 Lexical Search]
    D --> C[(ChromaDB)]
    S --> M[(Markdown Corpus)]
    D --> R[Task 7: RRF]
    S --> R

    P -->|Cosine dưới 0.48| F[Task 8: PageIndex Fallback]
    R --> O[Reorder front + back reversed]
    F --> O
    O --> L[OpenRouter hoặc OpenAI LLM]
    L -->|Answer + citations + sources| UI
```

## Nhóm 4 thành viên

| Thành viên | MSSV | Role | Trách nhiệm chính |
|---|---|---:|---|
| Ngô Hoàng Phú | 2A202601244 | 1 | Leader, kiến trúc và tích hợp tổng thể |
| Nguyễn Trung Long | 2A202601514 | 2 | Data, dense retrieval và pipeline integration |
| Đinh Quốc Việt | 2A202601891 | 3 | Sparse/PageIndex, generation và Streamlit UI |
| Nguyễn Xuân Kiên | 2A202601398 | 4 | Golden dataset, RAGAS A/B và QA |

## Chuẩn bị môi trường

Tạo `.env` từ `.env.example`. Không commit API key.

```env
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-v1-...       # tùy chọn, được ưu tiên cho generation
PAGEINDEX_API_KEY=pix_...             # tùy chọn cho fallback
PAGEINDEX_DOC_ID=pi-...               # hoặc chạy upload Task 8 một lần
EMBEDDING_PROVIDER=openai
```

Nếu dùng PageIndex key riêng và chưa có document ID:

```powershell
python -m src.task8_pageindex_vectorless
```

## Chạy ứng dụng

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Mở `http://localhost:8501` nếu trình duyệt không tự mở.

## Kịch bản demo đề xuất

1. Hỏi: “Thời gian thử việc tối đa và lương thử việc tối thiểu là bao nhiêu?”
2. Mở vùng **Tài liệu tham khảo**, chỉ ra nguồn, score và từ khóa highlight.
3. Hỏi nối tiếp: “Còn công việc khác thì sao?” để demo conversation memory.
4. Thay đổi `top_k` và giải thích trade-off độ bao phủ/context length.
5. Trình bày luồng Semantic + BM25 → RRF → PageIndex → Citation.

## Kiểm tra trước khi demo

```powershell
python -m pytest tests/test_individual.py -v
python -m pytest tests/test_bonus_and_integration.py -v
python -m pytest tests/test_individual.py::TestTask10 -v
python -m streamlit run app.py --server.headless=true
```

Checklist:

- [x] App mở không có exception.
- [x] Câu hỏi luật lao động trả lời có citation.
- [x] Danh sách nguồn hiển thị đúng số đoạn và retrieval channel.
- [x] Câu hỏi nối tiếp sử dụng history khi memory bật.
- [x] Không có API key trong Git hoặc giao diện.
- [x] Golden dataset 15 câu, 4 metric và báo cáo A/B đã hoàn tất.

## Evaluation

Các deliverable đánh giá nằm trong:

- `group_project/evaluation/golden_dataset.json`
- `group_project/evaluation/eval_pipeline.py`
- `group_project/evaluation/results.md`

Chạy theo hướng dẫn của nhóm QA:

```powershell
python -m group_project.evaluation.eval_pipeline
```

Kết quả benchmark ngày 04/08/2026 trên đủ 15 câu:

| Config | Faithfulness | Relevancy | Recall | Precision | Average |
|---|---:|---:|---:|---:|---:|
| Hybrid + BM25 + RRF + fallback | 0.7622 | 0.4166 | 0.8778 | 0.9740 | **0.7576** |
| Dense-only | 0.5881 | 0.3662 | 0.8111 | 0.9317 | **0.6743** |

## Bonus 20 điểm

- **5 điểm lexical khác BM25:** `tfidf_search()` dùng TF-IDF word/bigram + cosine;
  BM25 dùng term saturation (`k1`) và document-length normalization (`b`).
- **5 điểm Semantic Search nâng cao:** Query Expansion tích hợp vào Task 5, Task 9 và UI.
- **4 điểm deploy:** có `Dockerfile`, `render.yaml`, healthcheck và Streamlit production config;
  cần khai báo secrets trên nền tảng trước khi tạo URL công khai.
- **3 điểm memory:** lịch sử 4 tin nhắn giúp phân giải follow-up nhưng không được dùng làm evidence.
- **3 điểm UI/UX:** source, retrieval channel, score, highlight, suggestions và download chat.

## Deploy bằng Render Blueprint

1. Push repository lên GitHub.
2. Trên Render chọn **New → Blueprint**, trỏ tới repository này.
3. Khai báo bốn secrets theo `render.yaml`; tuyệt đối không commit `.env`.
4. Deploy và kiểm tra endpoint `/_stcore/health` trước khi demo.
