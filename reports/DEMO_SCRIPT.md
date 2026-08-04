# Kịch bản demo CP6 (5–8 phút)

## 0:00–1:00 — Kiến trúc (Role 1)

Mở sơ đồ trong `group_project/README.md`: Data → Chunk/Embedding → Dense + BM25
→ RRF → PageIndex fallback → Reorder → LLM citation → Streamlit.

## 1:00–2:30 — Retrieval (Role 2)

- Giải thích cosine gốc quyết định fallback; RRF score chỉ dùng xếp hạng.
- So sánh BM25 với TF-IDF: BM25 có saturation/length normalization, TF-IDF dùng
  sparse vector + cosine và bigram.
- Trình bày Query Expansion: “tăng ca” → “làm thêm giờ”.

## 2:30–4:30 — Live chatbot (Role 3)

1. Hỏi “Thời gian thử việc tối đa và lương thử việc tối thiểu là bao nhiêu?”.
2. Mở nguồn để chỉ citation, score và highlight.
3. Hỏi “Còn công việc khác thì sao?” để demo memory.
4. Bật/tắt Query Expansion và thay đổi `top_k`.

## 4:30–6:00 — Evaluation (Role 4)

Mở `results.md`: đủ 15 câu, 4 metrics, Hybrid vs Dense-only; Hybrid tăng trung
bình 0.0834. Trình bày bottom-three và khuyến nghị chunk theo Markdown headings.

## 6:00–7:00 — QA và kết luận

Chạy `pytest tests/test_individual.py -v` và `pytest tests/test_bonus_and_integration.py -v`.
Nhắc disclaimer: trợ lý hỗ trợ tra cứu, không thay thế tư vấn pháp lý.
