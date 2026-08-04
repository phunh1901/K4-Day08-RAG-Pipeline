# Báo cáo cá nhân — Đinh Quốc Việt

- MSSV: **2A202601891**
- Role: **3 — Frontend & Chatbot Developer**
- Đề tài: **Trợ lý Luật Lao động Việt Nam**

## Trách nhiệm và kết quả

- Hoàn thiện BM25 trên cùng chunks 800/100 với Dense retrieval.
- Tích hợp PageIndex, citation generation và chống lost-in-the-middle.
- Xây dựng Streamlit chat UI, nguồn/score/highlight, suggestions và tải lịch sử.
- Bổ sung conversation memory và Query Expansion toggle.

## Deliverable chính

- `src/task6_lexical_search.py`, Tasks 8–10 và `app.py`.
- TF-IDF alternative để so sánh với BM25.
- Streamlit production config và deployment files.

## Bằng chứng nghiệm thu

- Task 6 đạt 4/4 test; mỗi BM25 result tối đa 800 ký tự.
- Task 8 kiểm tra ổn định bằng mock đúng contract; API thật đã từng được xác minh,
  nhưng cần bổ sung PageIndex credits trước buổi demo fallback live.
- Follow-up “Còn công việc khác thì sao?” trả lời đúng ngữ cảnh thử việc, có citation.

## Tự đánh giá

Hoàn thành frontend, sparse/vectorless retrieval và các bonus memory/UI/TF-IDF.
