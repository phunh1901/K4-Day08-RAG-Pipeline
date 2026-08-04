# Báo cáo cá nhân — Ngô Hoàng Phú

- MSSV: **2A202601244**
- Role: **1 — Team Leader & RAG Architect**
- Đề tài: **Trợ lý Luật Lao động Việt Nam**

## Trách nhiệm và kết quả

- Điều phối tích hợp Tasks 1–10 thành một pipeline thống nhất.
- Kiểm tra kiến trúc Dense + BM25 → RRF và quy tắc PageIndex fallback theo cosine gốc.
- Chuẩn hóa README, checklist checkpoint và kịch bản demo.
- Kiểm tra không lộ API key; cấu hình triển khai dùng secrets.

## Deliverable chính

- `group_project/README.md`: kiến trúc, cách chạy, rubric và phân công.
- `src/task9_retrieval_pipeline.py`: orchestration, fallback và Query Expansion.
- `reports/GROUP_REPORT.md`, `reports/DEMO_SCRIPT.md`.

## Bằng chứng nghiệm thu

- CP0–CP5 có pass criteria và file tương ứng.
- Pipeline cá nhân được kiểm tra bằng 35 test.
- RAGAS A/B đủ 15 câu và 4 metric; Hybrid trung bình 0.7576.

## Tự đánh giá

Hoàn thành trách nhiệm kiến trúc và tích hợp. Phần triển khai URL công khai cần
thực hiện bằng tài khoản nền tảng của nhóm sau khi code được push lên `main`.
