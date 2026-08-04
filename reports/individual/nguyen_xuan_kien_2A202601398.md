# Báo cáo cá nhân — Nguyễn Xuân Kiên

- MSSV: **2A202601398**
- Role: **4 — Evaluation & QA Engineer**
- Đề tài: **Trợ lý Luật Lao động Việt Nam**

## Trách nhiệm và kết quả

- Xây dựng golden dataset 15 tình huống luật lao động có answer/context chuẩn.
- Implement RAGAS với Faithfulness, Answer Relevancy, Context Recall và Precision.
- Chạy A/B Hybrid + RRF so với Dense-only trên cùng dataset/judge.
- Phân tích bottom-three, nguyên nhân và đề xuất cải tiến.

## Deliverable chính

- `group_project/evaluation/golden_dataset.json`.
- `group_project/evaluation/eval_pipeline.py`.
- `results.md` và `evaluation_artifacts.json`.

## Kết quả

- Hybrid average: **0.7576**.
- Dense-only average: **0.6743**.
- Delta: **+0.0834**, Hybrid tốt hơn ở cả 4 metric.

## Tự đánh giá

Evaluation chạy thật đủ 15 câu, có raw artifact để tái kiểm tra; không sử dụng số
điểm mẫu hoặc kết quả khác domain.
