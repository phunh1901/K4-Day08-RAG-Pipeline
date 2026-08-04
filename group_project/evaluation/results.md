# RAGAS Evaluation Results

- Generated: `2026-08-04T22:24:25`
- Golden questions: **15**
- top_k: **5**
- Config A: **Hybrid + BM25 + RRF + PageIndex fallback**
- Config B: **Dense-only cosine baseline**

## A/B scores

| Metric | Config A | Config B | Δ A−B |
|---|---:|---:|---:|
| Faithfulness | 0.7622 | 0.5881 | +0.1741 |
| Answer Relevancy | 0.4166 | 0.3662 | +0.0505 |
| Context Recall | 0.8778 | 0.8111 | +0.0667 |
| Context Precision | 0.9740 | 0.9317 | +0.0423 |
| **Average** | **0.7576** | **0.6743** | **+0.0834** |

## Analysis

Config A combines exact Vietnamese legal keywords from BM25 with semantic matches, then fuses ranks using RRF. Config B is the controlled baseline and uses only cosine similarity.
The delta table above is generated from the same golden set and LLM judge, so the comparison is reproducible.

## Worst performers — Config A

| # | Question | Faithfulness | Relevancy | Recall | Precision | Average | Root cause |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | Công ty bảo cứ làm 2 tháng rồi tính, không ký hợp đồng lao động bằng văn bản có được không? | 0.6667 | 0.0000 | 0.6667 | 1.0000 | 0.5833 | Retrieval thiếu một phần điều kiện/ngoại lệ trong expected answer |
| 2 | Nếu em muốn nghỉ việc ở công ty hiện tại thì phải báo trước bao lâu? | 0.3333 | 0.0000 | 1.0000 | 1.0000 | 0.5833 | Generation chưa bám đủ chặt vào các mệnh đề trong context |
| 3 | Sếp nói sẽ sa thải em vì đi muộn nhiều lần, điều đó có hợp pháp không? | 0.5000 | 0.0000 | 1.0000 | 1.0000 | 0.6250 | Generation chưa bám đủ chặt vào các mệnh đề trong context |

## Root-cause analysis and recommendations

1. Inspect the saved contexts for the bottom-three questions in `evaluation_artifacts.json`; add missing statutes when recall is low.
2. Use Markdown heading-aware chunking for long statutes to avoid splitting an article from its conditions and exceptions.
3. Keep RRF for rank fusion, but use original cosine similarity—not RRF score—for PageIndex fallback decisions.
4. Expand short or colloquial questions before dense retrieval; retain the original query for citations and evaluation.

## Reproduce

```powershell
python -m group_project.evaluation.eval_pipeline --top-k 5
```

Raw per-question predictions, contexts, sources and metric scores are stored in `evaluation_artifacts.json`.
