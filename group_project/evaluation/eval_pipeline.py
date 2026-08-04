"""
RAG Evaluation Pipeline (Task Group Project) — RAGAS strict mode.

Luồng:
1. Đọc danh sách câu hỏi từ golden_dataset.json.
2. Với mỗi câu hỏi, gọi retrieve() rồi generate_with_citation().
3. Đóng gói thành HuggingFace Dataset (question, answer, contexts, ground_truth).
4. Chạy ragas.evaluate() với 4 metric chuẩn.
5. In bảng điểm ra terminal và ghi kết quả vào results.md.

Không có fallback. Mọi lỗi thư viện hoặc API đều ném ngoại lệ dừng chương trình.
"""

from __future__ import annotations

import statistics
from datetime import datetime
from pathlib import Path
from typing import Any
import json

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import generate_with_citation


GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------

def load_golden_dataset() -> list[dict[str, str]]:
    """Đọc danh sách câu hỏi từ golden_dataset.json."""
    data = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise Exception("golden_dataset.json phải là một JSON array.")
    if len(data) == 0:
        raise Exception("golden_dataset.json không có mục nào.")
    return data


# ---------------------------------------------------------------------------
# 2. Thu thập prediction từ RAG pipeline
# ---------------------------------------------------------------------------

def _normalize_contexts(chunks: list[dict[str, Any]]) -> list[str]:
    """Trích nội dung text từ danh sách chunk trả về bởi retrieve()."""
    contexts: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise Exception(
                f"retrieve() trả về phần tử không phải dict: {type(chunk)}"
            )
        content = chunk.get("content")
        if not isinstance(content, str):
            raise Exception(
                f"Chunk thiếu trường 'content' kiểu str: {chunk!r}"
            )
        if content.strip():
            contexts.append(content.strip())
    return contexts


def collect_predictions(
    golden_dataset: list[dict[str, str]],
) -> tuple[list[str], list[str], list[list[str]], list[str]]:
    """
    Với mỗi câu hỏi:
      - Gọi retrieve() để lấy contexts.
      - Gọi generate_with_citation() để lấy answer.

    Trả về 4 list song song: questions, answers, contexts, ground_truths.
    """
    questions: list[str] = []
    answers: list[str] = []
    contexts_list: list[list[str]] = []
    ground_truths: list[str] = []

    n = len(golden_dataset)
    for idx, item in enumerate(golden_dataset, start=1):
        question = str(item.get("question", "")).strip()
        expected_answer = str(item.get("expected_answer", "")).strip()

        if not question:
            raise Exception(
                f"Mục thứ {idx} trong golden_dataset.json không có trường 'question'."
            )
        if not expected_answer:
            raise Exception(
                f"Mục thứ {idx} trong golden_dataset.json không có trường 'expected_answer'."
            )

        print(f"[{idx:02d}/{n}] Retrieving contexts for: {question[:60]}...")
        retrieved = retrieve(question, top_k=5)
        contexts = _normalize_contexts(list(retrieved))

        print(f"[{idx:02d}/{n}] Generating answer...")
        generated = generate_with_citation(question)

        answer = str(generated.get("answer", "")).strip()
        if not answer:
            raise Exception(
                f"generate_with_citation() trả về answer rỗng cho câu hỏi: {question!r}"
            )

        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(expected_answer)

        print(f"[{idx:02d}/{n}] Done.")

    return questions, answers, contexts_list, ground_truths


# ---------------------------------------------------------------------------
# 3. Đóng gói Dataset và chạy RAGAS
# ---------------------------------------------------------------------------

def run_ragas(
    questions: list[str],
    answers: list[str],
    contexts_list: list[list[str]],
    ground_truths: list[str],
) -> dict[str, float]:
    """
    Đóng gói dữ liệu thành HuggingFace Dataset và chạy ragas.evaluate().

    Trả về dict: metric_name -> điểm trung bình.
    """
    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        }
    )

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    result = evaluate(dataset, metrics=metrics)

    # result là EvaluationResult; lấy dict điểm trung bình
    scores: dict[str, float] = {}
    for metric in metrics:
        key = metric.name  # tên chuẩn của metric trong RAGAS
        scores[key] = float(result[key])

    return scores


# ---------------------------------------------------------------------------
# 4. In bảng kết quả
# ---------------------------------------------------------------------------

METRIC_DISPLAY = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
}


def print_results_table(scores: dict[str, float], n_samples: int) -> None:
    print("\n" + "=" * 60)
    print(f"RAGAS Evaluation Results | samples={n_samples}")
    print("=" * 60)
    print(f"{'Metric':<25} | {'Score':>8}")
    print("-" * 60)
    for key, display in METRIC_DISPLAY.items():
        print(f"{display:<25} | {scores[key]:.4f}")
    print("-" * 60)
    avg_all = statistics.fmean(scores.values())
    print(f"{'Average (4 metrics)':<25} | {avg_all:.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 5. Ghi kết quả vào results.md
# ---------------------------------------------------------------------------

def export_results(scores: dict[str, float], n_samples: int) -> None:
    """Ghi đè báo cáo tổng hợp vào results.md."""
    avg_all = statistics.fmean(scores.values())

    lines: list[str] = [
        "# RAG Evaluation Results",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Evaluator: **RAGAS**",
        f"- Number of questions: **{n_samples}**",
        "",
        "## Overall Scores",
        "",
        "| Metric | Score |",
        "|---|---:|",
        f"| Faithfulness | {scores['faithfulness']:.4f} |",
        f"| Answer Relevancy | {scores['answer_relevancy']:.4f} |",
        f"| Context Precision | {scores['context_precision']:.4f} |",
        f"| Context Recall | {scores['context_recall']:.4f} |",
        f"| **Average (4 metrics)** | **{avg_all:.4f}** |",
        "",
        "## Notes",
        "",
        "- Faithfulness: câu trả lời bám sát context truy xuất.",
        "- Answer Relevancy: câu trả lời đúng trọng tâm câu hỏi.",
        "- Context Precision: context lấy ra thực sự liên quan.",
        "- Context Recall: context bao phủ thông tin cần thiết.",
    ]

    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to: {RESULTS_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} questions from {GOLDEN_DATASET_PATH.name}")

    questions, answers, contexts_list, ground_truths = collect_predictions(golden_dataset)

    print("\nRunning RAGAS evaluation...")
    scores = run_ragas(questions, answers, contexts_list, ground_truths)

    print_results_table(scores, n_samples=len(questions))
    export_results(scores, n_samples=len(questions))


if __name__ == "__main__":
    main()
