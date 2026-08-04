"""Reproducible RAGAS A/B evaluation for the Vietnamese labour-law RAG app.

Config A uses the production hybrid pipeline (dense + BM25 + RRF + PageIndex
fallback). Config B is a dense-only baseline. Both configs generate answers
from exactly the contexts recorded for RAGAS, preventing evidence mismatch.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from src.task10_generation import generate_with_citation
from src.task5_semantic_search import semantic_search
from src.task9_retrieval_pipeline import retrieve


BASE_DIR = Path(__file__).parent
GOLDEN_DATASET_PATH = BASE_DIR / "golden_dataset.json"
RESULTS_PATH = BASE_DIR / "results.md"
ARTIFACT_PATH = BASE_DIR / "evaluation_artifacts.json"
METRICS = [faithfulness, answer_relevancy, context_recall, context_precision]
METRIC_NAMES = [metric.name for metric in METRICS]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_golden_dataset() -> list[dict[str, str]]:
    data = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) < 15:
        raise ValueError("golden_dataset.json must contain at least 15 Q&A pairs")
    required = {"question", "expected_answer", "expected_context"}
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f"Golden item {index} is missing required fields")
        if any(not str(item[key]).strip() for key in required):
            raise ValueError(f"Golden item {index} contains an empty field")
    return data


def _hybrid(question: str, top_k: int) -> list[dict]:
    return retrieve(
        question, top_k=top_k, use_reranking=True, use_query_expansion=True
    )


def _dense_only(question: str, top_k: int) -> list[dict]:
    return [
        {**item, "source": "dense_only"}
        for item in semantic_search(question, top_k=top_k)
    ]


CONFIGS: dict[str, Callable[[str, int], list[dict]]] = {
    "hybrid_rrf_fallback": _hybrid,
    "dense_only": _dense_only,
}


def collect_predictions(
    golden_dataset: list[dict[str, str]], config_name: str, top_k: int = 5
) -> dict[str, list]:
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config {config_name!r}; choose from {list(CONFIGS)}")
    payload: dict[str, list] = {
        "question": [], "answer": [], "contexts": [], "ground_truth": [], "sources": []
    }
    retriever = CONFIGS[config_name]
    for index, item in enumerate(golden_dataset, start=1):
        question = item["question"].strip()
        print(f"[{config_name} {index:02d}/{len(golden_dataset)}] {question[:70]}", flush=True)
        chunks = retriever(question, top_k)
        if not chunks:
            raise RuntimeError(f"{config_name} returned no context for item {index}")
        result = generate_with_citation(
            question, top_k=top_k, context_chunks=chunks
        )
        payload["question"].append(question)
        payload["answer"].append(result["answer"])
        payload["contexts"].append([str(chunk["content"]) for chunk in chunks])
        payload["ground_truth"].append(item["expected_answer"])
        payload["sources"].append([
            str((chunk.get("metadata") or {}).get("source") or
                (chunk.get("metadata") or {}).get("filename") or chunk.get("source", "unknown"))
            for chunk in chunks
        ])
    return payload


def evaluate_predictions(payload: dict[str, list]) -> tuple[dict[str, float], list[dict]]:
    dataset = Dataset.from_dict({key: payload[key] for key in (
        "question", "answer", "contexts", "ground_truth"
    )})
    result = evaluate(dataset, metrics=METRICS, raise_exceptions=True)
    summary = {name: float(result[name]) for name in METRIC_NAMES}
    score_rows = result.scores.to_list()
    rows: list[dict] = []
    for index, scores in enumerate(score_rows):
        numeric = {name: float(scores.get(name, 0.0)) for name in METRIC_NAMES}
        rows.append({
            "question": payload["question"][index],
            "answer": payload["answer"][index],
            "sources": payload["sources"][index],
            **numeric,
            "average": statistics.fmean(numeric.values()),
        })
    return summary, rows


def compare_configs(
    golden_dataset: list[dict[str, str]], top_k: int = 5
) -> dict[str, dict[str, Any]]:
    comparison: dict[str, dict[str, Any]] = {}
    for config_name in CONFIGS:
        payload = collect_predictions(golden_dataset, config_name, top_k)
        comparison[config_name] = {"predictions": payload, "status": "predictions_ready"}
        ARTIFACT_PATH.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary, rows = evaluate_predictions(payload)
        comparison[config_name] = {"summary": summary, "rows": rows, "predictions": payload}
        ARTIFACT_PATH.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return comparison


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _diagnose(row: dict[str, Any]) -> str:
    """Turn the weakest metric into a concise, evidence-based failure stage."""
    if row["context_recall"] < 0.75:
        return "Retrieval thiếu một phần điều kiện/ngoại lệ trong expected answer"
    if row["faithfulness"] < 0.70:
        return "Generation chưa bám đủ chặt vào các mệnh đề trong context"
    if row["answer_relevancy"] < 0.60:
        return "Answer chưa trả lời trực diện theo cách diễn đạt của câu hỏi"
    if row["context_precision"] < 0.75:
        return "Top-k còn context nhiễu; cần rerank/chunking tốt hơn"
    return "Sai lệch nhỏ giữa answer sinh và ground truth"


def export_results(comparison: dict[str, dict[str, Any]], sample_count: int) -> None:
    config_a = comparison["hybrid_rrf_fallback"]
    config_b = comparison["dense_only"]
    a_scores, b_scores = config_a["summary"], config_b["summary"]
    a_avg = statistics.fmean(a_scores.values())
    b_avg = statistics.fmean(b_scores.values())
    display = {
        "faithfulness": "Faithfulness", "answer_relevancy": "Answer Relevancy",
        "context_recall": "Context Recall", "context_precision": "Context Precision",
    }
    lines = [
        "# RAGAS Evaluation Results", "",
        f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Golden questions: **{sample_count}**", "- top_k: **5**",
        "- Config A: **Hybrid + BM25 + RRF + PageIndex fallback**",
        "- Config B: **Dense-only cosine baseline**", "",
        "## A/B scores", "", "| Metric | Config A | Config B | Δ A−B |",
        "|---|---:|---:|---:|",
    ]
    for key in METRIC_NAMES:
        lines.append(
            f"| {display[key]} | {_fmt(a_scores[key])} | {_fmt(b_scores[key])} | "
            f"{a_scores[key] - b_scores[key]:+.4f} |"
        )
    lines.extend([
        f"| **Average** | **{_fmt(a_avg)}** | **{_fmt(b_avg)}** | **{a_avg-b_avg:+.4f}** |",
        "", "## Analysis", "",
        "Config A combines exact Vietnamese legal keywords from BM25 with semantic matches, "
        "then fuses ranks using RRF. Config B is the controlled baseline and uses only cosine similarity.",
        "The delta table above is generated from the same golden set and LLM judge, so the comparison is reproducible.",
        "", "## Worst performers — Config A", "",
        "| # | Question | Faithfulness | Relevancy | Recall | Precision | Average | Root cause |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ])
    worst = sorted(config_a["rows"], key=lambda row: row["average"])[:3]
    for index, row in enumerate(worst, start=1):
        lines.append(
            f"| {index} | {row['question'].replace('|', '/')} | {_fmt(row['faithfulness'])} | "
            f"{_fmt(row['answer_relevancy'])} | {_fmt(row['context_recall'])} | "
            f"{_fmt(row['context_precision'])} | {_fmt(row['average'])} | {_diagnose(row)} |"
        )
    lines.extend([
        "", "## Root-cause analysis and recommendations", "",
        "1. Inspect the saved contexts for the bottom-three questions in `evaluation_artifacts.json`; add missing statutes when recall is low.",
        "2. Use Markdown heading-aware chunking for long statutes to avoid splitting an article from its conditions and exceptions.",
        "3. Keep RRF for rank fusion, but use original cosine similarity—not RRF score—for PageIndex fallback decisions.",
        "4. Expand short or colloquial questions before dense retrieval; retain the original query for citations and evaluation.",
        "", "## Reproduce", "", "```powershell",
        "python -m group_project.evaluation.eval_pipeline --top-k 5", "```", "",
        "Raw per-question predictions, contexts, sources and metric scores are stored in `evaluation_artifacts.json`.",
    ])
    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run two-config RAGAS evaluation")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=15, help="Use <15 only for debugging")
    args = parser.parse_args()
    if args.top_k <= 0 or args.limit <= 0:
        parser.error("--top-k and --limit must be positive")
    golden = load_golden_dataset()[: args.limit]
    comparison = compare_configs(golden, top_k=args.top_k)
    export_results(comparison, len(golden))
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
