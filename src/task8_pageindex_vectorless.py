"""Task 8 - PageIndex vectorless retrieval for Vietnamese labour-law PDFs.

PageIndex's cloud SDK accepts PDF documents. This module uploads the canonical
labour-law PDF(s), stores their document IDs locally, and exposes
``pageindex_search`` for Task 9's low-confidence fallback path.

Environment variables:
    PAGEINDEX_API_KEY       Required API key (``pix_...``).
    PAGEINDEX_DOC_ID        Optional single pre-uploaded document ID.
    PAGEINDEX_DOC_IDS       Optional comma-separated document IDs.
    PAGEINDEX_UPLOAD_GLOB   PDF selection pattern. Defaults to ``*hop-nhat*.pdf``.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent.parent
LANDING_LEGAL_DIR = PROJECT_DIR / "data" / "landing" / "legal"
MANIFEST_PATH = PROJECT_DIR / "pageindex_doc_ids.json"

DEFAULT_UPLOAD_GLOB = "*hop-nhat*.pdf"
DOCUMENT_READY_TIMEOUT = 900.0
RETRIEVAL_TIMEOUT = 120.0
POLL_INTERVAL = 3.0

load_dotenv(PROJECT_DIR / ".env")


def _get_api_key() -> str:
    api_key = os.getenv("PAGEINDEX_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "PAGEINDEX_API_KEY is not configured. Add it to .env before using Task 8."
        )
    return api_key


def _get_client():
    try:
        from pageindex import PageIndexClient
    except ImportError as exc:
        raise RuntimeError(
            "Missing pageindex SDK. Install it with: python -m pip install pageindex"
        ) from exc
    return PageIndexClient(api_key=_get_api_key())


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"documents": []}
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid PageIndex manifest: {MANIFEST_PATH}") from exc
    if not isinstance(manifest.get("documents"), list):
        raise RuntimeError(f"PageIndex manifest has invalid schema: {MANIFEST_PATH}")
    return manifest


def _save_manifest(manifest: dict) -> None:
    temp_path = MANIFEST_PATH.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(MANIFEST_PATH)


def _selected_pdfs() -> list[Path]:
    pattern = os.getenv("PAGEINDEX_UPLOAD_GLOB", DEFAULT_UPLOAD_GLOB).strip()
    if not pattern:
        pattern = DEFAULT_UPLOAD_GLOB
    pdf_files = sorted(path for path in LANDING_LEGAL_DIR.glob(pattern) if path.is_file())
    if not pdf_files:
        raise FileNotFoundError(
            f"No legal PDF matched {pattern!r} in {LANDING_LEGAL_DIR}. "
            "Set PAGEINDEX_UPLOAD_GLOB=*.pdf to upload every legal PDF."
        )
    return pdf_files


def _wait_until_document_ready(client: Any, doc_id: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.is_retrieval_ready(doc_id):
            return
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(
        f"PageIndex document {doc_id} was not retrieval-ready after {timeout:.0f}s"
    )


def upload_documents(
    *,
    force: bool = False,
    wait_until_ready: bool = True,
    ready_timeout: float = DOCUMENT_READY_TIMEOUT,
) -> list[dict]:
    """Upload selected legal PDFs and persist their PageIndex document IDs.

    Existing manifest entries are reused unless ``force=True``. By default the
    canonical consolidated Labour Code PDF is selected; set
    ``PAGEINDEX_UPLOAD_GLOB=*.pdf`` to upload the full legal corpus.
    """
    if ready_timeout <= 0:
        raise ValueError("ready_timeout must be positive")

    client = _get_client()
    manifest = _load_manifest()
    documents = manifest["documents"]
    by_source = {
        record.get("source_path"): record
        for record in documents
        if isinstance(record, dict) and record.get("source_path")
    }
    selected_records: list[dict] = []

    for pdf_path in _selected_pdfs():
        relative_path = pdf_path.relative_to(PROJECT_DIR).as_posix()
        existing = by_source.get(relative_path)

        if existing and existing.get("doc_id") and not force:
            record = existing
            print(f"REUSE {pdf_path.name} -> {record['doc_id']}")
        else:
            print(f"UPLOAD {pdf_path.name}")
            response = client.submit_document(str(pdf_path))
            doc_id = response.get("doc_id") or response.get("id")
            if not doc_id:
                raise RuntimeError(
                    f"PageIndex upload response did not contain doc_id: {response}"
                )
            record = {
                "source_path": relative_path,
                "filename": pdf_path.name,
                "doc_id": str(doc_id),
                "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            if existing:
                documents[documents.index(existing)] = record
            else:
                documents.append(record)
            by_source[relative_path] = record
            _save_manifest(manifest)
            print(f"  SAVED doc_id={doc_id}")

        if wait_until_ready:
            print(f"  WAIT processing {record['doc_id']} ...")
            _wait_until_document_ready(client, record["doc_id"], ready_timeout)
            print("  READY")
        selected_records.append(record)

    return selected_records


def _configured_document_ids() -> list[str]:
    raw_ids = [
        os.getenv("PAGEINDEX_DOC_ID", ""),
        os.getenv("PAGEINDEX_DOC_IDS", ""),
    ]
    document_ids: list[str] = []
    for raw_value in raw_ids:
        document_ids.extend(part.strip() for part in raw_value.split(",") if part.strip())

    if not document_ids:
        document_ids.extend(
            str(record["doc_id"])
            for record in _load_manifest()["documents"]
            if isinstance(record, dict) and record.get("doc_id")
        )

    # De-duplicate while preserving document priority/order.
    return list(dict.fromkeys(document_ids))


def _wait_for_retrieval(client: Any, retrieval_id: str) -> dict:
    from pageindex import PageIndexAPIError

    deadline = time.monotonic() + RETRIEVAL_TIMEOUT
    while time.monotonic() < deadline:
        try:
            response = client.get_retrieval(retrieval_id)
        except PageIndexAPIError as exc:
            # The legacy API can return a retrieval_id before the asynchronous
            # task is visible to GET /retrieval/{id}. Retry only this transient
            # eventual-consistency response; auth, quota, and other API errors
            # must still fail immediately.
            if "retrieval task not found" in str(exc).lower():
                time.sleep(POLL_INTERVAL)
                continue
            raise
        status = str(response.get("status", "")).lower()

        # Some API versions omit status once retrieved_nodes is available.
        if response.get("retrieved_nodes") is not None or status in {
            "completed",
            "complete",
            "succeeded",
            "success",
        }:
            return response
        if status in {"failed", "error", "cancelled", "canceled"}:
            raise RuntimeError(
                f"PageIndex retrieval {retrieval_id} ended with status {status}: {response}"
            )
        time.sleep(POLL_INTERVAL)

    raise TimeoutError(
        f"PageIndex retrieval {retrieval_id} did not finish after {RETRIEVAL_TIMEOUT:.0f}s"
    )


def _iter_relevant_items(value: Any) -> Iterator[dict]:
    """Yield relevant-content objects from PageIndex's nested response shape."""
    if isinstance(value, list):
        for item in value:
            yield from _iter_relevant_items(item)
    elif isinstance(value, dict):
        if value.get("relevant_content") or value.get("content"):
            yield value
        else:
            for child in value.values():
                if isinstance(child, (list, dict)):
                    yield from _iter_relevant_items(child)


def _parse_retrieval(response: dict, doc_id: str) -> list[dict]:
    parsed: list[dict] = []
    seen_content: set[str] = set()

    for node in response.get("retrieved_nodes") or []:
        if not isinstance(node, dict):
            continue
        node_metadata = {
            "doc_id": doc_id,
            "node_id": node.get("node_id"),
            "node_title": node.get("title") or node.get("section_title"),
            "page_index": node.get("page_index"),
        }
        for item in _iter_relevant_items(node.get("relevant_contents", [])):
            content = str(item.get("relevant_content") or item.get("content") or "").strip()
            if not content or content in seen_content:
                continue
            seen_content.add(content)
            parsed.append(
                {
                    "content": content,
                    "score": 0.0,  # Filled globally from PageIndex result rank.
                    "metadata": {
                        **node_metadata,
                        "section": item.get("section_title") or node_metadata["node_title"],
                    },
                    "source": "pageindex",
                }
            )
    return parsed


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve structurally relevant content from configured PageIndex PDFs."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    document_ids = _configured_document_ids()
    if not document_ids:
        raise RuntimeError(
            "No PageIndex document ID is configured. Run upload_documents() first, "
            "or set PAGEINDEX_DOC_ID/PAGEINDEX_DOC_IDS in .env."
        )

    client = _get_client()
    combined: list[dict] = []
    seen_content: set[str] = set()

    for doc_id in document_ids:
        if not client.is_retrieval_ready(doc_id):
            raise RuntimeError(
                f"PageIndex document {doc_id} is not ready for retrieval yet"
            )
        submitted = client.submit_query(doc_id=doc_id, query=query.strip())
        retrieval_id = submitted.get("retrieval_id") or submitted.get("id")
        if not retrieval_id:
            raise RuntimeError(
                f"PageIndex query response did not contain retrieval_id: {submitted}"
            )
        response = _wait_for_retrieval(client, str(retrieval_id))
        for result in _parse_retrieval(response, doc_id):
            if result["content"] not in seen_content:
                seen_content.add(result["content"])
                combined.append(result)

    # Legacy PageIndex retrieval has no comparable numeric relevance score.
    # Use reciprocal rank so downstream components still receive a stable,
    # descending score without pretending it is cosine similarity.
    for rank, result in enumerate(combined, start=1):
        result["score"] = round(1.0 / rank, 6)
    return combined[:top_k]


if __name__ == "__main__":
    records = upload_documents()
    print(f"\nConfigured {len(records)} PageIndex document(s).")

    test_query = "Thời gian thử việc tối đa và lương thử việc là bao nhiêu?"
    print(f"\nTest query: {test_query}")
    for result in pageindex_search(test_query, top_k=3):
        print(f"[{result['score']:.3f}] {result['content'][:120]}...")
