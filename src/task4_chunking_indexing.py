"""Task 4 - chunk standardized documents and build a persistent Chroma index.

Run from the project root with::

    python -m src.task4_chunking_indexing

The implementation uses character-based recursive chunking because Vietnamese
legal documents mix Markdown headings, numbered articles, paragraphs, and long
tables.  An 800-character limit keeps citations focused, while a 100-character
overlap preserves article/clause context across chunk boundaries.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
CHROMA_DIR = PROJECT_DIR / "chroma_db"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"

# OpenAI keeps the indexing environment lightweight while providing a capable
# multilingual embedding model. The same helper must be reused by Task 5.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
EMBEDDING_BATCH_SIZE = 128

VECTOR_STORE = "chromadb"
COLLECTION_NAME = "vietnam_labor_law_docs"
UPSERT_BATCH_SIZE = 100


def _document_title(content: str, fallback: str) -> str:
    """Return the first Markdown heading without storing large metadata."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title[:500]
    return fallback


def load_documents() -> list[dict]:
    """Load every non-empty Markdown document in deterministic path order."""
    if not STANDARDIZED_DIR.exists():
        raise FileNotFoundError(f"Standardized data directory does not exist: {STANDARDIZED_DIR}")

    documents: list[dict] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "source_path": relative_path.as_posix(),
                    "document_id": md_file.stem,
                    "title": _document_title(content, md_file.stem),
                    "type": doc_type,
                },
            }
        )
    return documents


def _get_splitter():
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise RuntimeError(
            "Missing langchain-text-splitters. Install Task 4 dependencies from requirements.txt."
        ) from exc

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", "; ", ", ", " ", ""],
        keep_separator=True,
        strip_whitespace=True,
    )


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split documents and attach deterministic citation/index metadata."""
    splitter = _get_splitter()
    chunks: list[dict] = []

    for document in documents:
        content = document.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        base_metadata = dict(document.get("metadata", {}))
        splits = [text for text in splitter.split_text(content) if text.strip()]
        chunk_count = len(splits)

        for chunk_index, chunk_text in enumerate(splits):
            fingerprint = hashlib.sha256(
                f"{base_metadata.get('source_path', '')}\0{chunk_index}\0{chunk_text}".encode("utf-8")
            ).hexdigest()
            chunks.append(
                {
                    "id": fingerprint,
                    "content": chunk_text,
                    "metadata": {
                        **base_metadata,
                        "chunk_index": chunk_index,
                        "chunk_count": chunk_count,
                    },
                }
            )
    return chunks


@lru_cache(maxsize=1)
def get_embedding_model():
    """Create one OpenAI client for indexing and semantic search."""
    load_dotenv(PROJECT_DIR / ".env")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Missing openai. Install Task 4 dependencies from requirements.txt."
        ) from exc

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings")
    return OpenAI()


def embed_texts(texts: Iterable[str], *, show_progress_bar: bool = False) -> list[list[float]]:
    """Embed passages or queries with the model shared by Tasks 4 and 5."""
    text_list = list(texts)
    if not text_list:
        return []

    client = get_embedding_model()
    embeddings: list[list[float]] = []
    total_batches = (len(text_list) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
    for batch_number, start in enumerate(
        range(0, len(text_list), EMBEDDING_BATCH_SIZE), start=1
    ):
        batch = text_list[start : start + EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        batch_embeddings = [item.embedding for item in ordered]
        if len(batch_embeddings) != len(batch):
            raise ValueError(
                f"OpenAI returned {len(batch_embeddings)} embeddings for {len(batch)} texts"
            )
        if any(len(embedding) != EMBEDDING_DIM for embedding in batch_embeddings):
            dimensions = sorted({len(embedding) for embedding in batch_embeddings})
            raise ValueError(
                f"Unexpected embedding dimensions {dimensions}; expected {EMBEDDING_DIM}"
            )
        embeddings.extend(batch_embeddings)
        if show_progress_bar:
            print(f"Embedded batch {batch_number}/{total_batches}", flush=True)
    return embeddings


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Return copied chunk dictionaries containing 1536-dimensional vectors."""
    embeddings = embed_texts(
        (chunk["content"] for chunk in chunks),
        show_progress_bar=bool(chunks),
    )
    return [{**chunk, "embedding": embedding} for chunk, embedding in zip(chunks, embeddings)]


def get_chroma_client():
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("Missing chromadb. Install Task 4 dependencies from requirements.txt.") from exc

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    """Open the existing Task 4 collection for downstream retrieval."""
    return get_chroma_client().get_collection(name=COLLECTION_NAME)


def _batched(items: list[dict], size: int) -> Iterable[list[dict]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def index_to_vectorstore(chunks: list[dict]):
    """Replace the collection atomically at collection scope and index chunks."""
    if not chunks:
        raise ValueError("Cannot create a vector index from zero chunks")
    if any("embedding" not in chunk for chunk in chunks):
        raise ValueError("All chunks must contain an embedding before indexing")

    client = get_chroma_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception as exc:
        # Chroma versions expose different not-found exception classes.
        if "does not exist" not in str(exc).lower() and "not found" not in str(exc).lower():
            raise

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": EMBEDDING_DIM,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
    )

    for batch in _batched(chunks, UPSERT_BATCH_SIZE):
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=[chunk["metadata"] for chunk in batch],
        )
    return collection


def run_pipeline():
    """Run load -> chunk -> embed -> persistent Chroma indexing."""
    load_dotenv(PROJECT_DIR / ".env")
    provider = os.getenv("EMBEDDING_PROVIDER", "openai")
    if provider != "openai":
        raise ValueError(
            "Checkpoint 2 is configured for EMBEDDING_PROVIDER=openai "
            f"with {EMBEDDING_MODEL}; received {provider!r}"
        )

    print("=" * 60)
    print("Task 4: Vietnamese Labor Law Chunking & Chroma Indexing")
    print(f"Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"Vector store: {VECTOR_STORE} -> {CHROMA_DIR}")
    print("=" * 60)

    documents = load_documents()
    print(f"Loaded {len(documents)} documents")

    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    embedded_chunks = embed_chunks(chunks)
    print(f"Embedded {len(embedded_chunks)} chunks")

    collection = index_to_vectorstore(embedded_chunks)
    print(f"Indexed {collection.count()} chunks in collection {COLLECTION_NAME!r}")
    return collection


if __name__ == "__main__":
    run_pipeline()
