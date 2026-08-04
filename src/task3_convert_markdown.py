"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install "markitdown[pdf]"
    # Lưu ý: cần extra [pdf] để convert được file PDF. Chỉ "pip install markitdown"
    # (không có extra) sẽ báo MissingDependencyException khi convert PDF, dù JSON/DOCX
    # vẫn convert bình thường.

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON, HTML)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
from pathlib import Path

import dotenv

dotenv.find_dotenv = lambda *args, **kwargs: ""
dotenv.load_dotenv = lambda *args, **kwargs: False

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

LEGAL_EXTENSIONS = {".pdf", ".docx", ".doc"}
NEWS_EXTENSIONS = {".json", ".html", ".htm"}


def _convert_file_with_markitdown(markitdown: MarkItDown, filepath: Path) -> str:
    result = markitdown.convert(str(filepath))
    content = getattr(result, "text_content", None) or getattr(result, "markdown", None) or ""
    return content.strip()


def _build_news_markdown(data: dict, filepath: Path) -> str:
    title = (data.get("title") or data.get("headline") or filepath.stem).strip()
    source = data.get("url") or data.get("source") or data.get("link") or "N/A"
    crawled = data.get("date_crawled") or data.get("published_at") or data.get("date") or "N/A"
    publisher = data.get("publisher") or data.get("site_name") or "N/A"
    topic = data.get("topic") or data.get("category") or "N/A"

    body = ""
    for key in ("content_markdown", "content", "text", "body", "description"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            body = value.strip()
            break

    if not body:
        body = "No content extracted from source file."

    return (
        f"# {title}\n\n"
        f"**Source:** {source}\n\n"
        f"**Publisher:** {publisher}\n\n"
        f"**Topic:** {topic}\n\n"
        f"**Crawled:** {crawled}\n\n"
        "---\n\n"
        f"{body.strip()}\n"
    )


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    markitdown = MarkItDown()
    converted_files = []
    failed_files = []

    for filepath in sorted(legal_dir.iterdir()):
        if not filepath.is_file() or filepath.suffix.lower() not in LEGAL_EXTENSIONS:
            continue

        try:
            markdown = _convert_file_with_markitdown(markitdown, filepath)
            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(markdown + "\n", encoding="utf-8")
            converted_files.append(output_path)
            print(f"  ✓ {filepath.name} -> {output_path.relative_to(OUTPUT_DIR.parent)}")
        except Exception as exc:
            failed_files.append((filepath.name, str(exc)))
            print(f"  ✗ {filepath.name}: {exc}")

    if converted_files:
        print("  Converted legal files:")
        for output_path in converted_files:
            print(f"    - {output_path.relative_to(OUTPUT_DIR.parent)}")

    if failed_files:
        print("  Failed legal files:")
        for filename, error in failed_files:
            print(f"    - {filename}: {error}")


def convert_news():
    """Convert JSON/HTML crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    markitdown = MarkItDown()
    converted_files = []
    failed_files = []

    for filepath in sorted(news_dir.iterdir()):
        if not filepath.is_file() or filepath.suffix.lower() not in NEWS_EXTENSIONS:
            continue

        try:
            output_path = output_dir / f"{filepath.stem}.md"

            if filepath.suffix.lower() == ".json":
                data = json.loads(filepath.read_text(encoding="utf-8"))
                markdown = _build_news_markdown(data, filepath)
            else:
                markdown = _convert_file_with_markitdown(markitdown, filepath)
                if not markdown:
                    markdown = f"# {filepath.stem}\n"

            output_path.write_text(markdown + "\n", encoding="utf-8")
            converted_files.append(output_path)
            print(f"  ✓ {filepath.name} -> {output_path.relative_to(OUTPUT_DIR.parent)}")
        except Exception as exc:
            failed_files.append((filepath.name, str(exc)))
            print(f"  ✗ {filepath.name}: {exc}")

    if converted_files:
        print("  Converted news files:")
        for output_path in converted_files:
            print(f"    - {output_path.relative_to(OUTPUT_DIR.parent)}")

    if failed_files:
        print("  Failed news files:")
        for filename, error in failed_files:
            print(f"    - {filename}: {error}")


def convert_news_articles():
    """Backward-compatible alias for older entrypoints."""
    convert_news()


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
