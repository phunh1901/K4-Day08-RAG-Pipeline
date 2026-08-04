"""Task 2 - Crawl cac bai huong dan ve quyen loi nguoi lao dong.

Chu de nhom: Tro ly hoi dap Luat Lao dong cho nguoi tre.

Script crawl 5 bai viet cong khai tu Cong Thong tin dien tu Chinh phu va luu
moi bai thanh mot file JSON trong ``data/landing/news/``. Cac file JSON nay la
input truc tiep cua Task 3 (convert sang Markdown).

Chuan bi moi truong::

    python -m pip install crawl4ai
    python -m playwright install chromium

Chay tu thu muc goc cua project::

    python -m src.task2_crawl_news
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "landing" / "news"
MIN_CONTENT_CHARS = 500

# Dung slug co dinh de file output co y nghia va khong bi trung ten khi chay lai.
ARTICLES = [
    {
        "slug": "01_quyen_loi_khi_nghi_viec",
        "expected_title": "Nguoi lao dong nghi viec duoc nhan nhung khoan tien nao?",
        "url": (
            "https://xaydungchinhsach.chinhphu.vn/"
            "nguoi-lao-dong-nghi-viec-duoc-nhan-nhung-khoan-tien-nao-"
            "119240509063307003.htm"
        ),
    },
    {
        "slug": "02_xu_phat_vi_pham_tien_luong_gio_lam_viec",
        "expected_title": (
            "Nguoi su dung lao dong vi pham ve tien luong, gio lam viec, "
            "gio nghi ngoi, xu phat the nao?"
        ),
        "url": (
            "https://xaydungchinhsach.chinhphu.vn/"
            "nguoi-su-dung-lao-dong-vi-pham-ve-tien-luong-gio-lam-viec-"
            "gio-nghi-ngoi-xu-phat-the-nao-119260717182001954.htm"
        ),
    },
    {
        "slug": "03_muc_luong_toi_thieu",
        "expected_title": (
            "Nghi dinh 293/2025/ND-CP quy dinh muc luong toi thieu doi voi "
            "nguoi lao dong lam viec theo hop dong lao dong"
        ),
        "url": (
            "https://xaydungchinhsach.chinhphu.vn/"
            "nghi-dinh-so-293-2025-nd-cp-quy-dinh-muc-luong-toi-thieu-doi-voi-"
            "nguoi-lao-dong-lam-viec-theo-hop-dong-lao-dong-"
            "119251110172808433.htm"
        ),
    },
    {
        "slug": "04_can_cu_ty_le_dong_bhxh",
        "expected_title": "Quy dinh can cu, ty le dong BHXH",
        "url": (
            "https://xaydungchinhsach.chinhphu.vn/"
            "quy-dinh-can-cu-ty-le-dong-bhxh-119240726084122076.htm"
        ),
    },
    {
        "slug": "05_che_do_om_dau_dai_ngay",
        "expected_title": (
            "Nguoi mac benh dai ngay duoc huong che do om dau theo ngay lam viec"
        ),
        "url": (
            "https://xaydungchinhsach.chinhphu.vn/"
            "nguoi-mac-benh-dai-ngay-duoc-huong-che-do-om-dau-theo-ngay-lam-viec-"
            "119260627081647202.htm"
        ),
    },
]

# Giu ten bien cua starter code de nhung module khac (neu co) van import duoc.
ARTICLE_URLS = [article["url"] for article in ARTICLES]


def setup_directory() -> None:
    """Tao thu muc output neu chua ton tai."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _markdown_to_text(markdown: Any) -> str:
    """Chuan hoa output Markdown giua cac phien ban Crawl4AI.

    Crawl4AI ban cu tra ``result.markdown`` la chuoi; ban moi co the tra object
    chua ``raw_markdown``. Ham nay ho tro ca hai dang.
    """
    if isinstance(markdown, str):
        return markdown.strip()

    raw_markdown = getattr(markdown, "raw_markdown", None)
    if isinstance(raw_markdown, str):
        return raw_markdown.strip()

    return str(markdown or "").strip()


async def crawl_article(
    url: str,
    expected_title: str = "Unknown",
    crawler: Any | None = None,
) -> dict:
    """Crawl mot URL va tra ve record san sang de luu JSON.

    Neu ``crawler`` duoc truyen vao, ham tai su dung browser session do. Neu goi
    rieng ham nay, mot session tam thoi se duoc tao va dong tu dong.
    """
    from crawl4ai import AsyncWebCrawler

    if crawler is None:
        async with AsyncWebCrawler() as owned_crawler:
            return await crawl_article(url, expected_title, owned_crawler)

    result = await crawler.arun(url=url)
    success = getattr(result, "success", True)
    status_code = getattr(result, "status_code", None)

    if not success:
        error = getattr(result, "error_message", "Crawl4AI khong tra ve ly do")
        raise RuntimeError(f"Crawl that bai: {error}")
    if status_code is not None and not 200 <= int(status_code) < 300:
        raise RuntimeError(f"HTTP status khong hop le: {status_code}")

    content = _markdown_to_text(getattr(result, "markdown", ""))
    if len(content) < MIN_CONTENT_CHARS:
        raise ValueError(
            f"Noi dung chi co {len(content)} ky tu; co the trang bi chan hoac "
            "chua render xong"
        )

    metadata = getattr(result, "metadata", None) or {}
    title = metadata.get("title") if isinstance(metadata, dict) else None

    return {
        "url": url,
        "title": title or expected_title,
        "publisher": "Cong Thong tin dien tu Chinh phu",
        "topic": "Luat Lao dong va quyen loi nguoi lao dong",
        "date_crawled": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"
        ),
        "content_markdown": content,
    }


def save_article(article: dict, filepath: Path) -> None:
    """Luu JSON UTF-8 theo cach atomic de tranh file do neu chuong trinh bi ngat."""
    temp_path = filepath.with_suffix(filepath.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(article, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(filepath)


async def crawl_all() -> None:
    """Crawl toan bo 5 bai, tiep tuc bai sau neu mot bai gap loi."""
    from crawl4ai import AsyncWebCrawler

    setup_directory()
    failures: list[str] = []

    async with AsyncWebCrawler() as crawler:
        for index, article_config in enumerate(ARTICLES, start=1):
            url = article_config["url"]
            print(f"[{index}/{len(ARTICLES)}] Crawling: {url}")

            try:
                article = await crawl_article(
                    url=url,
                    expected_title=article_config["expected_title"],
                    crawler=crawler,
                )
                filepath = DATA_DIR / f"{article_config['slug']}.json"
                save_article(article, filepath)
                print(
                    f"  OK Saved: {filepath.name} "
                    f"({len(article['content_markdown']):,} chars)"
                )
            except Exception as exc:  # Ghi nhan loi tung URL, khong bo ca batch.
                message = f"{article_config['slug']}: {exc}"
                failures.append(message)
                print(f"  ERROR {message}")

    succeeded = len(ARTICLES) - len(failures)
    print(f"\nKet qua: {succeeded}/{len(ARTICLES)} bai crawl thanh cong.")

    if failures:
        details = "\n  - ".join(failures)
        raise RuntimeError(f"Co {len(failures)} bai crawl that bai:\n  - {details}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
