"""Task 1 - collect the Vietnamese labor-law source corpus.

The corpus intentionally uses original files published by official government
sources.  Run from the project root with::

    python -m src.task1_collect_legal_docs

The generated ``manifest.json`` records the legal identifier, effective-state
notes, official landing page, checksum, and retrieval priority for every file.
Expired instruments are not mixed into this active corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import requests


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "landing" / "legal"
MANIFEST_PATH = DATA_DIR / "manifest.json"
MIN_FILE_SIZE = 1_024


@dataclass(frozen=True)
class LegalDocument:
    filename: str
    identifier: str
    document_type: str
    title: str
    issuing_body: str
    issued_date: str
    effective_date: str | None
    legal_status: str
    retrieval_priority: int
    topics: tuple[str, ...]
    landing_page_url: str
    download_url: str
    note: str = ""


# Status was checked against the official pages on 2026-08-04.  A RAG answer
# must still state an "as of" date because Vietnamese law can change.
ACTIVE_DOCUMENTS: tuple[LegalDocument, ...] = (
    LegalDocument(
        filename="bo-luat-lao-dong-hop-nhat-18-vbhn-vpqh-2026.pdf",
        identifier="18/VBHN-VPQH",
        document_type="Văn bản hợp nhất",
        title="Văn bản hợp nhất Bộ luật Lao động",
        issuing_body="Văn phòng Quốc hội",
        issued_date="2026-02-12",
        effective_date=None,
        legal_status="current_consolidated_text",
        retrieval_priority=100,
        topics=("all", "employment_contract", "probation", "wages", "overtime", "leave", "termination", "discipline", "labor_disputes"),
        landing_page_url="https://congbao.chinhphu.vn/van-ban/van-ban-hop-nhat-so-18-vbhn-vpqh-468971/62878.htm",
        download_url="https://congbaocdn.chinhphu.vn/180507251028987904/2026/3/5/468971-1772684381_v1_1772690833_signed.pdf",
        note="Preferred source for the current text of the Labor Code.",
    ),
    LegalDocument(
        filename="bo-luat-lao-dong-45-2019-qh14.pdf",
        identifier="45/2019/QH14",
        document_type="Bộ luật",
        title="Bộ luật Lao động",
        issuing_body="Quốc hội",
        issued_date="2019-11-20",
        effective_date="2021-01-01",
        legal_status="partially_effective_use_consolidated_text",
        retrieval_priority=90,
        topics=("all", "employment_contract", "probation", "wages", "overtime", "leave", "termination", "discipline", "labor_disputes"),
        landing_page_url="https://vanban.chinhphu.vn/?classid=1&docid=198540&pageid=27160&typegroupid=3",
        download_url="https://datafiles.chinhphu.vn/cpp/files/vbpq/2019/12/45.signed.pdf",
        note="Original enacted text; prefer 18/VBHN-VPQH when provisions differ.",
    ),
    LegalDocument(
        filename="nghi-dinh-145-2020-nd-cp.pdf",
        identifier="145/2020/NĐ-CP",
        document_type="Nghị định",
        title="Quy định chi tiết Bộ luật Lao động về điều kiện lao động và quan hệ lao động",
        issuing_body="Chính phủ",
        issued_date="2020-12-14",
        effective_date="2021-02-01",
        legal_status="effective",
        retrieval_priority=95,
        topics=("employment_contract", "termination", "labor_records", "overtime", "rest", "female_employees", "labor_disputes"),
        landing_page_url="https://vanban.chinhphu.vn/?docid=201967&pageid=27160",
        download_url="https://datafiles.chinhphu.vn/cpp/files/vbpq/2020/12/145.signed.pdf",
    ),
    LegalDocument(
        filename="nghi-dinh-12-2022-nd-cp-xu-phat-lao-dong.pdf",
        identifier="12/2022/NĐ-CP",
        document_type="Nghị định",
        title="Xử phạt vi phạm hành chính trong lĩnh vực lao động và bảo hiểm xã hội",
        issuing_body="Chính phủ",
        issued_date="2022-01-17",
        effective_date="2022-01-17",
        legal_status="effective",
        retrieval_priority=90,
        topics=("penalties", "employment_contract", "wages", "overtime", "termination", "social_insurance"),
        landing_page_url="https://vanban.chinhphu.vn/?classid=1&docid=205182&orggroupid=2&pageid=27160",
        download_url="https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/01/12-2022-nd.signed.pdf",
    ),
    LegalDocument(
        filename="nghi-dinh-135-2020-nd-cp-tuoi-nghi-huu.pdf",
        identifier="135/2020/NĐ-CP",
        document_type="Nghị định",
        title="Quy định về tuổi nghỉ hưu",
        issuing_body="Chính phủ",
        issued_date="2020-11-18",
        effective_date="2021-01-01",
        legal_status="effective",
        retrieval_priority=80,
        topics=("retirement",),
        landing_page_url="https://vanban.chinhphu.vn/?pageid=27160&docid=201650",
        download_url="https://datafiles.chinhphu.vn/cpp/files/vbpq/2020/11/135.signed.pdf",
    ),
    LegalDocument(
        filename="nghi-dinh-293-2025-nd-cp-luong-toi-thieu.pdf",
        identifier="293/2025/NĐ-CP",
        document_type="Nghị định",
        title="Mức lương tối thiểu đối với người lao động làm việc theo hợp đồng lao động",
        issuing_body="Chính phủ",
        issued_date="2025-11-10",
        effective_date="2026-01-01",
        legal_status="effective",
        retrieval_priority=100,
        topics=("minimum_wage", "wages"),
        landing_page_url="https://vanban.chinhphu.vn/?classid=1&docid=215832&pageid=27160",
        download_url="https://datafiles.chinhphu.vn/cpp/files/vbpq/2025/11/293-cp.signed.pdf",
        note="Current minimum-wage decree; replaces 74/2024/NĐ-CP.",
    ),
    LegalDocument(
        filename="nghi-dinh-219-2025-nd-cp-lao-dong-nuoc-ngoai.pdf",
        identifier="219/2025/NĐ-CP",
        document_type="Nghị định",
        title="Quy định về người lao động nước ngoài làm việc tại Việt Nam",
        issuing_body="Chính phủ",
        issued_date="2025-08-07",
        effective_date="2025-08-07",
        legal_status="effective",
        retrieval_priority=85,
        topics=("foreign_workers", "work_permits"),
        landing_page_url="https://vanban.chinhphu.vn/?classid=1&docid=214840&orggroupid=2&pageid=27160",
        download_url="https://datafiles.chinhphu.vn/cpp/files/vbpq/2025/8/219-cp.signed.pdf",
        note="Current foreign-worker decree; replaces 152/2020/NĐ-CP and 70/2023/NĐ-CP.",
    ),
    LegalDocument(
        filename="thong-tu-09-2020-tt-bldtbxh-lao-dong-chua-thanh-nien.pdf",
        identifier="09/2020/TT-BLĐTBXH",
        document_type="Thông tư",
        title="Hướng dẫn Bộ luật Lao động về lao động chưa thành niên",
        issuing_body="Bộ Lao động - Thương binh và Xã hội",
        issued_date="2020-11-12",
        effective_date="2021-03-15",
        legal_status="partially_effective_with_amendment",
        retrieval_priority=80,
        topics=("minor_workers", "prohibited_work", "night_work"),
        landing_page_url="https://congbao.chinhphu.vn/van-ban/thong-tu-so-09-2020-tt-bldtbxh-33378.htm",
        download_url="https://g7.cdnchinhphu.vn/api/download/stream?Url=tm-8mq6BhNw0NbrKRhTDAQWsKg3tuqaY0aWypnY78U6M2BY68Ekp0Gvvr483flbRU4_c2kBEkEfTqlW7vXheR0xqkH9QqXR9Us5mEl1UZzbQ_oXqM-B7D4LNYJ4bCJZB-pvZXZk5kKUghSt39sV9-A~~",
        note="Read together with 08/2023/TT-BLĐTBXH.",
    ),
    LegalDocument(
        filename="thong-tu-08-2023-tt-bldtbxh-sua-doi-thu-tuc-cu-tru.pdf",
        identifier="08/2023/TT-BLĐTBXH",
        document_type="Thông tư",
        title="Sửa đổi quy định về giấy tờ cư trú trong thủ tục lao động",
        issuing_body="Bộ Lao động - Thương binh và Xã hội",
        issued_date="2023-08-29",
        effective_date="2023-10-12",
        legal_status="effective_amending_instrument",
        retrieval_priority=75,
        topics=("minor_workers", "administrative_procedures", "residence_documents"),
        landing_page_url="https://congbao.chinhphu.vn/van-ban/thong-tu-so-08-2023-tt-bldtbxh-40176.htm",
        download_url="https://g7.cdnchinhphu.vn/api/download/stream?Url=tm-8mq6BhNw0NbrKRhTDAQWsKg3tuqaY0aWypnY78U6M2BY68Ekp0Gvvr483flbRrbhNI-0YPlhGdNENPfHiNgvuUJuGUc_k9Dd4qCZtxlfMaz_S1IK0zY_g625r6_8ZlSNP4KZ_az8-X0TgooxWuA~~",
    ),
    LegalDocument(
        filename="thong-tu-11-2020-tt-bldtbxh-nghe-nang-nhoc-phan-1.pdf",
        identifier="11/2020/TT-BLĐTBXH (phần 1)",
        document_type="Thông tư",
        title="Danh mục nghề, công việc nặng nhọc, độc hại, nguy hiểm - phần 1",
        issuing_body="Bộ Lao động - Thương binh và Xã hội",
        issued_date="2020-11-12",
        effective_date="2021-03-01",
        legal_status="effective",
        retrieval_priority=75,
        topics=("hazardous_work", "occupational_safety", "early_retirement"),
        landing_page_url="https://congbao.chinhphu.vn/van-ban/thong-tu-so-11-2020-tt-bldtbxh-33183.htm",
        download_url="https://g7.cdnchinhphu.vn/api/download/stream?Url=tm-8mq6BhNw0NbrKRhTDAQWsKg3tuqaY0aWypnY78U6M2BY68Ekp0Gvvr483flbR3V6g9zoWzV_xWsiAMUSeMjj9D3QsKP6daKGEYkJiTCINKc6rbqi7xQ9mrovZRYKqjmPHMnfADny3_PK3Zss0_w~~",
        note="The official Gazette publishes this long annex in two files.",
    ),
    LegalDocument(
        filename="thong-tu-11-2020-tt-bldtbxh-nghe-nang-nhoc-phan-2.pdf",
        identifier="11/2020/TT-BLĐTBXH (phần 2)",
        document_type="Thông tư",
        title="Danh mục nghề, công việc nặng nhọc, độc hại, nguy hiểm - phần 2",
        issuing_body="Bộ Lao động - Thương binh và Xã hội",
        issued_date="2020-11-12",
        effective_date="2021-03-01",
        legal_status="effective",
        retrieval_priority=75,
        topics=("hazardous_work", "occupational_safety", "early_retirement"),
        landing_page_url="https://congbao.chinhphu.vn/van-ban/thong-tu-so-11-2020-tt-bldtbxh-33183/34674.htm",
        download_url="https://g7.cdnchinhphu.vn/api/download/stream?Url=tm-8mq6BhNw0NbrKRhTDAQWsKg3tuqaY0aWypnY78U6M2BY68Ekp0Gvvr483flbR3V6g9zoWzV_xWsiAMUSeMmgtkOUqYOclPUzP4CwPiOkGLN4ZUvrRkezpY6CMlyZIzH44-bh8Hl4-VnE05sPiyg~~",
        note="Continuation of the official Gazette document.",
    ),
    LegalDocument(
        filename="thong-tu-18-2021-tt-bldtbxh-thoi-gio-lam-viec-thoi-vu.pdf",
        identifier="18/2021/TT-BLĐTBXH",
        document_type="Thông tư",
        title="Thời giờ làm việc, nghỉ ngơi đối với công việc thời vụ và gia công theo đơn đặt hàng",
        issuing_body="Bộ Lao động - Thương binh và Xã hội",
        issued_date="2021-12-15",
        effective_date="2022-02-01",
        legal_status="effective",
        retrieval_priority=75,
        topics=("working_hours", "rest", "overtime", "seasonal_work"),
        landing_page_url="https://vanban.chinhphu.vn/?docid=204826&pageid=27160",
        download_url="https://datafiles.chinhphu.vn/cpp/files/vbpq/2021/12/18-bldtbxh.pdf",
    ),
)


def setup_directory() -> None:
    """Create the landing directory if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_pdf(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= MIN_FILE_SIZE:
        return False
    with path.open("rb") as file_obj:
        if file_obj.read(5) != b"%PDF-":
            return False
        # A network interruption can leave a large file with a valid header.
        # PDF writers place %%EOF close to the end of a complete document.
        file_obj.seek(max(0, path.stat().st_size - 2_048))
        return b"%%EOF" in file_obj.read()


def download_file(document: LegalDocument, force: bool = False) -> Path:
    """Download one official PDF atomically and reject HTML/error responses."""
    destination = DATA_DIR / document.filename
    if not force and _valid_pdf(destination):
        print(f"= Existing: {destination.name}")
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    headers = {"User-Agent": "K4-Day08-RAG-Pipeline/1.0 (educational legal corpus)"}
    try:
        with requests.get(document.download_url, headers=headers, stream=True, timeout=(15, 90)) as response:
            response.raise_for_status()
            with temporary.open("wb") as file_obj:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file_obj.write(chunk)
        if not _valid_pdf(temporary):
            raise ValueError(f"download is not a valid PDF or is too small: {document.download_url}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"+ Downloaded: {destination.name} ({destination.stat().st_size:,} bytes)")
    return destination


def write_manifest(documents: Iterable[LegalDocument]) -> None:
    """Write machine-readable provenance and retrieval metadata."""
    records = []
    for document in documents:
        path = DATA_DIR / document.filename
        record = asdict(document)
        record["topics"] = list(document.topics)
        record.update(
            {
                "retrieval_enabled": True,
                "local_path": str(path.relative_to(PROJECT_DIR)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
        records.append(record)

    manifest = {
        "topic": "Trợ lý hỏi đáp Luật Lao động Việt Nam",
        "jurisdiction": "Vietnam",
        "language": "vi",
        "last_verified": date.today().isoformat(),
        "status_policy": "Active/current instruments only. Prefer consolidated text and later amending instruments.",
        "citation_policy": "Cite identifier plus article/clause/point and source URL; include an as-of date.",
        "document_count": len(records),
        "documents": records,
        "excluded_historical_documents": [
            {
                "identifier": "17/2022/UBTVQH15",
                "title": "Temporary COVID-19 overtime-hours resolution",
                "reason": "Expired after 2022; excluded to prevent obsolete 60-hour/month rules from being presented as current law.",
                "official_url": "https://congbao.chinhphu.vn/detail/tai-ve?id=36956&slug=17-2022-ubtvqh15",
            },
            {
                "identifier": "74/2024/NĐ-CP",
                "title": "Former minimum-wage decree",
                "reason": "Replaced by 293/2025/NĐ-CP from 2026-01-01.",
            },
            {
                "identifier": "152/2020/NĐ-CP and 70/2023/NĐ-CP",
                "title": "Former foreign-worker rules",
                "reason": "Replaced by 219/2025/NĐ-CP from 2025-08-07.",
            },
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"+ Manifest: {MANIFEST_PATH.relative_to(PROJECT_DIR)}")


def collect_all(force: bool = False) -> list[Path]:
    """Download and validate the complete active corpus."""
    setup_directory()
    downloaded = [download_file(document, force=force) for document in ACTIVE_DOCUMENTS]
    write_manifest(ACTIVE_DOCUMENTS)
    return downloaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official Vietnamese labor-law documents")
    parser.add_argument("--force", action="store_true", help="redownload valid existing files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    files = collect_all(force=args.force)
    print(f"Done: {len(files)} active legal documents in {DATA_DIR}")
