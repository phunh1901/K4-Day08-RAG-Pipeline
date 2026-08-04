# Vietnamese labor-law landing corpus

This directory is Role 2's Checkpoint 1 / Task 1 deliverable for Topic 1.

- Run `python -m src.task1_collect_legal_docs` to reproduce the downloads.
- `manifest.json` is the source-of-truth for provenance, legal status, topics,
  checksums, and citation metadata.
- Prefer `18/VBHN-VPQH` (2026 consolidated Labor Code) over the original
  `45/2019/QH14` text when the two differ.
- Current wage and foreign-worker answers must use `293/2025/NĐ-CP` and
  `219/2025/NĐ-CP`, respectively.
- Historical or superseded instruments listed under
  `excluded_historical_documents` are deliberately not indexed.

Suggested legal citation format:

> Điểm a khoản 1 Điều 25 Bộ luật Lao động (Văn bản hợp nhất số
> 18/VBHN-VPQH, xác thực ngày 12/02/2026).

Every answer should include an “as of” date and should not present the chatbot
as a substitute for advice from a qualified lawyer or competent authority.
