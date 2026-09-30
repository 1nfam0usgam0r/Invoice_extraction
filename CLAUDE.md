# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Indian logistics invoice reconciliation tool. A transporter submits a combined PDF (approval form + bill table + tax invoice + LR documents). This tool extracts all fields, cross-checks bill rows against LR records, and produces a multi-sheet Excel report.

**Input:** Single combined PDF uploaded via browser  
**Output:** `.xlsx` workbook — Invoice sheet, Bill sheet, one sheet per LR, optional Tax Invoice sheet

---

## Running the app

```cmd
cd C:\Users\1nfam\Downloads\Invoice_extraction-update

:: Activate the venv
venv\Scripts\activate.bat

:: Start the Flask dev server
python app.py

:: Run the preprocessing test
python test_preprocessing.py

:: Debug bill extraction on a specific PDF (edit path inside)
python debug_bill.py

:: Debug grid detection
python debug_grid.py
```

The server listens on `http://localhost:5000`. Upload a PDF via the browser UI; poll `/status/<job_id>` for progress.

---

## GitHub

Repository: https://github.com/1nfam0usgam0r/Invoice_extraction

Branches: `main` (stable — do not touch), `update` (active development)

```cmd
cd C:\Users\1nfam\Downloads\Invoice_extraction-update

:: First-time setup — connect local folder to the remote
git init
git remote add origin https://github.com/1nfam0usgam0r/Invoice_extraction.git
git checkout -b update

:: Push changes to the update branch
git add .
git commit -m "your message here"
git push origin update
```

---

## Rules for Claude

**Before reporting any bug, Claude must have read the relevant lines in its own context — not delegated to a subagent.**

- Every bug report must quote the exact line(s) from the file.
- Subagents may be used to locate or search files, not to conclude what bugs exist.
- No bug list is presented to the user until each item has been personally verified by reading the source.

---

## Architecture

### Request flow

```
Browser → POST /upload → app.py          (saves PDF, creates job, returns job_id)
Browser → POST /process/<job_id> → app.py (starts pipeline in background thread)
  → _run_pipeline() in daemon thread
      → pdf_to_images()         (ocr/pdf_handler.py)
      → extract_invoice()       (ocr/invoice_extractor.py)
      → detect_client()         (ocr/column_config.py)
      → extract_bill()          (ocr/bill_extractor.py)
      → extract_tax_invoice()   (ocr/tax_invoice_extractor.py)
      → extract_lr() × N        (ocr/lr_extractor.py)
      → ink_separation()        (ocr/ink_separation.py) — handwritten ack box per LR
      → normalise_bill_row/lr_record  (ocr/normaliser.py)
      → reconcile()             (ocr/reconciler.py)
      → write_excel()           (ocr/excel_writer.py)
Browser polls  → GET /status/<job_id>
Browser        → GET /download/<job_id>
Browser        → GET /debug/<job_id>     (raw Tesseract detections for bill page)
GET /health                              (liveness check)
```

`POST /upload` returns immediately with a `job_id`. `POST /process/<job_id>` starts the pipeline in a daemon thread and also returns immediately. Progress is tracked via `job_store[job_id]['status']`, which the frontend polls.

### OCR layer (`ocr/reader.py`)

Single shared OCR engine: Tesseract via pytesseract. The module auto-detects `tesseract.exe` — first looks two levels up from `ocr/` (beside the project root, where the Windows installer drops it), then falls back to standard install paths. All extractors call `get_ocr_results(image_np, psm=...)` which returns phrase-level detections with boxes in EasyOCR's four-corner format.

### Bill extraction (`ocr/bill_extractor.py`)

Uses morphological line detection (OpenCV) to find the printed grid, extracts cell bounding boxes from intersections, then OCRs each cell individually with `psm=6`. The table spans two pages; both are read and their rows combined under the column names from page 1. Column identity is determined by `ocr/column_classifier.py` using fuzzy caption matching (rapidfuzz) and per-column validators.

### Ink separation (`ocr/ink_separation.py`)

Splits a scanned LR page into a printed layer and a coloured-ink layer (rubber stamps, blue/red signatures). Reads the "CUSTOMER ACKNOWLEDGEMENT DETAILS" box — received weight, date, remark — off the ink layer so handwritten entries are read without the printed form underneath interfering. Grayscale pages are detected and passed through unchanged. Called by `lr_extractor.py`, not directly by the pipeline.

### LR extractor legacy (`ocr/lr_extractor_legacy.py`)

Earlier implementation of LR extraction kept for reference. Not called by the pipeline — `ocr/lr_extractor.py` is the active one.

### Client/column configuration (`ocr/column_config.py`)

`detect_client()` fuzzy-matches the invoice text to identify the transporter. `get_columns(client_id)` returns the ordered column spec for that client's bill layout. Adding a new client means adding an entry here.

### Tax invoice (`ocr/tax_invoice_extractor.py`)

Located by scanning every page for its printed title (via `column_classifier.locate_pages_by_type`), not a fixed page slice — handed all pages, not a pre-sliced range. The continuation-page logic relies on LR pages being present in the full list as natural column-count stops; narrowing the list caused it to absorb unrelated pages as extra rows.

### Page layout (hardcoded to sample PDF)

```python
INVOICE_PAGES = slice(0, 1)   # page 1: approval form
BILL_PAGES    = slice(1, 3)   # pages 2-3: bill table
LR_PAGES      = slice(5, 36)  # pages 6-36: LR receipts
```

Defined in `app.py:77–79`. A different client's PDF structure will silently extract wrong data.

---

## Key dependencies

| Package | Purpose |
|---------|---------|
| PyMuPDF (`fitz`) | PDF → image rendering |
| pytesseract | OCR engine wrapper |
| opencv-contrib-python | Image preprocessing, grid detection |
| rapidfuzz | Fuzzy caption/field matching |
| openpyxl | Excel output |
| flask | Web server |

**Installed OpenCV version: 5.0.0** — `cv2.minAreaRect` returns angles in `[0, 90)` not `[-90, 0)`. The deskew fix in `ocr/preprocessing.py:51–56` handles both conventions.

---

## Measured performance (sample PDF — 38 pages, 31 LRs)

| Stage | Time |
|---|---|
| Converting + Invoice OCR | ~10s |
| Bill OCR | ~10s (known-client single data pass + 16 workers) |
| Tax Invoice OCR | ~20s (after parallel title scan) |
| LR OCR | ~30s (8 capped workers — unlimited caused Windows scheduling jitter) |
| Reconciling + Excel | <1s |
| **Total** | **~70s** (occasionally ~80s due to OS scheduling outliers) |

## Output accuracy (sample PDF)

| Sheet | Quality |
|---|---|
| Invoice | Good — all 9 fields clean |
| Bill (31 rows) | ~70–75% — systematic numeric misreads largely fixed by column whitelists; vehicle_no still has garbled alphanumeric reads |
| Tax Invoice | ~30% — small scanned print, heavy noise |
| LR sheets | ~75% — dates/LR nos mostly correct; 4/31 sheets lost their LR number (fallback name `LR_page_N`) |

Reconciliation result: `clear: 7, mismatch: 17, missing_lr: 7, unbilled: 7` — improvement from baseline (clear: 4, mismatch: 20) due to whitelist and vehicle normalisation fixes.

---

## Performance optimisations applied

1. **`app.py`** — LR OCR parallelised: replaced sequential `[extract_lr(img) for img in lr_images]` with `ThreadPoolExecutor.map`. 31 Tesseract subprocesses now run concurrently (each releases the GIL). ~4–5× faster on the LR stage.

2. **`ocr/column_classifier.py`** — `locate_pages_by_type` parallelised: replaced sequential per-page loop with `ThreadPoolExecutor.map`. All 38 page title-band scans now run concurrently. Tax invoice stage dropped from ~50s to ~20s.

3. **Tax invoice page narrowing — attempted and reverted.** Passing only non-LR pages to `extract_tax_invoice` caused the continuation-page logic to absorb pages 36–37 as extra tax invoice rows (28 → 65 rows found). The LR pages in the full list act as natural column-count stops for that logic; removing them broke it. No net time saving was measured either. Reverted.

---

## Known limitations

- Page layout hardcoded — a different client's PDF structure silently extracts wrong data (`app.py:77–79`). A runtime warning is now emitted if the page count doesn't cover the expected slices.
- GST rate not extracted from document — tax invoice totals must be verified manually.
- `job_store` is in-memory only, capped at 50 jobs — no persistence across restarts.
- Medium-priority issues not yet fixed: `job_store` has no thread lock (race condition on concurrent uploads), no concurrency cap on `ThreadPoolExecutor` (resource exhaustion under simultaneous jobs), no job timeout if Tesseract hangs, downloaded files use a short UUID prefix (`extraction_<8chars>.xlsx`) rather than a meaningful name.

---

## Accuracy improvements applied

1. **`ocr/column_config.py` + `ocr/bill_extractor.py`** — per-column Tesseract character whitelists: numeric columns (`amount`, `rate`, `gross_qty`, `balance_pay`, etc.) now restrict OCR to digits and punctuation, eliminating `B`/`$`/`S` misread as `8`/`8`/`5`. Text columns (`bill_no`, `delivery_date`, `party_name`, `delivery_station`) left unrestricted. `extract_cells()` now accepts `column_whitelists`, `row_start`, and `row_end` parameters. For known clients, `extract_bill()` reads only the first `HEADER_SEARCH_ROWS` rows (cheap) to locate the caption row, then does a single whitelist-restricted pass over data rows — skipping the full two-pass approach. Unknown clients still use two passes. Page 2+ read once directly with whitelists.

2. **`ocr/normaliser.py` — `normalise_vehicle_no()`** — position-aware OCR correction for Indian vehicle plates (`SS DD LLL NNNN` format): digits corrected to letters at state-code positions (0–1), letters corrected to digits at district (2–3) and serial (last 4) positions. Only visually ambiguous characters are substituted (`0/O`, `1/I`, `5/S`, `6/G`, `8/B`, `2/Z`). Lifted reconciliation from `clear: 4` to `clear: 7`.

3. **`ocr/bill_extractor.py`** — `CELL_WORKERS` increased from 8 to 16: fills more CPU cores during parallel cell OCR, saving ~2s on the bill stage.

4. **`app.py`** — LR `ThreadPoolExecutor` capped at 8 workers: previously unlimited (defaulted to ~35 on this machine), which caused Windows process-scheduling jitter and ~10s variance between runs. Capping at 8 gives consistent ~70s total runtime.

---

## Bugs fixed (history)

1. **`ocr/pdf_handler.py`** — fitz file handle leak: wrapped render loop in `try/finally` so `doc.close()` is always called.
2. **`ocr/preprocessing.py`** — deskew wrong on OpenCV 5.0: added `elif angle > 45: angle -= 90` to handle the new angle convention.
3. **`app.py`** — temp files leaked on pipeline error: added `_discard` calls to the except block.
4. **`templates/index.html` + `app.py`** — frontend stage names didn't match backend: updated `STATUS_TO_STEP` to use `invoice_ocr`, `bill_ocr`, `tax_invoice_ocr`, `lr_ocr`.
5. **`app.py`** — pipeline blocked the server: moved `_run_pipeline` to a `daemon=True` background thread.
6. **`app.py`** — `job_store` grew forever: added `_evict_old_jobs()` with a 50-job cap.
7. **`app.py`** — validator disabled: re-enabled `validate_bill_row` from `ocr/validator.py` after reconciliation. Checks arithmetic (`gross_qty × rate = amount`) and field shapes per row.
8. **`ocr/excel_writer.py` — `_write_bill_sheet`** — no row highlighting: bill rows with `validation_flags` are now coloured yellow. `validation_flags` is stripped from the data columns before writing.
9. **`ocr/excel_writer.py` — `_write_lr_sheet`** — silent empty LR sheets: sheets where `lr_no` could not be read now show a red warning cell at the top.
10. **`app.py` — `_run_pipeline`** — no layout warning: after rendering, page count is checked against hardcoded slices; mismatches are written to `job['layout_warnings']` and to the Invoice sheet header in red.
11. **`ocr/excel_writer.py` — `_autofit`** — crash on merged cells: `MergedCell` objects have no `column_letter`; added `hasattr` guard to skip them.
