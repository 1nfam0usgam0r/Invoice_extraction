`# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Indian logistics invoice reconciliation tool. A transporter submits a combined PDF (approval form + bill table + tax invoice + LR documents). This tool extracts all fields, cross-checks bill rows against LR records, and produces a multi-sheet Excel report.

**Input:** Single combined PDF uploaded via browser  
**Output:** `.xlsx` workbook — Invoice sheet, Bill sheet, one sheet per LR, optional Tax Invoice sheet

---

## Running the app

```powershell
# Activate the venv first
.\venv\Scripts\Activate.ps1

# Start the Flask dev server
python app.py

# Run the preprocessing test
python test_preprocessing.py

# Debug bill extraction on a specific PDF (edit path inside)
python debug_bill.py

# Debug grid detection
python debug_grid.py
```

The server listens on `http://localhost:5000`. Upload a PDF via the browser UI; poll `/status/<job_id>` for progress.

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
Browser → POST /process → app.py
  → _run_pipeline() in daemon thread
      → pdf_to_images()         (ocr/pdf_handler.py)
      → extract_invoice()       (ocr/invoice_extractor.py)
      → detect_client()         (ocr/column_config.py)
      → extract_bill()          (ocr/bill_extractor.py)
      → extract_tax_invoice()   (ocr/tax_invoice_extractor.py)
      → extract_lr() × N        (ocr/lr_extractor.py)
      → normalise_bill_row/lr_record  (ocr/normaliser.py)
      → reconcile()             (ocr/reconciler.py)
      → write_excel()           (ocr/excel_writer.py)
Browser polls → GET /status/<job_id>
Browser → GET /download/<job_id>
```

`/process` returns immediately; the pipeline runs in a background thread. Progress is tracked via `job_store[job_id]['status']`, which the frontend polls.

### OCR layer (`ocr/reader.py`)

Single shared OCR engine: Tesseract via pytesseract. The module auto-detects `tesseract.exe` — first looks two levels up from `ocr/` (beside the project root, where the Windows installer drops it), then falls back to standard install paths. All extractors call `get_ocr_results(image_np, psm=...)` which returns phrase-level detections with boxes in EasyOCR's four-corner format.

### Bill extraction (`ocr/bill_extractor.py`)

Uses morphological line detection (OpenCV) to find the printed grid, extracts cell bounding boxes from intersections, then OCRs each cell individually with `psm=6`. The table spans two pages; both are read and their rows combined under the column names from page 1. Column identity is determined by `ocr/column_classifier.py` using fuzzy caption matching (rapidfuzz) and per-column validators.

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
| Bill OCR | ~10s |
| Tax Invoice OCR | ~20s (after parallel title scan) |
| LR OCR | ~30s (parallel) |
| Reconciling + Excel | <1s |
| **Total** | **~77s** (down from ~102s) |

## Output accuracy (sample PDF)

| Sheet | Quality |
|---|---|
| Invoice | Good — all 9 fields clean |
| Bill (31 rows) | ~60% — amounts and vehicle nos have systematic OCR errors (`B`/`$` misread as `8`, garbled alphanumerics) |
| Tax Invoice | ~30% — small scanned print, heavy noise |
| LR sheets | ~75% — dates/LR nos mostly correct; 4/31 sheets lost their LR number (fallback name `LR_page_N`) |

Reconciliation result: `clear: 4, mismatch: 20, missing_lr: 7, unbilled: 7` — high mismatch count reflects Bill OCR errors in vehicle numbers and amounts.

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
- Medium-priority issues not yet fixed: `job_store` has no thread lock (race condition on concurrent uploads), no concurrency cap on `ThreadPoolExecutor` (resource exhaustion under simultaneous jobs), no job timeout if Tesseract hangs, downloaded files named by UUID.

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
