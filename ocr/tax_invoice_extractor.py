"""Read the Transportation Tax Invoice table: a real scanned document, not
the clean computer-typed "working sheet" ocr/bill_extractor.py reads.

Located by page title, not a fixed page index - see
``ocr.column_classifier.locate_pages_by_type`` and its ``PAGE_TYPE_SYNONYMS``
entry for ``"tax_invoice"``. Reuses bill_extractor.py's rule-finding,
cell-cropping and caption-reading machinery (none of it assumes anything
about which document it is pointed at), but with its own grid-detection
tuning and its own deskew step, because this scan needs both and
bill_extractor.py's page does not:

Confirmed on the sample scan: the horizontal rules' morphological-opening
projection peaked at 16.6% of the theoretical maximum (a long horizontal
kernel cannot survive a run that drifts vertically across the page, which a
skewed scan does even a little) - the same page deskewed first peaked at
88.2%, and grid detection went from finding 0 rules to finding all 29.
Column classification is ocr.column_classifier's tier system - the same
one bill_extractor.py uses, not a second copy - so this file adds no
per-field format assumptions of its own.
"""

import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image

try:
    from .bill_extractor import CELL_CONFIG, _group_lines, extract_cells, read_captions
    from .column_classifier import (
        classify_columns_by_content, filter_footer_rows, locate_pages_by_type,
        make_names_unique,
    )
    from .preprocessing import _deskew
    from .reader import TESSERACT_PATH  # noqa: F401  (sets tesseract_cmd)
except ImportError:  # running this file directly from inside ocr/
    from bill_extractor import CELL_CONFIG, _group_lines, extract_cells, read_captions
    from column_classifier import (
        classify_columns_by_content, filter_footer_rows, locate_pages_by_type,
        make_names_unique,
    )
    from preprocessing import _deskew
    from reader import TESSERACT_PATH  # noqa: F401

# Tuned against the sample scan's own rule strength - see the module
# docstring. Both ratios differ from bill_extractor.py's because this is a
# different document with a different table geometry and a genuine scan
# skew that page never has; nothing here assumes it also applies there.
H_KERNEL_RATIO = 0.12
H_THRESHOLD = 0.5
V_SPAN_RATIO = 0.20
V_THRESHOLD = 0.5
MIN_LINES = 3
GROUP_GAP = 10

# A data row fills nearly every column; the totals/footer rows below the
# table fill far fewer - the same generic shape bill_extractor.py's
# is_valid_data_row and column_classifier.filter_footer_rows already use,
# not a document-specific row count. No row count is assumed or checked
# anywhere in this file - however many rows the grid finds are read.
DATA_ROW_FILL = 0.6

# Longer than any real cell in this table (the longest genuine value is a
# vehicle number or a description, well under this) - past it, a cell is
# wrapped footer/notice text, not data. See is_valid_data_row.
MAX_DATA_CELL_LENGTH = 40

_DIGITS = re.compile(r"^\d+$")
_LONG_NUMBER = re.compile(r"\d{8}")

# This document's own company/address header block runs far deeper than the
# 4 rows bill_extractor.find_caption_row searches (its caption row is at
# index 8 on the sample scan, not near the top) - same scoring heuristic as
# that function (captions are short and mention no shipment/invoice number),
# just searched over more rows, since this table's header block genuinely
# is bigger, not because the caption itself is expected anywhere specific.
HEADER_SEARCH_ROWS = 15
HEADER_MAX_CELL = 25
HEADER_LONG_TOLERANCE = 0.2


def find_caption_row(cells: list) -> int:
    """Index of the row that names the columns - see
    ``bill_extractor.find_caption_row``, same heuristic, wider search."""
    best, best_score = 0, -1
    for index, row in enumerate(cells[:HEADER_SEARCH_ROWS]):
        filled = [cell for cell in row if cell]
        if not filled:
            continue
        if any(_LONG_NUMBER.search(cell) for cell in filled):
            continue
        long_cells = sum(1 for cell in filled if len(cell) > HEADER_MAX_CELL)
        if long_cells > len(filled) * HEADER_LONG_TOLERANCE:
            continue
        if len(filled) > best_score:
            best, best_score = index, len(filled)
    return best if best_score > 0 else 0


def _deskewed_rgb(image: Image.Image) -> np.ndarray:
    """The page, leveled, as an RGB array - what every downstream step in
    this file (grid detection, cell cropping, caption OCR) works from, so
    their pixel coordinates all agree with each other."""
    gray = np.array(image.convert("L"))
    return cv2.cvtColor(_deskew(gray), cv2.COLOR_GRAY2RGB)


# How much bigger than the table's own typical row spacing a gap has to be
# before the line on its far side is a stray detection (a border, a stamp
# edge) rather than the table's own next rule - a ratio of this table's own
# measured spacing, not a fixed pixel count.
OUTLIER_GAP_RATIO = 4.0


def _trim_trailing_outliers(lines: list) -> list:
    """Drop trailing lines whose gap from the previous one dwarfs the
    table's own row spacing - a line far below the last consistently-spaced
    row is not part of this table (on the sample scan, the real rows ran
    35-80px apart and one stray line turned up 1596px past the last of
    them, which then dragged vertical-line detection across the unruled
    footer text below the table along with it - see detect_grid)."""
    if len(lines) < 3:
        return lines
    gaps = [b - a for a, b in zip(lines, lines[1:])]
    median_gap = sorted(gaps)[len(gaps) // 2]
    if median_gap <= 0:
        return lines
    trimmed = list(lines)
    while len(trimmed) >= 3 and (trimmed[-1] - trimmed[-2]) > median_gap * OUTLIER_GAP_RATIO:
        trimmed.pop()
    return trimmed


def detect_grid(img_np: np.ndarray) -> tuple:
    """Find the table's printed rules - same algorithm as
    ``bill_extractor.detect_grid``, this document's own tuning (see the
    module docstring)."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )

    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(2, int(width * H_KERNEL_RATIO)), 1)
    )
    h_projection = np.sum(cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel), axis=1)
    h_lines = _group_lines(
        np.where(h_projection > width * H_KERNEL_RATIO * 255 * H_THRESHOLD)[0], GROUP_GAP
    )
    h_lines = _trim_trailing_outliers(h_lines)

    if len(h_lines) < MIN_LINES:
        return h_lines, []

    top, bottom = h_lines[0], h_lines[-1]
    band = binary[top:bottom, :]
    band_height = bottom - top

    v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(2, int(band_height * V_SPAN_RATIO)))
    )
    v_projection = np.sum(cv2.morphologyEx(band, cv2.MORPH_OPEN, v_kernel), axis=0)
    v_lines = _group_lines(
        np.where(v_projection > band_height * V_SPAN_RATIO * 255 * V_THRESHOLD)[0], GROUP_GAP
    )

    if len(v_lines) < 2:
        return h_lines, []
    return h_lines, v_lines


def is_valid_data_row(row_dict: dict) -> bool:
    """A row is a record rather than a blank line or the totals row: its
    serial cell is numeric, or (the serial having been lost to the scan)
    the row still fills most of its columns. Same shape as
    ``bill_extractor.is_valid_data_row`` - not a copy of its per-field
    assumptions, since this table has a different column set, only of the
    "a data row is dense, a totals/footer row is sparse" heuristic."""
    if not row_dict:
        return False
    values = list(row_dict.values())

    # A real data cell is a date, a code, a name, a number - never a
    # sentence. The footer below this table is mostly notice/terms
    # paragraphs, and the grid sometimes still finds enough rule-like
    # structure there to hand back a "row" - one with a stray digit
    # OCR'd into its first cell, which would otherwise pass the digit-
    # serial check below unconditionally. A row holding a long, wrapped
    # cell is that footer text, whatever its first cell looks like.
    if any(len(str(value or "")) > MAX_DATA_CELL_LENGTH for value in values):
        return False

    serial = str(values[0] or "").strip()
    if _DIGITS.match(serial):
        return True
    # Not blank-or-digit falls through to the same density check either
    # way, rather than only when blank: this scan's serial column reads as
    # short garbage ("Ba", "&", "rs") on many otherwise-clean rows, not
    # cleanly empty - a row that fills nearly every other column is still
    # data regardless of which failure mode its own serial cell hit.
    return sum(1 for value in values if value) >= len(values) * DATA_ROW_FILL


def _read_page(image: Image.Image) -> tuple:
    """``(deskewed_rgb, h_lines, v_lines, cells)`` for one page, or
    ``(deskewed_rgb, [], [], [])`` if it carries no table this file's grid
    detection can find."""
    deskewed = _deskewed_rgb(image)
    h_lines, v_lines = detect_grid(deskewed)
    if not h_lines or not v_lines:
        return deskewed, [], [], []
    cells = extract_cells(deskewed, h_lines, v_lines)
    return deskewed, h_lines, v_lines, cells


def extract_tax_invoice(pages: list) -> tuple:
    """Locate and read every Transportation Tax Invoice page in ``pages``.

    Args:
        pages: Every page of the combined PDF, in document order - not
            pre-sliced to a known range, since the page(s) are found by
            title (see ``locate_pages_by_type``), not a fixed index.

    Returns:
        ``(header, rows)``. ``header`` is empty - this table carries no
        label/value block above it the way the bill table does, only the
        caption row itself. ``rows`` is one dict per data row, keyed by
        column name (``ocr.column_classifier``'s tiered resolver - no
        per-field format assumed here), covering every Transportation Tax
        Invoice page found and however many rows its table turns out to
        hold, in order. Empty if no such page is found at all.
    """
    page_indices = locate_pages_by_type(pages, "tax_invoice")
    if not page_indices:
        return {}, []

    # The table can run onto the page(s) immediately after a located title
    # page, the same way bill_extractor.py's own table spans two pages -
    # kept only as long as this file's own grid detection still finds a
    # table there and that page did not start a new Transportation Tax
    # Invoice of its own (already in page_indices).
    ordered_pages = sorted(page_indices)
    to_read = list(ordered_pages)
    # A continuation page must have roughly the same column count as the
    # title page it follows - "has some grid at all" is not enough to tell
    # this table continuing from an unrelated one starting right after it
    # (on the sample PDF, individual LR receipts immediately follow and
    # each has its own small, unrelated table; checking for a grid alone
    # absorbed eight of them as "more of the tax invoice" before this).
    COLUMN_COUNT_TOLERANCE = 2
    for start in ordered_pages:
        _, _, start_v_lines, _ = _read_page(pages[start])
        expected_columns = len(start_v_lines)
        cursor = start + 1
        while cursor < len(pages) and cursor not in to_read:
            _, h_lines, v_lines, _ = _read_page(pages[cursor])
            if not h_lines or not v_lines:
                break
            if abs(len(v_lines) - expected_columns) > COLUMN_COUNT_TOLERANCE:
                break
            to_read.append(cursor)
            cursor += 1
    to_read.sort()

    column_names = None
    all_rows = []
    for index in to_read:
        deskewed, h_lines, v_lines, cells = _read_page(pages[index])
        if not cells:
            continue

        if column_names is None:
            caption = find_caption_row(cells)
            captions = read_captions(deskewed, h_lines, v_lines, caption)
            classified = classify_columns_by_content(
                filter_footer_rows(cells[caption + 1:]), captions
            )
            column_names = make_names_unique(
                [classified.get(i, f"col_{i + 1}") for i in range(len(v_lines) - 1)]
            )
            print(f"tax invoice ocr captions: {captions}")
            print(f"tax invoice columns: {column_names}")
            all_rows.extend(cells[caption + 1:])
        else:
            # A continuation page repeats no caption of its own.
            all_rows.extend(cells)

    if column_names is None:
        return {}, []

    rows = []
    for row in filter_footer_rows(all_rows):
        record = {
            name: (row[i].strip() or None) if i < len(row) else None
            for i, name in enumerate(column_names)
        }
        # Every genuine row on this table prints an amount; nothing below
        # the table does. Applied after is_valid_data_row rather than
        # folded into it, since it depends on a specific column existing
        # (amount, whatever column ended up named that) rather than the
        # generic row shape that function checks.
        if is_valid_data_row(record) and record.get("amount"):
            rows.append(record)

    return {}, rows
