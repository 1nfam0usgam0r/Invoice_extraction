"""Extract the bill table by finding its printed grid and reading each cell.

The table is ruled, so the rules themselves are the most reliable thing on the
page: morphology finds the horizontal and vertical lines, their intersections
give the cells, and each cell is read on its own as a single line of text. That
replaces img2table, which collapsed rows wholesale on these scans.

The table runs across two pages - captions and rows 1-26 on the first, the rest
and the totals line on the second - so both are read and their rows combined
under the column names the first page prints.
"""

import os

# Kept from the img2table setup: anything that shells out to the binary finds
# it on PATH, which the Windows installer does not set. Resolved from this
# file's location rather than hardcoded, so the checkout can live on any drive.
# ocr/ -> project/ -> Invoice Extraction/, where tesseract.exe sits.
os.environ["PATH"] = (
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    + os.pathsep
    + os.environ.get("PATH", "")
)

import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytesseract
from PIL import Image

try:
    from .column_classifier import (
        validate_amount, validate_date, validate_delivery_timestamp,
        validate_gross_qty, validate_shipment_no, validate_sr_no,
        validate_vehicle_no,
    )
    from .column_config import get_columns, get_whitelists
    from .invoice_extractor import extract_label_values, ocr_segments, preprocess
    from .reader import TESSERACT_PATH  # noqa: F401  (sets tesseract_cmd)
except ImportError:   # running this file directly from inside ocr/
    from column_classifier import (
        validate_amount, validate_date, validate_delivery_timestamp,
        validate_gross_qty, validate_shipment_no, validate_sr_no,
        validate_vehicle_no,
    )
    from column_config import get_columns, get_whitelists
    from invoice_extractor import extract_label_values, ocr_segments, preprocess
    from reader import TESSERACT_PATH  # noqa: F401

# Width of the opening kernel used to find horizontal rules, as a fraction of
# the page width. The 0.3 the debug script uses misses two faint rules on the
# first bill page, merging four rows into two - the giveaway is a 155px gap in
# a table whose rows are otherwise a uniform 77px apart. Halving the kernel
# recovers them and changes nothing on the second page. Threshold and dilation
# make no difference; only this does.
H_KERNEL_RATIO = 0.2
H_THRESHOLD = 0.5

# Column rules are looked for only between the top and bottom rules of the
# table, and must run this much of that height. Searching the whole page
# instead picks up the scanner's edge artefacts and the odd tall glyph, which
# is what made the two pages disagree on how many columns they had.
V_SPAN_RATIO = 0.6
V_THRESHOLD = 0.5

# Detections this close to the page edge are the scan border, not a rule.
EDGE_MARGIN = 5

# Rules within this many pixels of each other are one rule, thickened by the
# scan.
GROUP_GAP = 10

# Cells are cropped inside their rules by this much, so the border does not end
# up in the OCR. An inset of 3 leaves a sliver of the rule in the crop and
# Tesseract reads it as a pipe or a letter: on the first page's serial column,
# 3 read 12 of 26 values correctly against 23 at this inset. Anything from 10
# to 16 scores the same; past 20 it starts clipping the digits.
CELL_INSET = 12

# Below this the inset is eating the cell, so it is backed off.
MIN_CELL_SIDE = 8

# psm 6 rather than the 7 a single line suggests: measured over the same
# column, 6 read 23 of 26 and 7 read 22, and 8 collapsed to almost nothing.
CELL_CONFIG = "--psm 6 --oem 3"

# One tesseract process per cell is the cost of this approach: ~122ms each,
# which is 51 seconds for the first page alone. The calls are independent and
# spend their time waiting on a subprocess, so they run in a pool - 8 workers
# brings that page to 15 seconds.
CELL_WORKERS = 8

# A table needs at least this many rules each way before it is a table.
MIN_LINES = 3

# Metadata above the table is read down to the first horizontal rule; if no
# grid is found at all, this much of the page is used instead.
HEADER_FRACTION = 0.3

# The caption row is looked for among the first few rows rather than assumed to
# be the first. The bill prints a thin band above it carrying the bill date,
# which the grid picks up as a row of its own.
HEADER_SEARCH_ROWS = 4
HEADER_MAX_CELL = 25
HEADER_LONG_TOLERANCE = 0.2

# The caption row is re-read on its own, magnified, off the original page
# rather than the thresholded one. Measured against the fifteen captions on the
# sample page, scoring exact and partial matches out of 30: this scores 11,
# where 4x with psm 8 - single word - scores 2. Captions wrap onto two lines
# inside their cell, which is what psm 6 handles and the single-line and
# single-word modes do not.
CAPTION_SCALE = 2
CAPTION_CONFIG = "--psm 6 --oem 3"
CAPTION_PAD = 20

# A data row fills nearly every column. The totals line fills about a third,
# which is what separates the two when the serial number does not read.
DATA_ROW_FILL = 0.6

_DIGITS = re.compile(r"^\d+$")
_LONG_NUMBER = re.compile(r"\d{8}")


def _group_lines(positions, gap: int = GROUP_GAP) -> list:
    """Collapse runs of adjacent pixel rows/columns into one line each."""
    grouped = []
    if len(positions) == 0:
        return grouped
    start = previous = positions[0]
    for position in positions[1:]:
        if position - previous > gap:
            grouped.append(int((start + previous) // 2))
            start = position
        previous = position
    grouped.append(int((start + previous) // 2))
    return grouped


def detect_grid(img_np: np.ndarray) -> tuple:
    """Find the table's printed rules.

    Args:
        img_np: RGB page as a numpy array.

    Returns:
        ``(h_lines, v_lines)`` - sorted Y positions of the horizontal rules and
        X positions of the vertical ones, one entry per rule. Empty lists when
        the page carries no table.
    """
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
        np.where(h_projection > width * H_KERNEL_RATIO * 255 * H_THRESHOLD)[0]
    )

    if len(h_lines) < MIN_LINES:
        return h_lines, []

    # Only the band the table occupies, so page-edge artefacts cannot qualify.
    top, bottom = h_lines[0], h_lines[-1]
    band = binary[top:bottom, :]
    band_height = bottom - top

    v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(2, int(band_height * V_SPAN_RATIO)))
    )
    v_projection = np.sum(cv2.morphologyEx(band, cv2.MORPH_OPEN, v_kernel), axis=0)
    v_lines = _group_lines(
        np.where(v_projection > band_height * V_SPAN_RATIO * 255 * V_THRESHOLD)[0]
    )
    v_lines = [x for x in v_lines if EDGE_MARGIN <= x <= width - EDGE_MARGIN]

    return h_lines, v_lines


def extract_cells(img_np: np.ndarray, h_lines: list, v_lines: list,
                  column_whitelists: list = None, row_start: int = 0) -> list:
    """Read every cell the grid encloses.

    Args:
        img_np: RGB page as a numpy array.
        h_lines: Y positions of the horizontal rules.
        v_lines: X positions of the vertical rules.
        column_whitelists: Per-column Tesseract character whitelists. A non-empty
            string restricts recognition to those characters, eliminating common
            misreads (B→8, $→8) in numeric columns. Empty string or None means
            no restriction for that column.
        row_start: First grid row to read (0-indexed). Used to skip the caption
            row when re-reading data cells with whitelists applied.

    Returns:
        One list of cell strings per row, left to right, starting from
        ``row_start``.
    """
    if len(h_lines) < 2 or len(v_lines) < 2:
        return []

    # Thresholded once for the whole page rather than per cell: Otsu needs the
    # spread of a full page to pick a sensible cut, and a cell holding one
    # number does not have it.
    cleaned = np.array(preprocess(Image.fromarray(img_np)))

    num_data_rows = len(h_lines) - 1 - row_start
    if num_data_rows <= 0:
        return []

    boxes = []
    for row in range(row_start, len(h_lines) - 1):
        for column in range(len(v_lines) - 1):
            top, bottom = h_lines[row], h_lines[row + 1]
            left, right = v_lines[column], v_lines[column + 1]
            # A narrow column or a short row would be inset out of existence;
            # back off to a third of the smaller side for those.
            inset = min(CELL_INSET, max(0, (min(bottom - top, right - left)
                                            - MIN_CELL_SIDE) // 3))
            boxes.append((row - row_start, column, top + inset, bottom - inset,
                          left + inset, right - inset))

    def read(box):
        row, column, y1, y2, x1, x2 = box
        crop = cleaned[y1:y2, x1:x2]
        if crop.size == 0:
            return row, column, ""
        whitelist = (column_whitelists[column]
                     if column_whitelists and column < len(column_whitelists)
                     else "")
        config = (f"{CELL_CONFIG} -c tessedit_char_whitelist={whitelist}"
                  if whitelist else CELL_CONFIG)
        text = pytesseract.image_to_string(crop, config=config)
        return row, column, " ".join(text.split())

    cells = [["" for _ in range(len(v_lines) - 1)] for _ in range(num_data_rows)]
    with ThreadPoolExecutor(max_workers=CELL_WORKERS) as pool:
        for row, column, text in pool.map(read, boxes):
            cells[row][column] = text
    return cells


def ocr_header_cell(cell_img_np: np.ndarray) -> str:
    """Read one caption, magnified.

    Captions are the smallest type on the page, so this works off the original
    colour crop rather than the page-wide threshold the data cells use: it
    upscales first, then lets Otsu pick a cut from the enlarged glyphs.
    """
    if cell_img_np is None or cell_img_np.size == 0:
        return ""

    big = cv2.resize(cell_img_np, None, fx=CAPTION_SCALE, fy=CAPTION_SCALE,
                     interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_RGB2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    gray = cv2.copyMakeBorder(gray, CAPTION_PAD, CAPTION_PAD, CAPTION_PAD,
                              CAPTION_PAD, cv2.BORDER_CONSTANT, value=255)

    return " ".join(pytesseract.image_to_string(gray, config=CAPTION_CONFIG).split())


def read_captions(img_np: np.ndarray, h_lines: list, v_lines: list, row: int) -> list:
    """Re-read the caption row at magnification, one cell at a time."""
    captions = []
    for column in range(len(v_lines) - 1):
        top, bottom = h_lines[row], h_lines[row + 1]
        left, right = v_lines[column], v_lines[column + 1]
        inset = min(CELL_INSET, max(0, (min(bottom - top, right - left)
                                        - MIN_CELL_SIDE) // 3))
        captions.append(ocr_header_cell(
            img_np[top + inset:bottom - inset, left + inset:right - inset]
        ))
    return captions


def find_caption_row(cells: list) -> int:
    """Index of the row that names the columns.

    Captions are short, and no caption contains a shipment or invoice number -
    that is what separates the row from the data under it and from the band of
    stray text the bill prints above it.
    """
    best, best_score = 0, -1
    for index, row in enumerate(cells[:HEADER_SEARCH_ROWS]):
        filled = [cell for cell in row if cell]
        if not filled:
            continue
        if any(_LONG_NUMBER.search(cell) for cell in filled):
            continue
        # "Most cells short", not "all": the captions are set in small bold
        # type that the scan mangles, and one caption coming back as a long
        # smear of noise should not disqualify the row that names every column.
        long_cells = sum(1 for cell in filled if len(cell) > HEADER_MAX_CELL)
        if long_cells > len(filled) * HEADER_LONG_TOLERANCE:
            continue
        if len(filled) > best_score:
            best, best_score = index, len(filled)
    return best if best_score > 0 else 0


def is_valid_data_row(row_dict: dict) -> bool:
    """Whether a row is a record rather than a blank line or the totals.

    A data row is numbered. The totals line leaves its serial column empty and
    a signature line has words there, so both fall out on the same test.

    The serial is one small cell, though, and the scan does lose them - 12 and
    18 came back as "iz" and "i?" on the sample page. So a row that fills
    nearly every column counts too: the totals line fills about a third, and a
    signature line one or two.
    """
    if not row_dict:
        return False

    values = list(row_dict.values())
    serial = str(values[0] or "").strip()
    if _DIGITS.match(serial):
        return True

    if not serial:
        return False
    return sum(1 for value in values if value) >= len(values) * DATA_ROW_FILL


# Currently unused - kept dormant, not deleted, for a deliberate rebuild of
# the validation-flagging layer later (see extract_bill's own row loop,
# where the call to _validate_row below was removed on that same
# rollback: every field now keeps its raw extracted value unconditionally).
#
# Bill fields validated against a known shape before they reach the sheet;
# amount/ld_charges/balance_pay/short_qty_charges/rate are all "a number,
# maybe with commas", so one validator covers all of them. vehicle_no is
# deliberately not here - see _validate_row, it is never dropped.
_SIMPLE_VALIDATORS = {
    "sr_no": validate_sr_no,
    "shipment_no": validate_shipment_no,
    "gross_qty": validate_gross_qty,
    "rate": validate_amount,
    "amount": validate_amount,
    "short_qty_charges": validate_amount,
    "ld_charges": validate_amount,
    "balance_pay": validate_amount,
    "lr_date": validate_date,
    "delivery_date": validate_delivery_timestamp,
}


def _validate_row(record: dict) -> dict:
    """Check each field against its known shape; garbled text never survives.

    A field that fails its validator becomes ``None`` rather than keeping
    whatever Tesseract read - a manual reviewer can tell a blank cell needs
    filling in, but not tell a plausible-looking wrong number from a right
    one. ``vehicle_no`` is the one exception: it has been the least reliable
    field to OCR, so an unvalidated reading is kept, flagged, rather than
    thrown away - here a wrong-but-visible value is more useful than a blank
    one, since review means checking it against the LR anyway.
    """
    flags = []
    for field, validator in _SIMPLE_VALIDATORS.items():
        if field not in record or record[field] is None:
            continue
        value, ok = validator(record[field])
        record[field] = value if ok else None
        if not ok:
            flags.append(f"{field}_invalid")

    if record.get("vehicle_no") is not None:
        value, ok, vehicle_flags = validate_vehicle_no(record["vehicle_no"])
        record["vehicle_no"] = value
        flags.extend(vehicle_flags)

    if flags:
        record["flags"] = flags
    return record


def _page_header(image: Image.Image, h_lines: list) -> dict:
    """The label/value block printed above the table."""
    cutoff = h_lines[0] if h_lines else int(image.height * HEADER_FRACTION)
    region = image.crop((0, 0, image.width, max(1, cutoff)))
    return extract_label_values(ocr_segments(region))


def debug_ocr_output(image: Image.Image) -> list:
    """Every detection on the page, top to bottom, with where it sat.

    Kept for the ``/debug`` route: when a table comes back empty this is what
    says whether the text was read at all.
    """
    items = [
        {
            "y_center": round(segment.cy, 1),
            "x_left": round(segment.x, 1),
            "text": segment.text,
            "confidence": round(segment.conf, 4),
        }
        for segment in ocr_segments(image)
    ]
    items.sort(key=lambda item: (item["y_center"], item["x_left"]))
    for item in items:
        print(f"y={item['y_center']:.0f} x={item['x_left']:.0f} "
              f"conf={item['confidence']:.2f} text={item['text']}")
    return items


def extract_bill(bill_images: list, client_id: str = "") -> tuple:
    """Read the bill table across the pages it spans.

    Args:
        bill_images: PIL images of the bill pages in order, rendered at 300
            DPI. The first is expected to carry the caption row.
        client_id: Which client's bill this is, as ``column_config`` keys
            them. Their configured column names are used when it matches;
            otherwise the OCR'd captions are.

    Returns:
        ``(header, rows)``. ``header`` is the label/value block printed above
        the table on the first page. ``rows`` is one dict per data row, keyed
        by column name, with the rows of every page combined in order.
    """
    header: dict = {}
    column_names = None
    whitelists: list = []
    all_rows = []

    for page_index, image in enumerate(bill_images or []):
        img_np = np.array(image.convert("RGB"))
        h_lines, v_lines = detect_grid(img_np)

        if page_index == 0:
            header = _page_header(image, h_lines)
            # First pass without whitelists: needed to identify the caption row
            # and derive column names before whitelists can be assigned.
            cells = extract_cells(img_np, h_lines, v_lines)
            if not cells:
                continue

            caption = find_caption_row(cells)
            captions = read_captions(img_np, h_lines, v_lines, caption)
            column_names = get_columns(client_id, len(captions), captions,
                                        data_rows=cells[caption + 1:])
            whitelists = get_whitelists(column_names)
            print(f"ocr captions: {captions}")
            print(f"bill columns: {column_names}")

            for above in cells[:caption]:
                numbered = [cell for cell in above if cell]
                if numbered and all(_DIGITS.match(cell) for cell in numbered):
                    header["column_numbers"] = list(above)

            # Second pass: re-read only data rows (below caption) with
            # per-column whitelists so numeric columns reject B/$/S misreads.
            data_cells = extract_cells(img_np, h_lines, v_lines, whitelists,
                                       row_start=caption + 1)
            all_rows.extend(data_cells)
        else:
            if column_names is None:
                continue
            # Later pages repeat the table, not its captions — read directly
            # with whitelists since column identity is already known.
            data_cells = extract_cells(img_np, h_lines, v_lines, whitelists)
            all_rows.extend(data_cells)

    if column_names is None:
        return header, []

    rows = []
    for row in all_rows:
        record = {
            name: (row[index].strip() or None) if index < len(row) else None
            for index, name in enumerate(column_names)
        }
        if is_valid_data_row(record):
            # _validate_row is intentionally not called here - rolled back
            # per instruction: every field keeps its raw extracted value,
            # full stop, until the validation-flagging layer is rebuilt
            # deliberately. _validate_row/_SIMPLE_VALIDATORS are kept below,
            # unused, for that rebuild rather than deleted.
            rows.append(record)

    return header, rows


if __name__ == "__main__":
    import sys

    try:
        from .pdf_handler import pdf_to_images
    except ImportError:
        from pdf_handler import pdf_to_images

    path = sys.argv[1] if len(sys.argv) > 1 else "test_bill.pdf"
    pages = pdf_to_images(path, dpi=300)
    bill_header, bill_rows = extract_bill([pages[1], pages[2]])

    print("--- header ---")
    for name, value in bill_header.items():
        print(f"{name}: {value!r}")
    print(f"--- {len(bill_rows)} rows ---")
    for record in bill_rows:
        print(record)
