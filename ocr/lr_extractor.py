"""Extract a single LR (lorry receipt) page - fixed schema, zone by zone.

Full rebuild per the FINAL SPEC (2026-09-25): a deliberate exception to this
project's usual schema-free extraction, taken after diagnosing that
open-ended box/label detection produced unstable field names across scans of
the identical LR template (garbled ``lr_no`` values leaking into sheet
names, fields appearing under different keys page to page - see
``lr_extractor_legacy.py``, kept as the untouched pre-rebuild reference).

The page is divided into a small number of ZONES, each anchored on a fixed,
known printed label (e.g. "LR NO", "TRUCK NO"). Zones are being built one at
a time and wired in incrementally; only Zone 1 (LR Number box) and Zone 2
(Truck No box) exist so far - see ``ZONE_ANCHORS``/``ZONE_FIELD_SYNONYMS``
below for what is implemented and the module docstring of each helper for
how.

No LLM anywhere in this file. Every field is: fuzzy-match the zone's anchor
label (reusing column_classifier.py's Tier 1 ``best_synonym_matches`` /
``greedy_assign`` - the exact same functions bill_extractor.py and
tax_invoice_extractor.py use for their own caption matching, not a new
copy), locate the zone's crop box relative to that anchor, then read the
zone with Tesseract via invoice_extractor.py's generic label/value pass
(also reused, not reimplemented).

Zone cropping deliberately combines TWO existing detectors rather than
inventing a third (see ``_zone_crop_box``):
  - bill_extractor.detect_grid's horizontal rule positions bound a zone's
    top/bottom - confirmed against this form's actual pages that its
    h_lines line up with the printed section boundaries.
  - bill_extractor.detect_grid reports NO vertical rules on this form at
    all (confirmed against LR_1350/LR_1351: v_lines always ``[]`` - the
    form's box borders never reach the width/height fractions that
    function is tuned for, which assume one wide uniform table). Where
    that leaves no v_lines to use, the zone's left/right bounds fall back
    to whichever of this file's own existing ``detect_boxes()`` contour
    rectangles contains the anchor - the same function the pre-rebuild
    version of this file already used for section fencing, so this is
    still reuse, not a new detector.
"""

import re

import cv2
import numpy as np
from PIL import Image

try:
    from .bill_extractor import detect_grid, extract_cells
    from .column_classifier import best_synonym_matches, greedy_assign
    from .invoice_extractor import extract_label_values, ocr_segments
    from .preprocessing import preprocess_for_ocr
    from .reader import get_ocr_results
except ImportError:  # running this file directly from inside ocr/
    from bill_extractor import detect_grid, extract_cells
    from column_classifier import best_synonym_matches, greedy_assign
    from invoice_extractor import extract_label_values, ocr_segments
    from preprocessing import preprocess_for_ocr
    from reader import get_ocr_results

import pytesseract

# --------------------------------------------------------------------------
# This form's own printed section boxes - kept from the pre-rebuild version
# of this file (same contour technique, unchanged), and now used only as
# the x-bound fallback for zone cropping when detect_grid has no vertical
# rules to offer (see the module docstring).
# --------------------------------------------------------------------------

BOX_H_KERNEL_RATIO = 0.05
BOX_V_KERNEL_RATIO = 0.015
MIN_BOX_AREA_RATIO = 0.0002
MAX_BOX_AREA_RATIO = 0.35


def detect_boxes(img_np: np.ndarray) -> list:
    """This page's own printed section boxes, as ``(x1, y1, x2, y2)``
    pixel rectangles - found by contour. Order is not meaningful; every
    downstream use only tests containment."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )

    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(2, int(width * BOX_H_KERNEL_RATIO)), 1)
    )
    h_lines_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(2, int(height * BOX_V_KERNEL_RATIO)))
    )
    v_lines_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    grid = cv2.bitwise_or(h_lines_mask, v_lines_mask)
    grid = cv2.dilate(grid, np.ones((3, 3), np.uint8), iterations=1)

    cell_mask = cv2.bitwise_not(grid)
    contours, _hierarchy = cv2.findContours(cell_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    page_area = width * height
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < page_area * MIN_BOX_AREA_RATIO or area > page_area * MAX_BOX_AREA_RATIO:
            continue
        boxes.append((x, y, x + w, y + h))
    return boxes


# --------------------------------------------------------------------------
# Zone definitions.
# --------------------------------------------------------------------------
#
# Each zone is located by fuzzy-matching its OWN anchor label (ZONE_ANCHORS)
# against the whole page's OCR output, using the exact same best_synonym_
# matches/greedy_assign pair Tier 1 caption matching uses elsewhere in this
# project. Once a zone's crop is read, its own dynamically-discovered
# label/value pairs are aliased onto the zone's FIXED field list
# (ZONE_FIELD_SYNONYMS) with the same two functions again - so there is
# still only one fuzzy-matching implementation in the whole codebase, used
# twice (anchor location, then field aliasing) rather than duplicated.

# How close a label has to be to a synonym before it counts as a match -
# rapidfuzz's 0-100 scale. Same bar the pre-rebuild version of this file
# used for its own canonical aliasing.
ZONE_MATCH_THRESHOLD = 82

# zone_id -> synonyms for the label that marks where that zone starts.
# "zone3_weights" is not a top-level zone (not in ZONES_BUILT) - it is
# Zone 3's OWN internal sub-anchor, used only to locate the 3-column
# LORRY TARE WT/NET WT/GROSS WT table that sits below Zone 3's ROUTE
# line, separately from Zone 3's own outer "zone3" anchor (which stays
# "route"-or-"lorry tare wt", whichever is found, purely to bound the
# outer crop route is read from). See extract_lr's own zone3 handling.
ZONE_ANCHORS = {
    "zone1": ["lr no"],
    "zone2": ["truck no"],
    "zone3": ["route", "lorry tare wt"],
    "zone3_weights": ["lorry tare wt"],
    "zone4": ["packages"],
    # "STAMP & SIGNATURE OF THE TRANSPORTER" itself is often not OCR-able
    # at all - confirmed on both LR_1350/LR_1351, where a printed/rubber
    # stamp mark visually overlaps that exact line, defeating Tesseract
    # on both pages. "AUTHORISED SIGNATORY" (the OTHER printed label in
    # the same box, positioned below the stamp/signature area, never
    # itself overlapped by the stamp) is tried second - see extract_lr's
    # own zone5 handling for how the crop is derived once either is
    # found: the box CONTAINING whichever anchor matched, not a position
    # relative to one specific label's own line.
    "zone5": ["stamp and signature of the transporter"],
    "zone5_fallback": ["authorised signatory"],
    # "CUSTOMER'S SEAL & SIGN" reads reliably on both test pages (once
    # merged as one phrase, once split into "CUSTOMER'S" + "SEAL & SIGN"
    # by Tesseract's own line-merging - best_synonym_matches' partial_ratio
    # still matches the merged synonym against either). Internally still
    # called "zone7" (a pre-existing identifier, not user-facing - see
    # ZONE_FIELD_ORDER's own comment) even though the FINAL SPEC numbers
    # this Zone 6; the "Customer Acknowledgement Details" (RECEIVED WT/
    # DATE/REMARK) box the old zone6 anchor/table used to read is not in
    # the FINAL SPEC's field list at all and has been removed outright.
    "zone7": ["customer's seal and sign", "customer s seal and sign"],
}

# Zone 3's weight sub-table columns, fixed left-to-right order - see
# _extract_positional_table's own docstring for why this is read by
# position, not by caption-text fuzzy matching or the generic label/value
# pairer.
ZONE3_WEIGHT_FIELDS = ["lorry_tare_wt", "net_wt", "gross_wt"]

# This table's own crop is a small slice of the page (roughly 900x120px at
# this document's 300 DPI render) - too small for reliable digit OCR even
# though the printed text itself is clean (confirmed directly against the
# original scan). The same fix invoice_extractor.py's own _ocr_zone already
# applies to a small anchored crop (ZONE_SCALE).
ZONE3_WEIGHT_UPSCALE = 3

# zone_id -> {canonical_field: [synonyms]} - the FIXED field list for that
# zone. A label detected inside the zone's crop that matches none of these
# is simply not carried into the record (fixed schema, per the FINAL SPEC -
# a deliberate exception to this project's usual schema-free extraction,
# see the module docstring).
ZONE_FIELD_SYNONYMS = {
    "zone1": {
        "lr_no": ["lr no"],
        "lr_date": ["lr date"],
        "shipment_no": ["shipment no"],
        "from_origin": ["from origin", "from"],
        # "to" alone risked colliding with an unrelated key on this
        # project's earlier schema-free scans (see lr_extractor_legacy.py) -
        # kept here anyway because Zone 1's crop is fenced to just this
        # box, where nothing else printed is likely to say "to"; "to
        # destination" is tried first regardless since it is listed first
        # and best_synonym_matches takes the best synonym match, not the
        # first.
        "to_destination": ["to destination", "to"],
    },
    "zone2": {
        "truck_no": ["truck no"],
        "truck_type": ["truck type"],
        "incoterm": ["incoterm"],
        "freight": ["freight"],
        "eway_bill_no": ["eway bill no", "e way bill no", "eway bill number"],
        "mwr": ["mwr"],
    },
    "zone3": {
        # Only "route" is aliased here now - lorry_tare_wt/net_wt/gross_wt
        # used to be too (via this same fuzzy machinery), but that let the
        # generic label/value pairer pick the wrong adjacent number inside
        # an already-correctly-bounded crop (confirmed on LR_1350/LR_1351:
        # visually the LORRY TARE WT/NET WT/GROSS WT box is a genuine
        # 3-column ruled table, header row over one data row, not a
        # label-beside-its-value layout) - now read positionally instead,
        # see ZONE3_WEIGHT_FIELDS and _extract_positional_table.
        "route": ["route"],
    },
    # Zone 4 no longer has a synonyms entry: PACKAGES/DESCRIPTION OF
    # PRODUCT/NET WT/GROSS WT/AMOUNT used to be aliased by fuzzy-matching
    # each column's own OCR'd caption text, but that was confirmed
    # unreliable across the full batch (well under 10% completeness -
    # caption cells are small, bold type that this form's scan quality
    # garbles heavily). Read positionally instead, the same fixed-
    # left-to-right-column-order decision as Zone 3's weight table - see
    # _extract_positional_table.
}

# Explicit field order per zone, so the record (and later, the sheet) is
# stable regardless of dict/aliasing iteration order.
ZONE_FIELD_ORDER = {
    "zone1": ["lr_no", "lr_date", "shipment_no", "from_origin", "to_destination"],
    "zone2": ["truck_no", "truck_type", "incoterm", "freight", "eway_bill_no", "mwr"],
    "zone3": ["route", "lorry_tare_wt", "net_wt", "gross_wt"],
    "zone4": ["packages", "description_of_product", "net_wt_2", "gross_wt_2", "amount"],
    "zone5": ["text_above_transporter_stamp", "transporter_stamp_present"],
    # Still keyed "zone7" internally (see ZONE_ANCHORS' own comment) - this
    # is the FINAL SPEC's Zone 6, Customer's Seal & Sign.
    "zone7": ["customer_seal_text_handwritten", "customer_seal_present"],
}

# The FINAL SPEC's own field list, in its own order - exactly what
# extract_lr returns, regardless of internal zone id/order above. Kept
# separate from ZONE_FIELD_ORDER (which is keyed by internal zone id, for
# the per-zone code that fills each one in) since the spec's own order
# groups fields by MEANING (LR Number box, Truck No box, ...), not by this
# file's own zone-numbering history.
LR_OUTPUT_FIELDS = [
    "lr_no", "lr_date", "shipment_no", "from_origin", "to_destination",
    "truck_no", "truck_type", "incoterm", "freight", "eway_bill_no", "mwr",
    "route", "lorry_tare_wt", "net_wt", "gross_wt",
    "packages", "description_of_product", "net_wt_2", "gross_wt_2", "amount",
    "text_above_transporter_stamp", "transporter_stamp_present",
    "customer_seal_text_handwritten", "customer_seal_present",
]

# Padding fallback (fraction of page width/height) - the LAST resort when a
# zone's anchor falls inside no detect_boxes() rectangle AND no neighbouring
# OCR phrase on the page gives a content-derived bound either (see
# ``_zone_crop_box``). Kept only as a true last resort, not the normal path:
# a fixed fraction of the page fits whichever scan it was tuned against and
# nothing else - confirmed on this project's own LR_1350 scan, where this
# padding alone pulled an unrelated "Demurrage charges" notice box into
# Zone 3's crop. Whenever this path actually fires, the caller is told via
# the returned ``low_confidence`` flag, and the record carries an explicit
# ``<zone_id>_crop_low_confidence`` marker rather than silently trusting a
# guessed-wide crop. FALLBACK_PAD_Y_RATIO has no such problem - h_lines
# covers the y-axis almost everywhere on this form - but is kept for the
# same "some crop beats none" reason on a page whose lines were too faint
# to detect at all.
FALLBACK_PAD_X_RATIO = 0.18
FALLBACK_PAD_Y_RATIO = 0.06

# Small fixed-pixel anti-collision buffer between a zone's crop edge and
# whatever neighbouring content bounded it - the same kind of small fixed
# margin bill_extractor.py's own GROUP_GAP (10px) uses to avoid slicing a
# rule or a glyph in half, not a page-fraction guess about the crop's own
# size.
NEIGHBOR_MARGIN = 10

# A crop this small (fraction of page area) is not a real zone box - the
# anchor/line matching went wrong somewhere; fall back to the padded crop
# instead of reading a sliver.
MIN_ZONE_AREA_RATIO = 0.003


def _box_bounds(box) -> tuple:
    points = np.asarray(box, dtype=float)
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


_ALNUM_ONLY = re.compile(r"[^A-Za-z0-9]+")

# rapidfuzz's partial_ratio scores a candidate 100 against a synonym in two
# different degenerate cases, both confirmed on this project's own
# LR_1350/LR_1351 scans, and an absolute length floor cannot fix both at
# once: a bare 1-2 character noise fragment ("a", "te") scores 100 against
# almost any synonym purely because so little of it has to align to find
# SOME alignment - but ZONE_FIELD_SYNONYMS also has genuinely short real
# synonyms ("to"), so a floor high enough to exclude "te" also excludes a
# real "to" match outright. What actually tells a real (if short or OCR-
# noisy) match apart from noise or from a synonym merely BURIED inside a
# much longer unrelated sentence ("...delivered to or to the order of...”
# scoring 100 against the bare synonym "to") is not length in isolation,
# but length RELATIVE to the specific synonym it is being judged against:
# either the candidate actually STARTS WITH that synonym (so it may run
# arbitrarily longer after it - "ROUTE : NAHARPALI-VISAKHAPATNAM" is a
# real match many times "route"'s own length), or its length is at least
# roughly comparable to the synonym's own (neither a 1-character sliver of
# a multi-word synonym, nor a multi-sentence paragraph merely containing a
# short one). MAX_TEXT_TO_SYNONYM_RATIO reuses column_classifier.
# locate_pages_by_type()'s own "text more than 3x its synonym's length is
# not a plausible whole-phrase match" rule, applied per-candidate here
# instead of per-page-title-band.
MIN_TEXT_TO_SYNONYM_COVERAGE = 0.5
MAX_TEXT_TO_SYNONYM_RATIO = 3


def _alnum_len(text: str) -> int:
    return len(_ALNUM_ONLY.sub("", str(text or "")))


def _safe_synonym_matches(cleaned_texts: dict, synonyms_by_field: dict, threshold: float) -> list:
    """``best_synonym_matches``, filtered against the degenerate-match
    failure mode documented above - every fuzzy-match call site in this
    file goes through this, not the shared Tier 1 function directly, so
    the guard exists in exactly one place. Filters candidates; never
    changes a score, and never touches ``best_synonym_matches`` itself.
    """
    candidates = best_synonym_matches(cleaned_texts, synonyms_by_field, threshold)

    safe = []
    for score, index, field in candidates:
        text = cleaned_texts[index]
        text_len = _alnum_len(text)
        plausible = False
        for synonym in synonyms_by_field[field]:
            if text.startswith(synonym):
                plausible = True
                break
            synonym_len = _alnum_len(synonym)
            if synonym_len * MIN_TEXT_TO_SYNONYM_COVERAGE <= text_len <= synonym_len * MAX_TEXT_TO_SYNONYM_RATIO:
                plausible = True
                break
        if plausible:
            safe.append((score, index, field))

    return safe


def _locate_anchor(results: list, zone_id: str):
    """The OCR result box (from ``get_ocr_results``) that best matches
    ``ZONE_ANCHORS[zone_id]``, or ``None`` when nothing on the page does.

    Confirmed against this project's own scans: partial_ratio also lets an
    unrelated sentence that merely CONTAINS the synonym as a substring
    ("...diverted, re-routed" containing "route") tie a genuine anchor
    line's own top score. Both are real matches by the shared scoring
    function's own rules - the tie-break below, not a change to that
    function, is what tells them apart: prefer whichever tied candidate's
    text actually STARTS WITH one of the zone's own synonyms (a real
    printed label always leads its own line; a synonym merely buried
    inside an unrelated sentence never does), then the shortest text.
    """
    cleaned = {index: item["text"].strip().lower() for index, item in enumerate(results)}
    synonyms = {zone_id: ZONE_ANCHORS[zone_id]}
    candidates = _safe_synonym_matches(cleaned, synonyms, ZONE_MATCH_THRESHOLD)
    if not candidates:
        return None

    top_score = max(item[0] for item in candidates)
    tied = [item for item in candidates if item[0] == top_score]

    def _rank(candidate):
        _score, index, _field = candidate
        text = cleaned[index]
        starts = any(text.startswith(synonym) for synonym in ZONE_ANCHORS[zone_id])
        return (0 if starts else 1, len(text))

    _score, index, _field = min(tied, key=_rank)
    return results[index]["box"]


def _zone_crop_box(anchor_box, h_lines: list, boxes: list,
                    img_w: int, img_h: int) -> tuple:
    """``((x1, y1, x2, y2), low_confidence)`` - the pixel rectangle to read
    for a zone anchored on ``anchor_box``, and whether it had to fall all
    the way back to fixed page-fraction padding to get one. See the module
    docstring for how top/bottom and left/right are each sourced."""
    ax1, ay1, ax2, ay2 = _box_bounds(anchor_box)
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2

    below = [y for y in h_lines if y <= ay1]
    above = [y for y in h_lines if y >= ay2]
    top = max(below) if below else max(0, ay1 - img_h * FALLBACK_PAD_Y_RATIO)
    bottom = min(above) if above else min(img_h, ay2 + img_h * FALLBACK_PAD_Y_RATIO)

    low_confidence = False
    containing = [
        box for box in boxes
        if box[0] <= acx <= box[2] and box[1] <= acy <= box[3]
    ]
    if containing:
        # Tightest-fitting rectangle, in case the anchor sits inside more
        # than one (a small header cell nested inside a larger section).
        left, _box_top, right, _box_bottom = min(
            containing, key=lambda box: (box[2] - box[0]) * (box[3] - box[1])
        )
    else:
        # No printed section box CONTAINS this anchor (see the module
        # docstring for why that happens on this form). Rather than guess a
        # fixed page-fraction width, look for another detect_boxes()
        # rectangle sitting to the right of the anchor, in its own row band
        # - a real printed section already found on THIS page, not a raw
        # OCR word. Deliberately NOT individual OCR phrases: tried first,
        # confirmed unsafe on LR_1351's own scan - a single garbled 2-glyph
        # OCR fragment sitting a few pixels right of the anchor became the
        # bound and clipped off the label's own value entirely. A detected
        # BOX is a structural unit (an actual printed rectangle on the
        # page), immune to that failure mode.
        left = max(0, ax1 - NEIGHBOR_MARGIN)
        # Same PRINTED ROW as the anchor only - not the whole top..bottom
        # zone band (which can span several rows, e.g. Zone 1's LR/Date/
        # Shipment/From/To lines together) - confirmed necessary the same
        # way: without it, content on a LOWER row of this same zone was
        # mistaken for a neighbour bounding the anchor's own row.
        anchor_height = ay2 - ay1
        row_tolerance = max(anchor_height * 1.5, 20)
        row_top, row_bottom = ay1 - row_tolerance, ay2 + row_tolerance
        # A neighbouring box's left edge landing a little before the
        # anchor's own OCR box's right edge is common (a nearby border
        # slightly overlapping a wide text bounding box, not the anchor's
        # own section) - tolerated up to NEIGHBOR_MARGIN*2 so it still
        # counts as "to the right", and the final right edge is never let
        # go closer in than the anchor's own detected extent regardless.
        neighbor_box_lefts = [
            box[0] for box in boxes
            if box[0] > ax2 - (NEIGHBOR_MARGIN * 2)
            and box[1] < row_bottom and box[3] > row_top
        ]
        if neighbor_box_lefts:
            right = min(img_w, max(ax2 + NEIGHBOR_MARGIN, min(neighbor_box_lefts) - NEIGHBOR_MARGIN))
        else:
            # True last resort: nothing detected on the page bounds this
            # zone at all. Flagged, not silently trusted - see
            # FALLBACK_PAD_X_RATIO's own docstring.
            left = max(0, acx - img_w * FALLBACK_PAD_X_RATIO)
            right = min(img_w, acx + img_w * FALLBACK_PAD_X_RATIO)
            low_confidence = True

    left, right = int(max(0, left)), int(min(img_w, right))
    top, bottom = int(max(0, top)), int(min(img_h, bottom))

    if (right - left) <= 0 or (bottom - top) <= 0 or \
            (right - left) * (bottom - top) < img_w * img_h * MIN_ZONE_AREA_RATIO:
        left = int(max(0, acx - img_w * FALLBACK_PAD_X_RATIO))
        right = int(min(img_w, acx + img_w * FALLBACK_PAD_X_RATIO))
        top = int(max(0, acy - img_h * FALLBACK_PAD_Y_RATIO))
        bottom = int(min(img_h, acy + img_h * FALLBACK_PAD_Y_RATIO))
        low_confidence = True

    return (left, top, right, bottom), low_confidence


def _table_crop_box(anchor_box, h_lines: list, boxes: list, img_w: int, img_h: int) -> tuple:
    """The crop rule shared by every small ruled sub-table this file reads
    positionally (Zone 4's own Packages/Description/Net Wt/Gross Wt/
    Amount table, anchored on "PACKAGES"; Zone 3's own Lorry Tare Wt/Net
    Wt/Gross Wt table, anchored on "LORRY TARE WT" via the
    "zone3_weights" sub-anchor) - different from ``_zone_crop_box``
    because the anchor sits inside just its own narrow header CELL, not a
    box spanning the whole table row: this form's ruled tables are
    contoured by ``detect_boxes`` as many small per-column cells, not one
    outer rectangle. The right bound the other zones get from a single
    containing box or a lone neighbouring one would cut a table off after
    one column.

    Instead: take the UNION of every ``detect_boxes`` rectangle whose own
    height is close to the anchor's own containing cell (so a much taller,
    unrelated section box - like the notice/demurrage box that caused
    Zone 3's own ROUTE-row bleed problem - is excluded by shape, not just
    position) and whose y-range overlaps that cell's row.

    Confirmed on this project's own scans that this union can still
    include a genuinely NEIGHBOURING section's own header cells (on
    LR_1351, Zone 4's own crop this way also picks up the Customer
    Acknowledgement box's "RECEIVED WT"/"DATE"/"REMARK" columns, which
    sit in the same row band with no reliable gap to bound on
    geometrically) - deliberately not fought here with more position
    tuning. ``_extract_positional_table`` only ever reads the leftmost N
    columns it expects, so an unrelated column captured this way is
    simply never assigned to anything; reading a wider crop than strictly
    necessary is harmless.
    """
    ax1, ay1, ax2, ay2 = _box_bounds(anchor_box)
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2

    containing = [
        box for box in boxes
        if box[0] <= acx <= box[2] and box[1] <= acy <= box[3]
    ]
    if not containing:
        left = max(0, acx - img_w * FALLBACK_PAD_X_RATIO)
        right = min(img_w, acx + img_w * FALLBACK_PAD_X_RATIO)
        top = max(0, acy - img_h * FALLBACK_PAD_Y_RATIO)
        bottom = min(img_h, acy + img_h * FALLBACK_PAD_Y_RATIO)
        return (int(left), int(top), int(right), int(bottom)), True

    cell = min(containing, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    cell_height = cell[3] - cell[1]
    row_top, row_bottom = cell[1], cell[3]
    max_height = cell_height * 1.5

    row_boxes = [
        box for box in boxes
        if (box[3] - box[1]) <= max_height and box[1] < row_bottom and box[3] > row_top
    ]
    if not row_boxes:
        row_boxes = [cell]

    left = min(box[0] for box in row_boxes)
    right = max(box[2] for box in row_boxes)
    top = min(box[1] for box in row_boxes)
    bottom = max(box[3] for box in row_boxes)

    # Extend one row below the header union to reach the data row beneath
    # it - the next printed rule after the header's own bottom.
    below = [y for y in h_lines if y > bottom]
    bottom = min(below) if below else bottom

    left, right = int(max(0, left - NEIGHBOR_MARGIN)), int(min(img_w, right + NEIGHBOR_MARGIN))
    top, bottom = int(max(0, top - NEIGHBOR_MARGIN)), int(min(img_h, bottom + NEIGHBOR_MARGIN))

    return (left, top, right, bottom), False


# How many of the weight anchor's own header-cell widths its 3-column
# table spans in total - a structural fact (ZONE3_WEIGHT_FIELDS is a
# fixed, known 3-equal-width-column table, same category of knowledge as
# the field list itself), not a fitted pixel constant. Confirmed against
# LR_1351's own scan (where the table was NOT truncated): its full
# 3-column width (697px) is within 3% of 3x its own "LORRY TARE WT"
# header cell's width (239px x 3 = 717px).
ZONE3_WEIGHT_TABLE_WIDTH_MULTIPLIER = 3.3

# The anchor's own OCR box hugs its TEXT, not the printed cell's border -
# confirmed on both LR_1350/LR_1351: a small fixed margin (this file's
# usual NEIGHBOR_MARGIN) left the crop starting to the right of the
# column's own left-hand printed rule, so detect_grid never saw it as a
# line and the column split came up short by one edge. A margin
# proportional to the anchor's own width, not a fixed pixel count, is
# what actually clears the cell's own internal text padding.
ZONE3_WEIGHT_LEFT_MARGIN_RATIO = 0.2


def _zone3_weight_crop_box(anchor_box, zone3_crop_box, zone3_local_h_lines: list,
                            img_w: int, img_h: int) -> tuple:
    """Zone 3's own weight sub-table crop rule - not ``_table_crop_box``,
    because that one bounds a table's right edge using either a
    containing ``detect_boxes`` cell or a neighbouring one, and neither
    exists reliably here: this specific header cell was not found inside
    any ``detect_boxes`` rectangle on either LR_1350 or LR_1351, and this
    box's true neighbour (a "Demurrage charges" notice) sits close enough
    that its own text PHYSICALLY OVERLAPS the weight table's own right
    column in x - confirmed directly: Tesseract's own whole-page OCR
    pass glues the GROSS WT column's data value and the demurrage
    notice's own text into one merged phrase on both pages, with no gap
    to bound on at all. The right edge is therefore derived from the
    anchor's own measured header-cell width instead (see
    ZONE3_WEIGHT_TABLE_WIDTH_MULTIPLIER) - running a little into
    neighbouring content there is harmless: ``detect_grid`` finds the
    real PRINTED RULES inside whatever this crop hands it, and
    ``extract_cells``/``_extract_positional_table`` only ever read the 3
    columns they expect.

    The top/bottom edges, by contrast, reuse ``zone3_local_h_lines`` -
    ``detect_grid`` run once already on Zone 3's own OUTER crop (the one
    ROUTE is read from), passed in rather than recomputed. Confirmed
    necessary two ways: the page's own (whole-page) ``h_lines`` are this
    FORM's broad SECTION boundaries, far too sparse to bound one small
    table's own header/data rows (produced a crop 4x too tall); and
    guessing top/bottom purely from the anchor's own height instead left
    the crop so short that detect_grid's OWN kernel/threshold, run again
    on that sliver, could not find enough of a sample to detect its
    lines at all (2 lines, not the 3 needed before it will even attempt
    a column split). Zone 3's OUTER crop is large enough for its own
    detect_grid call to find these same row boundaries reliably (already
    confirmed: cleanly separates the ROUTE row from the header row from
    the data row on both LR_1350 and LR_1351), so those positions -
    translated from that crop's own coordinate space into the page's -
    are reused here instead of asking a much smaller image to rediscover
    them.
    """
    ax1, ay1, ax2, ay2 = _box_bounds(anchor_box)
    anchor_width = ax2 - ax1
    anchor_height = ay2 - ay1

    dx, dy = zone3_crop_box[0], zone3_crop_box[1]

    # Top pinned tight to the anchor's own top, NOT to whichever
    # zone3_local_h_lines entry happens to sit at or above it - confirmed
    # necessary on this project's own scans: the outer zone3 crop's h_lines
    # (this form's broad section rules, not this one small table's own row
    # boundaries - see the module docstring) sometimes have no detected rule
    # between the ROUTE line and the header row at all, so "nearest line at
    # or above the header's own top" walked back past ROUTE instead, pulling
    # it into what should have been the header's own row band.
    top = max(0, ay1 - NEIGHBOR_MARGIN)

    # Bottom: a rule genuinely BELOW the header row's own bottom (not merely
    # at-or-below the header's own TOP, which the header's own bottom rule
    # itself already satisfies) is what actually reaches the data row below
    # it; a generous multiple of the header's own height is the fallback
    # when no such rule is found at all - confirmed necessary the same way
    # Zone 6's own now-removed weight/received-wt table crop needed a
    # generous margin past its own detected boundary (see git history):
    # this form's printed rules under a data row read faint enough that
    # detect_grid's own kernel/threshold frequently misses them, and
    # trusting "only one rule found below the header's own top" as if it
    # were the data's own bottom - the previous version of this function -
    # left the crop stopping at the header's own bottom border on most of
    # this project's own scans (LR_1350/1351/1352 all confirmed cut off
    # before the data row), reading nothing but the header text itself.
    # More than a hair past the header's own bottom edge (a couple of pixels
    # would still catch the header's OWN border, or an antialiasing artifact
    # right beside it, rather than a genuine second-row separator below it) -
    # confirmed necessary on this project's own scans: a bare +2px margin let
    # a line sitting essentially on the header's own bottom border through,
    # producing a crop barely taller than the header row itself and no data
    # band at all. Half the header's own height is enough to clear the
    # header's own border while still catching a real rule a normal row's
    # height below it.
    header_bottom_local = (ay2 - dy) + anchor_height * 0.5
    lines_below_header = sorted(y for y in zone3_local_h_lines if y > header_bottom_local)
    if lines_below_header:
        bottom = min(img_h, dy + lines_below_header[0])
    else:
        bottom = min(img_h, ay2 + anchor_height * 5)

    left = max(0, ax1 - ZONE3_WEIGHT_LEFT_MARGIN_RATIO * anchor_width)
    # Never wider than the outer zone3 crop's own right edge - that crop
    # (``_zone_crop_box``) is already bounded by structural ``detect_boxes``/
    # ``h_lines`` geometry; this table's own right edge, derived purely from
    # the anchor's own OCR'd width, has no such guarantee: confirmed on one
    # of this project's own scans that whole-page OCR sometimes merges
    # several of the header row's own cells into ONE anchor box ("LORRY TARE
    # WT ... GROSS WT"), inflating anchor_width enough that 3.3x it
    # overshoots straight past the real table into the neighbouring
    # Demurrage Charges notice.
    right = min(img_w, zone3_crop_box[2], ax1 + ZONE3_WEIGHT_TABLE_WIDTH_MULTIPLIER * anchor_width)

    return int(left), int(top), int(right), int(bottom)


def _alias_zone_fields(zone_id: str, raw: dict) -> dict:
    """``raw`` (a zone crop's own generic label/value dict, keys already
    ``clean_key``-formatted by ``extract_label_values``) mapped onto
    ``ZONE_FIELD_SYNONYMS[zone_id]``'s fixed field list. Every field in
    that list is present in the result, ``None`` when nothing in ``raw``
    matched it confidently.

    Uses ``_safe_synonym_matches``, not the shared Tier 1 function
    directly - confirmed necessary on LR_1350's own Zone 1 scan: a
    90-character boilerplate disclaimer key ("...delivered to or to the
    order of...") scored a perfect 100 against the bare synonym "to" via
    plain substring containment, winning ``to_destination`` outright over
    the real (if OCR-garbled) destination line, which never even reached
    threshold under its own unrelated-looking key name.

    A raw key that fails to confidently match ANY of this zone's fields is
    simply left out of the result - the FINAL SPEC's fixed field list is
    the whole of what a zone may surface, never a raw key's own OCR'd name
    (see extract_lr's own closing whitelist reconstruction).
    """
    cleaned = {key: key.replace("_", " ").lower() for key in raw}
    candidates = _safe_synonym_matches(cleaned, ZONE_FIELD_SYNONYMS[zone_id], ZONE_MATCH_THRESHOLD)
    assigned = greedy_assign(candidates)  # {raw_key: canonical_field}

    by_field = {}
    for raw_key, canonical in assigned.items():
        by_field[canonical] = raw[raw_key]

    return {field: by_field.get(field) for field in ZONE_FIELD_ORDER[zone_id]}


def _extract_positional_table(crop_np: np.ndarray, canonical_fields: list) -> dict:
    """Read a small ruled sub-table (Zone 4's Packages/Description/Net Wt/
    Gross Wt/Amount table, or Zone 3's own Lorry Tare Wt/Net Wt/Gross Wt
    table) as one caption row over one data row - not a label/value form
    section like the rest of Zones 1-3, so it is read with the same grid/
    cell machinery bill_extractor.py's own table reading uses
    (``detect_grid``, ``extract_cells``), not the generic label/value
    pairer.

    Columns are assigned to ``canonical_fields`` by LEFT-TO-RIGHT
    POSITION, not by fuzzy-matching each column's own OCR'd caption text
    against a synonym list. Caption-text matching was tried first for
    Zone 4 and confirmed unreliable across the full 31-page batch (well
    under 10% completeness on packages/description_of_product/net_wt_2/
    gross_wt_2/amount) - this form's column captions are small, bold type
    that this scan's quality garbles heavily ("PACKAGES" read as
    unrecognisable noise), but the printed column ORDER itself is a fixed,
    known fact about this already-schema-fixed box, not business data -
    the same category of decision as fixing the caption row at 0 below,
    just applied to columns instead of rows.

    The caption row itself is fixed at row 0, NOT located with
    bill_extractor.find_caption_row(). That function's job is finding an
    unknown caption row's position within a page-spanning table where the
    header could be one of several rows above genuine data - a real
    unknown for the bill/tax-invoice tables it was built for, where it is
    still used unchanged (this file only imports it there; these
    sub-tables never call it). Their own crop is always built to start
    exactly at the header row (see ``_table_crop_box``), so "which row is
    the caption" is not a per-page unknown here. Confirmed necessary on
    LR_1350's own scan: OCR noise merged into two of Zone 4's header cells
    pushed them over find_caption_row's own HEADER_MAX_CELL length
    tolerance, which disqualified the true header row and picked a data
    row instead.

    A column beyond ``len(canonical_fields)`` (a genuinely different,
    neighbouring section's own trailing columns - see ``_table_crop_box``'s
    own docstring) is simply ignored - the FINAL SPEC's fixed field list is
    the whole of what this table may surface. If FEWER columns are detected
    than ``canonical_fields`` expects, no position is guessed at all -
    every field is ``None`` rather than force an assignment onto a shape
    that does not match what this box is known to look like.
    """
    result = {field: None for field in canonical_fields}

    h_lines, v_lines = detect_grid(crop_np)
    cells = extract_cells(crop_np, h_lines, v_lines)
    if not cells:
        return result

    caption_row = 0
    data_rows = cells[caption_row + 1:]
    data_row = next((row for row in data_rows if any(str(cell or "").strip() for cell in row)), None)
    if data_row is None or len(data_row) < len(canonical_fields):
        return result

    for column_index, field in enumerate(canonical_fields):
        value = str(data_row[column_index]).strip()
        result[field] = value or None

    return result


# Digit/decimal-point-only whitelist for the weight sub-table's own data row -
# confirmed appropriate the same way ink_separation.py's own (now-removed)
# RECEIVED_WT_CONFIG was for a different table's own weight cell: a value
# here is always a number, never free text, so nothing is lost by refusing
# every other glyph Tesseract might otherwise guess at.
_WEIGHT_VALUE_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789.,"


def _extract_positional_row_by_proportion(crop_np: np.ndarray, canonical_fields: list,
                                           header_height: float) -> dict:
    """Read Zone 3's own LORRY TARE WT | NET WT | GROSS WT data row by fixed
    LEFT-TO-RIGHT COLUMN POSITION - the same principle
    ``_extract_positional_table`` already uses for Zone 4, but not that
    function itself: this table's own crop reliably comes back with only 2
    ``detect_grid`` h_lines (this specific 2-row, likely bottom-unbordered
    table has no third rule for ``detect_grid`` to find), and
    ``detect_grid``'s own v_lines detection refuses to even run below its
    ``MIN_LINES`` (3) gate on h_lines - confirmed directly on this
    project's own scans: v_lines comes back ``[]`` every time this gate is
    not met, regardless of whether the printed vertical rules are actually
    there, so ``extract_cells`` never has a column split to work from at
    all. Splitting the crop into ``len(canonical_fields)`` EQUAL width
    columns sidesteps that gate entirely - a fixed, known structural fact
    about this table (it is a 3-equal-width-column table, the same
    assumption ``ZONE3_WEIGHT_TABLE_WIDTH_MULTIPLIER`` already relies on),
    not a fitted guess.

    ``header_height`` (the anchor's own printed height, in the crop's own
    pixel space) locates where the header row ends and the data row begins
    when ``detect_grid`` found no usable line at all - a fixed multiple of
    it is the fallback there, the same "some crop beats none" reasoning
    ``_zone3_weight_crop_box``'s own fallback already uses.

    The FIRST ``detect_grid`` line found, not the second, is what marks
    that boundary here - confirmed directly on this project's own scans:
    this crop is already tightly bounded (top pinned to the header's own
    top, bottom padded generously past the data row - see
    ``_zone3_weight_crop_box``), so the one rule ``detect_grid`` reliably
    finds inside it is the header's own underline; a SECOND entry, when one
    comes back at all, sits essentially on the crop's own bottom edge (an
    artifact of that generous padding, not a real second divider), and
    using it as the data row's own top left almost no height to read at
    all - confirmed directly: a crop this reduced to read the header's
    OWN underline pixels rather than the data digits below it.
    """
    result = {field: None for field in canonical_fields}
    height, width = crop_np.shape[:2]
    if height <= 0 or width <= 0:
        return result

    gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
    h_lines, _v_lines = detect_grid(crop_np)
    if h_lines:
        data_top = h_lines[0]
    else:
        data_top = int(header_height * 1.3)
    data_top = min(data_top, height - 1)

    column_width = width / len(canonical_fields)
    for index, field in enumerate(canonical_fields):
        left, right = int(index * column_width), int((index + 1) * column_width)
        cell = gray[data_top:height, left:right]
        if cell.size == 0:
            continue
        text = pytesseract.image_to_string(cell, config=_WEIGHT_VALUE_CONFIG).strip()
        result[field] = text or None

    return result


# --------------------------------------------------------------------------
# Zone 5 (Transporter stamp) - pixel-density presence checks, no OCR
# involved for the presence flag itself.
# --------------------------------------------------------------------------
#
# Fraction of a crop's own pixels that read as "ink" (Otsu-threshold
# darkness, not OCR, not the HSV colour separation ink_separation.py's own
# split_ink_layers uses - that one needs a COLOUR difference between print
# and ink, which this project's own scans do not have: confirmed on
# LR_1350/LR_1351, both read as fully grayscale, so split_ink_layers
# returns an EMPTY ink layer on both - see Zone 6/7's own module comment).
# A plain darkness-density check works regardless of colour, which is why
# it is used for a presence FLAG here rather than for reading handwritten
# TEXT (Zone 6/7's own fields) - a flag only needs "is there materially
# more ink here than a blank cell", not to isolate what kind of ink it is.
#
# Calibrated against this project's own scans: an empty, blank cell reads
# under 1% density; a printed rule crossing the crop reads a few percent;
# a rubber-stamp seal or a handwritten signature reads well past 10%.
INK_PRESENCE_DENSITY_THRESHOLD = 0.05


def _ink_density(pil_image: Image.Image, bbox) -> float:
    """Fraction of ``bbox`` on ``pil_image`` that reads as ink, by a plain
    Otsu darkness threshold - no OCR, no colour separation."""
    left, top, right, bottom = (int(v) for v in bbox)
    left, top = max(0, left), max(0, top)
    right, bottom = min(pil_image.width, right), min(pil_image.height, bottom)
    if right <= left or bottom <= top:
        return 0.0
    crop_np = np.array(pil_image.crop((left, top, right, bottom)).convert("RGB"))
    gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
    _threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return float(np.count_nonzero(binary)) / float(binary.size)


def _extract_zone5(page_results: list, boxes: list, proc_image: Image.Image,
                    img_w: int, img_h: int) -> dict:
    """Zone 5 (Transporter stamp): ``text_above_transporter_stamp`` (OCR,
    printed text) and ``transporter_stamp_present`` (pixel density, no
    OCR).

    Anchored on "STAMP & SIGNATURE OF THE TRANSPORTER" when that reads at
    all, else on "AUTHORISED SIGNATORY" - the other printed label in the
    same box, positioned below the stamp/signature area - which is what
    actually anchors this on both LR_1350/LR_1351: the stamp mark itself
    visually overlaps "STAMP & SIGNATURE..." on both, defeating Tesseract
    there specifically, confirmed directly (checked both the preprocessed
    and the original page - neither reads it).

    Whichever anchor is found, the crop used for both fields is the
    ``detect_boxes`` rectangle CONTAINING it - the printed box bordering
    this whole section - not a position relative to one specific label's
    own line, so the result does not depend on which of the two anchors
    actually matched. ``text_above_transporter_stamp`` reads the row
    directly above that box's own top edge (confirmed on LR_1350: this is
    "STO NO : 44003797555", printed just above the box); the stamp/
    signature ink itself sits in the box's own vertical middle, between
    whichever label anchored it at the top and "AUTHORISED SIGNATORY" (or
    the box's own bottom edge) - so the ink-density check is restricted
    to that middle band, not the whole box, so the printed labels' own
    ink does not by itself register as "stamp present".
    """
    result = {field: None for field in ZONE_FIELD_ORDER["zone5"]}

    anchor = _locate_anchor(page_results, "zone5") or _locate_anchor(page_results, "zone5_fallback")
    if anchor is None:
        return result

    ax1, ay1, ax2, ay2 = _box_bounds(anchor)
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    anchor_height = ay2 - ay1

    containing = [
        box for box in boxes
        if box[0] <= acx <= box[2] and box[1] <= acy <= box[3]
    ]
    if containing:
        box_left, box_top, box_right, box_bottom = min(
            containing, key=lambda box: (box[2] - box[0]) * (box[3] - box[1])
        )
    else:
        box_left = max(0, ax1 - NEIGHBOR_MARGIN)
        box_right = min(img_w, ax2 + NEIGHBOR_MARGIN)
        box_top = max(0, ay1 - anchor_height)
        box_bottom = min(img_h, ay2 + anchor_height * 3)

    above_top = max(0, box_top - anchor_height * 2)
    above_box = (box_left, above_top, box_right, box_top)
    segments = ocr_segments(proc_image.crop(above_box))
    text_above = " ".join(segment.text for segment in sorted(segments, key=lambda s: (s.y, s.x))).strip()
    result["text_above_transporter_stamp"] = text_above or None

    band_top = box_top + (box_bottom - box_top) * 0.25
    band_bottom = box_top + (box_bottom - box_top) * 0.85
    density = _ink_density(proc_image, (box_left, band_top, box_right, band_bottom))
    result["transporter_stamp_present"] = density >= INK_PRESENCE_DENSITY_THRESHOLD

    return result


# --------------------------------------------------------------------------
# Zone 6 (Customer's Seal & Sign, per the FINAL SPEC - still keyed "zone7"
# internally, see ZONE_ANCHORS' own comment) - handwritten, not read through
# ink_separation.py's own HSV colour-layer split.
# --------------------------------------------------------------------------
#
# That approach was tried first (reusing find_acknowledgement_zone() /
# extract_acknowledgement_box() unchanged) and confirmed unusable on this
# project's own scans: split_ink_layers()'s own _is_grayscale() check
# reads True for both LR_1350 and LR_1351 (every pixel's R, G and B
# channel already equal - this PDF's own content has no colour signal at
# all, confirmed directly against fitz's own pixmap, not merely on the
# rendered PIL image), so the ink layer it returns is entirely blank
# regardless of what is actually handwritten on the page.
#
# The real fix is not a masking-technique swap, though: this box is an
# EMPTY ANSWER CELL on the printed template - unlike Zone 5's own "STAMP &
# SIGNATURE OF THE TRANSPORTER", which genuinely has a stamp overlapping
# PRINTED text that must be told apart, this one has nothing printed
# underneath where the handwriting goes. Isolating an "ink layer" was never
# structurally necessary here - there is no printed baseline to separate
# from. It is read directly off the ordinary (grayscale) crop instead,
# located the same anchor-based way every other zone in this file already
# is (``_locate_anchor`` against ``page_results``).

# How many multiples of the "CUSTOMER'S SEAL & SIGN" label's own printed
# width/height a CONTAINING detect_boxes rectangle is allowed to be before
# it is rejected as a much larger, unrelated section's own outer border
# rather than this specific box - see _extract_zone7's own comment.
# Height gets a larger allowance than width: the box is taller than the
# label to hold a stamp and a short handwritten note, but not much wider.
ZONE7_MAX_WIDTH_MULTIPLIER = 3
ZONE7_MAX_HEIGHT_MULTIPLIER = 15


def _extract_zone7(page_results: list, boxes: list, proc_image: Image.Image,
                    img_w: int, img_h: int) -> dict:
    """Zone 6 (Customer's Seal & Sign) - located the same anchor-based way
    every other zone is, crop taken as the ``detect_boxes`` rectangle
    CONTAINING the anchor (the printed box bordering this section, same
    technique Zone 5 uses for its own box), then sub-cropped to ONLY the
    region BELOW the anchor's own bottom edge - "CUSTOMER'S SEAL & SIGN" is
    always this box's own top printed line (per the FINAL SPEC), so this is
    what keeps the printed header out of ``customer_seal_text_handwritten``,
    reading only the handwriting/stamp beneath it. Both the free-text OCR
    and the presence density check read that same sub-crop, so the ink a
    signature happens to make is never counted twice under two different
    boundaries.
    """
    result = {field: None for field in ZONE_FIELD_ORDER["zone7"]}
    result["customer_seal_present"] = False

    anchor = _locate_anchor(page_results, "zone7")
    if anchor is None:
        return result

    ax1, ay1, ax2, ay2 = _box_bounds(anchor)
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    anchor_width = ax2 - ax1
    anchor_height = ay2 - ay1

    fallback_box = (
        max(0, ax1 - NEIGHBOR_MARGIN), max(0, ay1 - NEIGHBOR_MARGIN),
        min(img_w, ax2 + NEIGHBOR_MARGIN * 20), min(img_h, ay2 + anchor_height * 6),
    )

    # A CONTAINING box this much bigger than the label's own printed size
    # is a much larger section's own outer border (the whole bottom
    # block of the form, ODN NO through AUTHORISED SIGNATORY together),
    # not this specific small seal/signature box - confirmed on LR_1351,
    # where the only containing box found spans nearly the page's own
    # full width and 10x the label's own height, and dumped that entire
    # section's text into the result. Rejected by shape, the same
    # principle Zone 4's own crop already uses for a much-too-tall
    # neighbouring section.
    containing = [
        box for box in boxes
        if box[0] <= acx <= box[2] and box[1] <= acy <= box[3]
        and (box[2] - box[0]) <= anchor_width * ZONE7_MAX_WIDTH_MULTIPLIER
        and (box[3] - box[1]) <= anchor_height * ZONE7_MAX_HEIGHT_MULTIPLIER
    ]
    if containing:
        box = min(containing, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    else:
        box = fallback_box

    # "CUSTOMER'S SEAL & SIGN" is this box's own top printed line - the
    # sub-crop starts at the anchor's own bottom edge, never higher, so the
    # printed label itself is excluded regardless of which box (containing
    # or fallback) was used above.
    below_label_box = (box[0], max(box[1], ay2), box[2], box[3])

    text, _confidence = _read_free_text(proc_image, below_label_box)
    result["customer_seal_text_handwritten"] = text or None
    result["customer_seal_present"] = (
        _ink_density(proc_image, below_label_box) >= INK_PRESENCE_DENSITY_THRESHOLD
    )

    return result


def _read_free_text(pil_image: Image.Image, bbox) -> tuple:
    """``(text, mean_word_confidence)`` for a free-text crop - no
    whitelist, no grid-cell boundary math, since this box has no internal
    columns."""
    left, top, right, bottom = (int(v) for v in bbox)
    left, top = max(0, left), max(0, top)
    right, bottom = min(pil_image.width, right), min(pil_image.height, bottom)
    if right <= left or bottom <= top:
        return "", 0.0

    crop = pil_image.crop((left, top, right, bottom)).convert("L")
    data = pytesseract.image_to_data(crop, config="--psm 6", output_type=pytesseract.Output.DICT)
    words = [
        (text, float(conf)) for text, conf in zip(data["text"], data["conf"])
        if text.strip() and str(conf) not in ("-1", "")
    ]
    if not words:
        return "", 0.0
    text = " ".join(word for word, _ in words)
    confidence = sum(conf for _, conf in words) / len(words)
    return text, confidence


ZONES_BUILT = ("zone1", "zone2", "zone3", "zone4")
TABLE_ZONES = ("zone4",)


def _mean_phrase_confidence(results: list) -> float:
    """Mean confidence across ``get_ocr_results``' own phrases (0-1 scale
    each) - the comparison ``extract_lr`` uses to pick between the
    original page and its own preprocessed version, computed from OCR
    results it needed anyway rather than a second, separate metric call."""
    if not results:
        return 0.0
    return sum(item["confidence"] for item in results) / len(results)


def extract_lr(image: Image.Image) -> dict:
    """Extract one LR page into a flat dict - the public entry point
    ``app.py`` imports and calls once per LR page (``extract_lr(image)``),
    same name and signature as the pre-rebuild version of this file.

    Extracts the FINAL SPEC's 6 zones - LR Number, Truck No, Weight,
    Packages, Transporter stamp and Customer's Seal & Sign - and returns
    exactly ``LR_OUTPUT_FIELDS``, ``None`` (or ``False`` for the two
    presence flags, once a zone's box itself was located) when a field or
    its zone was not found. Nothing else - no unclassified_* fallback, no
    confidence score, no low-confidence flag - ever reaches the result.

    Picks between the original page and preprocessing.preprocess_for_ocr's
    own cleaned-up version itself, rather than calling
    preprocessing.preprocess_for_ocr_if_better() - that function already
    makes the identical choice (OCR both, keep whichever reads better),
    but throws its own OCR results away once it has decided, so every
    caller ends up running a THIRD whole-page OCR pass just to get
    results back. Confirmed as real, measured overhead, not a guess:
    profiled at 4.62s (preprocess_for_ocr_if_better's own two internal
    passes) + 2.62s (this file's own separate get_ocr_results call) of an
    18.29s total single-page extraction. Reusing whichever pass already
    won removes that third pass entirely; every zone below reads from
    whichever of ``page_results``/``proc_image`` this comparison kept.
    """
    import os, time
    _DEBUG_TIMING = os.environ.get("LR_DEBUG_TIMING")
    _t = time.perf_counter()
    def _lap(label):
        nonlocal _t
        if _DEBUG_TIMING:
            now = time.perf_counter()
            print(f"  [{label}] {now - _t:.2f}s")
            _t = now

    try:
        processed_image = preprocess_for_ocr(image)
    except Exception as exc:  # noqa: BLE001 - must never break extraction
        print(f"warning: preprocess_for_ocr failed, using original page: {exc}")
        processed_image = None
    _lap("preprocess_for_ocr")

    original_np = np.array(image.convert("RGB"))
    original_results = get_ocr_results(original_np)
    _lap("get_ocr_results(original)")

    if processed_image is not None:
        processed_np = np.array(processed_image.convert("RGB"))
        processed_results = get_ocr_results(processed_np)
    else:
        processed_np = None
        processed_results = []
    _lap("get_ocr_results(processed)")

    if processed_results and _mean_phrase_confidence(processed_results) >= _mean_phrase_confidence(original_results):
        proc_image, img_np, page_results = processed_image, processed_np, processed_results
    else:
        proc_image, img_np, page_results = image, original_np, original_results

    img_h, img_w = img_np.shape[:2]
    h_lines, _v_lines = detect_grid(img_np)
    boxes = detect_boxes(img_np)
    _lap("detect_grid+detect_boxes")

    record: dict = {}
    zone3_crop_box = None
    for zone_id in ZONES_BUILT:
        anchor_box = _locate_anchor(page_results, zone_id)
        if anchor_box is None:
            record.update({field: None for field in ZONE_FIELD_ORDER[zone_id]})
            continue

        if zone_id in TABLE_ZONES:
            crop_box, _low_confidence = _table_crop_box(anchor_box, h_lines, boxes, img_w, img_h)
        else:
            crop_box, _low_confidence = _zone_crop_box(anchor_box, h_lines, boxes, img_w, img_h)
        # _low_confidence (whether the crop fell back to fixed page-fraction
        # padding - see _zone_crop_box/_table_crop_box) is no longer
        # surfaced in the output; the FINAL SPEC's field list has no room
        # for a per-zone flag alongside it.
        if zone_id == "zone3":
            zone3_crop_box = crop_box
        crop = proc_image.crop(crop_box)

        if zone_id in TABLE_ZONES:
            record.update(_extract_positional_table(np.array(crop.convert("RGB")), ZONE_FIELD_ORDER[zone_id]))
        else:
            segments = ocr_segments(crop)
            raw = extract_label_values(segments)
            record.update(_alias_zone_fields(zone_id, raw))
        _lap(f"zone {zone_id}")

    # Zone 3's own weight sub-table (LORRY TARE WT | NET WT | GROSS WT) is
    # read separately from the rest of Zone 3, anchored on its own header
    # ("zone3_weights", not the outer "zone3" anchor "route" was just
    # found through above) - see ZONE_ANCHORS' own comment and
    # _table_crop_box's docstring for why: it is a distinct 3-column ruled
    # table sitting below the ROUTE line, not more label/value content in
    # the same crop. Overwrites the ``None`` placeholders the loop above
    # already put in the record for lorry_tare_wt/net_wt/gross_wt (Zone
    # 3's own ZONE_FIELD_SYNONYMS no longer names them, on purpose - see
    # that dict's own comment).
    #
    # The "LORRY TARE WT" anchor is NOT searched for in the WHOLE-PAGE OCR
    # pass (``page_results``) the way every other zone's anchor is -
    # confirmed on LR_1350's own scan that whole-page OCR merges this
    # label with an unrelated printed clause several columns away ("...
    # PIOMBINO - 57025 LORRY TARE WT"), long and disproportionate enough
    # relative to its own synonym that ``_safe_synonym_matches`` correctly
    # rejects it as a likely buried-substring false positive (the same
    # guard that fixed Zone 1's own to_destination bug) - it just also,
    # rightly, has no way to tell that THIS particular over-long match
    # happens to be real. Re-OCR'ing Zone 3's own already-correctly-
    # bounded outer crop instead - confirmed on both LR_1350 and LR_1351 -
    # reads "LORRY TARE WT" cleanly, because the unrelated clause was
    # never inside that crop to begin with. The anchor box this finds is
    # in the CROP's own coordinate space, so it is translated back to the
    # full page's coordinate space (adding the crop's own top-left offset)
    # before handing it to ``_table_crop_box``, which expects whole-page
    # anchor/h_lines/boxes - the same mechanism Zone 4's own anchor
    # already goes through, just sourced from a second, narrower OCR pass
    # instead of the first, whole-page one.
    if zone3_crop_box is not None:
        # Extended a bit past the outer zone3 crop's own bottom edge before
        # searching for the weight table's own header - confirmed necessary
        # on this project's own scans: ``_zone_crop_box``'s bottom edge for
        # "zone3" comes from the WHOLE PAGE's own h_lines (this form's broad
        # section rules, sparse by the module docstring's own admission),
        # which on some pages puts a rule between the ROUTE line and the
        # LORRY TARE WT header itself, stopping the crop before the header
        # ever appears in it at all - the sub-anchor search below then finds
        # nothing to anchor on, and the whole weight table is lost rather
        # than merely mis-bounded. A fixed fraction of the outer crop's own
        # height is a safety margin only for THIS search, not a claim about
        # where the table actually sits - _zone3_weight_crop_box derives the
        # table's own top/bottom from the header's own OCR'd position once
        # found, not from this margin.
        zone3_crop_box = (
            zone3_crop_box[0], zone3_crop_box[1], zone3_crop_box[2],
            min(img_h, zone3_crop_box[3] + int((zone3_crop_box[3] - zone3_crop_box[1]) * 0.6)),
        )
        zone3_crop_np = np.array(proc_image.crop(zone3_crop_box).convert("RGB"))
        zone3_local_h_lines, _zone3_local_v_lines = detect_grid(zone3_crop_np)
        zone3_sub_results = get_ocr_results(zone3_crop_np)
        sub_anchor = _locate_anchor(zone3_sub_results, "zone3_weights")
    else:
        sub_anchor = None

    if sub_anchor is not None:
        dx, dy = zone3_crop_box[0], zone3_crop_box[1]
        weight_anchor = [[x + dx, y + dy] for x, y in sub_anchor]
        weight_crop_box = _zone3_weight_crop_box(
            weight_anchor, zone3_crop_box, zone3_local_h_lines, img_w, img_h
        )
        weight_crop = proc_image.crop(weight_crop_box)
        # Upscaled before either read is attempted - the same fix
        # invoice_extractor.py's own _ocr_zone already applies to a small
        # anchored crop (ZONE_SCALE), confirmed necessary here too: this
        # table's own crop is only ~900x120px at this page's native 300 DPI,
        # too small for Tesseract to read its digits reliably even though
        # the printed text itself is clean (confirmed by eye against the
        # original scan) - extract_cells's own per-cell OCR read pure noise
        # off the native-resolution crop ('42 E07' for a printed "13.660")
        # and read it perfectly once upscaled 3x, nothing else changed.
        weight_crop_np = np.array(weight_crop.convert("RGB"))
        weight_crop_np = cv2.resize(
            weight_crop_np, None, fx=ZONE3_WEIGHT_UPSCALE, fy=ZONE3_WEIGHT_UPSCALE,
            interpolation=cv2.INTER_CUBIC,
        )
        weight_fields = _extract_positional_table(weight_crop_np, ZONE3_WEIGHT_FIELDS)
        if not any(weight_fields.values()):
            # detect_grid's own v_lines detection refuses to run at all
            # below its MIN_LINES gate on h_lines (bill_extractor.py's own
            # constant) - this table's own crop reliably comes back with
            # only 2 h_lines (a 2-row, likely bottom-unbordered table has no
            # third rule to find), so extract_cells never gets a column
            # split and _extract_positional_table above returns every field
            # None. Falling back to a fixed-proportion column split (see
            # _extract_positional_row_by_proportion's own docstring) is what
            # actually reaches the data in that case.
            anchor_height = (sub_anchor[2][1] - sub_anchor[0][1]) * ZONE3_WEIGHT_UPSCALE
            weight_fields = _extract_positional_row_by_proportion(
                weight_crop_np, ZONE3_WEIGHT_FIELDS, anchor_height
            )
        record.update(weight_fields)
    else:
        record.update({field: None for field in ZONE3_WEIGHT_FIELDS})
    _lap("zone3 weight sub-table")

    record.update(_extract_zone5(page_results, boxes, proc_image, img_w, img_h))
    _lap("zone5")
    record.update(_extract_zone7(page_results, boxes, proc_image, img_w, img_h))
    _lap("zone7")

    # Final whitelist: exactly the FINAL SPEC's 24 fields, in its own
    # order, nothing else - regardless of what any zone above happened to
    # add to `record` along the way (this is also what keeps a stray
    # unclassified_*/low-confidence key from ever reaching the sheet, even
    # if some future edit to a zone above reintroduces one).
    return {field: record.get(field) for field in LR_OUTPUT_FIELDS}


if __name__ == '__main__':
    from pdf_handler import pdf_to_images
    imgs = pdf_to_images('test_lr.pdf')
    result = extract_lr(imgs[0])
    print(result)
