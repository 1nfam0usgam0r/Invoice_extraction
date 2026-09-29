"""Extract a single LR (lorry receipt) page.

Schema-free: this page's own label/value pairs are detected the same way
ocr/invoice_extractor.py already detects the cover form's - by position
(a value beside or under its label), by an inline "LABEL: value" split, by
nothing more than what the page itself prints - never against a fixed,
enumerated list of expected labels. A field name is the label's own OCR'd
text, cleaned to a key; a page that prints 30 fields yields 30 keys, one
that prints 15 yields 15, and neither is a bug. ocr/column_classifier.py's
LR_FIELD_SYNONYMS only ALIASES whichever of those keys the reconciler and
normaliser specifically need (lr_no, truck_no, net_wt, ...) to the exact
snake_case name they already expect - it never decides what gets extracted.

One engine reads the printed form (reused wholesale from invoice_extractor.py,
not reimplemented); a second, separate one reads the handwritten
acknowledgement box (ink_separation.py).
"""

import re

import cv2
import numpy as np
from PIL import Image

try:
    from .column_classifier import (
        LR_FIELD_SYNONYMS, best_synonym_matches, greedy_assign, looks_like_plausible_value,
    )
    from .ink_separation import (
        extract_acknowledgement_box, find_acknowledgement_zone, split_ink_layers,
    )
    from .invoice_extractor import extract_label_values, ocr_segments
    from .preprocessing import preprocess_for_ocr_if_better
    from .reader import get_ocr_results
except ImportError:  # running this file directly from inside ocr/
    from column_classifier import (
        LR_FIELD_SYNONYMS, best_synonym_matches, greedy_assign, looks_like_plausible_value,
    )
    from ink_separation import (
        extract_acknowledgement_box, find_acknowledgement_zone, split_ink_layers,
    )
    from invoice_extractor import extract_label_values, ocr_segments
    from preprocessing import preprocess_for_ocr_if_better
    from reader import get_ocr_results

# --------------------------------------------------------------------------
# This form's own printed section boxes - a fencing mechanism, not a table.
# --------------------------------------------------------------------------
#
# Unlike the bill/tax invoice tables (one uniform grid of rows and columns,
# see ocr/bill_extractor.py's detect_grid), this form's printed rules mark
# out a handful of irregularly-sized SECTIONS at different heights (the
# transporter block, the LR No/Date/Shipment block, the truck block, the
# notice block, the weight/amount table, ...) - each one internally just
# stacked "LABEL : value" lines or a label beside its value, not a further
# ruled sub-grid. Detecting these sections by contour (rather than by
# intersecting one global row/column grid, which assumes a single uniform
# table) and fencing the generic label/value pass to run separately inside
# each one is what stops a label in one section finding its value in a
# neighbouring one - confirmed on this project's own scan, where "AMOUNT"'s
# real value used to lose to an unrelated insurance-disclaimer box's own
# use of the word "Amount", and "LORRY TARE WT" used to bleed into a
# "Demurrage Chargeable after" notice several columns away, both because
# nothing bounded the search to the label's own printed box.

# Kernel width/height for finding this form's own printed section lines, as
# a fraction of the page's own dimensions - not a fixed pixel count, so
# this holds at any scan DPI.
BOX_H_KERNEL_RATIO = 0.05
BOX_V_KERNEL_RATIO = 0.015

# A detected rectangle smaller than this fraction of the page's own area is
# scan noise (a stray mark, a thin line fragment), not a real section. Small
# but genuine cells (the little weight-table column headers, each well
# under a hundredth of the page) still have to clear this, hence the low
# bound - confirmed against this project's own scan, where several real
# header cells measured 0.0018-0.0031 of the page area.
MIN_BOX_AREA_RATIO = 0.0002
# A detected rectangle bigger than this fraction of the page's own area is
# the page's own outer border, not one section worth fencing on its own.
MAX_BOX_AREA_RATIO = 0.35


def detect_boxes(img_np: np.ndarray) -> list:
    """This page's own printed section boxes, as ``(x1, y1, x2, y2)``
    pixel rectangles - found by contour, not by intersecting one global
    row/column grid (see the module docstring for why). Order is not
    meaningful; every downstream use only tests containment.
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )

    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(2, int(width * BOX_H_KERNEL_RATIO)), 1)
    )
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(2, int(height * BOX_V_KERNEL_RATIO)))
    )
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    grid = cv2.bitwise_or(h_lines, v_lines)
    grid = cv2.dilate(grid, np.ones((3, 3), np.uint8), iterations=1)

    # Inverted: each enclosed (unruled) area becomes one connected white
    # blob, found as its own contour - the standard technique for reading
    # bordered cells off a scanned form, and the reason this differs from
    # bill_extractor.py's own grid detection (which instead intersects
    # separately-found horizontal/vertical line positions, the right model
    # for one uniform table, not a form's irregular local boxes).
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


def _box_for_segment(segment, boxes: list):
    """The first detected box whose rectangle contains ``segment``'s own
    centre, or ``None`` when it falls inside none of them (the page's own
    section lines were too faint to detect there, or it genuinely sits
    outside every section - the bottom ODN NO/INV VALUE block prints as
    plain inline text with no box around it at all)."""
    cx, cy = segment.cx, segment.cy
    for x1, y1, x2, y2 in boxes:
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return (x1, y1, x2, y2)
    return None


def _merge_dynamic_fields(pieces: list) -> dict:
    """Combine several small label/value dicts (one per fenced section,
    see ``extract_lr``) into one, numbering a key that more than one
    section happened to produce (two sections both printing a field this
    form's own template names "STATE", say) rather than letting the later
    one silently overwrite the earlier - the same rule invoice_extractor.
    py's own _store uses for a label printed twice on the page."""
    merged: dict = {}
    for piece in pieces:
        for key, value in piece.items():
            final_key = key
            suffix = 2
            while final_key in merged:
                final_key = f"{key}_{suffix}"
                suffix += 1
            merged[final_key] = value
    return merged


# --------------------------------------------------------------------------
# Noise floor and paragraph ceiling.
# --------------------------------------------------------------------------
#
# The generic pairer above sometimes turns a stray OCR glyph into its own
# "field" (a lone punctuation mark, a single garbled letter it mistook for
# a label) and, on this page's dense boilerplate paragraphs, sometimes
# mistakes an arbitrary early clause for a label and grabs the rest of the
# sentence as its "value" - two failure modes with the same root cause
# (treating noise or prose as if it were a form field) and the same fix:
# recognise the SHAPE of what a real short value or a real short label
# looks like, and drop whatever does not clear that bar, rather than
# emitting it as a fake field.

# A cleaned key/value shorter than this many alphanumeric characters is
# noise UNLESS it matches one of the generic value shapes (see
# column_classifier.looks_like_plausible_value) - a state code ("37")
# survives that check; a lone punctuation mark or single garbled letter
# does not. Not a length cutoff alone - the shape check is what actually
# decides.
MIN_ALNUM_CHARS = 3

# A value is boilerplate prose, not a field's value, once it runs this
# many words or carries this many sentence-ending marks - a real label's
# value is a short answer (a number, a name, a code), never a paragraph.
MAX_VALUE_WORDS = 20
MAX_SENTENCE_ENDS = 2
# A period is only a sentence end when it is not sandwiched between two
# digits - "13.05.2026" is a date, not two sentences, and would otherwise
# count 2 "sentence ends" on a value that is neither long nor prose.
_SENTENCE_END = re.compile(r"(?<!\d)[.!?](?!\d)")

# A "label" this many words long is not a label at all - it is a clause of
# prose the generic pairer mistook for one (see the section docstring).
MAX_LABEL_WORDS = 5

_ALNUM_ONLY = re.compile(r"[^A-Za-z0-9]+")


def _alnum_len(text: str) -> int:
    return len(_ALNUM_ONLY.sub("", str(text or "")))


def _is_noise(text: str) -> bool:
    """Too short to be a real field's key or value, and not shaped like
    one of the generic short-but-legitimate values either.

    A single character is never rescued by shape alone - a state code
    ("37") is the shortest genuine value this exception is meant for, and
    it is already two characters; a lone digit or letter reaching here is
    consistently a stray OCR glyph, not a real one-character field.
    """
    length = _alnum_len(text)
    if length >= MIN_ALNUM_CHARS:
        return False
    return length < 2 or not looks_like_plausible_value(text)


def _is_paragraph(text: str) -> bool:
    """Boilerplate prose, not a field's value - see the section docstring."""
    text = str(text or "")
    return len(text.split()) > MAX_VALUE_WORDS or len(_SENTENCE_END.findall(text)) >= MAX_SENTENCE_ENDS


def _filter_generic_fields(result: dict) -> dict:
    """Drop noise-floor and paragraph-ceiling entries from this page's own
    generically-detected fields - never called on the handful of already-
    canonicalised, already-trusted keys added later (lr_no, net_wt, ...),
    only on the raw output of the generic pairer, which is exactly where a
    stray glyph or a mis-split sentence would otherwise reach the sheet as
    if it were a real field.

    ``Unmatched_Text_*``/``Stamp_*`` keys are dropped outright, not merely
    filtered by shape: both are invoice_extractor.py's own explicit "this
    is not a label:value pair" fallbacks (leftover text nothing consumed,
    a dense cluster it guessed was a rubber stamp) - by definition not a
    form field, genuine or noise, just unassociated text. Confirmed on
    this project's own scan: of 93 total keys before this cut, only 33
    were real label:value pairs; the other 60 were exactly these two
    fallback categories.
    """
    filtered: dict = {}
    for key, value in result.items():
        if key.startswith("Unmatched_Text") or key.startswith("Stamp_"):
            continue
        label_text = key.replace("_", " ")
        if _is_noise(label_text) or len(label_text.split()) > MAX_LABEL_WORDS:
            continue
        if value is not None and (_is_noise(value) or _is_paragraph(value)):
            continue
        filtered[key] = value
    return filtered


def _split_by_gap(segments: list, gap_ratio: float = 3.0) -> list:
    """``segments`` (already known to share no detected box - see
    ``extract_lr``), split into groups separated by an unusually large
    vertical gap.

    Two pieces of unboxed content from opposite ends of the page (this
    form's own bottom inline "LABEL : value" block, and a stray fragment
    near the top that no detected box happened to enclose) would otherwise
    all run through one ``extract_label_values`` call as if they were one
    contiguous region - exactly the cross-section bleed detecting boxes in
    the first place was meant to stop, just for whatever fell outside every
    detected box rather than inside one.
    """
    if not segments:
        return []
    ordered = sorted(segments, key=lambda item: (item.y, item.x))
    heights = sorted(item.height for item in ordered if item.height > 0)
    unit = heights[len(heights) // 2] if heights else 15.0

    groups = [[ordered[0]]]
    for segment in ordered[1:]:
        previous = groups[-1][-1]
        gap = segment.y - (previous.y + previous.height)
        if gap > gap_ratio * unit:
            groups.append([])
        groups[-1].append(segment)
    return groups

# How close a detected label key has to be to one of LR_FIELD_SYNONYMS'
# entries before it is aliased - rapidfuzz's 0-100 scale, the same
# convention ocr/column_classifier.py's own CAPTION_MATCH_THRESHOLD uses.
LR_ALIAS_THRESHOLD = 82

# A value failing these is garbage, not a reading worth passing downstream.
VALID_LR_NO = re.compile(r"^[A-Z]{0,3}\d{7,12}$")
VALID_SHIPMENT_NO = re.compile(r"^\d{7,9}$")
HAS_LETTER = re.compile(r"[A-Za-z]")

_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")


def _validated(value, pattern):
    """Return ``value`` when it matches ``pattern``, else None.

    Separators the OCR may have inserted ("LR-848501350") are ignored when
    matching, so a recoverable number is not thrown away.
    """
    text = str(value or "").strip()
    if not text:
        return None
    joined = _NON_ALNUM.sub("", text).upper()
    return value if pattern.match(joined) else None


# --------------------------------------------------------------------------
# Value-shape refinement for the handful of numeric fields proximity gets
# wrong most often - a correction of an already-discovered field's VALUE,
# not a decision about which fields exist (that stays entirely with the
# generic pass above). Kept from the previous, fixed-schema version of this
# file largely unchanged, since it was already proven against this
# project's own scans across many pages; the one thing it no longer does is
# consult a separate hardcoded label list for its anchor - it now reuses
# LR_FIELD_SYNONYMS, the same canonical pool _alias_canonical_fields()
# already draws from, so there is still only one place this form's field
# names are named as data.
# --------------------------------------------------------------------------

LR_NO_PATTERNS = (re.compile(r"^\d{7,12}$"), re.compile(r"^[A-Z]{1,3}\d{7,12}$"))
SHIPMENT_NO_PATTERN = re.compile(r"^\d{8}$")
WEIGHT_PATTERN = re.compile(r"^\d{1,3}\.\d{2,3}$")

# key -> (patterns, region, anchor_to_label)
TARGETED_FIELDS = {
    "lr_no": (LR_NO_PATTERNS, "top", True),
    "shipment_no": ((SHIPMENT_NO_PATTERN,), "top", False),
    "net_wt": ((WEIGHT_PATTERN,), "bottom", True),
    "gross_wt": ((WEIGHT_PATTERN,), "bottom", True),
}

# Fraction of the page height splitting the header area from the weight table.
TOP_REGION_FRACTION = 0.4


def _box_bounds(box) -> tuple:
    """Return ``(x1, y1, x2, y2)`` for a 4-point box."""
    points = np.asarray(box, dtype=float)
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def _locate_canonical_anchor(results: list, canonical_field: str):
    """The box of whichever OCR phrase (from ``get_ocr_results``) best
    matches ``LR_FIELD_SYNONYMS[canonical_field]`` - an anchor point for
    ``targeted_search``, drawn from the same canonical synonym pool
    ``_alias_canonical_fields`` already uses, not a separate label list."""
    cleaned = {index: item["text"].strip().lower() for index, item in enumerate(results)}
    synonyms = {canonical_field: LR_FIELD_SYNONYMS[canonical_field]}
    candidates = best_synonym_matches(cleaned, synonyms, LR_ALIAS_THRESHOLD)
    if not candidates:
        return None
    _score, index, _field = max(candidates, key=lambda item: item[0])
    return results[index]["box"]


def targeted_search(results, page_height, patterns, region, anchor=None, taken=()):
    """Find a value by pattern and page region instead of by proximity.

    ``region`` is "top" (above ``TOP_REGION_FRACTION`` of the page) or
    "bottom" (at or below it). Without an ``anchor`` the first match in
    reading order wins; with one - the field's own anchor box - the nearest
    match to it wins, which is how two fields sharing a pattern stay apart.
    Boxes listed in ``taken`` are already claimed and are skipped.

    Returns ``(text, confidence, box)`` or None.
    """
    cutoff = page_height * TOP_REGION_FRACTION

    candidates = []
    for item in results:
        box = item["box"]
        confidence = item["confidence"]
        candidate = item["text"].strip()
        if not candidate:
            continue

        bounds = _box_bounds(box)
        if bounds in taken:
            continue

        x1, y1, _x2, y2 = bounds
        y_center = (y1 + y2) / 2
        if region == "top" and y_center >= cutoff:
            continue
        if region == "bottom" and y_center < cutoff:
            continue

        # OCR splits "L 848501350" as often as not, and sometimes glues a
        # stray punctuation character onto the front ("0) 000001359" for a
        # genuine "0000001359") - stripped, not just whitespace, before
        # judging the joined form against the pattern. The decimal point is
        # kept, not stripped as "punctuation" - net_wt/gross_wt's own
        # WEIGHT_PATTERN needs it to match at all.
        joined = re.sub(r"[^A-Za-z0-9.]+", "", candidate).upper()
        if not any(pattern.match(joined) for pattern in patterns):
            continue

        # joined, not the raw candidate: it already passed the pattern
        # check above, so it is guaranteed clean of whatever stray
        # punctuation the raw OCR text carried.
        candidates.append((y_center, x1, joined, confidence, box))

    if not candidates:
        return None

    if anchor is None:
        order = lambda item: (item[0], item[1])
    else:
        ax1, ay1, ax2, ay2 = _box_bounds(anchor)
        anchor_x, anchor_y = (ax1 + ax2) / 2, (ay1 + ay2) / 2
        # The y-offset is weighted far more heavily than x: a label and its
        # value routinely sit a couple hundred pixels apart in x (a wide
        # label box, a value column start well to its right), which plain
        # unweighted distance treats as equally significant as the same
        # couple-hundred-pixel gap in y - readily large enough to prefer an
        # unrelated value one row down/up that happens to sit more directly
        # below the label's own x-centre.
        Y_WEIGHT = 10
        order = lambda item: (item[1] - anchor_x) ** 2 + (Y_WEIGHT * (item[0] - anchor_y)) ** 2

    _y, _x, text, confidence, box = min(candidates, key=order)
    return text, confidence, box


def _refine_numeric_fields(record: dict, image_np: np.ndarray) -> dict:
    """Override ``lr_no``/``shipment_no``/``net_wt``/``gross_wt`` with a
    value-shape-and-region match when one is found - see the section
    docstring above. Returns ``record`` (mutated in place, for the
    caller's convenience)."""
    results = get_ocr_results(image_np)
    page_height = float(image_np.shape[0])
    claimed: set = set()
    for key, (patterns, region, anchor_to_label) in TARGETED_FIELDS.items():
        anchor = _locate_canonical_anchor(results, key) if anchor_to_label else None
        found = targeted_search(results, page_height, patterns, region, anchor, claimed)
        if found is None:
            continue
        value_text, _confidence, value_box = found
        record[key] = value_text
        claimed.add(_box_bounds(value_box))
    return record


def _alias_canonical_fields(result: dict) -> dict:
    """A copy of ``result`` with the reconciler/normaliser's own snake_case
    keys added wherever one of ``result``'s own dynamically-discovered keys
    fuzzy-matches an entry in ``LR_FIELD_SYNONYMS`` - additive only, the
    original key is kept exactly as extracted alongside its alias.

    Runs the same fuzzy-synonym, global-greedy-assignment machinery tier 1
    table-caption matching uses (see ocr/column_classifier.py's
    best_synonym_matches/greedy_assign), just over this page's own detected
    label keys instead of a table's column captions - so two keys competing
    for the same canonical field (say a genuine "Lr No" and an unrelated
    "L C No") resolve by score, not by whichever happened to be seen first.
    """
    cleaned = {key: key.replace("_", " ").lower() for key in result}
    candidates = best_synonym_matches(cleaned, LR_FIELD_SYNONYMS, LR_ALIAS_THRESHOLD)
    assigned = greedy_assign(candidates)

    aliased = dict(result)
    for key, canonical in assigned.items():
        aliased.setdefault(canonical, result[key])
    return aliased


def extract_lr(image: Image.Image) -> dict:
    """Extract one LR page into a flat dict.

    Every printed field this page's own generic label/value pass (see the
    module docstring) finds reaches the sheet under its own cleaned key;
    ``LR_FIELD_SYNONYMS`` additionally aliases whichever of those the
    reconciler/normaliser need to their expected snake_case names. The
    handwritten acknowledgement box (RECEIVED WT / DATE / REMARK) is read
    separately, off the ink layer, and merged in under its own keys.
    """
    # Kept from before any preprocessing below touches it - the acknowledgement
    # box read at the end needs the page's real color, and preprocess_for_ocr
    # can binarize to grayscale.
    original_image = image

    # Optional cleanup pass, kept only on pages Tesseract itself reads better
    # this way (see ocr/preprocessing.py). Never allowed to be the reason
    # extraction fails - any exception here falls back to the original page.
    try:
        image = preprocess_for_ocr_if_better(image)
    except Exception as exc:  # noqa: BLE001 - must never break extraction
        print(f"warning: preprocess_for_ocr_if_better failed, using original page: {exc}")

    segments = ocr_segments(image)
    img_np = np.array(image.convert("RGB"))
    boxes = detect_boxes(img_np)

    # Run the generic label/value pass separately inside each detected
    # section (so a label's search for its value never crosses into a
    # neighbouring section), plus once more per gap-separated cluster of
    # whatever fell inside no detected section at all (this form's own
    # bottom block of plain "LABEL : value" lines has no box around it, and
    # a section the page's own print quality made too faint to detect
    # still needs to fall back to the whole-page behaviour rather than
    # losing its content outright - but never lumped in with unrelated
    # unboxed content from elsewhere on the page, see _split_by_gap).
    by_box: dict = {}
    leftover = []
    for segment in segments:
        box = _box_for_segment(segment, boxes)
        if box is None:
            leftover.append(segment)
        else:
            by_box.setdefault(box, []).append(segment)

    groups = list(by_box.values()) + _split_by_gap(leftover)
    pieces = [extract_label_values(group) for group in groups if group]
    generic = _filter_generic_fields(_merge_dynamic_fields(pieces))
    record = _alias_canonical_fields(generic)
    record = _refine_numeric_fields(record, img_np)

    # Reject values that cannot be what the field is meant to hold; an
    # explicit None beats plausible-looking garbage reaching the workbook.
    if "lr_no" in record:
        record["lr_no"] = _validated(record["lr_no"], VALID_LR_NO)
    if "shipment_no" in record:
        record["shipment_no"] = _validated(record["shipment_no"], VALID_SHIPMENT_NO)
    for key in ("net_wt", "gross_wt"):
        if key in record and (not record[key] or HAS_LETTER.search(str(record[key]))):
            record[key] = None

    scores = [segment.conf for segment in segments if segment.conf > 0]
    record["confidence_avg"] = round(float(np.mean(scores)), 4) if scores else 0.0

    # The customer acknowledgement box (RECEIVED WT / DATE / REMARK), read
    # off the colored-ink layer of the original (pre-preprocessing) page so a
    # handwritten entry there is not competing with the printed form under
    # it. Merged in under new keys only - nothing above this is touched, and
    # any failure here falls back to blank values rather than breaking
    # extraction.
    try:
        printed_layer, ink_layer = split_ink_layers(original_image)
        zone = find_acknowledgement_zone(printed_layer)
        acknowledgement = extract_acknowledgement_box(ink_layer, zone)
    except Exception as exc:  # noqa: BLE001 - must never break extraction
        print(f"warning: acknowledgement box extraction failed: {exc}")
        acknowledgement = {
            "received_wt": "", "received_date": "", "remark": "",
            "remark_confidence": 0.0, "low_confidence": False, "zone_found": False,
        }
    record["received_wt_handwritten"] = acknowledgement["received_wt"]
    record["received_date_handwritten"] = acknowledgement["received_date"]
    record["remark_handwritten"] = acknowledgement["remark"]
    record["remark_handwritten_confidence"] = acknowledgement["remark_confidence"]
    record["remark_handwritten_low_confidence"] = acknowledgement["low_confidence"]
    record["acknowledgement_zone_found"] = acknowledgement["zone_found"]

    return record


if __name__ == '__main__':
    from pdf_handler import pdf_to_images
    imgs = pdf_to_images('test_lr.pdf')
    result = extract_lr(imgs[0])
    print(result)
