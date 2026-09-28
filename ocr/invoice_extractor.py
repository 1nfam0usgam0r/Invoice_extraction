"""Read any key-value form by pairing the labels it prints with their values.

Nothing here knows what an invoice looks like. The page is OCR'd, each piece
of text is judged a LABEL or a VALUE by where it sits and how it reads, and
every label found becomes a key in the returned dict. A form that prints
"Vendor Code" yields ``Vendor_Code``; one that prints something else yields
that instead.

The pixel tolerances the pairing needs - how far right a value may sit, how
close a tick has to be to its box - are all expressed as multiples of the
page's median text height rather than as raw pixels, so the same constants
hold whether the page was rendered at 150 or 300 DPI. The comment on each one
gives the pixel figure it works out to on a 150 DPI scan, where text is
roughly 15px tall.
"""

import re
from collections import defaultdict

import cv2
import numpy as np
import pytesseract
from PIL import Image

try:
    # Importing reader points pytesseract at the bundled binary.
    from .reader import TESSERACT_PATH  # noqa: F401
    from .column_classifier import (
        CAPTION_MATCH_THRESHOLD, INVOICE_FORM_FIELD_SYNONYMS, _clean_caption,
        best_synonym_matches, greedy_assign,
    )
except ImportError:   # running this file directly from inside ocr/
    from reader import TESSERACT_PATH  # noqa: F401
    from column_classifier import (
        CAPTION_MATCH_THRESHOLD, INVOICE_FORM_FIELD_SYNONYMS, _clean_caption,
        best_synonym_matches, greedy_assign,
    )

# bill_extractor.py imports FROM this module (extract_label_values,
# ocr_segments, preprocess) - a top-level "from .bill_extractor import
# detect_grid" here would be circular. detect_grid is only needed inside
# extract_invoice_cover's own helpers below, so it is imported there
# instead, once both modules have already finished loading.


def _detect_grid(img_np):
    try:
        from .bill_extractor import detect_grid
    except ImportError:
        from bill_extractor import detect_grid
    return detect_grid(img_np)

# The page is read twice and the detections merged. psm 11 - sparse text, no
# assumed order - is what actually finds text inside a ruled form, where the
# block modes read the rules as structure and drop whole rows; psm 6 then
# fills in the few phrases sparse mode splits or misses. Measured on the
# sample cover form: psm 6 alone found 8 of 19 checked fields, psm 11 alone
# 15, the two merged 16.
INVOICE_PSMS = (11, 6)
OEM = 3

# A detection is a duplicate of one an earlier pass already found when this
# much of the smaller of the two boxes lies inside the other.
DUPLICATE_OVERLAP = 0.5

# Contrast boost applied before thresholding. Faxed forms come through grey.
CONTRAST_ALPHA = 1.5

# Words on one line are joined into a phrase until the gap between them is
# wider than this many text heights. A word space is ~0.3 heights; the gutter
# between a form's label column and its value column is several.
WORD_GAP = 1.2

# Two detections belong to the same printed row within this much of each
# other vertically. Sparse mode numbers a row's label and its value as
# separate text lines, so rows are rebuilt from geometry instead.
ROW_BAND = 0.7

# Text height is the unit every tolerance below is measured in. A page whose
# median height is nonsense - a nearly blank one - falls back to this.
FALLBACK_UNIT = 15.0

# Two pieces of text count as being on the same line within this much of each
# other vertically. (~15px at 150 DPI.)
Y_TOLERANCE = 1.0

# How far right of a label its value may start. (~300px.)
RIGHT_SEARCH = 20.0

# A value printed under its label rather than beside it must start within this
# vertical distance, and this far left or right of the label. (~45px, ~40px.)
BELOW_BAND = 3.0
BELOW_X_TOLERANCE = 2.7

# A label with no other label closer than this below it owns the lines under
# it too, which is how a four-line approver block stays in one field. (~60px.)
CONTINUATION_GAP = 4.0

# The right-hand neighbour rule only treats a gap this wide as a label/value
# gutter. Phrases are already split at ``WORD_GAP``, so this only has to be
# wide enough that a run-on line is not mistaken for two columns.
GUTTER_MIN = 1.5

# A tick belongs to the code it sits nearest, within this. (~25px.)
TICK_DISTANCE = 1.7

# Stamps are read as a dense cluster of text inside a box this big carrying at
# least this many pieces of text. (~150px square.)
STAMP_BOX = 10.0
STAMP_MIN_ITEMS = 4

# OCR this unsure of itself is not trusted to name a field.
LABEL_MIN_CONF = 0.6

# Lines of short codes with tick boxes beside them, e.g. PO1X PO2X PO2O. The
# empty box beside a code usually comes back glued to it as bracket and
# underscore characters, so those are stripped before matching. Requiring a
# digit is what keeps an ordinary row of short words from being read as a
# set of codes.
CHECKBOX_BOX_GLYPHS = "[]{}()|_.:;"
CHECKBOX_CODE = re.compile(r"^(?=[A-Z0-9]*\d)[A-Z0-9]{2,5}$", re.IGNORECASE)
CHECKBOX_MIN_CODES = 3
TICK_MARKS = {"v", "V", "/", "✓", "✔", "√"}

# Words that mark a phrase as a label when it ends with one of them.
LABEL_SUFFIX = re.compile(
    r"\b(no|nos|number|name|date|code|type|centre|center|amount|by|id)[.:]?$",
    re.IGNORECASE,
)

# Forms print some fields inline - "Company Name: JSW STEEL LIMITED",
# "Amount :- 23,30,645/-" - rather than in two columns. Those are split into
# a label and a value before pairing, so they are read as fields like any
# other rather than being left as loose text.
INLINE_PAIR = re.compile(r"^(?P<label>[^:]{2,60}):[\s\-]*(?P<value>\S.*)$")

# Text shaped like data rather than like a caption: a number, an amount, a
# date. Checked first, so a value is never mistaken for the label of the field
# below it.
VALUE_ONLY = [
    re.compile(r"^[\d,]+(\.\d+)?[/\-]*$"),                  # 23,30,645/-  84008.60
    re.compile(r"^\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}$"),     # 10.06.2026
    re.compile(r"^\d+$"),                                   # 20050923
]

# Stripped off a label before it is turned into a key.
_LEADING_JUNK = re.compile(r"^[\s*\-.•]+")
_TRAILING_JUNK = re.compile(r"[\s:*\-.]+$")
_NON_KEY = re.compile(r"[^0-9A-Za-z_\s]")

# --------------------------------------------------------------------------
# Zonal / anchor-based re-read of this form's own known fields
# --------------------------------------------------------------------------
#
# The generic label/value pairing above reads the whole page at once and
# never assumes what a field is called - it is what keeps this file working
# on any form. But this particular cover form is read from the same
# template on every run, and its own field labels (INVOICE_FORM_FIELD_
# SYNONYMS, in ocr/column_classifier.py) are stable anchors: once one of
# them is found on the page, its value sits in a small, predictable box
# relative to that anchor (the same right-then-below geometry the generic
# pass already searches, see _zone_box), worth re-OCRing tightly and at
# higher fidelity rather than trusting whatever the whole-page sparse/block
# passes happened to read there. This never invents a field the generic
# pass did not already find a label for, and only overwrites that label's
# own value when the zonal re-read actually produced text - nothing here
# risks losing what Unmatched_Text already preserves.

# A zonal crop is anchored tight around one label's own value - by
# construction a single printed line, not a whole cell or column - so this
# is read at psm 7 (single line), unlike the sparse/block passes ocr_
# segments runs over the whole page.
ZONE_SCALE = 3
ZONE_CONFIG = "--psm 7 --oem 3"

# Padding above/below a right-hand zone, so a value's ascenders/descenders
# are not clipped the way a tight box flush to the label's own line would.
ZONE_Y_PAD = 6


def _zone_box(label: "Segment", segments: list, unit: float) -> tuple:
    """The pixel box this label's own value should sit in.

    Same shape as the generic pass's own search (_collect_right, then
    _collect_below): to the right of the label first, and only below it
    when nothing else on the page sits to its right on the same line -
    returned as one box to crop, rather than the matched segments
    themselves, so the value can be re-read straight off the original page
    at higher fidelity.
    """
    right_candidates = [
        segment for segment in segments
        if segment is not label and segment.page == label.page
        and abs(segment.cy - label.cy) <= Y_TOLERANCE * unit
        and segment.x >= label.right
    ]
    if right_candidates:
        left = label.right
        right = label.right + RIGHT_SEARCH * unit
        top = label.y - ZONE_Y_PAD
        bottom = label.bottom + ZONE_Y_PAD
        return left, top, right, bottom

    left = label.x - BELOW_X_TOLERANCE * unit
    right = label.x + BELOW_X_TOLERANCE * unit + RIGHT_SEARCH * unit
    top = label.bottom
    bottom = label.bottom + BELOW_BAND * unit
    return left, top, right, bottom


def _ocr_zone(image: Image.Image, box: tuple) -> str:
    """Re-read one anchored box off the original page: magnified, its own
    Otsu cut, a single line. Empty string, never an exception, for a box
    that lands off the page or on a sliver too thin to hold anything."""
    left, top, right, bottom = box
    width, height = image.size
    left, top = max(0, int(left)), max(0, int(top))
    right, bottom = min(width, int(right)), min(height, int(bottom))
    if right <= left or bottom <= top:
        return ""

    crop = np.array(image.convert("RGB"))[top:bottom, left:right]
    if crop.size == 0:
        return ""

    big = cv2.resize(crop, None, fx=ZONE_SCALE, fy=ZONE_SCALE, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_RGB2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return " ".join(pytesseract.image_to_string(gray, config=ZONE_CONFIG).split())


def _refine_known_fields(label_keys: dict, segments: list, labels: set,
                          unit: float, image: Image.Image, result: dict) -> None:
    """Overwrite this form's own known fields with a tight, anchor-relative
    re-read - see the module docstring above for why.

    ``label_keys`` maps each label ``Segment`` the generic pass already
    paired to the exact key it stored the pairing under (handles a label
    printed twice on the page, where the second occurrence's key carries a
    numeric suffix - see ``_store``). Every label is matched against
    ``INVOICE_FORM_FIELD_SYNONYMS`` by the same fuzzy-synonym, global-
    greedy-assignment machinery tier 1 table-caption matching uses (see
    ocr/column_classifier.py's ``best_synonym_matches``/``greedy_assign``),
    so a label this form's own template does not define competes for
    nothing and is simply left as the generic pass already read it.
    """
    label_list = sorted(labels, key=lambda item: (item.y, item.x))
    cleaned = {index: _clean_caption(label.text) for index, label in enumerate(label_list)}
    candidates = best_synonym_matches(cleaned, INVOICE_FORM_FIELD_SYNONYMS,
                                      CAPTION_MATCH_THRESHOLD)
    assigned = greedy_assign(candidates)

    for index, _field in assigned.items():
        label = label_list[index]
        key = label_keys.get(label) or clean_key(label.text)
        if not key or result.get(key):
            # A gap-filler, not an override: the generic whole-page passes
            # read this same text with the benefit of dictionary-aware,
            # multi-pass context (psm 11 sparse + psm 6 block, see
            # ocr_segments) that a tight single-line psm 7 crop does not
            # have. Confirmed on this form's own scan - overwriting
            # unconditionally replaced a correct "INLAND WORLD LOGISTICS P
            # LTD." with a garbled "| TIN LE" read off the exact same
            # region, because a ruled form's grid line sitting right at the
            # zone's left edge reads as stray punctuation once nothing else
            # on the line gives Tesseract context to correct it. The zonal
            # read is only trusted to fill in a field the generic pass
            # found nothing for at all.
            continue
        value = _ocr_zone(image, _zone_box(label, segments, unit))
        if value:
            result[key] = value


class Segment:
    """One phrase of text on the page, with where it sits."""

    __slots__ = ("text", "page", "line", "x", "y", "width", "height", "conf")

    def __init__(self, text, page, line, x, y, width, height, conf):
        self.text = text
        self.page = page
        self.line = line
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.conf = conf

    @property
    def right(self):
        return self.x + self.width

    @property
    def bottom(self):
        return self.y + self.height

    @property
    def cx(self):
        return self.x + self.width / 2

    @property
    def cy(self):
        return self.y + self.height / 2

    def __repr__(self):
        return f"<Segment p{self.page} y={self.y:.0f} x={self.x:.0f} {self.text!r}>"


def _to_segment(word: dict, page: int) -> Segment:
    return Segment(
        text=word["text"],
        page=page,
        line=0,
        x=float(word["left"]),
        y=float(word["top"]),
        width=float(word["width"]),
        height=float(word["height"]),
        conf=float(word["conf"]),
    )


def preprocess(image: Image.Image) -> Image.Image:
    """Clean a scan up for OCR: grayscale, contrast boost, Otsu threshold.

    Otsu picks its threshold per page, so a pale fax and a dark photocopy both
    come out as black text on white. On the sample form this roughly doubled
    what Tesseract found.
    """
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    boosted = cv2.convertScaleAbs(gray, alpha=CONTRAST_ALPHA, beta=0)
    _threshold, binary = cv2.threshold(
        boosted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return Image.fromarray(binary)


def _duplicate(segment: Segment, kept: list) -> bool:
    """True when an earlier pass already found this piece of text."""
    for other in kept:
        overlap_x = min(segment.right, other.right) - max(segment.x, other.x)
        overlap_y = min(segment.bottom, other.bottom) - max(segment.y, other.y)
        if overlap_x <= 0 or overlap_y <= 0:
            continue
        smaller = min(segment.width * segment.height, other.width * other.height)
        if smaller and (overlap_x * overlap_y) / smaller > DUPLICATE_OVERLAP:
            return True
    return False


def ocr_segments(image: Image.Image, page: int = 0) -> list:
    """Read one page and return its text as phrases, in reading order.

    The page is cleaned up, read once per entry in ``INVOICE_PSMS``, and the
    passes merged - a later pass contributes only the text no earlier one
    found in that spot.
    """
    cleaned = preprocess(image)

    merged: list = []
    for psm in INVOICE_PSMS:
        for segment in _read(cleaned, page, psm):
            if not _duplicate(segment, merged):
                merged.append(segment)

    merged.sort(key=lambda segment: (segment.y, segment.x))
    return merged


def _read(image: Image.Image, page: int, psm: int) -> list:
    """One Tesseract pass.

    Tesseract reports one box per word. Words on the same text line are
    re-joined into phrases, splitting wherever the gap between them widens
    past ``WORD_GAP`` - that gap is what separates a label from its value.
    """
    data = pytesseract.image_to_data(
        image, config=f"--psm {psm} --oem {OEM}", output_type=pytesseract.Output.DICT
    )

    lines = defaultdict(list)
    for index in range(len(data["text"])):
        text = str(data["text"][index]).strip()
        conf = float(data["conf"][index])
        # Tesseract emits its layout boxes - blocks, paragraphs, lines - with
        # a confidence of -1 and no text.
        if not text or conf < 0:
            continue
        key = (data["block_num"][index], data["par_num"][index], data["line_num"][index])
        lines[key].append(
            {
                "text": text,
                "conf": conf / 100.0,
                "left": int(data["left"][index]),
                "top": int(data["top"][index]),
                "width": int(data["width"][index]),
                "height": int(data["height"][index]),
            }
        )

    segments = []
    for _key, words in sorted(lines.items()):
        words.sort(key=lambda word: word["left"])
        current = None
        for word in words:
            if current is not None:
                gap = word["left"] - (current["left"] + current["width"])
                if gap <= WORD_GAP * max(current["height"], word["height"]):
                    right = max(current["left"] + current["width"],
                                word["left"] + word["width"])
                    bottom = max(current["top"] + current["height"],
                                 word["top"] + word["height"])
                    current["top"] = min(current["top"], word["top"])
                    current["width"] = right - current["left"]
                    current["height"] = bottom - current["top"]
                    current["text"] += " " + word["text"]
                    current["conf"] = min(current["conf"], word["conf"])
                    continue
                segments.append(_to_segment(current, page))
            current = dict(word)
        if current is not None:
            segments.append(_to_segment(current, page))

    return segments


def _split_inline_pairs(segments: list) -> list:
    """Split "Label: value" text into a label segment and a value segment.

    The box is divided across the colon in proportion to the text either side
    of it - close enough, since both halves sit on one printed row and all the
    pairing needs is that the value starts to the right of the label.
    """
    split = []
    for segment in segments:
        match = INLINE_PAIR.match(segment.text.strip())
        if match is None or not any(char.isalpha() for char in match.group("label")):
            split.append(segment)
            continue

        label_text = match.group("label").strip() + ":"
        value_text = match.group("value").strip()
        share = len(label_text) / (len(label_text) + len(value_text))
        boundary = segment.x + segment.width * share

        split.append(Segment(label_text, segment.page, segment.line, segment.x,
                             segment.y, boundary - segment.x, segment.height,
                             segment.conf))
        split.append(Segment(value_text, segment.page, segment.line, boundary,
                             segment.y, segment.right - boundary, segment.height,
                             segment.conf))
    return split


def _assign_rows(segments: list, unit: float) -> None:
    """Number the printed rows, so a value's own lines can be told apart."""
    row = -1
    anchor = None
    page = None
    for segment in sorted(segments, key=lambda item: (item.page, item.cy, item.x)):
        if page != segment.page or anchor is None or abs(segment.cy - anchor) > ROW_BAND * unit:
            row += 1
            anchor = segment.cy
            page = segment.page
        segment.line = row


def _unit(segments: list) -> float:
    """Median text height, the unit every tolerance is expressed in."""
    heights = sorted(segment.height for segment in segments if segment.height > 0)
    if not heights:
        return FALLBACK_UNIT
    return float(heights[len(heights) // 2]) or FALLBACK_UNIT


def _looks_like_value(text: str) -> bool:
    """True when the text reads as data rather than as a caption."""
    stripped = text.strip()
    return any(pattern.match(stripped) for pattern in VALUE_ONLY)


def _is_label(segment: Segment, segments: list, unit: float) -> bool:
    """Judge one phrase a label.

    A phrase is a label when it ends in a colon, ends in one of the words
    forms use to name a field, or has a value sitting across a gutter to its
    right. Text that reads as a number, an amount or a date never is.

    The confidence floor guards the two inferred rules only. A trailing colon
    is the form itself saying this is a caption, and bold headings on a faxed
    page routinely score below the floor while reading perfectly well.
    """
    text = segment.text.strip()
    if not text:
        return False
    if _looks_like_value(text):
        return False
    if text.endswith(":"):
        return True
    if segment.conf < LABEL_MIN_CONF:
        return False
    if LABEL_SUFFIX.search(text):
        return True

    for other in segments:
        if other is segment or other.page != segment.page:
            continue
        if abs(other.cy - segment.cy) > Y_TOLERANCE * unit:
            continue
        gap = other.x - segment.right
        if GUTTER_MIN * unit <= gap <= RIGHT_SEARCH * unit:
            return True
    return False


def clean_key(text: str) -> str:
    """Turn label text into a key.

    Drops the leading asterisks and trailing colons forms decorate labels
    with, throws away punctuation, and joins what is left with underscores:
    ``*Penalties & any other deduction Applicable`` becomes
    ``Penalties_Any_Other_Deduction_Applicable``.
    """
    cleaned = _TRAILING_JUNK.sub("", _LEADING_JUNK.sub("", text.strip()))
    cleaned = _NON_KEY.sub(" ", cleaned)
    return "_".join(word.title() for word in cleaned.split())


def _store(result: dict, key: str, value) -> str:
    """Add a field, numbering the key when the page prints the label twice.

    Returns the key actually written under - the caller's own ``key`` most
    of the time, but a numbered ``key_2``, ``key_3``, ... when the page
    printed that label more than once - so a later pass (the zonal re-read
    below) can find exactly the same field it is meant to refine rather
    than always the first occurrence.
    """
    if not key:
        return ""
    if key not in result:
        result[key] = value
        return key
    suffix = 2
    while f"{key}_{suffix}" in result:
        suffix += 1
    final_key = f"{key}_{suffix}"
    result[final_key] = value
    return final_key


def _join(segments: list) -> str:
    """Render collected segments as one string, lines separated by ``|``."""
    lines = defaultdict(list)
    for segment in segments:
        lines[segment.line].append(segment)

    rendered = []
    for _line, items in sorted(lines.items(), key=lambda pair: min(i.y for i in pair[1])):
        rendered.append(" ".join(item.text for item in sorted(items, key=lambda i: i.x)))
    return " | ".join(part for part in rendered if part)


def _collect_right(label, segments, labels, consumed, unit):
    """Value segments beside the label, stopping at the next label on the line."""
    candidates = [
        segment for segment in segments
        if segment is not label
        and segment.page == label.page
        and segment not in consumed
        and abs(segment.cy - label.cy) <= Y_TOLERANCE * unit
        and segment.x >= label.right
        and segment.x - label.right <= RIGHT_SEARCH * unit
    ]
    collected = []
    for segment in sorted(candidates, key=lambda item: item.x):
        if segment in labels:
            break
        collected.append(segment)
    return collected


def _collect_below(label, segments, labels, consumed, unit):
    """Value segments under the label, stopping at the next label beneath it."""
    candidates = [
        segment for segment in segments
        if segment is not label
        and segment.page == label.page
        and segment not in consumed
        and label.bottom <= segment.cy <= label.bottom + BELOW_BAND * unit
        and abs(segment.x - label.x) <= BELOW_X_TOLERANCE * unit
    ]
    collected = []
    for segment in sorted(candidates, key=lambda item: (item.y, item.x)):
        if segment in labels:
            break
        collected.append(segment)
    return collected


def _collect_continuation(label, values, segments, labels, consumed, unit):
    """The extra lines of a multi-line value.

    Only runs when no other label starts within ``CONTINUATION_GAP`` below
    this one - a label with a neighbour right under it owns one line, not the
    rest of the form.
    """
    below = [other.y for other in labels
             if other.page == label.page and other.y > label.bottom]
    nearest = min(below) if below else None
    if nearest is not None and nearest - label.bottom <= CONTINUATION_GAP * unit:
        return []

    anchor_x = min(segment.x for segment in values)
    floor = max(segment.bottom for segment in values)
    ceiling = nearest if nearest is not None else float("inf")

    # ``bottom`` rather than ``cy``, so text sharing a row with the next label
    # - that label's own value - is left for it.
    return [
        segment for segment in segments
        if segment.page == label.page
        and segment not in consumed
        and segment not in labels
        and floor < segment.cy
        and segment.bottom <= ceiling
        and segment.x >= anchor_x - BELOW_X_TOLERANCE * unit
    ]


def _extract_checkboxes(segments, consumed, unit, result) -> None:
    """Read a row of tick boxes, recording which code was ticked.

    A row of short codes - ``PO1X PO2X PO2O ...`` - is found by its shape, and
    whichever code has a tick mark within ``TICK_DISTANCE`` is the answer. The
    codes and the tick are marked used, so they are not also read as fields.
    """
    rows = defaultdict(list)
    for segment in segments:
        rows[(segment.page, segment.line)].append(segment)

    found = 0
    for (_page, _line), items in sorted(rows.items()):
        codes = [item for item in items
                 if CHECKBOX_CODE.match(item.text.strip(CHECKBOX_BOX_GLYPHS + " "))]
        if len(codes) < CHECKBOX_MIN_CODES:
            continue

        ticks = [
            item for item in segments
            if item.text.strip() in TICK_MARKS
            and item.page == codes[0].page
            and abs(item.cy - codes[0].cy) <= Y_TOLERANCE * unit * 2
        ]

        selected = None
        for tick in ticks:
            nearest = min(codes, key=lambda code: abs(code.cx - tick.cx))
            if abs(nearest.cx - tick.cx) <= (TICK_DISTANCE * unit) + nearest.width:
                selected = nearest.text.strip(CHECKBOX_BOX_GLYPHS + " ")
                consumed.add(tick)
                break

        found += 1
        key = "PO_Type_Selected" if found == 1 else f"PO_Type_Selected_{found}"
        _store(result, key, selected)
        _store(result, key.replace("Selected", "Options"),
               " ".join(code.text.strip(CHECKBOX_BOX_GLYPHS + " ") for code in codes))
        consumed.update(codes)


def _extract_stamps(segments, consumed, unit, result) -> None:
    """Read rubber stamps as dense clusters of leftover text.

    Only text that no field claimed is considered, which is what keeps the
    form's own densely printed rows from being read as stamps. Each cluster is
    named after its largest text - the company line on the stamp.
    """
    available = [segment for segment in segments if segment not in consumed]
    box = STAMP_BOX * unit

    while True:
        best = None
        for seed in available:
            members = [
                segment for segment in available
                if abs(segment.cx - seed.cx) <= box / 2
                and abs(segment.cy - seed.cy) <= box / 2
            ]
            if len(members) >= STAMP_MIN_ITEMS and (best is None or len(members) > len(best)):
                best = members
        if best is None:
            return

        headline = max(best, key=lambda segment: segment.height)
        key = clean_key(headline.text)
        if key:
            _store(
                result,
                f"Stamp_{key}",
                " ".join(segment.text for segment in
                         sorted(best, key=lambda item: (item.y, item.x))),
            )
        consumed.update(best)
        available = [segment for segment in available if segment not in best]


def extract_label_values(segments: list, page_images: dict = None) -> dict:
    """Pair up the labels and values in a page's OCR segments.

    Shared with the bill's metadata block, which is the same problem over a
    smaller region - that caller passes no ``page_images``, so the zonal
    re-read below simply never runs there (it targets this cover form's own
    fields, which do not appear on the bill's metadata block anyway).

    Pages are handled one at a time because every tolerance is scaled by the
    page's own text height: a landscape table page and a portrait form read at
    the same DPI have very different text sizes, and one median across both
    fits neither.

    Args:
        segments: OCR'd phrases across every page, in reading order.
        page_images: ``{page: PIL.Image}`` for the zonal/anchor-based
            re-read of this form's own known fields (see _refine_known_
            fields) - omitted, the pairing runs exactly as before.
    """
    result: dict = {}
    if not segments:
        return result

    pages = defaultdict(list)
    for segment in _split_inline_pairs(segments):
        pages[segment.page].append(segment)

    for page in sorted(pages):
        image = (page_images or {}).get(page)
        _extract_page(pages[page], result, image)
    return result


def _extract_page(segments: list, result: dict, image: Image.Image = None) -> None:
    """Run the label/value pass over one page, adding to ``result``."""
    unit = _unit(segments)
    _assign_rows(segments, unit)
    consumed: set = set()

    _extract_checkboxes(segments, consumed, unit, result)

    labels = {
        segment for segment in segments
        if segment not in consumed and _is_label(segment, segments, unit)
    }

    label_keys: dict = {}
    for label in sorted(labels, key=lambda item: (item.page, item.y, item.x)):
        if label in consumed:
            continue
        consumed.add(label)

        values = _collect_right(label, segments, labels, consumed, unit)
        if not values:
            values = _collect_below(label, segments, labels, consumed, unit)

        if values:
            values = values + _collect_continuation(
                label, values, segments, labels, consumed, unit
            )
            consumed.update(values)

        label_keys[label] = _store(
            result, clean_key(label.text), _join(values) if values else None
        )

    if image is not None:
        _refine_known_fields(label_keys, segments, labels, unit, image, result)

    _extract_stamps(segments, consumed, unit, result)

    # Nothing is thrown away: whatever is left is flagged by where it sat.
    for segment in sorted(
        (item for item in segments if item not in consumed),
        key=lambda item: (item.page, item.y, item.x),
    ):
        _store(result, f"Unmatched_Text_{segment.page + 1}_{round(segment.cy)}",
               segment.text)


def extract_form_generic(images: list) -> dict:
    """Extract every label/value pair from a form, across all its pages.

    Kept as the module's original, schema-free reading of ANY key-value
    form (position, colon-splitting, gutter-adjacency - see the module
    docstring); not called by app.py any more (see extract_invoice_cover
    below, per the FINAL SPEC, 2026-09-26), but left available for a form
    this project has no fixed schema for yet.

    Args:
        images: PIL images, one per page of the form.

    Returns:
        A flat dict whose keys are the labels the pages actually print,
        cleaned into ``Title_Case_With_Underscores``. A label with no value
        beside or under it maps to ``None``. Text that matched no label comes
        back under ``Unmatched_Text_<page>_<y>``, so nothing is lost.
    """
    segments = []
    page_images = {}
    for page, image in enumerate(images or []):
        segments.extend(ocr_segments(image, page))
        page_images[page] = image
    return extract_label_values(segments, page_images=page_images)


# --------------------------------------------------------------------------
# Invoice cover form - fixed 8-field extraction (FINAL SPEC, 2026-09-26)
# --------------------------------------------------------------------------
#
# The JSW GBS PO Expenses Approval Template is one fixed layout used for
# every submission, not an open-ended form - so unlike extract_form_generic
# above (which pairs whatever labels a page happens to print, and keeps
# whatever it could not pair under Unmatched_Text_*), this reads exactly 8
# known fields and nothing else. A whole-page OCR pass measurably drops this
# form's own "Invoice Date" label outright (confirmed directly on this
# project's own sample scan: neither psm 6 nor psm 11 - individually or
# merged via ocr_segments - find it at all, though the identical crop reads
# it perfectly on its own), so this instead reads the form's own printed
# table row by row: each row is bounded by the ruled lines detect_grid
# (bill_extractor.py's own table-line detector, reused here on a plain
# label/value form instead of a repeating table) can find reliably, then
# split at the table's own internal label/value column rule - also
# detect_grid, run again on a narrow two-row band - so a small crop reads
# far more accurately than one whole-page pass ever does, the same
# principle ocr/lr_extractor.py's own zone crops already rely on.

INVOICE_COVER_FIELDS = (
    "vendor_name", "vendor_code", "po_number", "grn_srn_number",
    "invoice_number", "invoice_date", "penalty_amount", "invoice_amount",
)

# The rapidfuzz 0-100 score a row's own label crop must clear against
# INVOICE_FORM_FIELD_SYNONYMS before that row counts as one of the 7 fields
# above - same bar this form's own zonal re-read (_refine_known_fields)
# already uses.
COVER_MATCH_THRESHOLD = CAPTION_MATCH_THRESHOLD

# Divider/right-edge search: only a v-line strictly between these two
# fractions of the page width is trusted as the table's own internal
# label/value rule - a v-line nearer either edge is the page's own outer
# border (or the barcode's own trailing rule further right), not this.
DIVIDER_SEARCH_MIN_FRACTION = 0.25
DIVIDER_SEARCH_MAX_FRACTION = 0.75

# A v-line this close to one detect_grid already found across the WHOLE
# page (the page's own outer left/right borders) is that border, not a
# real internal rule - see _find_column_divider.
DIVIDER_EDGE_MARGIN = 50

# Values are read with a small left margin off the divider itself, so the
# rule's own ink is never included in the crop.
CELL_MARGIN = 6


def _find_column_divider(image: Image.Image, h_lines: list, page_v_lines: list) -> tuple:
    """``(divider_x, right_edge_x)`` for this form's own label/value column
    rule, or ``(None, None)`` if no band of the page reveals one.

    detect_grid's own v-line search needs a rule to span most of whatever
    band it is given (see bill_extractor.detect_grid's own V_SPAN_RATIO) -
    run on the whole page that band is the full page height, which the
    label/value rule does not reach (it only runs the height of this one
    inner table), so detect_grid there only ever finds the page's own outer
    border. Run again on a narrow two-row-tall crop instead - any two
    adjacent rows of the same table share this same rule its own full
    height, easily clearing that span requirement - and it is found
    immediately. Every pair of adjacent rows is tried in turn (not just the
    first) since a row landing on the barcode graphic can itself register
    spurious vertical bars as v-lines.
    """
    width = image.width
    for index in range(len(h_lines) - 2):
        top, bottom = h_lines[index], h_lines[index + 2]
        band = image.crop((0, top, width, bottom))
        _band_h, band_v = _detect_grid(np.array(band.convert("RGB")))
        interior = [x for x in band_v if all(abs(x - edge) > DIVIDER_EDGE_MARGIN for edge in page_v_lines)]
        central = [x for x in interior
                   if width * DIVIDER_SEARCH_MIN_FRACTION <= x <= width * DIVIDER_SEARCH_MAX_FRACTION]
        if central:
            divider = min(central)
            further = [x for x in interior if x > divider]
            right_edge = min(further) if further else max(page_v_lines) - DIVIDER_EDGE_MARGIN
            return divider, right_edge
    return None, None


def _row_cell_text(image: Image.Image, top: int, bottom: int, left: int, right: int) -> str:
    """One row's own cell, OCR'd on its own (psm 6, the whole cell as one
    block) - reused for both the label and the value half of a row."""
    if right <= left or bottom <= top:
        return ""
    crop = image.crop((left, top, right, bottom))
    return " ".join(pytesseract.image_to_string(crop, config="--psm 6 --oem 3").split())


_COVER_LABEL_NOISE = re.compile(r"[^a-z ]+")


def _clean_row_label(text: str) -> str:
    """Lowercased, alphabetic-only - a row's OCR'd label almost always picks
    up a stray ruled-border pipe or two at either edge (``"| | Vendor
    Name"``), which would otherwise dilute its own fuzzy-match score for
    nothing; the degenerate-match guard below still runs on this cleaned
    text afterwards."""
    return _COVER_LABEL_NOISE.sub(" ", text.lower()).strip()


def _safe_field_matches(cleaned_texts: dict, synonyms_by_field: dict, threshold: float) -> list:
    """``best_synonym_matches``, filtered the same way ocr/lr_extractor.py's
    own ``_safe_synonym_matches`` filters its zone-anchor matches (that
    function cannot be imported here - lr_extractor.py imports from this
    module, not the other way around, so importing it back would be
    circular - hence this small copy of the same guard, rather than a third
    shared module for one function): a bare, short synonym ("srn number")
    can score 100 against a long, unrelated row purely because it sits in
    there as a substring; a real match either leads its own row's label
    text, or is not wildly longer or shorter than the synonym it matched.
    """
    candidates = best_synonym_matches(cleaned_texts, synonyms_by_field, threshold)
    safe = []
    for score, index, field in candidates:
        text = cleaned_texts[index]
        text_len = len(text.replace(" ", ""))
        for synonym in synonyms_by_field[field]:
            synonym_len = len(synonym.replace(" ", ""))
            if text.startswith(synonym) or synonym_len * 0.5 <= text_len <= synonym_len * 3:
                safe.append((score, index, field))
                break
    return safe


# Per-field cleanup of a row's own OCR'd value text - the barcode graphic
# sitting inside the JSW PO Number/GRN-SRN Number rows' own value cell reads
# as extra noise alongside the real value (confirmed on this project's own
# sample scan: "4500237065 A", "... 1012632938 STRHPO6X260001146"), so those
# two - and every other numeric/date field, which can pick up a stray ruled
# border character the same way - are read out with a shape-based regex
# instead of trusted as the whole cell string. Free-text fields (vendor
# name, invoice number) are only trimmed of leading/trailing punctuation the
# ruled border itself contributes.
_NUMERIC_TOKEN = re.compile(r"\d[\d,]{3,}")
_DATE_TOKEN = re.compile(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}")
_PENALTY_AMOUNT = re.compile(r"amount[^0-9]{0,6}([\d,]+\s*/?-?)", re.IGNORECASE)
# Invoice Amount's own value cell is read straight off its own row (unlike
# penalty_amount, whose row sometimes carries a repeated "Amount :-" label
# fragment inside the value cell itself, per _PENALTY_AMOUNT above) - just
# the digits/commas and an optional trailing "/-", the same shape this
# form prints an amount in ("23,30,645/-").
_AMOUNT_TOKEN = re.compile(r"[\d,]{3,}\s*/?-?")
_EDGE_JUNK = re.compile(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$")


def _clean_free_text_value(text: str):
    return _EDGE_JUNK.sub("", text).strip() or None


def _clean_numeric_value(text: str):
    match = _NUMERIC_TOKEN.search(text)
    return match.group(0) if match else _clean_free_text_value(text)


def _clean_date_value(text: str):
    match = _DATE_TOKEN.search(text)
    return match.group(0) if match else _clean_free_text_value(text)


def _clean_penalty_value(text: str):
    match = _PENALTY_AMOUNT.search(text)
    if match:
        return match.group(1).replace(" ", "")
    return _clean_free_text_value(text)


def _clean_amount_value(text: str):
    match = _AMOUNT_TOKEN.search(text)
    return match.group(0).replace(" ", "") if match else _clean_free_text_value(text)


_COVER_VALUE_CLEANERS = {
    "vendor_name": _clean_free_text_value,
    "vendor_code": _clean_numeric_value,
    "po_number": _clean_numeric_value,
    "grn_srn_number": _clean_numeric_value,
    "invoice_number": _clean_free_text_value,
    "invoice_date": _clean_date_value,
    "penalty_amount": _clean_penalty_value,
    "invoice_amount": _clean_amount_value,
}


# --------------------------------------------------------------------------
# Field 8 - the number printed under the barcode.
# --------------------------------------------------------------------------
#
# Found by shape, not position: a barcode is the one region on this page
# dense with tall, thin, tightly-packed vertical bars, which a horizontal
# Sobel gradient (edges running vertical score high on dx, near-zero on dy,
# since a barcode's own bars are perfectly vertical) picks out from
# ordinary printed text - text's own edges run every which way, so it
# scores much lower once the two gradients are subtracted. Closing that
# gradient over a wide-but-short kernel then fuses the many individual bars
# into one solid blob a contour can bound - never a hardcoded pixel
# position, confirmed against this project's own sample scan.
BARCODE_GRADIENT_BLUR = (9, 9)
BARCODE_THRESHOLD = 225
BARCODE_CLOSE_WIDTH_RATIO = 0.06
BARCODE_CLOSE_HEIGHT = 3
BARCODE_DILATE_KERNEL = (3, 25)
BARCODE_DILATE_ITER = 3

# A candidate in this aspect-ratio band, within this fraction of the page's
# own area, is the barcode - a printed rule or an underline (this form has
# several) passes a bare "wider than tall" check just as easily but reads
# far more extreme on both ends: confirmed on this project's own scan, the
# real barcode is 2.75 wide-to-tall and every decorative line on the same
# page is 14-55, so the upper bound is what actually tells the two apart (a
# lower bound alone let the page's own title underline, wider still, win on
# raw area every time).
BARCODE_MIN_ASPECT = 2.0
BARCODE_MAX_ASPECT = 6.0
BARCODE_MIN_AREA_RATIO = 0.0005
BARCODE_MAX_AREA_RATIO = 0.05
# A genuine barcode's own bars run tall enough to hold ~30-40 of them
# side by side; a thin printed rule or underline this same gradient/close
# pass also turns into a wide blob is only a few pixels tall regardless of
# how long it runs - confirmed on this project's own scan (barcode 137px,
# every decorative line under 40px at this page's own 3491px height).
BARCODE_MIN_HEIGHT_RATIO = 0.02

# The printed human-readable number sits directly under the bars, often
# running wider than the bars themselves (confirmed on this project's own
# scan: barcode 377px wide, its own text 490px) - read with a generous
# padding on every side instead of the bars' own tight width.
BARCODE_TEXT_LEFT_PAD = 20
BARCODE_TEXT_RIGHT_MULTIPLIER = 1.7
BARCODE_TEXT_HEIGHT_MULTIPLIER = 0.9


def _detect_barcode_box(gray: np.ndarray):
    """The barcode's own bounding box ``(x, y, w, h)`` in ``gray``'s pixel
    space, or ``None`` if nothing on the page reads as one."""
    height, width = gray.shape
    grad_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_y = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradient = cv2.convertScaleAbs(cv2.subtract(grad_x, grad_y))
    blurred = cv2.blur(gradient, BARCODE_GRADIENT_BLUR)
    _threshold, thresholded = cv2.threshold(blurred, BARCODE_THRESHOLD, 255, cv2.THRESH_BINARY)

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(2, int(width * BARCODE_CLOSE_WIDTH_RATIO)), BARCODE_CLOSE_HEIGHT)
    )
    closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, close_kernel)
    closed = cv2.dilate(closed, np.ones(BARCODE_DILATE_KERNEL, np.uint8), iterations=BARCODE_DILATE_ITER)

    contours, _hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_area = width * height

    best = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area_ratio = (w * h) / page_area
        aspect = w / max(h, 1)
        if not (BARCODE_MIN_ASPECT <= aspect <= BARCODE_MAX_ASPECT):
            continue
        if h < height * BARCODE_MIN_HEIGHT_RATIO:
            continue
        if not (BARCODE_MIN_AREA_RATIO < area_ratio < BARCODE_MAX_AREA_RATIO):
            continue
        if best is None or w * h > best[2] * best[3]:
            best = (x, y, w, h)
    return best


# The human-readable line under a barcode is printed in one run of capital
# letters/digits with no spaces (this project's own sample: the underlying
# barcode's own crop otherwise also OCRs a stray leading/trailing punctuation
# character off the crop's own edge - "___STRHPO6X260001146 |").
_BARCODE_CODE = re.compile(r"[A-Z0-9]{6,}")


def _read_barcode_number(image: Image.Image):
    """Field 8: the human-readable text printed directly under the
    barcode's own bars, located by ``_detect_barcode_box`` - never a
    hardcoded pixel position."""
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    box = _detect_barcode_box(gray)
    if box is None:
        return None

    x, y, w, h = box
    left = max(0, x - BARCODE_TEXT_LEFT_PAD)
    right = min(image.width, x + int(w * BARCODE_TEXT_RIGHT_MULTIPLIER))
    top = y + h
    bottom = min(image.height, top + int(h * BARCODE_TEXT_HEIGHT_MULTIPLIER))
    text = _ocr_zone(image, (left, top, right, bottom))
    if not text:
        return None
    match = _BARCODE_CODE.search(text.upper())
    return match.group(0) if match else text


def extract_invoice_cover(images: list) -> dict:
    """Extract the JSW GBS PO Expenses Approval Template's fixed 8 fields
    from the cover page - see the module comment above this function.

    Args:
        images: PIL images, one per page of the form; only the first page
            (the cover form itself) is read.

    Returns:
        A flat dict with exactly these 9 keys, in this order -
        ``vendor_name``, ``vendor_code``, ``po_number``, ``grn_srn_number``,
        ``invoice_number``, ``invoice_date``, ``penalty_amount``,
        ``invoice_amount``, ``barcode_number`` - ``None`` for any field this
        page's own scan did not yield. Nothing else (no Unmatched_Text, no
        confidence, no flags) is ever added.
    """
    result = {field: None for field in INVOICE_COVER_FIELDS}
    result["barcode_number"] = None
    if not images:
        return result

    image = images[0]
    np_img = np.array(image.convert("RGB"))
    h_lines, page_v_lines = _detect_grid(np_img)

    if len(h_lines) >= 3 and len(page_v_lines) >= 2:
        divider, right_edge = _find_column_divider(image, h_lines, page_v_lines)
        if divider is not None:
            left_border = page_v_lines[0]
            cleaned_labels = {}
            for index in range(len(h_lines) - 1):
                top, bottom = h_lines[index], h_lines[index + 1]
                label_text = _row_cell_text(image, top, bottom, left_border + CELL_MARGIN, divider)
                cleaned_labels[index] = _clean_row_label(label_text)

            synonyms = {field: INVOICE_FORM_FIELD_SYNONYMS[field] for field in INVOICE_COVER_FIELDS}
            candidates = _safe_field_matches(cleaned_labels, synonyms, COVER_MATCH_THRESHOLD)
            assigned = greedy_assign(candidates)  # {row_index: field}

            for row_index, field in assigned.items():
                top, bottom = h_lines[row_index], h_lines[row_index + 1]
                value_text = _row_cell_text(image, top, bottom, divider + CELL_MARGIN, right_edge)
                result[field] = _COVER_VALUE_CLEANERS[field](value_text) if value_text else None

    result["barcode_number"] = _read_barcode_number(image)
    return result


def extract_invoice(images: list) -> dict:
    """app.py's own entry point - the cover form's fixed 8 fields, per the
    FINAL SPEC (2026-09-26). Same name/signature the pipeline already
    imports and calls; see extract_invoice_cover for the implementation and
    extract_form_generic for this module's original schema-free reading."""
    return extract_invoice_cover(images)


if __name__ == "__main__":
    import sys

    try:
        from .pdf_handler import pdf_to_images
    except ImportError:
        from pdf_handler import pdf_to_images

    path = sys.argv[1] if len(sys.argv) > 1 else "test_invoice.pdf"
    for name, value in extract_invoice(pdf_to_images(path, dpi=300)[0:2]).items():
        print(f"{name}: {value!r}")
