"""Split a scanned page into printed text and colored-ink layers.

A rubber stamp or a signature written in blue, purple or red sits at a
different saturation and hue than the black-printed form under it, and that
is the only signal this uses - nothing here tries to tell a black pen stroke
from black print, since HSV genuinely cannot. A page scanned in grayscale
carries none of that signal at all (every pixel's R, G and B channels are
identical), and is detected and passed through unchanged as the printed
layer, with an empty ink layer, rather than run through masks that would
never find anything saturated to separate.

Also locates and reads the "CUSTOMER ACKNOWLEDGEMENT DETAILS" box - the
RECEIVED WT / DATE / REMARK columns an LR is signed back with - off the
ink layer, so a handwritten entry there is read without the printed form
underneath it interfering. Nothing here touches how ocr/lr_extractor.py
reads the LR's own printed fields.
"""

import re

import cv2
import numpy as np
import pytesseract
from PIL import Image

try:
    from .reader import TESSERACT_PATH  # noqa: F401  (sets tesseract_cmd)
except ImportError:  # running this file directly from inside ocr/
    from reader import TESSERACT_PATH  # noqa: F401

# A pixel this close to black, with this little saturation, is printed or
# handwritten in black ink - not the colored ink this module separates out.
PRINTED_HSV_LOW = (0, 0, 0)
PRINTED_HSV_HIGH = (180, 60, 90)

# Blue through red-violet in OpenCV's 0-179 hue range, saturated and not
# near-black or near-white: a rubber stamp or a ballpoint signature.
INK_HSV_LOW = (90, 40, 40)
INK_HSV_HIGH = (170, 255, 255)

# Below this, a page's R/G/B channels are treated as equal - genuinely
# grayscale, not merely a page with little colored content (which would
# still read a small but nonzero diff; a true grayscale scan reads ~0).
GRAYSCALE_CHANNEL_DIFF = 2.0

_ACKNOWLEDGEMENT_LABEL = re.compile(r"ACKNOWLEDGEMENT", re.IGNORECASE)
_RECEIVED_LABEL = re.compile(r"RECEIVED", re.IGNORECASE)
_DATE_LABEL = re.compile(r"^DATE$", re.IGNORECASE)
_REMARK_LABEL = re.compile(r"REMARK", re.IGNORECASE)

RECEIVED_WT_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789."
RECEIVED_DATE_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789./"
REMARK_CONFIG = "--psm 7"

# A REMARK reading below this word confidence is passed through flagged,
# not discarded and not "corrected" - see extract_acknowledgement_box.
LOW_CONFIDENCE_THRESHOLD = 60

# How many header-line-heights below the column labels their data is
# searched - generous, since this box holds at most one or two short
# entries (a weight, a date, a short note) on every LR sampled.
DATA_ROW_HEIGHT_LINES = 4

# How many header-line-heights below "ACKNOWLEDGEMENT" (or its RECEIVED
# fallback) the RECEIVED WT / DATE / REMARK sub-labels are searched for.
SUBLABEL_SEARCH_LINES = 6


def _is_grayscale(rgb: np.ndarray) -> bool:
    diff = (
        np.abs(rgb[:, :, 0].astype(int) - rgb[:, :, 1].astype(int))
        + np.abs(rgb[:, :, 1].astype(int) - rgb[:, :, 2].astype(int))
        + np.abs(rgb[:, :, 0].astype(int) - rgb[:, :, 2].astype(int))
    )
    return diff.mean() < GRAYSCALE_CHANNEL_DIFF


def split_ink_layers(pil_image_bgr: Image.Image):
    """Return ``(printed_layer, colored_ink_layer)``, same size as the input.

    Each is a white-background image holding only the pixels its mask
    claimed: near-black, low-saturation "printed" pixels in the first,
    saturated blue-through-red-violet "ink" pixels in the second. A
    grayscale page is returned unchanged as the printed layer, with an
    all-white ink layer - there is no color signal on it to split on.
    """
    rgb = np.array(pil_image_bgr.convert("RGB"))

    if _is_grayscale(rgb):
        return Image.fromarray(rgb), Image.new("RGB", pil_image_bgr.size, "white")

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    printed_mask = cv2.inRange(hsv, PRINTED_HSV_LOW, PRINTED_HSV_HIGH)
    ink_mask = cv2.inRange(hsv, INK_HSV_LOW, INK_HSV_HIGH)

    printed_layer = np.full_like(rgb, 255)
    printed_layer[printed_mask > 0] = rgb[printed_mask > 0]

    ink_layer = np.full_like(rgb, 255)
    ink_layer[ink_mask > 0] = rgb[ink_mask > 0]

    return Image.fromarray(printed_layer), Image.fromarray(ink_layer)


def _boxes_matching(data: dict, pattern: re.Pattern) -> list:
    return [
        {"left": data["left"][i], "top": data["top"][i],
         "width": data["width"][i], "height": data["height"][i],
         "text": text}
        for i, text in enumerate(data["text"])
        if text and pattern.search(text)
    ]


def find_acknowledgement_zone(printed_layer: Image.Image):
    """Locate the acknowledgement box from its own printed labels.

    Runs ``pytesseract.image_to_data`` on the printed layer - these labels
    are printed, not handwritten - and looks for "ACKNOWLEDGEMENT" (falling
    back to "RECEIVED" alone, for a bill format that prints the column
    label but not the fuller heading), then the RECEIVED WT / DATE / REMARK
    sub-labels in a band below it. No pixel coordinate here is hardcoded;
    a layout shifted between clients still locates from its own text.

    Returns:
        ``{"outer": (l, t, r, b), "received_wt": (...), "received_date":
        (...), "remark": (...)}`` in the layer's own pixel coordinates, or
        ``None`` if the header (or any of the three sub-labels) cannot be
        found at all - callers must not guess a position in that case.
    """
    data = pytesseract.image_to_data(printed_layer, output_type=pytesseract.Output.DICT)

    header = _boxes_matching(data, _ACKNOWLEDGEMENT_LABEL) or _boxes_matching(data, _RECEIVED_LABEL)
    if not header:
        return None
    header_box = min(header, key=lambda box: box["top"])
    line_height = header_box["height"]

    band_top = header_box["top"] + header_box["height"]
    band_bottom = band_top + line_height * SUBLABEL_SEARCH_LINES

    def _first_in_band(pattern):
        candidates = [
            box for box in _boxes_matching(data, pattern)
            if band_top <= box["top"] <= band_bottom
        ]
        return min(candidates, key=lambda box: box["top"]) if candidates else None

    received = _first_in_band(_RECEIVED_LABEL)
    date = _first_in_band(_DATE_LABEL)
    remark = _first_in_band(_REMARK_LABEL)
    if not (received and date and remark):
        return None

    field_names = {id(received): "received_wt", id(date): "received_date", id(remark): "remark"}
    ordered = sorted([received, date, remark], key=lambda box: box["left"])
    data_row_height = line_height * DATA_ROW_HEIGHT_LINES

    zone = {}
    for index, label in enumerate(ordered):
        left = label["left"]
        right = (ordered[index + 1]["left"] if index + 1 < len(ordered)
                 else label["left"] + label["width"] * 4)
        top = label["top"] + label["height"]
        zone[field_names[id(label)]] = (left, top, right, top + data_row_height)

    zone["outer"] = (
        ordered[0]["left"], header_box["top"],
        ordered[-1]["left"] + ordered[-1]["width"] * 4,
        max(box[3] for name, box in zone.items()),
    )
    return zone


def _ocr_cell(ink_layer: Image.Image, bbox: tuple, config: str) -> tuple:
    """``(text, mean_word_confidence)`` for one cropped cell, "" / 0.0 if blank."""
    left, top, right, bottom = bbox
    left, top = max(0, left), max(0, top)
    right, bottom = min(ink_layer.width, right), min(ink_layer.height, bottom)
    if right <= left or bottom <= top:
        return "", 0.0

    crop = ink_layer.crop((left, top, right, bottom))
    data = pytesseract.image_to_data(crop, config=config, output_type=pytesseract.Output.DICT)
    words = [
        (text, float(conf)) for text, conf in zip(data["text"], data["conf"])
        if text.strip() and str(conf) not in ("-1", "")
    ]
    if not words:
        return "", 0.0
    text = " ".join(word for word, _ in words)
    confidence = sum(conf for _, conf in words) / len(words)
    return text, confidence


def extract_acknowledgement_box(ink_layer_image: Image.Image, zone_bbox) -> dict:
    """Read RECEIVED WT / DATE / REMARK out of the acknowledgement box.

    Args:
        ink_layer_image: The colored-ink layer from ``split_ink_layers()`` -
            a handwritten entry here is what gets read, with the printed
            form underneath it already masked away.
        zone_bbox: What ``find_acknowledgement_zone()`` returned, or
            ``None``.

    Returns:
        ``received_wt`` and ``received_date`` are read against a digit (plus
        ``.`` or ``/``) whitelist - anything Tesseract reads outside that
        alphabet cannot come through. ``remark`` is free text: it is never
        "corrected" against a whitelist or guessed at, only flagged -
        garbled handwriting a whitelist would just silently mangle is more
        useful to a reviewer shown as-is than smoothed into something
        plausible-looking and wrong. If the zone could not be located at
        all, every value comes back empty rather than guessed.
    """
    if not zone_bbox:
        print("warning: acknowledgement box labels not found; returning empty values")
        return {
            "received_wt": "", "received_date": "", "remark": "",
            "remark_confidence": 0.0, "low_confidence": False, "zone_found": False,
        }

    received_wt, _ = _ocr_cell(ink_layer_image, zone_bbox["received_wt"], RECEIVED_WT_CONFIG)
    received_date, _ = _ocr_cell(ink_layer_image, zone_bbox["received_date"], RECEIVED_DATE_CONFIG)
    remark, remark_confidence = _ocr_cell(ink_layer_image, zone_bbox["remark"], REMARK_CONFIG)

    return {
        "received_wt": received_wt,
        "received_date": received_date,
        "remark": remark,
        "remark_confidence": round(remark_confidence, 1),
        "low_confidence": bool(remark) and remark_confidence < LOW_CONFIDENCE_THRESHOLD,
        "zone_found": True,
    }
