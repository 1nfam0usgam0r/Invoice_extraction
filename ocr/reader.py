"""One OCR engine for the whole project: Tesseract via pytesseract.

Replaces EasyOCR, which pulled in torch and spent most of a page's time in
model load. Every detection comes back in the shape the extractors already
expect — text, confidence on a 0-1 scale, and a four-corner box.
"""

import os

import numpy as np
import pytesseract
from PIL import Image

# pytesseract shells out to the binary and only finds it when it is on PATH,
# which the Windows installer does not do. Tesseract ships with this repo —
# the installer was pointed at the root, so the exe and its DLLs sit there
# loose, beside project/. Resolved relative to this file rather than hardcoded
# so the checkout can live on any drive. tessdata is found by Tesseract
# itself, next to the exe.
_LOCAL_PATH = os.path.join(
    os.path.dirname(          # Invoice Extraction root
        os.path.dirname(      # project/
            os.path.dirname(  # project/ocr/
                os.path.abspath(__file__)
            )
        )
    ),
    "tesseract.exe",
)

_FALLBACK_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]

if os.path.exists(_LOCAL_PATH):
    TESSERACT_PATH = _LOCAL_PATH
else:
    for _p in _FALLBACK_PATHS:
        if os.path.exists(_p):
            TESSERACT_PATH = _p
            break
    else:
        raise FileNotFoundError(
            f"Tesseract not found at {_LOCAL_PATH} or default install paths"
        )

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# psm 6 — "a single uniform block of text" — keeps Tesseract from trying to
# find its own reading order on a form; oem 3 is the default LSTM engine.
# Callers override psm per document type; the bill table uses 4.
DEFAULT_PSM = 6
OEM = 3

# ``image_to_data`` reports one box per WORD, where EasyOCR reported one per
# phrase. Both callers depend on phrases: the bill's header cell "LR No" has
# to stay one cell or it defines two columns, and the LR's labels are matched
# whole against "SHIPMENT NO" and "DESCRIPTION OF PRODUCT". So words on the
# same text line are re-joined when the gap between them is no wider than
# this many times their height — a word space, not a column gap. Expressed as
# a ratio of text height so it holds at any scan DPI.
WORD_GAP_RATIO = 0.8


def _merge_line(words: list[dict]) -> list[dict]:
    """Join words on one text line into phrases, splitting on wide gaps.

    ``words`` must be a single Tesseract line, left to right. Returns one
    dict per phrase, its box the union of the boxes merged into it and its
    confidence the lowest of them — a phrase is only as trustworthy as its
    worst word.
    """
    phrases = []
    for word in words:
        if phrases:
            previous = phrases[-1]
            gap = word["left"] - (previous["left"] + previous["width"])
            if gap <= WORD_GAP_RATIO * max(previous["height"], word["height"]):
                right = max(
                    previous["left"] + previous["width"], word["left"] + word["width"]
                )
                bottom = max(
                    previous["top"] + previous["height"], word["top"] + word["height"]
                )
                previous["top"] = min(previous["top"], word["top"])
                previous["width"] = right - previous["left"]
                previous["height"] = bottom - previous["top"]
                previous["text"] += " " + word["text"]
                previous["conf"] = min(previous["conf"], word["conf"])
                continue
        phrases.append(dict(word))
    return phrases


# A second merge pass, looser than WORD_GAP_RATIO and blind to Tesseract's
# own line/block ids: on a ruled/boxed form, one printed label is sometimes
# split across two of Tesseract's own lines even though both halves sit on
# the same visual row - confirmed on this project's own LR scans ("TRUCK"
# and "NO" of "TRUCK NO", "FROM" and "ORIGIN: Raigarh" of "FROM ORIGIN",
# each one printed line but read as two detections a Tesseract line/block
# apart). Looser than WORD_GAP_RATIO because a ruled cell's own border adds
# a few extra pixels to that gap on top of the ordinary word space -
# confirmed measured: "TRUCK"/"NO" sit 0.84x their own height apart, just
# past WORD_GAP_RATIO's 0.8x. Bounded by a real y-overlap requirement (not
# just "roughly the same y"), so two different phrases in two different
# table columns that merely happen to reach the same page height (a
# frequent case on a multi-column form) are not merged just because they're
# geometrically close - column headers evidenced this session run at
# hundreds of pixels apart, far past this ratio.
ROW_MERGE_Y_OVERLAP = 0.5
ROW_MERGE_GAP_RATIO = 1.2


def _row_merge_candidate(first: dict, second: dict) -> bool:
    """Whether two phrases, in whichever order, share one printed row and
    sit close enough together to merge - see ``_merge_same_row``."""
    left, right = (first, second) if first["left"] <= second["left"] else (second, first)
    left_bottom = left["top"] + left["height"]
    right_bottom = right["top"] + right["height"]
    overlap = min(left_bottom, right_bottom) - max(left["top"], right["top"])
    shorter = min(left["height"], right["height"])
    gap = right["left"] - (left["left"] + left["width"])
    return (shorter > 0 and overlap >= ROW_MERGE_Y_OVERLAP * shorter
            and 0 <= gap <= ROW_MERGE_GAP_RATIO * max(left["height"], right["height"]))


def _merge_pair(first: dict, second: dict) -> dict:
    """The single phrase two merged phrases become, left-to-right by
    their own ``left`` regardless of which argument order they arrived in
    - sort order alone is not a reliable left/right indicator when two
    same-row phrases differ by only a pixel or two in ``top`` (ordinary
    OCR box jitter), which is exactly the case this exists to handle."""
    left, right = (first, second) if first["left"] <= second["left"] else (second, first)
    new_right = max(left["left"] + left["width"], right["left"] + right["width"])
    new_bottom = max(left["top"] + left["height"], right["top"] + right["height"])
    new_top = min(left["top"], right["top"])
    return {
        "text": f"{left['text']} {right['text']}",
        "conf": min(left["conf"], right["conf"]),
        "left": left["left"],
        "top": new_top,
        "width": new_right - left["left"],
        "height": new_bottom - new_top,
    }


def _merge_same_row(phrases: list) -> list:
    """Merge phrases across Tesseract's own line/block boundaries when
    they clearly share one printed row: their vertical extents overlap by
    at least ``ROW_MERGE_Y_OVERLAP`` of the shorter one's height, and the
    gap between them is within ``ROW_MERGE_GAP_RATIO`` of their height.

    A plain single sorted pass merging only adjacent-in-sort-order pairs is
    not enough: sorting by (top, left) puts two same-row phrases in the
    wrong relative order whenever their ``top`` differs by even a pixel or
    two (ordinary OCR box jitter - confirmed on this project's own scan,
    where "LR" read top=205 and its own "NO" read top=204), which then
    computes a nonsensical negative gap and silently fails to merge them.
    Repeatedly re-scanning every pair (not just adjacent list entries) and
    picking left/right by their own coordinates, not iteration order,
    avoids that entirely, at the cost of being O(n^2) per pass - fine at a
    single page's phrase count.
    """
    remaining = [dict(phrase) for phrase in phrases]
    changed = True
    while changed:
        changed = False
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                if _row_merge_candidate(remaining[i], remaining[j]):
                    remaining[i] = _merge_pair(remaining[i], remaining[j])
                    del remaining[j]
                    changed = True
                    break
            if changed:
                break
    return remaining


def get_ocr_results(image_np: np.ndarray, psm: int = DEFAULT_PSM) -> list[dict]:
    """Run Tesseract on a numpy image array.

    Args:
        image_np: RGB (or grayscale) page image as a numpy array.
        psm: Tesseract page segmentation mode. 6 treats the page as one
            uniform block; 4 as a single column of variable-sized text,
            which holds a table's line structure together better.

    Returns:
        One dict per detected phrase, each with:
        ``text`` (str), ``confidence`` (float, 0-1) and ``box``, the four
        corners ``[[x1,y1], [x2,y1], [x2,y2], [x1,y2]]`` — the same corner
        order EasyOCR produced, so the callers' coordinate maths is unchanged.
    """
    pil_img = Image.fromarray(image_np)

    data = pytesseract.image_to_data(
        pil_img,
        config=f"--psm {psm} --oem {OEM}",
        output_type=pytesseract.Output.DICT,
    )

    # Collect the words, keyed by the text line Tesseract assigned them to.
    lines: dict[tuple, list] = {}
    for i in range(len(data["text"])):
        text = str(data["text"][i]).strip()
        # Tesseract reports -1 for the layout boxes (blocks, paragraphs,
        # lines) it emits alongside the words; those carry no text.
        confidence = float(data["conf"][i])
        if not text or confidence < 0:
            continue

        line_id = (
            data["block_num"][i],
            data["par_num"][i],
            data["line_num"][i],
        )
        lines.setdefault(line_id, []).append(
            {
                "text": text,
                "conf": confidence,
                "left": int(data["left"][i]),
                "top": int(data["top"][i]),
                "width": int(data["width"][i]),
                "height": int(data["height"][i]),
            }
        )

    phrases = []
    for _line_id, words in lines.items():
        words.sort(key=lambda word: word["left"])
        phrases.extend(_merge_line(words))

    results = []
    for phrase in _merge_same_row(phrases):
        left, top = phrase["left"], phrase["top"]
        right = left + phrase["width"]
        bottom = top + phrase["height"]
        results.append(
            {
                "text": phrase["text"],
                "confidence": phrase["conf"] / 100.0,
                "box": [
                    [left, top],
                    [right, top],
                    [right, bottom],
                    [left, bottom],
                ],
            }
        )

    return results
