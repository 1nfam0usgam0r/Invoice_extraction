"""Standalone check: does preprocess_for_ocr actually raise OCR confidence?

Run directly: python test_preprocessing.py

Renders the bill page and three real Lorry Receipt pages (not the "working
sheet" pages app.py's LR_PAGES slice currently and wrongly includes - see the
note at the bottom of this file), saves before/after crops to
tmp_preprocess_check/, and prints pytesseract's mean confidence for each,
raw vs. both threshold methods. Nothing is wired into the real extractors
from here; that only happens if the numbers below say to.
"""

import os

import numpy as np
import pytesseract
from PIL import Image

os.environ["PATH"] = (
    os.path.dirname(os.path.abspath(__file__)).rsplit(os.sep, 1)[0]
    + os.pathsep
    + os.environ.get("PATH", "")
)

from ocr.pdf_handler import pdf_to_images
from ocr.preprocessing import (
    preprocess_for_ocr, preprocess_for_ocr_if_better,
    THRESHOLD_ADAPTIVE, THRESHOLD_OTSU,
)
from ocr.reader import TESSERACT_PATH  # noqa: F401  (sets tesseract_cmd)

SOURCE_PDF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "STRHPO6X260001146 (Inland - Outbound Logistics).pdf",
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_preprocess_check")

# Page indices in the combined PDF. 1 = first bill page (what BILL_PAGE
# names). 5, 6, 7 = real individual Lorry Receipts - NOT 3/4, which are a
# second, cleanly-typed rendition of the bill table, not LR pages; app.py's
# LR_PAGES = slice(3, None) is wrong by two pages, but app.py's route logic
# is off-limits here, so this script just points at the real thing.
BILL_PAGE_INDEX = 1
LR_PAGE_INDICES = [5, 6, 7]

PAGES = {
    "bill_page": BILL_PAGE_INDEX,
    "lr_page_1": LR_PAGE_INDICES[0],
    "lr_page_2": LR_PAGE_INDICES[1],
    "lr_page_3": LR_PAGE_INDICES[2],
}


def mean_confidence(image: Image.Image) -> float:
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    scores = [float(c) for c in data["conf"] if str(c) not in ("-1", "")]
    return float(np.mean(scores)) if scores else 0.0


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    pages = pdf_to_images(SOURCE_PDF, dpi=300)

    results = {}
    for label, index in PAGES.items():
        raw = pages[index]
        raw.save(os.path.join(OUT_DIR, f"{label}_before.png"))
        raw_conf = mean_confidence(raw)

        scored = {"raw": raw_conf}
        for method in (THRESHOLD_ADAPTIVE, THRESHOLD_OTSU):
            processed = preprocess_for_ocr(raw, threshold_method=method)
            processed.save(os.path.join(OUT_DIR, f"{label}_after_{method}.png"))
            scored[method] = mean_confidence(processed)
        results[label] = scored

        print(f"{label} (page {index}):")
        for method, conf in scored.items():
            print(f"  {method:>10s}: {conf:6.2f}")

    print("\n--- summary (mean confidence, raw vs adaptive vs otsu) ---")
    header = f"{'page':<12}{'raw':>10}{'adaptive':>10}{'otsu':>10}"
    print(header)
    improved_over_raw = {THRESHOLD_ADAPTIVE: 0, THRESHOLD_OTSU: 0}
    worst_drop = {THRESHOLD_ADAPTIVE: 0.0, THRESHOLD_OTSU: 0.0}
    for label, scored in results.items():
        print(f"{label:<12}{scored['raw']:>10.2f}"
              f"{scored[THRESHOLD_ADAPTIVE]:>10.2f}{scored[THRESHOLD_OTSU]:>10.2f}")
        for method in (THRESHOLD_ADAPTIVE, THRESHOLD_OTSU):
            delta = scored[method] - scored["raw"]
            if delta > 0:
                improved_over_raw[method] += 1
            worst_drop[method] = min(worst_drop[method], delta)

    print("\n--- gate (static method, whole-run choice) ---")
    for method in (THRESHOLD_ADAPTIVE, THRESHOLD_OTSU):
        pages_improved = improved_over_raw[method]
        drop = -worst_drop[method]
        passed = pages_improved >= 2 and drop <= 2.0
        print(f"{method}: improved on {pages_improved}/4 pages, "
              f"worst drop {drop:.2f} points -> {'PASS' if passed else 'FAIL'}")

    print("\n--- confidence-guarded selector (keeps whichever each page reads "
          "better as) ---")
    guarded_improved = 0
    guarded_worst_drop = 0.0
    for label, index in PAGES.items():
        raw = pages[index]
        raw_conf = results[label]["raw"]
        guarded = preprocess_for_ocr_if_better(raw)
        guarded_conf = mean_confidence(guarded)
        delta = guarded_conf - raw_conf
        picked = "processed" if guarded_conf > raw_conf else "raw (fell back)"
        print(f"  {label:<12}raw={raw_conf:6.2f}  guarded={guarded_conf:6.2f}  "
              f"delta={delta:+6.2f}  picked={picked}")
        if delta > 0:
            guarded_improved += 1
        guarded_worst_drop = min(guarded_worst_drop, delta)
    guarded_drop = -guarded_worst_drop
    guarded_pass = guarded_improved >= 2 and guarded_drop <= 2.0
    print(f"  -> improved on {guarded_improved}/4 pages, worst drop "
          f"{guarded_drop:.2f} points -> {'PASS' if guarded_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
