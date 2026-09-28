"""Optional image cleanup pass, tried ahead of OCR.

Nothing in the existing extractors calls this yet on its own authority - it is
wired in at exactly one point in each (the image-loading step) and only after
``test_preprocessing.py`` showed it helps. Everything here is pure: given a
PIL image it returns a new one, and never touches the original.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image

# Below this width the page is upscaled 2x before anything else runs; text
# that small loses too much to thresholding and denoising otherwise.
DEFAULT_UPSCALE_THRESHOLD = 2400

# A rotation smaller than this is scanner jitter, not skew - rotating for it
# does more damage (resampling blur) than the tilt itself.
MIN_SKEW_DEGREES = 0.3

# cv2.adaptiveThreshold(ADAPTIVE_THRESH_GAUSSIAN_C) parameters. Chosen over
# Otsu by test_preprocessing.py: see its output for the comparison this was
# picked from.
ADAPTIVE_BLOCK_SIZE = 31
ADAPTIVE_C = 15

# cv2.medianBlur kernel for the final light denoise. Must be odd.
DENOISE_KERNEL = 3

THRESHOLD_OTSU = "otsu"
THRESHOLD_ADAPTIVE = "adaptive"


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Rotate the page level, using its largest ink blob as the text region.

    A blank or near-blank page (no contours) is returned unchanged rather
    than raising - there is nothing to deskew against.
    """
    inverted = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < gray.size * 0.01:
        # Too small a blob to trust for an angle - noise, not text.
        return gray

    angle = cv2.minAreaRect(largest)[-1]
    # OpenCV < 4.5 reports angles in [-90, 0); OpenCV >= 4.5 / 5.x reports
    # them in [0, 90). Normalise both to [-45, 45] so a nearly-upright page
    # reads as ~0 skew regardless of which convention is in use.
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90

    if abs(angle) < MIN_SKEW_DEGREES:
        return gray

    height, width = gray.shape
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray, matrix, (width, height),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )


def _threshold(gray: np.ndarray, method: str) -> np.ndarray:
    if method == THRESHOLD_OTSU:
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C,
    )


def preprocess_for_ocr(
    pil_image: Image.Image,
    upscale_if_narrower_than: int = DEFAULT_UPSCALE_THRESHOLD,
    threshold_method: str = THRESHOLD_ADAPTIVE,
) -> Image.Image:
    """Upscale, deskew, threshold and denoise a page ahead of OCR.

    Args:
        pil_image: A page image, grayscale or color.
        upscale_if_narrower_than: Pages narrower than this (pixels) are
            resized 2x with cubic interpolation before anything else.
        threshold_method: ``"adaptive"`` (the default, tuned against the
            sample pages) or ``"otsu"``, kept selectable so
            ``test_preprocessing.py`` can score both.

    Returns:
        A new, single-channel PIL image; ``pil_image`` is left untouched.
    """
    gray = np.array(pil_image.convert("L"))

    if gray.shape[1] < upscale_if_narrower_than:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray = _deskew(gray)
    thresholded = _threshold(gray, threshold_method)
    denoised = cv2.medianBlur(thresholded, DENOISE_KERNEL)

    return Image.fromarray(denoised)


def page_confidence(pil_image: Image.Image) -> float:
    """Mean Tesseract word confidence for a page.

    ``-1`` entries (no text found at that box) are dropped rather than
    counted as zero, so a mostly-blank margin does not make a page look
    worse than it read.
    """
    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
    scores = [float(score) for score in data["conf"] if str(score) not in ("-1", "")]
    return float(np.mean(scores)) if scores else 0.0


def preprocess_for_ocr_if_better(
    pil_image: Image.Image,
    upscale_if_narrower_than: int = DEFAULT_UPSCALE_THRESHOLD,
    threshold_method: str = THRESHOLD_ADAPTIVE,
) -> Image.Image:
    """``preprocess_for_ocr``, kept only when it actually reads better.

    test_preprocessing.py found thresholding helps a genuinely degraded scan
    a great deal (+5 to +12 confidence points on the two worst pages tested)
    and costs an already-clean one a little (-2 to -3 points) - and that no
    image statistic tried (Laplacian variance, contrast) told the two apart
    ahead of time; a page's own sharpness did not predict which case it was
    in. So this asks Tesseract directly: OCR both versions, keep whichever it
    reads better. That doubles the OCR cost of the page this runs on, but is
    the only thing that gets a "never regress a page" guarantee rather than a
    hope that a fixed method generalizes past the pages it was tuned on.
    """
    processed = preprocess_for_ocr(pil_image, upscale_if_narrower_than, threshold_method)
    if page_confidence(processed) >= page_confidence(pil_image):
        return processed
    return pil_image
