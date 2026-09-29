import io

import fitz  # PyMuPDF
from PIL import Image


def pdf_to_images(pdf_path: str, dpi: int = 300) -> list[Image.Image]:
    """
    Convert every page of a PDF to a PIL Image using PyMuPDF.
    DPI controls render resolution.
    Returns list of PIL Images, one per page.
    Raises ValueError if PDF has no pages.
    """
    doc = fitz.open(pdf_path)
    try:
        if len(doc) == 0:
            raise ValueError(f"PDF has no pages: {pdf_path}")

        images = []
        zoom = dpi / 72          # 72 is PyMuPDF default DPI
        matrix = fitz.Matrix(zoom, zoom)

        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            images.append(img)
    finally:
        doc.close()

    return images


def get_page_count(pdf_path: str) -> int:
    """
    Return number of pages without rendering all pages.
    """
    doc = fitz.open(pdf_path)
    try:
        count = len(doc)
    finally:
        doc.close()
    return count
