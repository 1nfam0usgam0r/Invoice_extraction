import os
import subprocess

# Set Tesseract PATH before img2table is imported.
# img2table shells out to the tesseract binary and finds it only on PATH -
# it ignores TESSERACT_CMD, which is what this script used to set.
# ocr/ -> project/ -> Invoice Extraction/, where the bundled exe sits.
TESSERACT_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
os.environ["PATH"] = TESSERACT_DIR + os.pathsep + os.environ.get("PATH", "")

result = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
if result.returncode != 0:
    raise EnvironmentError(
        f"Tesseract not found after PATH update. "
        f"Check {os.path.join(TESSERACT_DIR, 'tesseract.exe')} exists."
    )
print(f"Tesseract found: {result.stdout.split()[1]}")

import io

import pymupdf
from PIL import Image
from img2table.document import Image as TableImage
from img2table.ocr import TesseractOCR

# Load PDF - update path to your actual PDF
doc = pymupdf.open(
    r"E:\Invoice Extraction\project\STRHPO6X260001146 (Inland - Outbound Logistics).pdf"
)

# Render page 3 (index 2) at 300 DPI
zoom = 300 / 72
mat = pymupdf.Matrix(zoom, zoom)
pix = doc[2].get_pixmap(matrix=mat)
img = Image.open(io.BytesIO(pix.tobytes("png")))
img.save("debug_bill_page.png")
print(f"Bill page size: {img.size}")

ocr = TesseractOCR(lang='eng')

# The full page, uncropped. img2table's Image takes a path, bytes or BytesIO -
# it rejects a numpy array outright ("src must be a str, Path, BytesIO, or
# bytes"), so the same pixels go in as PNG bytes.
img_np = io.BytesIO()
img.convert("RGB").save(img_np, format="PNG")
doc_table = TableImage(src=img_np)


def report(tables):
    print(f"Tables found: {len(tables)}")
    for i, t in enumerate(tables):
        print(f"  Table {i}: {len(t.df)} rows x {len(t.df.columns)} cols "
              f"at y={t.bbox.y1}-{t.bbox.y2}")


# Test 1: default borderless
print("=== TEST 1: borderless=True ===")
tables = doc_table.extract_tables(
    ocr=ocr,
    implicit_rows=True,
    borderless_tables=True,
    min_confidence=30
)
report(tables)

# Test 2: bordered
print("=== TEST 2: borderless=False ===")
tables2 = doc_table.extract_tables(
    ocr=ocr,
    implicit_rows=True,
    borderless_tables=False,
    min_confidence=30
)
report(tables2)

# Test 3: lower confidence, borderless. img2table rejects min_confidence=0
# outright ("must be a positive int"), so 1 is as low as this goes.
print("=== TEST 3: borderless=True, min_conf=1 ===")
tables3 = doc_table.extract_tables(
    ocr=ocr,
    implicit_rows=True,
    borderless_tables=True,
    min_confidence=1
)
report(tables3)
