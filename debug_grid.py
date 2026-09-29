import os
os.environ["PATH"] = (
    r"E:\Invoice Extraction" + os.pathsep +
    os.environ.get("PATH", "")
)

import pymupdf
import cv2
import numpy as np
from PIL import Image
import io

# Load page 3 (bill) at 300 DPI
doc = pymupdf.open(
    r"E:\Invoice Extraction\project\STRHPO6X260001146 (Inland - Outbound Logistics).pdf"
)
zoom = 300 / 72
mat = pymupdf.Matrix(zoom, zoom)
pix = doc[2].get_pixmap(matrix=mat)
img = Image.open(io.BytesIO(pix.tobytes("png")))
img_np = np.array(img.convert('RGB'))
gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
h, w = gray.shape
print(f"Page size: {w}x{h}")

# Binarise
binary = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV, 15, 10
)

# Detect horizontal lines
h_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT, (int(w * 0.3), 1)
)
h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

# Detect vertical lines
v_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT, (1, int(h * 0.02))
)
v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

# Count horizontal lines (rows)
h_projection = np.sum(h_lines, axis=1)
h_line_rows = np.where(h_projection > w * 0.3 * 255 * 0.5)[0]
print(f"Max h_projection value: {h_projection.max()}")
print(f"Threshold used: {w * 0.3 * 255 * 0.5}")
# group adjacent rows
h_groups = []
if len(h_line_rows) > 0:
    start = h_line_rows[0]
    prev = h_line_rows[0]
    for y in h_line_rows[1:]:
        if y - prev > 10:
            h_groups.append((start + prev) // 2)
            start = y
        prev = y
    h_groups.append((start + prev) // 2)

# Count vertical lines (columns)
v_projection = np.sum(v_lines, axis=0)
v_line_cols = np.where(v_projection > h * 0.02 * 255 * 0.5)[0]
v_groups = []
if len(v_line_cols) > 0:
    start = v_line_cols[0]
    prev = v_line_cols[0]
    for x in v_line_cols[1:]:
        if x - prev > 10:
            v_groups.append((start + prev) // 2)
            start = x
        prev = x
    v_groups.append((start + prev) // 2)

print(f"Horizontal lines detected: {len(h_groups)}")
print(f"  Y positions: {h_groups}")
print(f"Vertical lines detected: {len(v_groups)}")
print(f"  X positions: {v_groups}")

# Save visual
vis = img_np.copy()
for y in h_groups:
    cv2.line(vis, (0, y), (w, y), (0, 0, 255), 3)
for x in v_groups:
    cv2.line(vis, (x, 0), (x, h), (0, 255, 0), 3)
Image.fromarray(vis).save("debug_grid_lines.png")
print("Saved debug_grid_lines.png")