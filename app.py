import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image

from ocr.pdf_handler import pdf_to_images, get_page_count
from ocr.invoice_extractor import extract_invoice
from ocr.bill_extractor import extract_bill, debug_ocr_output
from ocr.column_config import detect_client
from ocr.lr_extractor import extract_lr
from ocr.normaliser import normalise_bill_row, normalise_lr_record
from ocr.reconciler import reconcile
from ocr.tax_invoice_extractor import extract_tax_invoice
from ocr.validator import validate_bill_row
from ocr.excel_writer import write_excel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Created at import time so the app also works under a WSGI server.
for _directory in (UPLOAD_DIR, OUTPUT_DIR, MODEL_DIR):
    os.makedirs(_directory, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB per request

job_store = {}   # in-memory, fine for Phase 1
MAX_STORED_JOBS = 50


def _evict_old_jobs() -> None:
    """Drop finished jobs when the store grows too large.

    Prefers evicting completed or errored jobs first. Falls back to the
    oldest entry (insertion order) when everything is still in-flight.
    """
    if len(job_store) < MAX_STORED_JOBS:
        return
    for target_status in ('done', 'error'):
        for jid, job in list(job_store.items()):
            if job.get('status') == target_status:
                _discard(job.get('bill_image_path'))
                del job_store[jid]
                if len(job_store) < MAX_STORED_JOBS:
                    return
    # Still over limit — evict the oldest job regardless of status.
    oldest = next(iter(job_store))
    _discard(job_store[oldest].get('bill_image_path'))
    del job_store[oldest]

# Statuses the pipeline passes through, in order. No 'validating' stage for
# now - see _run_pipeline, rolled back along with the validation-flagging
# layer until it is rebuilt deliberately.
PIPELINE_STAGES = [
    'uploaded', 'converting', 'invoice_ocr', 'bill_ocr', 'tax_invoice_ocr',
    'lr_ocr', 'reconciling', 'writing_excel', 'done',
]

# Keys held only for server-side bookkeeping.
_PRIVATE_KEYS = ('combined_path', 'bill_image_path', 'bill_ocr')

# Page layout of the combined PDF: the approval form, the bill table (across
# two pages - captions and the first rows on the earlier one, the rest and
# the totals line on the later), a second clean rendition of that same bill
# table (also two pages - not read; bill_images above is what is OCR'd),
# then the individual LR receipts, one per page (31 of them, matching the
# bill's 31 rows), then more of that second bill rendition again. Confirmed
# against this file's actual page images - LR_PAGES used to start at 3,
# which is the second bill rendition, not an LR receipt; the real LRs are
# pages 5-35. This slice is specific to this sample PDF's layout (there is
# nothing in the file to detect page 36 not being an LR the way LR_PAGES'
# own start was corrected from a wrong assumption) - another client's
# combined PDF may not repeat the bill a second time at all.
INVOICE_PAGES = slice(0, 1)
BILL_PAGES = slice(1, 3)
LR_PAGES = slice(5, 36)

# The page /debug reads: the first of the bill's pages.
BILL_PAGE = 1

# Pages are rendered at this resolution for every stage.
RENDER_DPI = 300


def _public_job(job_id: str) -> dict:
    """The job record as the client sees it, without local upload paths."""
    job = job_store[job_id]
    payload = {key: value for key, value in job.items() if key not in _PRIVATE_KEYS}
    payload['job_id'] = job_id
    return payload


def _is_pdf(path: str) -> bool:
    """Check the file really starts with a PDF header, not just the name."""
    try:
        with open(path, 'rb') as handle:
            return handle.read(5) == b'%PDF-'
    except OSError:
        return False


def _discard(path) -> None:
    """Remove an upload we no longer need, ignoring a missing/locked file."""
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _run_pipeline(job_id: str) -> None:
    """Run the OCR pipeline, updating job status as each stage begins."""
    job = job_store[job_id]
    combined_path = job['combined_path']

    job['status'] = 'converting'
    images = pdf_to_images(combined_path, dpi=RENDER_DPI)
    invoice_images = images[INVOICE_PAGES]
    bill_images = images[BILL_PAGES]
    lr_images = images[LR_PAGES]
    job['total_pages'] = len(images)
    job['lr_pages'] = len(lr_images)

    # Warn if the PDF page count doesn't match the hardcoded layout. This
    # catches a different client's PDF silently producing wrong data.
    layout_warnings = []
    if len(images) < LR_PAGES.stop:
        layout_warnings.append(
            f"PDF has {len(images)} pages but layout expects at least "
            f"{LR_PAGES.stop}. LR pages may be missing or misaligned."
        )
    if len(lr_images) == 0:
        layout_warnings.append("No LR pages found — check page layout constants.")
    if len(bill_images) == 0:
        layout_warnings.append("No bill pages found — check page layout constants.")
    job['layout_warnings'] = layout_warnings

    # /process discards the upload at the end, but /debug is usually reached
    # after a run - keep the bill page so it still has something to read.
    bill_image_path = os.path.join(UPLOAD_DIR, f'{job_id}_bill.png')
    bill_images[0].save(bill_image_path)
    job['bill_image_path'] = bill_image_path

    job['status'] = 'invoice_ocr'
    job['detail'] = f'{len(invoice_images)} form page(s)'
    invoice_data = extract_invoice(invoice_images)

    job['status'] = 'bill_ocr'
    job['detail'] = f'{len(bill_images)} bill page(s)'
    # The transporter is named on the approval form, not on the bill, so the
    # client is identified from what the form read and handed to the bill
    # reader - it decides which column names the table gets.
    client_id = detect_client(' '.join(
        str(value) for value in invoice_data.values() if value
    ))
    job['client_id'] = client_id
    bill_header, bill_rows = extract_bill(bill_images, client_id=client_id)

    job['status'] = 'tax_invoice_ocr'
    job['detail'] = ''
    # Located by its own printed title (see ocr/column_classifier.py's
    # locate_pages_by_type and PAGE_TYPE_SYNONYMS), not a fixed page slice
    # the way invoice/bill/LR above are - handed every page, not a
    # pre-sliced range.
    tax_invoice_header, tax_invoice_rows = extract_tax_invoice(images)
    job['tax_invoice_rows_found'] = len(tax_invoice_rows)

    job['status'] = 'lr_ocr'
    job['detail'] = f'{len(lr_images)} LR page(s)'
    with ThreadPoolExecutor() as pool:
        lr_records = list(pool.map(extract_lr, lr_images))

    job['status'] = 'reconciling'
    job['detail'] = ''
    normalised_bill_rows = [normalise_bill_row(row) for row in bill_rows]
    normalised_lr_records = [normalise_lr_record(record) for record in lr_records]
    reconciliation = reconcile(normalised_bill_rows, normalised_lr_records)

    validated_bill_rows = [validate_bill_row(row, client_id=client_id) for row in bill_rows]

    job['status'] = 'writing_excel'
    job['detail'] = ''
    excel_path = os.path.join(OUTPUT_DIR, f'{job_id}.xlsx')
    write_excel(invoice_data, bill_header, validated_bill_rows, excel_path,
                tax_invoice_rows=tax_invoice_rows, tax_invoice_header=tax_invoice_header,
                lr_records=lr_records, layout_warnings=layout_warnings)

    job.update({
        'status': 'done',
        'detail': '',
        'invoice_fields_found': len(invoice_data),
        'invoice_fields_none': sum(1 for value in invoice_data.values() if value is None),
        'bill_rows_found': len(bill_rows),
        'bill_columns_found': list(bill_rows[0]) if bill_rows else [],
        'lr_records_found': len(lr_records),
        'reconciliation_summary': {
            'clear': sum(1 for r in reconciliation if r['status'] == 'clear'),
            'mismatch': sum(1 for r in reconciliation if r['status'] == 'mismatch'),
            'missing_lr': sum(1 for r in reconciliation if r['status'] == 'missing_lr'),
            'unbilled': sum(1 for r in reconciliation if r['status'] == 'unbilled'),
        },
        'excel_path': excel_path,
    })


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({"status": "ok"})


@app.route('/upload', methods=['POST'])
def upload():
    combined_pdf = request.files.get('combined_pdf')

    if combined_pdf is None or not combined_pdf.filename:
        return jsonify({'error': 'missing file(s): combined_pdf'}), 400

    if not combined_pdf.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'combined_pdf must be a PDF'}), 400

    job_id = str(uuid.uuid4())
    combined_path = os.path.join(UPLOAD_DIR, f'{job_id}_combined.pdf')
    combined_pdf.save(combined_path)

    # A renamed .txt would otherwise fail deep inside the pipeline.
    if not _is_pdf(combined_path):
        _discard(combined_path)
        return jsonify({'error': 'combined_pdf is not a valid PDF'}), 400

    try:
        page_count = get_page_count(combined_path)
    except Exception as exc:
        # The header already passed, so a failure here is the reader itself
        # failing to parse it - don't report that as a bad upload.
        _discard(combined_path)
        message = str(exc) or exc.__class__.__name__
        return jsonify({'error': f'could not read PDF: {message}'}), 500

    # Pages 1-2 are the approval form and page 3 is the bill.
    if page_count < 3:
        _discard(combined_path)
        return jsonify({'error': 'PDF must have at least 3 pages'}), 400

    _evict_old_jobs()
    job_store[job_id] = {
        'status': 'uploaded',
        'detail': '',
        'page_count': page_count,
        'combined_path': combined_path,
    }
    return jsonify({'job_id': job_id, 'page_count': page_count})


@app.route('/process/<job_id>', methods=['POST'])
def process(job_id):
    job = job_store.get(job_id)
    if job is None:
        return jsonify({'error': 'unknown job_id'}), 404

    status = job.get('status')
    if status == 'done':
        return jsonify(_public_job(job_id))
    if status not in ('uploaded', 'error'):
        # Guards against a double-click starting the pipeline twice.
        return jsonify({'error': f'job already {status}'}), 409

    def _run_in_background():
        try:
            _run_pipeline(job_id)
        except Exception as exc:
            job['last_stage'] = job.get('status')
            job['status'] = 'error'
            job['detail'] = ''
            job['error'] = str(exc) or exc.__class__.__name__
            _discard(job.get('combined_path'))
            _discard(job.get('bill_image_path'))
        else:
            # Uploads are only needed until the workbook exists.
            _discard(job.get('combined_path'))

    threading.Thread(target=_run_in_background, daemon=True).start()
    return jsonify(_public_job(job_id))


@app.route('/debug/<job_id>')
def debug(job_id):
    """Raw Tesseract detections for the bill page, top-to-bottom.

    Each entry carries the text with the ``y_center``, ``x_left`` and
    confidence that row grouping and column assignment key off, so a table
    that came out empty can be traced back to what was actually detected.
    """
    job = job_store.get(job_id)
    if job is None:
        return jsonify({'error': 'unknown job_id'}), 404

    if 'bill_ocr' in job:
        return jsonify({'job_id': job_id, 'items': job['bill_ocr']})

    bill_image_path = job.get('bill_image_path')
    if bill_image_path and os.path.isfile(bill_image_path):
        image = Image.open(bill_image_path)
    else:
        # Not processed yet - render the bill page out of the upload.
        combined_path = job.get('combined_path')
        if not combined_path or not os.path.isfile(combined_path):
            return jsonify({'error': 'no bill page available for this job'}), 404
        image = pdf_to_images(combined_path, dpi=RENDER_DPI)[BILL_PAGE]

    # OCR is the slow part; a job's page never changes, so read it once.
    job['bill_ocr'] = debug_ocr_output(image)
    return jsonify({'job_id': job_id, 'items': job['bill_ocr']})


@app.route('/status/<job_id>')
def status(job_id):
    if job_id not in job_store:
        return jsonify({'error': 'unknown job_id'}), 404
    return jsonify(_public_job(job_id))


@app.route('/download/<job_id>')
def download(job_id):
    job = job_store.get(job_id)
    if job is None:
        return jsonify({'error': 'unknown job_id'}), 404

    excel_path = job.get('excel_path')
    if not excel_path or not os.path.isfile(excel_path):
        return jsonify({'error': 'no report available for this job'}), 404

    return send_file(
        excel_path,
        as_attachment=True,
        download_name=f'extraction_{job_id[:8]}.xlsx',
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)
