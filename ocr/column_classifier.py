"""Name a bill table's columns without assuming any client's specific format.

Three tiers, most to least trusted, each only asked to resolve what the one
before it could not:

TIER 1 - fuzzy caption match: the printed caption row is small bold type a
photocopier destroys, but what survives is still text, and a garbled
"DELIVERY | STATION" is closer to "delivery station" than to anything else -
so it is matched against a plain-language synonym per field (data, never
regex) with rapidfuzz, not required to match exactly.

TIER 2 - generic data-type fingerprinting: whatever tier 1 could not name,
by what the column's *values* structurally look like - a date shape, a
decimal-with-two-places shape, digits mixed with letters - with no field
name, magnitude, country or client baked into the fingerprint itself. The
one field name TIER 2 is allowed to assign is done by relationship, not
shape: among decimal-currency columns, whichever three satisfy
qty * rate =~ amount row over row *are* qty, rate and amount, however large
or small the numbers happen to run for this client.

TIER 3 - the per-client config, i.e. ``column_config.CLIENT_COLUMNS``: the
only place a client name, plate format or currency magnitude may live, as
plain data. It is consulted by ``column_config.get_columns()`` *before*
this module runs at all for a configured client, so from here it is what
resolves anything tiers 1-2 still left ambiguous when the caller already
knows which client this is - see that function's own docstring.

Field names here match what ``ocr/normaliser.py`` and ``ocr/reconciler.py``
already expect off a bill row (``lr_no``, ``lr_date``, ``gross_qty``,
``rate``, ``amount``, ``vehicle_no``, ``shipment_no``) - see
``normalise_bill_row`` and its ``BILL_NUMERIC_FIELDS`` - plus a few
descriptive-only fields those two files never read. Changing any of the
first group's names would silently break reconciliation.
"""

import re

import pytesseract
from rapidfuzz import fuzz

try:
    from .normaliser import normalise_date
    from .reader import TESSERACT_PATH  # noqa: F401  (sets tesseract_cmd)
except ImportError:  # running this file directly from inside ocr/
    from normaliser import normalise_date
    from reader import TESSERACT_PATH  # noqa: F401

# How much of a column's non-empty cells must fit a shape for the column to
# be called that type. Anything less is coincidence.
MATCH_THRESHOLD = 0.7

# --------------------------------------------------------------------------
# TIER 1 - fuzzy caption match
# --------------------------------------------------------------------------

# Plain-language names a caption for this field might OCR close to - not a
# format, just words. Data, same as column_config.CLIENT_COLUMNS is data.
FIELD_SYNONYMS = {
    "sr_no": ["sr no", "s no", "serial no", "sl no"],
    "bill_no": ["bill no", "bill number"],
    "invoice_no": ["invoice no", "invoice number"],
    "shipment_no": ["shipment no", "shipment number"],
    "lr_no": ["lr no", "lr number", "lorry receipt no"],
    "lr_date": ["lr date", "date"],
    "arrival_date": ["arrival date"],
    "vehicle_no": ["vehicle no", "truck no", "vehicle number"],
    "delivery_station": ["delivery station"],
    "from_destination": ["from destination", "from origin"],
    "to_destination": ["to destination"],
    "gross_qty": ["gross qty", "invoice qty", "quantity"],
    "charge_qty": ["charge qty", "charged qty"],
    "rate": ["rate"],
    "amount": ["amount"],
    "short_qty_charges": ["short qty charges", "short qty"],
    "ld_charges": ["ld charges"],
    "balance_pay": ["balance pay", "balancepay"],
    "delivery_date": ["delivery date"],
    "party_name": ["party name", "consignee"],
    "vehicle_type": ["vehicle type", "truck type"],
    "description": ["description of product", "description"],
}

# The JSW GBS PO Expenses Approval Template's own key-value field labels -
# a single-instance cover FORM, not a repeating table, so kept in its own
# synonym pool rather than folded into FIELD_SYNONYMS above. Merging the two
# would reintroduce the exact cross-field collision _fuzzy_header_match's
# own docstring describes (a bare, generic synonym scoring a tied match
# against more than one field): this form's "invoice_number" and the bill
# table's "invoice_no" have no business competing over one shared pool when
# the two document types never share a page. Consulted only by
# ocr/invoice_extractor.py's zonal/anchor-based re-read of this form's own
# fields, never by classify_columns_by_content.
INVOICE_FORM_FIELD_SYNONYMS = {
    "vendor_name": ["vendor name"],
    "vendor_code": ["vendor code"],
    "po_number": ["po number", "jsw po number", "purchase order number"],
    "grn_srn_number": ["grn / srn number", "grn/srn number", "grn srn number",
                       "grn number", "srn number"],
    "invoice_number": ["invoice number"],
    "invoice_date": ["invoice date"],
    "invoice_amount": ["invoice amount incl taxes", "invoice amount"],
    # The cover form's own printed label for its one Amount field the FINAL
    # SPEC's "Penalties or any other deduction applicable" maps onto -
    # "*Penalties & any other deduction Applicable" is a Yes/No row, not a
    # plain label/value pair (see ocr/invoice_extractor.py's own
    # extract_invoice, which pulls just the "If Yes Amount:-" figure out of
    # that row rather than trusting a caption/value split here).
    "penalty_amount": ["penalties any other deduction applicable",
                       "penalties or any other deduction applicable",
                       "penalty amount"],
    "deduction_reason": ["reason for deduction above", "reason for deduction"],
    "company_name": ["company name"],
    "company_code": ["company code"],
    "location": ["location"],
    "currency": ["currency"],
    "cost_centre_wbs": ["cost centre wbs", "cost centre"],
    "profit_center": ["profit center", "profit centre"],
}

# The Lorry Receipt's own field labels - consulted only to ALIAS an
# already-detected label to the snake_case name ocr/normaliser.py and ocr/
# reconciler.py specifically key on (LR_TEXT_FIELDS/LR_NUMERIC_FIELDS), never
# to decide whether a label gets extracted in the first place. That
# decision belongs entirely to ocr/invoice_extractor.py's own generic
# label/value pass (position, colon-splitting, gutter-adjacency - the same
# engine already used for the cover form), which lr_extractor.py now reuses
# instead of matching every phrase on the page against a fixed, enumerated
# list of ~20 expected labels - the same schema-level hardcoding this
# session already replaced value-level hardcoding for elsewhere (bill_
# extractor.py's old per-field regex dict). A label this dict does not
# name is not dropped: it still reaches the sheet under its own cleaned key,
# whatever that turns out to be.
LR_FIELD_SYNONYMS = {
    "lr_no": ["lr no"],
    "lr_date": ["lr date"],
    "shipment_no": ["shipment no"],
    "from_origin": ["from origin"],
    # Not a bare "to": this form's own label for this field really is just
    # the two letters "TO", and best_synonym_matches has no length guard
    # of its own (that lives in lr_extractor.py's now-removed per-field
    # wrapper, not this shared primitive) - a bare two-letter synonym
    # collided with an unrelated key ("STO No") on this project's own scan,
    # stealing the alias. Left unaliased rather than risk that again;
    # reconciler.py does not key on to_destination anyway (see
    # MATCH_FIELDS), only normaliser.py's own text cleanup would have used it.
    "truck_no": ["truck no"],
    "truck_type": ["truck type"],
    "incoterm": ["incoterm"],
    "lorry_tare_wt": ["lorry tare wt"],
    "net_wt": ["net wt"],
    "gross_wt": ["gross wt"],
    "amount": ["amount"],
}

# A page's own title/caption, matched the same way a column's caption is -
# fuzzy, against plain-language variants, never a page's text embedded
# directly in extraction code. See locate_pages_by_type().
PAGE_TYPE_SYNONYMS = {
    "tax_invoice": [
        "transportation tax invoice", "gst tax invoice", "freight invoice",
        "tax invoice",
    ],
}

# How far down a page its title could plausibly sit, scanned as separate
# thin slices rather than one tall crop: Tesseract's own page segmentation
# garbles a crop mixing a title line with a logo, a folded-corner scan
# artefact and an address block into one image ('TRANSPORTATION TAX
# INVOICE' read as unrecognisable noise there), but reads the same title
# cleanly once it is the only thing in a narrow band on its own. Neither
# number is a per-client assumption - together they just bound how far down
# a page a title is worth looking, at a resolution fine enough to isolate
# one line from its neighbours.
PAGE_TITLE_REGION = 0.20
PAGE_TITLE_SLICE = 0.04


def locate_pages_by_type(pages: list, page_type: str, threshold: int = None) -> list:
    """Indices of ``pages`` whose own title reads as ``page_type``.

    The same mechanism tier 1 column matching uses (rapidfuzz against a
    plain-language synonym list in ``PAGE_TYPE_SYNONYMS``, never the page's
    caption text embedded directly in code) applied to a whole page instead
    of one cell: OCR the top ``PAGE_TITLE_REGION`` of each page in thin
    ``PAGE_TITLE_SLICE`` bands, and keep whichever pages have any slice that
    fuzzy-matches one of ``page_type``'s known variants at or above
    ``threshold``.

    Args:
        pages: PIL page images, in document order.
        page_type: A key of ``PAGE_TYPE_SYNONYMS``.
        threshold: Defaults to ``CAPTION_MATCH_THRESHOLD``, the same bar
            tier 1 column matching uses.

    Returns:
        Matching page indices, in document order. Empty if ``page_type`` is
        not configured or no page's title matches - never a guess.
    """
    synonyms = PAGE_TYPE_SYNONYMS.get(page_type)
    if not synonyms:
        return []
    if threshold is None:
        threshold = CAPTION_MATCH_THRESHOLD

    def _page_matches(args):
        index, page = args
        slice_height = max(1, int(page.height * PAGE_TITLE_SLICE))
        region_bottom = int(page.height * PAGE_TITLE_REGION)
        for top in range(0, region_bottom, slice_height):
            band = page.crop((0, top, page.width, min(top + slice_height, page.height)))
            text = " ".join(pytesseract.image_to_string(band).split()).lower()
            if not text:
                continue
            for synonym in synonyms:
                # partial_ratio alone will find some well-aligned window
                # inside a long, unrelated sentence and score it deceptively
                # high (an approval form's boilerplate paragraph scored 82
                # against "tax invoice" this way) - a real title is short,
                # not a paragraph, so a band whose text runs much longer
                # than the synonym itself is not a plausible title match
                # regardless of score.
                if len(text) > len(synonym) * 3:
                    continue
                score = max(fuzz.token_sort_ratio(text, synonym), fuzz.partial_ratio(text, synonym))
                if score >= threshold:
                    return index
        return None

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor() as pool:
        results = pool.map(_page_matches, enumerate(pages))
    return sorted(idx for idx in results if idx is not None)


# A caption's best synonym match below this score is not trusted at all -
# tier 2 gets the column instead.
CAPTION_MATCH_THRESHOLD = 70


def _clean_caption(text) -> str:
    """Drop tokens that are mostly punctuation; lowercase what is left."""
    tokens = []
    for token in str(text or "").split():
        alnum = sum(1 for char in token if char.isalnum())
        if alnum < 2 or alnum < len(token) * 0.5:
            continue
        tokens.append(token)
    return " ".join(tokens).lower()


def best_synonym_matches(cleaned_texts: dict, synonyms_by_field: dict,
                          threshold: float) -> list:
    """``(score, index, field)`` triples at or above ``threshold``, for
    every (index, field) pair - not just each index's single best field.

    ``cleaned_texts`` is ``{index: cleaned_text}``. Shared scoring core
    behind both tier 1 table-caption matching and the cover-form's own
    zonal label matching (ocr/invoice_extractor.py): token_sort_ratio and
    partial_ratio, the better of the two, against every synonym of every
    field. token_sort_ratio alone is too strict when a clean keyword
    survives OCR but is surrounded by unrelated noise tokens (a
    whole-string comparison against "amount" is dragged down by three
    extra noise words even though "amount" is intact); partial_ratio finds
    the best-aligned substring instead, so the two together catch both
    "the text reads close to the whole synonym" and "the synonym survived
    intact inside noisier text".
    """
    candidates = []
    for index, cleaned in cleaned_texts.items():
        if not cleaned:
            continue
        for field, synonyms in synonyms_by_field.items():
            score = max(
                max(fuzz.token_sort_ratio(cleaned, synonym),
                    fuzz.partial_ratio(cleaned, synonym))
                for synonym in synonyms
            )
            if score >= threshold:
                candidates.append((score, index, field))
    return candidates


def greedy_assign(candidates: list) -> dict:
    """``{index: field}`` from ``(score, index, field)`` triples, highest
    score first, each index and each field name used at most once.

    Every candidate pair must already be in the running - not just each
    index's single best-scoring field - for this to behave correctly: an
    index's top pick losing a tie to another index must not knock it out of
    the running entirely, it still deserves its own next-best, more
    specific field rather than going unassigned. A bare, generic synonym
    shared across several fields (say "date" inside ``lr_date``'s own list)
    can score a tied 100 against more than one column's caption - see the
    tax-invoice table's Arrival Date / Delivery Date / To Destination
    columns, which used to be mislabelled this exact way before every
    candidate pair (not just each caption's argmax) was fed in here.
    """
    assigned: dict = {}
    taken_fields: set = set()
    for score, index, field in sorted(candidates, key=lambda item: -item[0]):
        if field in taken_fields or index in assigned:
            continue
        assigned[index] = field
        taken_fields.add(field)
    return assigned


def _fuzzy_header_match(captions: list) -> dict:
    """TIER 1: each caption against every field's synonyms, best scores win.

    Returns ``{column_index: field_name}`` for whatever cleared
    ``CAPTION_MATCH_THRESHOLD`` - see ``best_synonym_matches`` and
    ``greedy_assign`` for the mechanics.
    """
    cleaned = {index: _clean_caption(caption)
               for index, caption in enumerate(captions or [])}
    candidates = best_synonym_matches(cleaned, FIELD_SYNONYMS, CAPTION_MATCH_THRESHOLD)
    return greedy_assign(candidates)


# --------------------------------------------------------------------------
# TIER 2 - generic data-type fingerprinting
# --------------------------------------------------------------------------

TYPE_DATE_LIKE = "date_like"
TYPE_TIMESTAMP_LIKE = "timestamp_like"
TYPE_DECIMAL_CURRENCY = "decimal_currency"
TYPE_ALPHANUMERIC_CODE = "alphanumeric_code"
TYPE_SMALL_SEQUENTIAL_INT = "small_sequential_int"
TYPE_LARGE_NUMERIC_CODE = "large_numeric_code"
TYPE_FREE_TEXT = "free_text"

GENERIC_COLUMN_TYPES = (
    TYPE_DATE_LIKE, TYPE_TIMESTAMP_LIKE, TYPE_DECIMAL_CURRENCY,
    TYPE_ALPHANUMERIC_CODE, TYPE_SMALL_SEQUENTIAL_INT,
    TYPE_LARGE_NUMERIC_CODE, TYPE_FREE_TEXT,
)

# Shape only - no field name, no magnitude, no locale beyond "a date has
# numbers separated by . / or -" and "a timestamp has a month name and a
# colon-separated time".
_DATE_SHAPE = re.compile(r"^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$")
_TIMESTAMP_SHAPE = re.compile(r"^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\s+\d{1,2}:\d{2}\s*(am|pm)?$",
                              re.IGNORECASE)
_DECIMAL_CURRENCY_SHAPE = re.compile(r"^\d+(,\d{2,3})*\.\d{2,3}$")
_ALL_DIGITS = re.compile(r"^\d+$")
_ALPHANUMERIC_MIXED = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9]+$")
_MOSTLY_ALPHA = re.compile(r"^[A-Za-z .]+$")

# A short numeric or alphanumeric code (a state code "37", a two-letter
# unit) is a plausible value in its own right; a lone punctuation mark or
# OCR-garbled letter is not. Reused wherever a caller needs to tell a short
# real value apart from short noise without hardcoding a length cutoff
# alone - see ocr/lr_extractor.py's own noise-floor filter, which consults
# this instead of assuming "short" always means "noise".
_SHORT_VALUE_SHAPES = (_ALL_DIGITS, _ALPHANUMERIC_MIXED, _DATE_SHAPE, _DECIMAL_CURRENCY_SHAPE)


def looks_like_plausible_value(text: str) -> bool:
    """Whether ``text`` matches one of the generic shapes TIER 2 column
    fingerprinting already tests table cells against (a number, a date, an
    alphanumeric code, a decimal amount) - used to tell a short but
    legitimate value (a state code "37") apart from short noise (a stray
    punctuation mark, a single OCR-garbled character), never a hardcoded
    allowlist of specific values."""
    stripped = str(text or "").strip()
    return bool(stripped) and any(shape.match(stripped) for shape in _SHORT_VALUE_SHAPES)


def _hit_rate(values: list, test) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if test(value.strip())) / len(values)


def _is_sequential(values: list) -> bool:
    """Behavioural, not magnitude-based: mostly-increasing by a small step -
    a serial number - rather than an arbitrary large reference number that
    happens to also be short. Needs a few rows to say anything."""
    try:
        numbers = [int(value) for value in values]
    except ValueError:
        return False
    if len(numbers) < 3:
        return False
    steps = [b - a for a, b in zip(numbers, numbers[1:])]
    reasonable = sum(1 for step in steps if 0 <= step <= 5)
    return reasonable >= len(steps) * MATCH_THRESHOLD


def _fingerprint_column(values: list) -> str:
    """One of the ``TYPE_*`` constants for what this column's data looks
    like structurally - no field semantics, see the module docstring."""
    if not values:
        return TYPE_FREE_TEXT
    if _hit_rate(values, lambda v: bool(_TIMESTAMP_SHAPE.match(v))) >= MATCH_THRESHOLD:
        return TYPE_TIMESTAMP_LIKE
    if _hit_rate(values, lambda v: bool(_DATE_SHAPE.match(v))) >= MATCH_THRESHOLD:
        return TYPE_DATE_LIKE
    if _hit_rate(values, lambda v: bool(_DECIMAL_CURRENCY_SHAPE.match(v))) >= MATCH_THRESHOLD:
        return TYPE_DECIMAL_CURRENCY
    if _hit_rate(values, lambda v: bool(_ALL_DIGITS.match(v))) >= MATCH_THRESHOLD:
        return TYPE_SMALL_SEQUENTIAL_INT if _is_sequential(values) else TYPE_LARGE_NUMERIC_CODE
    if _hit_rate(values, lambda v: bool(_ALPHANUMERIC_MIXED.match(v))) >= MATCH_THRESHOLD:
        return TYPE_ALPHANUMERIC_CODE
    if _hit_rate(values, lambda v: bool(_MOSTLY_ALPHA.match(v))) >= MATCH_THRESHOLD:
        return TYPE_FREE_TEXT
    return TYPE_FREE_TEXT


# A candidate qty/rate/amount triple needs this fraction of rows to satisfy
# the multiplication within this relative tolerance before it counts as a
# real relationship rather than coincidence - same tolerance
# ocr/validator.py's own amount_mismatch check uses, for the same reason.
TRIPLE_MATCH_THRESHOLD = 0.7
TRIPLE_TOLERANCE = 0.02


_NUMERIC_JUNK = re.compile(r"[^0-9.]")
_HAS_DECIMAL_POINT = re.compile(r"^\d+\.\d+$")


def _as_floats(values: list) -> list:
    """Best-effort float per value, tolerant of a stray OCR character the
    strict ``decimal_currency`` shape would reject outright (a garbled
    leading digit reading as a letter - "B4008.60", "$9386.85" - a real
    pattern on this scan's amount column, at a rate that fails a 70%
    shape-match gate outright even though the arithmetic relationship below
    still holds on the rows that read cleanly). A value that still will not
    parse becomes ``None``, which the per-row product check already skips.
    """
    out = []
    for value in values:
        cleaned = _NUMERIC_JUNK.sub("", value)
        if not _HAS_DECIMAL_POINT.match(cleaned):
            out.append(None)
            continue
        try:
            out.append(float(cleaned))
        except ValueError:
            out.append(None)
    return out


def _numeric_ish_columns(data_rows: list) -> dict:
    """Every column at least half of whose non-empty cells look like a
    decimal number once stray OCR junk is stripped - the candidate pool for
    the arithmetic triple check, deliberately looser than
    ``TYPE_DECIMAL_CURRENCY`` (see ``_as_floats``): the relationship is
    strong enough evidence on its own that a column should not be excluded
    from even being tried just for reading noisily.

    Row-aligned: ``values[i]`` is row ``i`` of ``data_rows`` for every
    column returned, blank cells included as ``""`` rather than dropped -
    the triple check compares three columns row by row, so column A's
    ``values[5]`` and column B's ``values[5]`` must be the same physical
    table row, which independently filtering each column's own blanks out
    (as the shape/hit-rate tests elsewhere in this module safely do, since
    they never compare across columns) would silently break.
    """
    if not data_rows:
        return {}
    num_columns = max(len(row) for row in data_rows)
    result = {}
    for index in range(num_columns):
        values = [str(row[index]).strip() if index < len(row) else "" for row in data_rows]
        non_blank = [value for value in values if value]
        if not non_blank:
            continue
        parseable = sum(1 for value in non_blank
                        if _HAS_DECIMAL_POINT.match(_NUMERIC_JUNK.sub("", value)))
        if parseable >= len(non_blank) * 0.5:
            result[index] = values
    return result


def _product_matches(x: float, y: float, z: float) -> bool:
    """Whether ``z`` is plausibly ``x * y`` - tolerant of one dropped
    leading digit on ``z``.

    A common failure on this scan: the leading digit of an amount reads as
    a currency symbol or a letter and is then stripped rather than
    recovered ("$9386.85" -> 9386.85, correctly 69386.85). Retried against
    every digit 0-9 prepended to ``z`` - not a guess at what the digit *is*,
    only whether *some* single missing leading digit would explain the gap;
    the value itself is never changed, only this relationship test's
    tolerance for reading it.
    """
    if abs(x * y - z) <= abs(z) * TRIPLE_TOLERANCE:
        return True
    for digit in "0123456789":
        try:
            candidate = float(f"{digit}{z}")
        except ValueError:
            continue
        if abs(x * y - candidate) <= candidate * TRIPLE_TOLERANCE:
            return True
    return False


def _find_qty_rate_amount(decimal_columns: dict) -> dict:
    """Among columns TIER 2 typed decimal_currency, find the one triple
    where one column is the product of the other two, row over row - that
    is what makes it qty/rate/amount rather than, say, balance_pay (which
    is also decimal_currency-shaped but not a product of anything else on
    the row). No magnitude assumption: this holds however large or small
    the client's own quantities and rates run.

    Returns ``{"amount": index, "gross_qty": index, "rate": index}`` -
    gross_qty/rate are unordered between themselves by this test alone
    (multiplication doesn't say which factor is which); the caller resolves
    that from tier 1 where it can and leaves both unlabelled otherwise.
    """
    parsed = {index: _as_floats(values) for index, values in decimal_columns.items()}
    indices = list(parsed)
    best = None
    for a in indices:
        for b in indices:
            if b == a:
                continue
            for c in indices:
                if c in (a, b):
                    continue
                va, vb, vc = parsed[a], parsed[b], parsed[c]
                n = min(len(va), len(vb), len(vc))
                attempted = hits = 0
                for row in range(n):
                    x, y, z = va[row], vb[row], vc[row]
                    if x is None or y is None or z is None or z == 0:
                        continue
                    attempted += 1
                    if _product_matches(x, y, z):
                        hits += 1
                # Rate is over rows actually attempted, not every row - a
                # column that failed to parse at all on some rows (a blank,
                # an unrecoverable OCR read) should not silently count those
                # as relationship failures; there needs to be enough
                # attempted rows for the rate to mean anything, though.
                if attempted < 3:
                    continue
                rate = hits / attempted
                if rate >= TRIPLE_MATCH_THRESHOLD and (best is None or rate > best[0]):
                    best = (rate, a, b, c)
    if not best:
        return {}
    _, a, b, c = best
    return {"amount": c, "factor_a": a, "factor_b": b}


# The only tier 1 field names trusted to name a column tier 2's arithmetic
# check has already confirmed is part of the qty/rate/amount relationship -
# a caption match to anything outside this set on one of those columns is
# weaker evidence than the arithmetic (see classify_columns_by_content for
# why: a badly garbled "INVOICE QTY" caption fuzzy-matched "invoice_no" at
# just-over-threshold confidence, on the very column the triple check had
# already tied to amount by a >90%-of-rows multiplication match).
_QTY_RATE_FIELDS = frozenset({"gross_qty", "charge_qty", "rate"})


def _fingerprint_columns(data_rows: list) -> tuple:
    """TIER 2 over every column. Returns ``(types, triple)``:
    ``types`` is ``{column_index: TYPE_* name}`` for every column;
    ``triple`` is what ``_find_qty_rate_amount`` returned, or ``{}``.
    """
    if not data_rows:
        return {}, {}
    num_columns = max(len(row) for row in data_rows)
    columns = {
        index: [str(row[index]).strip() for row in data_rows
                if index < len(row) and str(row[index]).strip()]
        for index in range(num_columns)
    }
    types = {index: _fingerprint_column(values) for index, values in columns.items()}
    return types, _find_qty_rate_amount(_numeric_ish_columns(data_rows))


def classify_columns_by_content(data_rows: list, ocr_captions: list = None) -> dict:
    """Map each column index to a field name, tiers 1 and 2 only.

    Args:
        data_rows: Raw cell text, one list per row, already limited to rows
            that are data (see ``filter_footer_rows``).
        ocr_captions: What the caption row read as, for tier 1. Omitted or
            all-blank falls straight through to tier 2 alone.

    Returns:
        ``{column_index: field_name}`` for every column tier 1 or 2 could
        name (tier 2's own generic type name, such as ``"date_like"``, when
        neither tier reached a specific field; ``"qty_or_rate"`` for an
        arithmetic-triple factor tier 1 could not tell apart from the other
        one). A column neither tier said anything about at all is named
        ``unclassified_col_N``. Tier 3 (a configured client's own column
        list) is not consulted here - see the module docstring for why that
        is column_config.get_columns()'s job, not this function's.
    """
    if not data_rows:
        return {}

    tier1 = _fuzzy_header_match(ocr_captions or [])
    types, triple = _fingerprint_columns(data_rows)

    result = dict(types)
    if triple:
        result[triple["amount"]] = "amount"
        result[triple["factor_a"]] = "qty_or_rate"
        result[triple["factor_b"]] = "qty_or_rate"

    triple_columns = set(triple.values()) if triple else set()
    for index, field in tier1.items():
        # Trust tier 1 everywhere, except on a triple column where it names
        # something outside the quantity/rate/amount family - there, the
        # arithmetic already confirmed what kind of column this is, and a
        # caption fuzzy-matching to an unrelated field is the weaker signal.
        if index in triple_columns and field not in _QTY_RATE_FIELDS | {"amount"}:
            continue
        result[index] = field

    # Process of elimination: multiplication alone cannot say which factor
    # is qty and which is rate, but if tier 1 confidently named one of the
    # two - from its own caption, independent of the other - the remaining
    # factor can only be whichever of gross_qty/rate that leaves open.
    if triple:
        factors = (triple["factor_a"], triple["factor_b"])
        # Only gross_qty/rate, not the broader _QTY_RATE_FIELDS used above -
        # a factor pair has exactly two roles to fill, and charge_qty is a
        # distinct third field on formats that print it separately, not a
        # second name for one of these two.
        two_roles = {"gross_qty", "rate"}
        named = {result[index] for index in factors if result[index] != "qty_or_rate"}
        still_open = [index for index in factors if result[index] == "qty_or_rate"]
        remaining = list(two_roles - named)
        if len(still_open) == 1 and len(remaining) == 1:
            result[still_open[0]] = remaining[0]

    num_columns = max(len(row) for row in data_rows)
    for index in range(num_columns):
        if index not in result:
            result[index] = f"unclassified_col_{index + 1}"

    return result


_ALPHA = re.compile(r"[A-Za-z]")


def make_names_unique(names: list) -> list:
    """Suffix every repeat of a name with ``_2``, ``_3``, ... so each one
    keys a distinct field. Column naming can legitimately produce the same
    name twice - two columns tier 2 could type no further than "free_text",
    say - and a bare dict built straight from those names would silently
    let the second overwrite the first."""
    unique = []
    seen: dict = {}
    for name in names:
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        unique.append(name)
    return unique


def filter_footer_rows(data_rows: list, sr_no_column: int = 0) -> list:
    """Drop rows that are not table data: totals, signatures, page notes.

    A data row is numbered. A row is dropped when its serial cell holds a
    letter (a signature line reading into the serial column) or is empty,
    UNLESS the row's other cells are still mostly filled - the same density
    fallback ``is_valid_data_row`` (both the one in bill_extractor.py and
    tax_invoice_extractor.py's own copy) already applies to the named dict
    built later. Without that fallback here, a noisy scan's OCR misreading a
    genuine digit as a letter in the serial cell ("8" -> "B", "1" -> "l") is
    indistinguishable from an actual signature line and the whole data row
    is lost - confirmed on the tax invoice table's own sample scan, where
    unconditional alpha-rejection here dropped 9 of 28 real rows whose only
    fault was a garbled serial, before this fallback was added.
    """
    kept = []
    for row in data_rows:
        if not row:
            continue
        serial = str(row[sr_no_column]).strip() if sr_no_column < len(row) else ""
        others = [value for i, value in enumerate(row) if i != sr_no_column]
        filled = sum(1 for value in others if str(value).strip())
        dense = bool(others) and filled > len(others) * 0.5
        if _ALPHA.search(serial) and not dense:
            continue
        if not serial and not dense:
            continue
        kept.append(row)
    return kept


# --------------------------------------------------------------------------
# Per-field value validators - a different job from the classification
# above: these check a value already known to belong to a given field
# against that field's real shape, which validation cannot avoid knowing
# (there is no format-free way to ask "is this a well-formed date"). None
# of these identify which *column* holds a field - see classify_columns_
# by_content above for that.
# --------------------------------------------------------------------------


def validate_sr_no(value):
    """``(value, ok)``; ``ok`` False (value kept as None) outside 1-999."""
    text = str(value or "").strip()
    if not text.isdigit():
        return None, False
    number = int(text)
    if not 1 <= number <= 999:
        return None, False
    return text, True


def validate_shipment_no(value):
    """``(value, ok)``; ``ok`` False (value kept as None) unless 7-9 digits."""
    text = str(value or "").strip()
    if not re.fullmatch(r"\d{7,9}", text):
        return None, False
    return text, True


# Generic shape only - letters and digits, no spaces, no country's plate
# format assumed. See ocr/validator.py's own vehicle_no fix for why: the
# Indian-plate-specific pattern this used to be flagged 20/31 rows on the
# sample bill, of which 29/30 non-blank values were a perfectly plausible
# alphanumeric code and only one was genuine OCR garbage.
_ALPHANUMERIC_CODE = re.compile(r"^[A-Za-z0-9]+$")


def validate_vehicle_no(value):
    """``(value, valid, flags)``.

    Unlike the other validators, a value that fails is never turned into
    None - a wrong-but-visible vehicle number is more useful for manual
    review than a blank cell, and this has been the least reliable field to
    OCR. ``flags`` carries ``"vehicle_no_unvalidated"`` only when the value
    is not even a plausible alphanumeric code (spaces, punctuation, a
    genuine OCR garbage character) - not for failing to match any specific
    country's plate format, which nothing here assumes.
    """
    text = re.sub(r"[\s\-]", "", str(value or "")).upper()
    if not text:
        return None, False, []
    if _ALPHANUMERIC_CODE.match(text):
        return text, True, []
    return text, False, ["vehicle_no_unvalidated"]


def validate_amount(value):
    """``(value, ok)`` for a rate/amount cell: digits and one decimal point
    once commas and whitespace are stripped."""
    text = str(value or "").strip()
    if not text:
        return None, False
    cleaned = re.sub(r"[,\s]", "", text)
    if not re.fullmatch(r"\d+(\.\d+)?", cleaned):
        return None, False
    return cleaned, True


def validate_gross_qty(value):
    """``(value, ok)`` for gross_qty: a plain decimal, comma not allowed.

    Unlike amount/rate, gross_qty is always a small number (this bill's
    values run 30-45) that never legitimately carries a thousands-separator
    comma - so a comma here is an OCR misread of the decimal point, not a
    separator. ``ocr.normaliser.normalise_number`` cannot tell the two apart
    (it strips every comma before parsing) and would turn "33,96" into
    3396.0, a hundredfold error that then reaches reconciler.py's tolerance
    check as a wrong number instead of a missing one.
    """
    text = str(value or "").strip()
    if not text:
        return None, False
    return (text, True) if re.fullmatch(r"\d{1,3}\.\d{1,2}", text) else (None, False)


def validate_date(value):
    """``(value, ok)``; ``ok`` False (value kept as None) if it will not
    parse as a day-first date at all - see ``ocr.normaliser.normalise_date``.

    For ``lr_date``, which that format covers.
    """
    text = str(value or "").strip()
    if not text:
        return None, False
    _normalised, ok = normalise_date(text)
    return (text, True) if ok else (None, False)


# The shape delivery_date prints in - a full timestamp ("15 May 26 05:55
# am") - rather than the bare day-first date normalise_date parses.
_DELIVERY_TIMESTAMP_SHAPE = re.compile(
    r"^\d{1,2} \w{3} \d{2} \d{2}:\d{2} (am|pm)$", re.IGNORECASE
)


def validate_delivery_timestamp(value):
    """``(value, ok)`` for ``delivery_date``, checked against its own shape
    rather than run through ``normalise_date``, which does not parse it."""
    text = str(value or "").strip()
    if not text:
        return None, False
    return (text, True) if _DELIVERY_TIMESTAMP_SHAPE.match(text) else (None, False)
