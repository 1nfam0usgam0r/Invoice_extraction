"""Column names for bill tables, per client, with an OCR fallback.

A bill's caption row is set in small bold type and is the first thing a
photocopier destroys - on the Inland scans the left-hand captions come back as
speckle no amount of magnification recovers, while the data rows under them
read fine. The column order, though, is a property of the client's bill format
and does not change from invoice to invoice, so it is recorded here once per
client rather than guessed at every run.

Onboarding a new client means adding one entry. Until then the reader falls
back to whatever the captions OCR'd to, and then to position, so an unknown
format still produces a usable sheet.
"""

import re

from .column_classifier import make_names_unique

# Client identifier -> the columns their bill prints, left to right.
CLIENT_COLUMNS = {
    # Read off the caption row of the Inland bill at magnification: S-NO,
    # BILL NO, INVOICE NO, DATE, LR NO, VEHICLE NO, DELIVERY STATION,
    # INVOICE QTY, RATE, AMOUNT, SHORT qty CHARGES, LD CHARGES, BALANCEPAY,
    # DELIVERY DATE, PARTY NAME.
    #
    # "DATE" and "INVOICE QTY" are named lr_date/gross_qty rather than after
    # their own captions: ocr/normaliser.py's normalise_bill_row() reads a
    # bill row by those exact keys (see its BILL_NUMERIC_FIELDS and the
    # lr_date it pulls for reconciler.py's date match), and this column's
    # values are the LR's own date and quantity, not a second, bill-specific
    # one - confirmed against the LR receipts themselves (13.05.2026 /
    # 41.080 on shipment 14479935 matches this column exactly).
    "inland_world_logistics": [
        "sr_no", "bill_no", "invoice_no", "lr_date",
        "lr_no", "vehicle_no", "delivery_station",
        "gross_qty", "rate", "amount", "short_qty_charges",
        "ld_charges", "balance_pay", "delivery_date",
        "party_name",
    ],
}

# Words too short or too junk-ridden to be part of a caption. A caption token
# has to be mostly letters and digits; OCR speckle is mostly neither.
_MIN_TOKEN_ALNUM = 2
_MIN_TOKEN_RATIO = 0.6

# A cleaned caption longer than this is noise that happened to contain letters.
MAX_CAPTION_LENGTH = 30

_CLEAN_KEY = re.compile(r"[^a-z0-9]+")
_SQUEEZE = re.compile(r"[^a-z0-9]+")


def clean_caption(text) -> str:
    """Turn one OCR'd caption into a column key, or "" if it is unusable.

    Tokens that are mostly punctuation are dropped first - a caption read as
    ``'. fe) wee . AMOUNT , 4 : OTe'`` should not carry its speckle into the
    key. What survives is lowercased and joined with underscores.
    """
    tokens = []
    for token in str(text or "").split():
        alnum = sum(1 for char in token if char.isalnum())
        if alnum < _MIN_TOKEN_ALNUM or alnum < len(token) * _MIN_TOKEN_RATIO:
            continue
        tokens.append(token)

    name = _CLEAN_KEY.sub("_", " ".join(tokens).lower()).strip("_")
    if sum(1 for char in name if char.isalpha()) < 2:
        return ""
    return name[:MAX_CAPTION_LENGTH].strip("_")


def detect_client(text) -> str:
    """Match page text against the configured clients; "" when none fit.

    Compared with the spacing squeezed out, because OCR is unreliable about
    exactly where the spaces in a company name fall.
    """
    squeezed = _SQUEEZE.sub("", str(text or "").lower())
    if not squeezed:
        return ""

    for client_id in CLIENT_COLUMNS:
        if _SQUEEZE.sub("", client_id) in squeezed:
            return client_id
    return ""


def get_columns(client_id, num_columns: int, ocr_captions: list,
                 data_rows: list = None) -> list:
    """Column names for this table.

    Args:
        client_id: A key of ``CLIENT_COLUMNS``, or anything else to fall back.
        num_columns: How many columns the grid actually found.
        ocr_captions: What the caption row read as. Used as a last-resort
            fallback for a column ``data_rows`` could not classify, since the
            caption row is at least sometimes readable even where the data
            pattern is ambiguous.
        data_rows: Raw cell text from the data rows under the caption, one
            list per row, for an unconfigured client. What a column *holds* -
            a date shape, a plate shape - survives the scan far better than
            the small bold caption above it, so this is tried before falling
            back to the caption text itself. See ``column_classifier.py``.

    Returns:
        Exactly ``num_columns`` names: the client's own where configured;
        otherwise the data-classified name per column, the cleaned caption
        for whatever that left unclassified, and ``col_N`` for anything
        both leave unclassified. Names are made unique, so every column
        keys a distinct field.
    """
    if client_id in CLIENT_COLUMNS:
        names = list(CLIENT_COLUMNS[client_id])[:num_columns]
    elif data_rows:
        try:
            from .column_classifier import (
                GENERIC_COLUMN_TYPES, classify_columns_by_content, filter_footer_rows,
            )
        except ImportError:  # running this file directly from inside ocr/
            from column_classifier import (
                GENERIC_COLUMN_TYPES, classify_columns_by_content, filter_footer_rows,
            )

        classified = classify_columns_by_content(filter_footer_rows(data_rows), ocr_captions)
        captions_cleaned = [clean_caption(caption) for caption in ocr_captions]
        # A generic tier-2 type name ("date_like", "free_text", ...) is
        # still not a specific field - the caption is worth one more try
        # before falling all the way to col_N, same as an outright miss.
        generic_names = set(GENERIC_COLUMN_TYPES) | {"qty_or_rate"}
        names = []
        for index in range(num_columns):
            name = classified.get(index, "")
            not_specific = name.startswith("unclassified_col_") or name in generic_names
            if not_specific and index < len(captions_cleaned):
                name = captions_cleaned[index] or name
            names.append(name)
    else:
        names = [clean_caption(caption) for caption in ocr_captions][:num_columns]

    # Pad to the column count, and name anything unreadable by position, so
    # the columns either side still line up.
    names += [""] * (num_columns - len(names))
    names = [name or f"col_{index + 1}" for index, name in enumerate(names)]

    return make_names_unique(names)
