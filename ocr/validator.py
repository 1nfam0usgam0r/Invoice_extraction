"""Cross-check what has already been extracted; change nothing about it.

Everything here runs after bill_extractor.py, lr_extractor.py and
reconciler.py have all already produced their output. It adds
``validation_flags`` and (where available) ``confidence_score`` to a copy of
each bill row, and a batch-level total check - it never edits a value, only
flags one it disagrees with. A flag here catching something Phase 2's own
per-field validation (ocr/column_classifier.py) already caught is not a bug:
this is meant to stand on its own even if that upstream check changes.
"""

import re

try:
    from . import column_config
    from .column_classifier import (
        validate_amount, validate_date, validate_delivery_timestamp,
    )
except ImportError:  # running this file directly from inside ocr/
    import column_config
    from column_classifier import (
        validate_amount, validate_date, validate_delivery_timestamp,
    )

# Each field's own dedicated shape validator (see column_classifier.py) -
# these check a value already known to belong to this field against its
# real shape, a different job from classify_columns_by_content's tiered
# *identification* of which column a field lives in, which no longer uses
# any hardcoded per-field regex at all.
_FIELD_VALIDATORS = {
    "lr_date": validate_date,
    "delivery_date": validate_delivery_timestamp,
    "rate": validate_amount,
    "amount": validate_amount,
}

# vehicle_no is checked separately from _FIELD_VALIDATORS above, against no
# fixed plate format. It used to reuse a hardcoded Indian-plate regex
# (removed from column_classifier.py entirely, not just here - see that
# file's own history) that flagged 20/31 rows on the sample bill, where 29
# of those 30 non-blank values were still a perfectly plausible alphanumeric
# code (one genuine OCR garbage character was the other). The only thing to
# check a value against without assuming a specific country's plate format
# is its own generic shape - letters and digits, no spaces - plus whatever
# length/shape a client's own config specifies, if any. There is no such
# per-client format table anywhere in this codebase yet
# (column_config.CLIENT_COLUMNS holds column names, not field formats); this
# looks for one under column_config.VEHICLE_NO_FORMATS so a caller can add
# one later without this changing, but finds nothing today - so nothing
# beyond the generic shape check ever flags vehicle_no as things stand.
_ALPHANUMERIC_CODE = re.compile(r"^[A-Za-z0-9]+$")


def _validate_vehicle_no(value, client_id: str = "") -> bool:
    """``True`` if ``value`` should be trusted; ``False`` to flag it.

    A value that fails even the generic alphanumeric-code shape (spaces,
    punctuation, an OCR garbage character) is flagged. One that fits it is
    trusted unless the client this row belongs to has its own format
    configured - there being no assumed format is not itself an error.
    """
    text = str(value).strip()
    if not _ALPHANUMERIC_CODE.match(text):
        return False
    client_format = getattr(column_config, "VEHICLE_NO_FORMATS", {}).get(client_id)
    if client_format is not None:
        return bool(re.match(client_format, text))
    return True

# A row's computed amount (gross_qty * rate) may differ from its printed
# amount by up to this fraction before it is flagged - small rounding in how
# a rate is quoted (per-tonne to two vs. three decimals) is not an error.
AMOUNT_MISMATCH_TOLERANCE = 0.02

# India's smallest currency unit, for the batch total's accumulated rounding
# tolerance - every row can be off by up to half a paisa in either direction
# from its own rounding, so the tolerance grows with row count.
CURRENCY_SMALLEST_UNIT = 0.01

_STRIP_TO_NUMBER = re.compile(r"[^0-9.]")


def _to_float(value):
    """Best-effort float out of a raw OCR'd number; ``None`` if it will not
    parse. Strips everything but digits and the decimal point, so
    "23,30,645/-" and "69,448.20" both come through."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = _STRIP_TO_NUMBER.sub("", str(value))
    if not text or text == ".":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def validate_bill_row(row: dict, client_id: str = "") -> dict:
    """A copy of ``row`` with ``validation_flags`` (and, where available,
    ``confidence_score``) added. Every existing key and value is untouched.

    Checks:
        - Arithmetic: ``gross_qty * rate`` against the printed ``amount``,
          flagged ``amount_mismatch`` if they disagree by more than
          ``AMOUNT_MISMATCH_TOLERANCE``.
        - Shape re-checks of ``lr_date``, ``delivery_date``, ``rate`` and
          ``amount`` against each field's own dedicated validator in
          ``column_classifier.py`` - independent of whether
          bill_extractor.py's own per-field validation ran or agreed.
          ``vehicle_no`` is checked separately, against no fixed plate
          format - see ``_validate_vehicle_no``.
        - ``confidence_score``: bill_extractor.py does not currently keep a
          per-cell Tesseract confidence anywhere (its OCR calls are
          ``image_to_string``, not ``image_to_data`` - there is nothing to
          pull), so this is omitted rather than fabricated. A caller that
          starts capturing per-cell confidence can add it without this
          function changing.
    """
    result = dict(row)
    # Start from whatever bill_extractor.py's own per-field validation
    # (ocr/column_classifier.py's _SIMPLE_VALIDATORS, run inside
    # extract_bill's _validate_row) already flagged, carried on the row as
    # "flags" - that stage nulls a field the moment it fails its shape
    # check, so by the time this function runs the value is already gone
    # and the field-validator loop below (which skips a None value outright)
    # can never independently re-derive that same flag. Without folding
    # "flags" in here, a row Phase 1 already caught reaches
    # ocr/excel_writer.py's needs_review check - which reads only
    # validation_flags - looking perfectly clean, and its whole-row yellow
    # highlight never fires despite the row genuinely needing review.
    flags = list(row.get("flags") or [])

    gross_qty = _to_float(row.get("gross_qty"))
    rate = _to_float(row.get("rate"))
    amount = _to_float(row.get("amount"))
    if gross_qty is not None and rate is not None and amount is not None:
        expected = gross_qty * rate
        if amount == 0:
            if expected != 0:
                flags.append("amount_mismatch")
        elif abs(expected - amount) > abs(amount) * AMOUNT_MISMATCH_TOLERANCE:
            flags.append("amount_mismatch")

    for field, validator in _FIELD_VALIDATORS.items():
        value = row.get(field)
        if value is None:
            continue
        _cleaned, ok = validator(value)
        if not ok:
            flags.append(f"{field}_invalid")

    vehicle_no = row.get("vehicle_no")
    if vehicle_no is not None and not _validate_vehicle_no(vehicle_no, client_id):
        flags.append("vehicle_no_invalid")

    result["validation_flags"] = flags
    return result


def validate_batch_total(bill_rows: list, stated_invoice_total) -> dict:
    """Sum every row's ``amount`` and compare it to the invoice's own total.

    Args:
        bill_rows: The bill rows (validated or not; only ``amount`` is read).
        stated_invoice_total: The total off the cover page - a raw OCR'd
            string ("23,30,645/-") is accepted directly.

    Returns:
        ``summed_amount``, ``stated_total`` (both ``None`` if they could not
        be parsed at all), ``discrepancy`` (``summed - stated``, ``None`` if
        either side is), ``tolerance`` and ``batch_flag`` - ``True`` when the
        discrepancy exceeds the accumulated rounding tolerance
        (``CURRENCY_SMALLEST_UNIT`` per row), or when the total could not be
        checked at all (nothing to compare against is itself worth flagging,
        not silently passing).
    """
    amounts = [_to_float(row.get("amount")) for row in (bill_rows or [])]
    parsed_amounts = [value for value in amounts if value is not None]
    summed = round(sum(parsed_amounts), 2) if parsed_amounts else None
    stated = _to_float(stated_invoice_total)
    tolerance = round(CURRENCY_SMALLEST_UNIT * len(bill_rows or []), 2)

    if summed is None or stated is None:
        return {
            "summed_amount": summed, "stated_total": stated,
            "discrepancy": None, "tolerance": tolerance, "batch_flag": True,
        }

    discrepancy = round(summed - stated, 2)
    return {
        "summed_amount": summed, "stated_total": stated,
        "discrepancy": discrepancy, "tolerance": tolerance,
        "batch_flag": abs(discrepancy) > tolerance,
    }
