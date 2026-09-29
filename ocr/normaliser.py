"""Normalise raw bill rows and LR records so they can be compared.

Every function returns a new dict with ``_norm`` / ``_num`` keys added; the
raw OCR values are never touched.
"""

import re
from datetime import datetime

# Separators that appear inside LR and vehicle numbers.
_SEPARATORS = re.compile(r"[^A-Za-z0-9]+")

# Position-aware OCR correction for Indian vehicle plates (SS DD LLL NNNN).
# At letter positions digits are misread as visually similar letters, and vice
# versa at digit positions. Only the characters that commonly flip are mapped.
_DIG_TO_LET = str.maketrans("01568", "OISGB")   # used at state-code positions (0-1)
_LET_TO_DIG = str.maketrans("OISGBZ", "015682")  # used at district (2-3) and serial (last 4)

# Currency markers, usually a prefix.
_CURRENCY_PREFIX = re.compile(r"^(?:RS|INR|₹|\$)\.?")

# Unit suffixes, longest alternative first so "KGS" is not left as a bare "S".
_UNIT_SUFFIX = re.compile(r"(?:TONNES|TONNE|TONS|TON|KGS|KG|MTS|MT|RS|INR)\.?$")

# Day-first formats, plus the ISO form so re-normalising is a no-op.
_DATE_FORMATS = [
    "%d.%m.%Y",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%y",
    "%d/%m/%y",
    "%d-%m-%y",
    "%Y-%m-%d",
]

# Bill field -> normalised key.
BILL_NUMERIC_FIELDS = {
    "gross_qty": "gross_qty_num",
    "charge_qty": "charge_qty_num",
    "rate": "rate_num",
    "amount": "amount_num",
}

# LR field -> normalised key.
LR_NUMERIC_FIELDS = {
    "net_wt": "net_wt_num",
    "gross_wt": "gross_wt_num",
    "lorry_tare_wt": "lorry_tare_wt_num",
    "amount": "amount_num",
}

# LR fields cleaned of leading punctuation before anything else runs. These
# are read from beside a printed label, so a colon or dash the OCR attached
# to the value rather than the label is the common failure.
LR_TEXT_FIELDS = (
    "lr_no", "shipment_no", "lr_date", "truck_no",
    "from_origin", "to_destination", "net_wt", "gross_wt",
    "lorry_tare_wt", "incoterm", "truck_type",
)

# Junk Tesseract leaves on the front of a value: label colons, bullet dashes,
# stray quotes.
_LEADING_JUNK = ":.-_>*'\"` "

# A date at the start of a value, ignoring whatever noise trails it.
_LEADING_DATE = re.compile(r"\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}")


def clean_extracted_text(text: str) -> str:
    """Strip leading punctuation the OCR carried over from a field label.

    ``": 848501350"`` -> ``"848501350"``. Anything that is not a string -
    including the None the LR extractor writes for a rejected field - is
    returned untouched.
    """
    if not text or not isinstance(text, str):
        return text

    text = text.strip()
    text = text.lstrip(_LEADING_JUNK)
    return text.strip()


def normalise_lr_number(value) -> str:
    """Reduce an LR number to comparable digits.

    Drops separators, upper-cases, then strips leading alphabetic prefixes
    and leading zeros::

        "L848501350"  -> "848501350"
        "0000001350"  -> "1350"
        "LR-8485/013" -> "8485013"
    """
    text = re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
    if not text:
        return ""

    text = text.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    stripped = text.lstrip("0")

    # An all-zero number is still a zero, not an empty string.
    if not stripped and text:
        return "0"
    return stripped


def lr_number_digits(normalised: str, length: int = 7) -> str:
    """Last ``length`` digits of a normalised LR number, for suffix matching."""
    digits = re.sub(r"\D", "", normalised or "")
    return digits[-length:]


def normalise_date(value) -> tuple:
    """Parse a day-first date into ``YYYY-MM-DD``.

    Returns ``(normalised, ok)``. On failure the original string comes back
    with ``ok`` False. A blank input returns ``("", True)`` — there was
    nothing to parse, which is not a parse failure.
    """
    text = str(value or "").strip()
    if not text:
        return "", True

    # Collapse "05 . 09 . 2026" and similar OCR spacing.
    candidate = re.sub(r"\s+", "", text)

    # Drop trailing OCR noise: "13.05.2026 qP' l?" -> "13.05.2026".
    leading_date = _LEADING_DATE.match(candidate)
    if leading_date:
        candidate = leading_date.group(0)

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(candidate, fmt).strftime("%Y-%m-%d"), True
        except ValueError:
            continue

    return text, False


def normalise_number(value) -> tuple:
    """Parse a quantity or amount into a rounded float.

    Strips commas, spaces, currency prefixes and unit suffixes (MT, KG, TON,
    RS, INR). Returns ``(number, ok)``; on failure ``(None, False)``. A blank
    input returns ``(None, True)`` — nothing was there to parse.
    """
    if value is None:
        return None, True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), 3), True

    text = str(value).strip()
    if not text:
        return None, True

    cleaned = re.sub(r"[,\s]+", "", text).upper()
    cleaned = _CURRENCY_PREFIX.sub("", cleaned)
    cleaned = _UNIT_SUFFIX.sub("", cleaned)

    try:
        return round(float(cleaned), 3), True
    except ValueError:
        return None, False


def normalise_vehicle_no(value) -> str:
    """Upper-case a vehicle number, drop separators, and correct common OCR
    character flips at known positions of the Indian plate format.

    Indian plates follow SS DD [L..L] NNNN:
      - positions 0-1: state code — always letters; digits corrected to letters
      - positions 2-3: district code — always digits; letters corrected to digits
      - last 4:        serial number — always digits; letters corrected to digits
      - middle:        series letters — left untouched (variable length, 1-3 chars)

    Only characters that are visually ambiguous are substituted (0/O, 1/I,
    5/S, 6/G, 8/B, 2/Z) to avoid corrupting genuinely correct reads.
    """
    text = _SEPARATORS.sub("", str(value or "")).upper()
    n = len(text)
    if n < 6:
        return text

    chars = list(text)

    # State code (positions 0-1): must be letters.
    for i in range(min(2, n)):
        chars[i] = chars[i].translate(_DIG_TO_LET)

    # District (positions 2-3): must be digits.
    for i in range(2, min(4, n)):
        chars[i] = chars[i].translate(_LET_TO_DIG)

    # Serial number (last 4 positions): must be digits.
    for i in range(max(4, n - 4), n):
        chars[i] = chars[i].translate(_LET_TO_DIG)

    return "".join(chars)


def _normalise(source: dict, numeric_fields: dict, vehicle_field: str, vehicle_key: str) -> dict:
    """Shared body for the bill-row and LR-record normalisers."""
    result = dict(source)
    warnings = []

    lr_norm = normalise_lr_number(source.get("lr_no", ""))
    result["lr_no_norm"] = lr_norm
    result["lr_no_digits"] = lr_number_digits(lr_norm)
    if str(source.get("lr_no", "")).strip() and not lr_norm:
        warnings.append("lr_no")

    date_norm, date_ok = normalise_date(source.get("lr_date", ""))
    result["lr_date_norm"] = date_norm
    if not date_ok:
        warnings.append("lr_date")

    for field, key in numeric_fields.items():
        number, ok = normalise_number(source.get(field, ""))
        result[key] = number
        if not ok:
            warnings.append(field)

    result[vehicle_key] = normalise_vehicle_no(source.get(vehicle_field, ""))

    result["parse_warnings"] = warnings
    return result


def normalise_bill_row(row: dict) -> dict:
    """Return a copy of a bill row with normalised keys added.

    Adds ``lr_no_norm``, ``lr_no_digits``, ``lr_date_norm``,
    ``gross_qty_num``, ``charge_qty_num``, ``rate_num``, ``amount_num``,
    ``vehicle_no_norm`` and ``parse_warnings``.
    """
    return _normalise(row, BILL_NUMERIC_FIELDS, "vehicle_no", "vehicle_no_norm")


def normalise_lr_record(record: dict) -> dict:
    """Return a copy of an LR record with normalised keys added.

    Adds ``lr_no_norm``, ``lr_no_digits``, ``lr_date_norm``, ``net_wt_num``,
    ``gross_wt_num``, ``lorry_tare_wt_num``, ``amount_num``, ``truck_no_norm``
    and ``parse_warnings``.

    The fields in ``LR_TEXT_FIELDS`` are cleaned of leading punctuation
    first, so the cleaned value is what both the normalisers and the
    workbook see.
    """
    cleaned = dict(record)
    for field in LR_TEXT_FIELDS:
        if field in cleaned:
            cleaned[field] = clean_extracted_text(cleaned[field])

    return _normalise(cleaned, LR_NUMERIC_FIELDS, "truck_no", "truck_no_norm")


if __name__ == '__main__':
    failures = []

    def check(label, actual, expected):
        ok = actual == expected
        if not ok:
            failures.append(label)
        print(f"[{'ok  ' if ok else 'FAIL'}] {label}: {actual!r}" + ("" if ok else f" != {expected!r}"))

    print("--- LR number ---")
    check("alpha prefix", normalise_lr_number("L848501350"), "848501350")
    check("leading zeros", normalise_lr_number("0000001350"), "1350")
    check("dots and dashes", normalise_lr_number("LR-8485.0135"), "84850135")
    check("spaces", normalise_lr_number(" lr 848 501 350 "), "848501350")
    check("alpha then zeros", normalise_lr_number("LR0000123"), "123")
    check("all zeros keeps one", normalise_lr_number("0000"), "0")
    check("letters only", normalise_lr_number("LR"), "")
    check("empty", normalise_lr_number(""), "")
    check("none", normalise_lr_number(None), "")
    check("trailing letter kept", normalise_lr_number("L848501350A"), "848501350A")

    print("\n--- LR digits (suffix match) ---")
    check("last 7 of 9", lr_number_digits("848501350"), "8501350")
    check("shorter than 7", lr_number_digits("1350"), "1350")
    check("embedded letters", lr_number_digits("848501350A"), "8501350")
    check("empty", lr_number_digits(""), "")

    print("\n--- dates ---")
    check("dotted", normalise_date("05.09.2026"), ("2026-09-05", True))
    check("slashed", normalise_date("05/09/2026"), ("2026-09-05", True))
    check("dashed", normalise_date("05-09-2026"), ("2026-09-05", True))
    check("unpadded", normalise_date("5/9/2026"), ("2026-09-05", True))
    check("two-digit year", normalise_date("05/09/26"), ("2026-09-05", True))
    check("iso passthrough", normalise_date("2026-09-05"), ("2026-09-05", True))
    check("ocr spacing", normalise_date("05 . 09 . 2026"), ("2026-09-05", True))
    check("day-first not month-first", normalise_date("03/09/2026"), ("2026-09-03", True))
    check("impossible date", normalise_date("32/13/2026"), ("32/13/2026", False))
    check("garbage", normalise_date("N/A"), ("N/A", False))
    check("blank is not a failure", normalise_date(""), ("", True))

    print("\n--- numbers ---")
    check("comma thousands", normalise_number("1,200"), (1200.0, True))
    check("decimal", normalise_number("10,625.50"), (10625.5, True))
    check("MT suffix", normalise_number("12.5 MT"), (12.5, True))
    check("KG suffix", normalise_number("8500KG"), (8500.0, True))
    check("TON suffix", normalise_number("3 TON"), (3.0, True))
    check("RS prefix", normalise_number("RS. 98,750.00"), (98750.0, True))
    check("INR prefix", normalise_number("INR 1200"), (1200.0, True))
    check("rupee symbol", normalise_number("₹1,500.25"), (1500.25, True))
    check("rounds to 3dp", normalise_number("12.34567"), (12.346, True))
    check("negative", normalise_number("-45.5"), (-45.5, True))
    check("already numeric", normalise_number(1200), (1200.0, True))
    check("garbage", normalise_number("ABC"), (None, False))
    check("unit only", normalise_number("MT"), (None, False))
    check("blank is not a failure", normalise_number(""), (None, True))

    print("\n--- vehicle number ---")
    check("spaces", normalise_vehicle_no("MH 12 AB 1234"), "MH12AB1234")
    check("hyphens", normalise_vehicle_no("mh-12-ab-1234"), "MH12AB1234")
    check("dots", normalise_vehicle_no("MH.12.AB.1234"), "MH12AB1234")
    check("empty", normalise_vehicle_no(""), "")

    print("\n--- bill row ---")
    raw_row = {
        "lr_no": "L848501350",
        "lr_date": "05.09.2026",
        "gross_qty": "1,200 MT",
        "charge_qty": "1250",
        "rate": "8.50",
        "amount": "RS. 10,625.00",
        "vehicle_no": "MH 12 AB 1234",
        "description": "Steel Coils",
    }
    row = normalise_bill_row(raw_row)
    check("lr_no_norm", row["lr_no_norm"], "848501350")
    check("lr_no_digits", row["lr_no_digits"], "8501350")
    check("lr_date_norm", row["lr_date_norm"], "2026-09-05")
    check("gross_qty_num", row["gross_qty_num"], 1200.0)
    check("charge_qty_num", row["charge_qty_num"], 1250.0)
    check("rate_num", row["rate_num"], 8.5)
    check("amount_num", row["amount_num"], 10625.0)
    check("vehicle_no_norm", row["vehicle_no_norm"], "MH12AB1234")
    check("no warnings", row["parse_warnings"], [])
    check("raw untouched", raw_row["lr_no"], "L848501350")
    check("original has no norm keys", "lr_no_norm" in raw_row, False)
    check("other keys survive", row["description"], "Steel Coils")

    print("\n--- bill row with bad data ---")
    bad_row = {
        "lr_no": "0000001350",
        "lr_date": "31/02/2026",
        "gross_qty": "eight hundred",
        "charge_qty": "",
        "rate": "12.00",
        "amount": "9,600.00",
        "vehicle_no": "gj-05-cd-9876",
    }
    bad = normalise_bill_row(bad_row)
    check("leading zeros stripped", bad["lr_no_norm"], "1350")
    check("bad date kept raw", bad["lr_date_norm"], "31/02/2026")
    check("bad number is None", bad["gross_qty_num"], None)
    check("blank number is None", bad["charge_qty_num"], None)
    check("warnings list", bad["parse_warnings"], ["lr_date", "gross_qty"])

    print("\n--- LR record ---")
    raw_lr = {
        "lr_no": "LR 0848501350",
        "lr_date": "5-9-2026",
        "net_wt": "12,000 KG",
        "gross_wt": "20500",
        "lorry_tare_wt": "8,500 KG",
        "amount": "INR 98,750.00",
        "truck_no": "MH-12-AB-1234",
        "remark": "received ok",
    }
    lr = normalise_lr_record(raw_lr)
    check("lr_no_norm", lr["lr_no_norm"], "848501350")
    check("lr_no_digits", lr["lr_no_digits"], "8501350")
    check("lr_date_norm", lr["lr_date_norm"], "2026-09-05")
    check("net_wt_num", lr["net_wt_num"], 12000.0)
    check("gross_wt_num", lr["gross_wt_num"], 20500.0)
    check("lorry_tare_wt_num", lr["lorry_tare_wt_num"], 8500.0)
    check("amount_num", lr["amount_num"], 98750.0)
    check("truck_no_norm", lr["truck_no_norm"], "MH12AB1234")
    check("no warnings", lr["parse_warnings"], [])
    check("raw untouched", raw_lr["truck_no"], "MH-12-AB-1234")
    check("other keys survive", lr["remark"], "received ok")

    print("\n--- suffix match across bill and LR ---")
    check(
        "differing prefixes agree on digits",
        normalise_bill_row({"lr_no": "L848501350"})["lr_no_digits"]
        == normalise_lr_record({"lr_no": "0000008501350"})["lr_no_digits"],
        True,
    )

    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
    else:
        print("all checks passed")
