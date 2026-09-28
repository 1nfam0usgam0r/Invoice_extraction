"""Reconcile normalised bill rows against normalised LR records.

Category A: bill row matched to an LR record.
Category B: bill row with no LR record  -> missing_lr
Category C: LR record never billed      -> unbilled
"""

NET_WT_TOLERANCE = 0.01
AMOUNT_TOLERANCE = 1.0

MATCH_FIELDS = ["shipment_no", "vehicle_no", "lr_date", "net_wt", "amount"]
DIFF_FIELDS = ["net_wt_diff", "amount_diff"]


def _empty_matches() -> dict:
    return {field: None for field in MATCH_FIELDS}


def _empty_diffs() -> dict:
    return {field: None for field in DIFF_FIELDS}


def _compare_text(bill_value, lr_value):
    """Exact stripped comparison; None when either side is blank."""
    left = str(bill_value or "").strip()
    right = str(lr_value or "").strip()
    if not left or not right:
        return None
    return left == right


def _compare_number(bill_value, lr_value, tolerance):
    """Return ``(within_tolerance, signed_diff)``; ``(None, None)`` if unparsed.

    The epsilon absorbs binary float noise only: 100.01 - 100.00 evaluates to
    0.010000000000005, which would otherwise fail a 0.01 tolerance.
    """
    if bill_value is None or lr_value is None:
        return None, None
    diff = float(bill_value) - float(lr_value)
    return abs(diff) <= tolerance + 1e-9, round(diff, 3)


def _build_lookups(lr_records: list) -> tuple:
    """Index LR records by ``lr_no_norm`` and by ``lr_no_digits``.

    Values are lists of indices, not single records, so that duplicate LR
    numbers are detected instead of silently overwriting each other.
    """
    primary: dict = {}
    fallback: dict = {}

    for index, record in enumerate(lr_records):
        norm = str(record.get("lr_no_norm") or "").strip()
        digits = str(record.get("lr_no_digits") or "").strip()
        if norm:
            primary.setdefault(norm, []).append(index)
        if digits:
            fallback.setdefault(digits, []).append(index)

    return primary, fallback


def _run_diff_checks(bill_row: dict, lr_record: dict) -> tuple:
    """Compare a matched pair. Returns ``(matches, diffs)``."""
    matches = _empty_matches()
    diffs = _empty_diffs()

    matches["shipment_no"] = _compare_text(
        bill_row.get("shipment_no"), lr_record.get("shipment_no")
    )
    # The LR normaliser emits truck_no_norm for the same physical field.
    matches["vehicle_no"] = _compare_text(
        bill_row.get("vehicle_no_norm"), lr_record.get("truck_no_norm")
    )
    matches["lr_date"] = _compare_text(
        bill_row.get("lr_date_norm"), lr_record.get("lr_date_norm")
    )

    matches["net_wt"], diffs["net_wt_diff"] = _compare_number(
        bill_row.get("gross_qty_num"), lr_record.get("net_wt_num"), NET_WT_TOLERANCE
    )
    matches["amount"], diffs["amount_diff"] = _compare_number(
        bill_row.get("amount_num"), lr_record.get("amount_num"), AMOUNT_TOLERANCE
    )

    return matches, diffs


def _display_lr_no(bill_row, lr_record) -> str:
    """Prefer the raw LR number off the bill, else off the LR."""
    for source in (bill_row, lr_record):
        if not source:
            continue
        for key in ("lr_no", "lr_no_norm"):
            value = str(source.get(key) or "").strip()
            if value:
                return value
    return ""


def _parse_warning_flags(bill_row, lr_record) -> list:
    """Surface upstream parse failures so they are visible on the report."""
    flags = []
    if bill_row and bill_row.get("parse_warnings"):
        flags.append("bill_parse_warnings")
    if lr_record and lr_record.get("parse_warnings"):
        flags.append("lr_parse_warnings")
    return flags


def reconcile(bill_rows: list, lr_records: list) -> list:
    """Match bill rows to LR records and report the differences.

    Primary match is an exact ``lr_no_norm`` equality; the fallback is an
    ``lr_no_digits`` (last 7 digits) match, which adds the
    ``lr_no_fuzzy_matched`` flag.
    """
    primary, fallback = _build_lookups(lr_records)
    matched_indices: dict = {}
    results = []

    for bill_row in bill_rows:
        flags = []
        norm = str(bill_row.get("lr_no_norm") or "").strip()
        digits = str(bill_row.get("lr_no_digits") or "").strip()

        # a/b. Primary on lr_no_norm, then fallback on the 7-digit suffix.
        candidates = primary.get(norm) if norm else None
        if not candidates and digits:
            candidates = fallback.get(digits)
            if candidates:
                flags.append("lr_no_fuzzy_matched")

        if not norm and not digits:
            flags.append("bill_lr_no_missing")

        # c. No LR on file for this bill row.
        if not candidates:
            flags.extend(_parse_warning_flags(bill_row, None))
            results.append(
                {
                    "lr_no": _display_lr_no(bill_row, None),
                    "category": "B",
                    "bill_row": bill_row,
                    "lr_record": None,
                    "matches": _empty_matches(),
                    "diffs": _empty_diffs(),
                    "flags": flags,
                    "status": "missing_lr",
                }
            )
            continue

        if len(candidates) > 1:
            flags.append("ambiguous_lr_match")

        index = candidates[0]
        lr_record = lr_records[index]

        # The same LR answering two bill rows is a possible double-billing.
        matched_indices[index] = matched_indices.get(index, 0) + 1
        if matched_indices[index] > 1:
            flags.append("duplicate_bill_for_lr")

        # d. Category A: compare the pair field by field.
        matches, diffs = _run_diff_checks(bill_row, lr_record)
        if any(value is None for value in matches.values()):
            flags.append("unverified_fields")
        flags.extend(_parse_warning_flags(bill_row, lr_record))

        status = "mismatch" if any(value is False for value in matches.values()) else "clear"
        results.append(
            {
                "lr_no": _display_lr_no(bill_row, lr_record),
                "category": "A",
                "bill_row": bill_row,
                "lr_record": lr_record,
                "matches": matches,
                "diffs": diffs,
                "flags": flags,
                "status": status,
            }
        )

    # 4. LR records nothing billed against.
    for index, lr_record in enumerate(lr_records):
        if index in matched_indices:
            continue

        flags = []
        if not str(lr_record.get("lr_no_norm") or "").strip():
            flags.append("lr_lr_no_missing")
        flags.extend(_parse_warning_flags(None, lr_record))

        results.append(
            {
                "lr_no": _display_lr_no(None, lr_record),
                "category": "C",
                "bill_row": None,
                "lr_record": lr_record,
                "matches": _empty_matches(),
                "diffs": _empty_diffs(),
                "flags": flags,
                "status": "unbilled",
            }
        )

    return results


if __name__ == '__main__':
    try:
        from .normaliser import normalise_bill_row, normalise_lr_record
    except ImportError:
        from normaliser import normalise_bill_row, normalise_lr_record

    failures = []

    def check(label, actual, expected):
        ok = actual == expected
        if not ok:
            failures.append(label)
        print(f"[{'ok  ' if ok else 'FAIL'}] {label}: {actual!r}" + ("" if ok else f" != {expected!r}"))

    def bill(**kw):
        row = {
            "lr_no": "", "shipment_no": "", "lr_date": "", "vehicle_no": "",
            "gross_qty": "", "charge_qty": "", "rate": "", "amount": "",
        }
        row.update(kw)
        return normalise_bill_row(row)

    def lr(**kw):
        record = {
            "lr_no": "", "shipment_no": "", "lr_date": "", "truck_no": "",
            "net_wt": "", "gross_wt": "", "lorry_tare_wt": "", "amount": "",
        }
        record.update(kw)
        return normalise_lr_record(record)

    bill_rows = [
        # 0: exact match, everything agrees -> clear
        bill(lr_no="L848501350", shipment_no="SHP-001", lr_date="05.09.2026",
             vehicle_no="MH 12 AB 1234", gross_qty="12,000 KG", amount="RS. 98,750.00"),
        # 1: norms differ but last 7 digits agree -> fallback match, weight and amount off
        bill(lr_no="L99848501351", shipment_no="SHP-002", lr_date="06/09/2026",
             vehicle_no="GJ-05-CD-9876", gross_qty="8000", amount="50,000.00"),
        # 2: no LR record at all -> Category B
        bill(lr_no="L999999999", shipment_no="SHP-003", lr_date="07/09/2026",
             vehicle_no="RJ14EF5555", gross_qty="500", amount="1000"),
        # 3: both differences sit inside tolerance -> clear
        bill(lr_no="848501352", shipment_no="SHP-004", lr_date="08/09/2026",
             vehicle_no="KA01GH7777", gross_qty="1000.005", amount="2000.50"),
        # 4: LR side unparseable amount -> comparison not possible
        bill(lr_no="848501353", shipment_no="SHP-005", lr_date="09/09/2026",
             vehicle_no="TN09IJ3333", gross_qty="700", amount="3000"),
        # 5: bill row with no LR number at all
        bill(shipment_no="SHP-006", lr_date="10/09/2026", vehicle_no="AP11KL2222",
             gross_qty="100", amount="500"),
    ]

    lr_records = [
        lr(lr_no="0000848501350", shipment_no="SHP-001", lr_date="05/09/2026",
           truck_no="MH-12-AB-1234", net_wt="12000", amount="98750"),
        lr(lr_no="8501351", shipment_no="SHP-002", lr_date="06/09/2026",
           truck_no="GJ05CD9876", net_wt="8500", amount="52,000.00"),
        lr(lr_no="848501352", shipment_no="SHP-004", lr_date="08/09/2026",
           truck_no="KA 01 GH 7777", net_wt="1000.009", amount="2001.20"),
        lr(lr_no="848501353", shipment_no="SHP-005", lr_date="09/09/2026",
           truck_no="TN09IJ3333", net_wt="700", amount="not readable"),
        # never billed -> Category C
        lr(lr_no="L848509999", shipment_no="SHP-099", lr_date="11/09/2026",
           truck_no="WB20MN1111", net_wt="450", amount="7000"),
    ]

    results = reconcile(bill_rows, lr_records)

    print("--- shape ---")
    check("one record per bill row plus unmatched LRs", len(results), 7)
    check("categories", [r["category"] for r in results], ["A", "A", "B", "A", "A", "B", "C"])
    check("statuses", [r["status"] for r in results],
          ["clear", "mismatch", "missing_lr", "clear", "clear", "missing_lr", "unbilled"])
    expected_keys = {"lr_no", "category", "bill_row", "lr_record", "matches", "diffs", "flags", "status"}
    check("every record has the full key set", all(set(r) == expected_keys for r in results), True)
    check("matches sub-keys", set(results[0]["matches"]), set(MATCH_FIELDS))
    check("diffs sub-keys", set(results[0]["diffs"]), set(DIFF_FIELDS))

    print("\n--- 0: exact match, all clear ---")
    check("lr_no", results[0]["lr_no"], "L848501350")
    check("matches", results[0]["matches"],
          {"shipment_no": True, "vehicle_no": True, "lr_date": True, "net_wt": True, "amount": True})
    check("diffs", results[0]["diffs"], {"net_wt_diff": 0.0, "amount_diff": 0.0})
    check("no flags", results[0]["flags"], [])

    print("\n--- 1: suffix match with real differences ---")
    check("fuzzy flag", "lr_no_fuzzy_matched" in results[1]["flags"], True)
    check("net_wt mismatch", results[1]["matches"]["net_wt"], False)
    check("net_wt_diff signed", results[1]["diffs"]["net_wt_diff"], -500.0)
    check("amount mismatch", results[1]["matches"]["amount"], False)
    check("amount_diff signed", results[1]["diffs"]["amount_diff"], -2000.0)
    check("vehicle still agrees", results[1]["matches"]["vehicle_no"], True)
    check("status", results[1]["status"], "mismatch")
    check("norms really do differ", results[1]["bill_row"]["lr_no_norm"] != lr_records[1]["lr_no_norm"], True)
    check("digits agree", results[1]["bill_row"]["lr_no_digits"], lr_records[1]["lr_no_digits"])

    print("\n--- 2: no LR on file ---")
    check("category", results[2]["category"], "B")
    check("lr_record is None", results[2]["lr_record"], None)
    check("matches all None", set(results[2]["matches"].values()), {None})
    check("diffs all None", set(results[2]["diffs"].values()), {None})

    print("\n--- 3: inside tolerance ---")
    check("net_wt within 0.01", results[3]["matches"]["net_wt"], True)
    check("net_wt_diff", results[3]["diffs"]["net_wt_diff"], -0.004)
    check("amount within 1.0", results[3]["matches"]["amount"], True)
    check("amount_diff still reported", results[3]["diffs"]["amount_diff"], -0.7)
    check("status is clear", results[3]["status"], "clear")

    print("\n--- 4: unparseable LR amount ---")
    check("amount not comparable", results[4]["matches"]["amount"], None)
    check("amount_diff None", results[4]["diffs"]["amount_diff"], None)
    check("unverified flag", "unverified_fields" in results[4]["flags"], True)
    check("lr parse warning surfaced", "lr_parse_warnings" in results[4]["flags"], True)
    check("status still clear (no False)", results[4]["status"], "clear")

    print("\n--- 5: bill row with no LR number ---")
    check("flag", "bill_lr_no_missing" in results[5]["flags"], True)
    check("category", results[5]["category"], "B")

    print("\n--- 6: unbilled LR ---")
    check("category", results[6]["category"], "C")
    check("bill_row is None", results[6]["bill_row"], None)
    check("lr_no", results[6]["lr_no"], "L848509999")
    check("status", results[6]["status"], "unbilled")

    print("\n--- duplicate and ambiguous handling ---")
    dup_bills = [bill(lr_no="848501360", gross_qty="10", amount="10"),
                 bill(lr_no="848501360", gross_qty="10", amount="10")]
    dup_lrs = [lr(lr_no="848501360", net_wt="10", amount="10")]
    dup = reconcile(dup_bills, dup_lrs)
    check("two bill rows hitting one LR", len(dup), 2)
    check("second flagged", "duplicate_bill_for_lr" in dup[1]["flags"], True)
    check("first not flagged", "duplicate_bill_for_lr" in dup[0]["flags"], False)
    check("LR not reported unbilled", [r["category"] for r in dup], ["A", "A"])

    amb = reconcile([bill(lr_no="848501370", gross_qty="5", amount="5")],
                    [lr(lr_no="848501370", net_wt="5", amount="5"),
                     lr(lr_no="848501370", net_wt="9", amount="9")])
    check("ambiguous flagged", "ambiguous_lr_match" in amb[0]["flags"], True)
    check("unused duplicate becomes Category C", amb[1]["category"], "C")

    print("\n--- tolerance boundaries ---")
    edge = reconcile(
        [bill(lr_no="900001", gross_qty="100.01", amount="501.00"),
         bill(lr_no="900002", gross_qty="100.02", amount="502.00")],
        [lr(lr_no="900001", net_wt="100.00", amount="500.00"),
         lr(lr_no="900002", net_wt="100.00", amount="500.00")])
    check("net_wt exactly 0.01 passes", edge[0]["matches"]["net_wt"], True)
    check("amount exactly 1.0 passes", edge[0]["matches"]["amount"], True)
    check("net_wt 0.02 fails", edge[1]["matches"]["net_wt"], False)
    check("amount 2.0 fails", edge[1]["matches"]["amount"], False)

    print("\n--- empty inputs ---")
    check("both empty", reconcile([], []), [])
    check("bills only", [r["status"] for r in reconcile([bill(lr_no="1")], [])], ["missing_lr"])
    check("LRs only", [r["status"] for r in reconcile([], [lr(lr_no="1")])], ["unbilled"])

    print("\n--- inputs untouched ---")
    check("bill row not mutated", "category" in bill_rows[0], False)
    check("lr record not mutated", "category" in lr_records[0], False)

    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
    else:
        print("all checks passed")
