#!/usr/bin/env python
"""Phase 5 T2: unified item-ID ranking evaluator with synthetic self-test.
SCHEMA: phase5-v2_evaluate-1.0

LOCKED METRIC SEMANTICS (boss protocol 2026-11-11; do not extend metric set):
1. Hit position = 1-based position of FIRST occurrence of target. Illegal
   (out-of-catalog) candidates KEEP their original rank position (no shift,
   no removal). Duplicate candidates are NOT deduplicated/reordered.
2. NDCG@K = 1/log2(pos+1) if pos <= K else 0; HR@K = 1 if pos <= K else 0.
   pos beyond list length or absent -> 0.
3. Short lists (len < K): HR/NDCG computed on actual positions only;
   duplicate-rate@K denominator stays K. DISCLOSURE (boss ruling 2026-11-11):
   by this formula missing candidates RAISE the duplicate rate — a short list
   is penalized for unfilled slots, never silently truncated. This does NOT
   license short lists: formal generative evaluation must return exactly 50
   candidates (Beam50 contract, T3); short lists cannot game dup-rate down.
4. Duplicate-rate@K per row = (K - unique_count_first_K) / K, averaged over rows.
5. Illegal rate = (# candidates across ALL rows/positions not in catalog) /
   (# candidates total)  [global, protocol table wording].
6. Coverage@10 = |union over rows of LEGAL items in first 10| / |catalog| (3686);
   denominator fixed by catalog, independent of method.
7. User macro average = mean over users of (per-user mean over rows);
   reported separately from row average. NDCG@10 is the primary metric.
8. Natural slice label = target item's original_description_missing flag
   (Phase-1 quality_features semantics), independent of cleaned text.
9. HARD BLOCKS (raise RowIdentityError / UnknownTargetError / EmptyRankedListError /
   LabelCoverageError, never silently intersect): duplicate row_id, prediction row
   count != bound row set, row_id set mismatch (missing/extra), target not in
   catalog, empty ranked list, identity mismatch against INDEPENDENT reference
   (row_id correct but user_id or target_item_id tampered to another legal value),
   missing-label mapping lacking an aligned target (never defaults to False).
   Illegal candidates are legal input (counted, not blocked).
10. Reference binding: row_id -> (user_id, target_item_id) ground truth MUST come
   from an independent frozen source (future T4/T5: derived from Phase-4 CSVs /
   rows manifest), NEVER from the predictions under test. All identity checks
   run before any metric output is written.

11 synthetic guide scenarios: rank1, rank3, not-in-topK, illegal-first (no shift),
duplicate-target-first-hit, history-contains-target (no filtering), row vs macro
divergence, original-label slicing, Coverage unique-legal fixed denominator,
short list, global illegal rate. Guide scenarios user-cluster-multiplicity /
identical-predictions-CI / empty-slice-round are T3 bootstrap scope (boss
ruling 2026-11-11: centralized in T3, no duplicate construction).
Absolute tolerance for pure metric math: 1e-12.
No test-split evaluation anywhere in Phase 5.
"""
import argparse, json, math, sys, hashlib, inspect
from pathlib import Path
from collections import OrderedDict

TOL = 1e-12
DEFAULT_KS = (3, 5, 10)


class RowIdentityError(RuntimeError):
    """row_id duplication / row-count mismatch / set misalignment."""

class UnknownTargetError(RuntimeError):
    """target_item_id not in catalog."""

class EmptyRankedListError(RuntimeError):
    """empty ranked_item_ids (degenerate prediction, not silently dropped)."""

class LabelCoverageError(RuntimeError):
    """missing-label mapping lacks an aligned target (never defaulted to False)."""


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# ---------------- pure metric layer (importable; reused by T3/T4/T5) ----------------

def score_row(ranked, target, ks=DEFAULT_KS):
    """Per-row hits. ranked: list of item ids (legal or illegal); target: legal id.
    Returns dict: pos (1-based first occurrence or None), hr[k], ndcg[k],
    dup[k] = (K - unique_first_K)/K, n_candidates, n_illegal."""
    pos = None
    for i, it in enumerate(ranked):
        if it == target:
            pos = i + 1
            break
    out = {"pos": pos}
    for k in ks:
        out[f"hr@{k}"] = 1.0 if (pos is not None and pos <= k) else 0.0
        out[f"ndcg@{k}"] = (1.0 / math.log2(pos + 1)) if (pos is not None and pos <= k) else 0.0
    for k in ks:
        first_k = ranked[:k]
        uniq = len(set(first_k))
        out[f"dup@{k}"] = (k - uniq) / k
    out["n_candidates"] = len(ranked)
    return out


def validate_and_align(prediction_rows, bound_row_ids, catalog, reference=None):
    """Identity gate. prediction_rows: list of dicts with row_id/user_id/
    target_item_id/ranked_item_ids. bound_row_ids: list (the evaluation row set,
    e.g. valid.csv row_ids in original order). reference: optional INDEPENDENT
    {row_id: {"user_id","target_item_id"}} ground truth (from frozen source, NOT
    from predictions); when given, per-row user_id AND target_item_id must equal
    the reference exactly — catches row_id-correct but user/target-tampered
    predictions even when the tampered target is a legal catalog item.
    Returns rows aligned to bound_row_ids order. Raises on any identity violation."""
    seen = set()
    for r in prediction_rows:
        rid = r.get("row_id")
        if rid in seen:
            raise RowIdentityError(f"duplicate row_id in predictions: {rid!r}")
        seen.add(rid)
    bound_set = set(bound_row_ids)
    if len(bound_set) != len(bound_row_ids):
        raise RowIdentityError("bound row set itself contains duplicate row_id")
    pred_set = set(seen)
    if len(prediction_rows) != len(bound_row_ids) or pred_set != bound_set:
        missing = sorted(bound_set - pred_set)[:5]
        extra = sorted(pred_set - bound_set)[:5]
        raise RowIdentityError(
            f"row set misalignment: n_pred={len(prediction_rows)} n_bound={len(bound_row_ids)}"
            f" missing={missing} extra={extra}")
    by_id = {r["row_id"]: r for r in prediction_rows}
    aligned = []
    for rid in bound_row_ids:
        r = by_id[rid]
        tgt = r["target_item_id"]
        if tgt not in catalog:
            raise UnknownTargetError(f"row {rid!r}: target {tgt!r} not in catalog")
        if reference is not None:
            ref = reference[rid]
            if r["user_id"] != ref["user_id"]:
                raise RowIdentityError(
                    f"identity mismatch at {rid!r}: user_id {r['user_id']!r} != reference {ref['user_id']!r}")
            if tgt != ref["target_item_id"]:
                raise RowIdentityError(
                    f"identity mismatch at {rid!r}: target {tgt!r} != reference {ref['target_item_id']!r}")
        ranked = r["ranked_item_ids"]
        if not isinstance(ranked, list) or len(ranked) == 0:
            raise EmptyRankedListError(f"row {rid!r}: empty ranked_item_ids")
        aligned.append(r)
    return aligned


def evaluate_rows(aligned_rows, catalog, ks=DEFAULT_KS, labels=None):
    """Full metric set. aligned_rows: validate_and_align output.
    labels: optional {item_id: bool original_description_missing} for slices.
    catalog: set of legal item ids. Returns metrics dict (row mean, user macro,
    coverage@10, illegal rate global, dup@K row-mean, optional slices)."""
    if labels is not None:
        lacking = sorted({r["target_item_id"] for r in aligned_rows if r["target_item_id"] not in labels})
        if lacking:
            raise LabelCoverageError(
                f"missing-label mapping lacks {len(lacking)} aligned target(s), first={lacking[:5]}; "
                "refusing to default any label to False")
    n = len(aligned_rows)
    per = []
    for r in aligned_rows:
        s = score_row(r["ranked_item_ids"], r["target_item_id"], ks)
        s["n_illegal"] = sum(1 for it in r["ranked_item_ids"] if it not in catalog)
        s["user_id"] = r["user_id"]
        per.append(s)
    m = {"n_rows": n}
    for k in ks:
        m[f"hr@{k}"] = sum(s[f"hr@{k}"] for s in per) / n
        m[f"ndcg@{k}"] = sum(s[f"ndcg@{k}"] for s in per) / n
        m[f"duplicate_rate@{k}"] = sum(s[f"dup@{k}"] for s in per) / n
    total_candidates = sum(s["n_candidates"] for s in per)
    total_illegal = sum(s["n_illegal"] for s in per)
    m["illegal_rate_global"] = total_illegal / total_candidates
    # user macro: per-user row mean, then mean over users
    users = OrderedDict()
    for s in per:
        users.setdefault(s["user_id"], []).append(s)
    macro = {"n_users": len(users)}
    for k in ks:
        macro[f"hr@{k}"] = sum(sum(u[f"hr@{k}"] for u in rows) / len(rows) for rows in users.values()) / len(users)
        macro[f"ndcg@{k}"] = sum(sum(u[f"ndcg@{k}"] for u in rows) / len(rows) for rows in users.values()) / len(users)
    m["user_macro"] = macro
    # coverage@10: unique legal items in first 10 across rows / |catalog|
    union = set()
    for r in aligned_rows:
        for it in r["ranked_item_ids"][:10]:
            if it in catalog:
                union.add(it)
    m["coverage@10"] = {"unique_legal_top10": len(union), "catalog_size": len(catalog),
                        "value": len(union) / len(catalog)}
    if labels is not None:
        m["slices"] = slice_metrics(aligned_rows, labels, catalog, ks)
    return m


def slice_metrics(aligned_rows, labels, catalog, ks=DEFAULT_KS):
    """Natural slices by TARGET's original_description_missing (original label,
    not cleaned text). Reports n_rows/n_users/HR@10/NDCG@10 per slice."""
    out = {}
    for name, want in (("missing", True), ("non_missing", False)):
        rows = [r for r in aligned_rows if bool(labels[r["target_item_id"]]) == want]
        if rows:
            per = [score_row(r["ranked_item_ids"], r["target_item_id"], ks) for r in rows]
            us = set(r["user_id"] for r in rows)
            out[name] = {"n_rows": len(rows), "n_users": len(us),
                         "hr@10": sum(p["hr@10"] for p in per) / len(rows),
                         "ndcg@10": sum(p["ndcg@10"] for p in per) / len(rows)}
        else:
            out[name] = {"n_rows": 0, "n_users": 0, "hr@10": None, "ndcg@10": None,
                         "note": "no rows in this evaluation set for the slice"}
    return out


def check_no_history_filtering():
    """API-level guarantee: scoring never takes user history. Returns evidence."""
    sigs = {fn: list(inspect.signature(f).parameters) for fn, f in
            [("score_row", score_row), ("evaluate_rows", evaluate_rows), ("slice_metrics", slice_metrics)]}
    offending = {k: v for k, v in sigs.items() if any("hist" in p for p in v)}
    return {"signatures": sigs, "no_history_parameter": len(offending) == 0}


# ---------------- synthetic self-test ----------------

def run_self_test():
    catalog = set(range(12))
    ill = 9000  # illegal probe (int, not in catalog)
    cases = []
    CAT = "set(range(12))  # synthetic catalog {0..11}"
    def rec(cid, name, got, want, note="", inp=None):
        err = max(abs(got[key] - want[key]) for key in want) if want else 0.0
        cases.append({"case": cid, "scenario": name, "input": inp or {}, "got": got,
                      "expected": want, "max_abs_error": err, "pass": err <= TOL, "note": note})
        print(f"S {cid} {name}: err={err:.3e} pass={err <= TOL}", flush=True)

    # S1 rank1
    s = score_row([5, 1, 2], 5)
    rec("S1", "rank1", {k: s[k] for k in ("hr@3", "hr@5", "hr@10", "ndcg@3", "ndcg@5", "ndcg@10")},
        {"hr@3": 1.0, "hr@5": 1.0, "hr@10": 1.0, "ndcg@3": 1.0, "ndcg@5": 1.0, "ndcg@10": 1.0},
        inp={"ranked": [5, 1, 2], "target": 5, "catalog": CAT})
    # S2 rank3
    s = score_row([1, 2, 5, 3], 5)
    rec("S2", "rank3", {"hr@3": s["hr@3"], "ndcg@3": s["ndcg@3"], "ndcg@5": s["ndcg@5"]},
        {"hr@3": 1.0, "ndcg@3": 0.5, "ndcg@5": 0.5},
        inp={"ranked": [1, 2, 5, 3], "target": 5, "catalog": CAT})
    # S3 not in topK (absent entirely)
    s = score_row([1, 2, 3, 4], 5)
    rec("S3", "not_in_topK", {k: s[k] for k in ("hr@3", "hr@5", "hr@10", "ndcg@3", "ndcg@5", "ndcg@10")},
        {k: 0.0 for k in ("hr@3", "hr@5", "hr@10", "ndcg@3", "ndcg@5", "ndcg@10")},
        inp={"ranked": [1, 2, 3, 4], "target": 5, "catalog": CAT})
    # S4 illegal first, target second -> NO shift, NDCG@3 = 1/log2(3)
    s = score_row([ill, 5, 1], 5)
    rec("S4", "illegal_first_no_shift",
        {"ndcg@3": s["ndcg@3"], "hr@3": s["hr@3"], "pos": float(s["pos"])},
        {"ndcg@3": 1.0 / math.log2(3), "hr@3": 1.0, "pos": 2.0},
        note="illegal keeps position 1; target scored at rank 2",
        inp={"ranked": [9000, 5, 1], "target": 5, "catalog": CAT, "9000": "illegal probe"})
    # S5 duplicate target at positions 2&3 -> first occurrence only; dup@3=1/3
    s = score_row([1, 5, 5, 2], 5)
    rec("S5", "duplicate_target_first_hit",
        {"ndcg@3": s["ndcg@3"], "hr@3": s["hr@3"], "dup@3": s["dup@3"], "pos": float(s["pos"])},
        {"ndcg@3": 1.0 / math.log2(3), "hr@3": 1.0, "dup@3": 1.0 / 3.0, "pos": 2.0},
        note="second occurrence not counted again; unique{1,5}=2 -> (3-2)/3",
        inp={"ranked": [1, 5, 5, 2], "target": 5, "catalog": CAT})
    # S6 history contains target -> no filtering (score unaffected; API has no history input)
    hist = [5, 7]
    s = score_row([5, 1], 5)
    rec("S6", "history_contains_target_no_removal",
        {"hr@3": s["hr@3"], "ndcg@3": s["ndcg@3"]},
        {"hr@3": 1.0, "ndcg@3": 1.0}, note=f"history={hist} recorded but never used in scoring",
        inp={"ranked": [5, 1], "target": 5, "catalog": CAT, "history_context": hist})
    # S7 row mean vs user macro divergence (non-trivial)
    rows7 = [
        {"row_id": "a", "user_id": "U1", "target_item_id": 1, "ranked_item_ids": [1, 2, 3]},
        {"row_id": "b", "user_id": "U1", "target_item_id": 2, "ranked_item_ids": [1, 3, 4]},
        {"row_id": "c", "user_id": "U2", "target_item_id": 3, "ranked_item_ids": [3, 1, 2]},
    ]
    m7 = evaluate_rows(rows7, catalog)
    rec("S7", "row_vs_macro_divergence",
        {"row_ndcg@10": m7["ndcg@10"], "macro_ndcg@10": m7["user_macro"]["ndcg@10"]},
        {"row_ndcg@10": 2.0 / 3.0, "macro_ndcg@10": 0.75},
        note="U1 rows ndcg=1,0 -> user mean 0.5; U2=1; macro=(0.5+1)/2=0.75 != 2/3",
        inp={"rows": rows7, "catalog": CAT})
    # S8 natural label binding (original flag, cleaned text irrelevant)
    labels = {i: False for i in catalog}
    labels[5] = True   # original missing, cleaned NON-empty
    labels[6] = False  # original present, cleaned EMPTY
    rows8 = [
        {"row_id": "a", "user_id": "U1", "target_item_id": 5, "ranked_item_ids": [5]},
        {"row_id": "b", "user_id": "U2", "target_item_id": 6, "ranked_item_ids": [6]},
    ]
    m8 = evaluate_rows(rows8, catalog, labels=labels)
    ok8 = (m8["slices"]["missing"]["n_rows"] == 1 and m8["slices"]["missing"]["n_users"] == 1
           and m8["slices"]["non_missing"]["n_rows"] == 1)
    cases.append({"case": "S8", "scenario": "original_label_slicing",
                  "got": m8["slices"], "expected": {"missing_rows": 1, "non_missing_rows": 1},
                  "max_abs_error": 0.0 if ok8 else 1.0, "pass": ok8,
                  "note": "item5 original-missing(cleaned nonempty)->missing slice; item6 original-present(cleaned empty)->non_missing",
                  "input": {"rows": rows8, "catalog": CAT, "labels": {5: "True(original-missing,cleaned-nonempty)", 6: "False(original-present,cleaned-empty)"}}})
    print(f"S S8 original_label_slicing: pass={ok8}", flush=True)
    # S9 Coverage: duplicates+illegal in top10, unique legal only, fixed denominator
    rows9 = [
        {"row_id": "a", "user_id": "U1", "target_item_id": 0,
         "ranked_item_ids": [0, 1, 1, 2, ill, 3, 4, 5, 6, 7]},
        {"row_id": "b", "user_id": "U2", "target_item_id": 8,
         "ranked_item_ids": [8, 1, 2, 3, 4, 5, 6, 7, ill, ill]},
    ]
    m9 = evaluate_rows(rows9, catalog)
    # legal unique union: row a {0,1,2,3,4,5,6,7}, row b {8,1..7} -> union {0..8} = 9
    rec("S9", "coverage_unique_legal_fixed_denominator",
        {"unique_legal_top10": float(m9["coverage@10"]["unique_legal_top10"]),
         "catalog_size": float(m9["coverage@10"]["catalog_size"]),
         "value": m9["coverage@10"]["value"]},
        {"unique_legal_top10": 9.0, "catalog_size": 12.0, "value": 9.0 / 12.0},
        note="duplicates counted once, illegal excluded, denominator=|catalog|=12 for any method",
        inp={"rows": rows9, "catalog": CAT})
    # S10 short list: denominator stays K
    s = score_row([1, 2], 9)
    rec("S10", "short_list_denominator_K",
        {"hr@3": s["hr@3"], "hr@10": s["hr@10"], "dup@3": s["dup@3"]},
        {"hr@3": 0.0, "hr@10": 0.0, "dup@3": 1.0 / 3.0},
        note="len=2<K: HR by actual positions; dup@3=(3-2)/3 by locked denominator",
        inp={"ranked": [1, 2], "target": 9, "catalog": CAT})
    # S11 global illegal rate
    rows11 = [
        {"row_id": "a", "user_id": "U1", "target_item_id": 1, "ranked_item_ids": [ill, 1]},
        {"row_id": "b", "user_id": "U2", "target_item_id": 2, "ranked_item_ids": [1, 2]},
    ]
    m11 = evaluate_rows(rows11, catalog)
    rec("S11", "illegal_rate_global", {"illegal_rate_global": m11["illegal_rate_global"]},
        {"illegal_rate_global": 0.25}, note="1 illegal / 4 candidates total",
        inp={"rows": rows11, "catalog": CAT})
    # shared catalog check: same catalog object, same denominator
    m9b = evaluate_rows(rows9, catalog)
    shared_ok = m9b["coverage@10"]["catalog_size"] == m9["coverage@10"]["catalog_size"] == len(catalog)
    cases.append({"case": "S12", "scenario": "shared_catalog_same_denominator",
                  "input": {"rows": rows9, "catalog": CAT, "calls": "evaluate_rows(rows9, catalog) x2 (same object)"},
                  "got": {"catalog_size": m9b["coverage@10"]["catalog_size"]},
                  "expected": {"catalog_size": 12}, "max_abs_error": 0.0 if shared_ok else 1.0,
                  "pass": shared_ok, "note": "two evaluate_rows calls share one catalog object"})
    print(f"S S12 shared_catalog_same_denominator: pass={shared_ok}", flush=True)
    hist_check = check_no_history_filtering()
    cases.append({"case": "S13", "scenario": "no_history_parameter_in_api",
                  "got": hist_check, "expected": {"no_history_parameter": True},
                  "max_abs_error": 0.0 if hist_check["no_history_parameter"] else 1.0,
                  "pass": hist_check["no_history_parameter"],
                  "note": "signatures of score_row/evaluate_rows/slice_metrics contain no history arg"})
    print(f"S S13 no_history_parameter: pass={hist_check['no_history_parameter']}", flush=True)

    # ---- failure paths F1-F5: each must raise ----
    INPUTS = {
        "F1": {"predictions": "3 rows, row_id 'a' duplicated", "bound": ["a", "b", "c"], "catalog": CAT},
        "F2": {"predictions": "2 rows (a,b)", "bound": ["a", "b", "c"], "catalog": CAT, "violation": "row dropped"},
        "F3": {"predictions": "3 rows (a,b,zz)", "bound": ["a", "b"], "catalog": CAT, "violation": "extra row"},
        "F4": {"predictions": "1 row target=999", "bound": ["a"], "catalog": CAT, "violation": "target not in catalog"},
        "F5": {"predictions": "1 row ranked_item_ids=[]", "bound": ["a"], "catalog": CAT, "violation": "empty list"},
        "F6": {"predictions": "row 'a' user U1->U9 (legal-looking), target unchanged", "bound": ["a", "b"],
               "reference": "independent {a:(U1,1), b:(U2,2)}", "catalog": CAT, "violation": "user_id != reference"},
        "F7": {"predictions": "row 'a' target 1->2 (BOTH legal in catalog)", "bound": ["a", "b"],
               "reference": "independent {a:(U1,1), b:(U2,2)}", "catalog": CAT, "violation": "target != reference"},
        "F8": {"labels": "entry for target 2 deleted", "bound": ["a", "b"], "catalog": CAT,
               "violation": "LabelCoverageError, no default False"},
    }
    def expect_raise(cid, name, fn, exc_type, key):
        try:
            fn()
            raised, msg = False, "no exception raised"
        except exc_type as e:
            raised, msg = True, str(e)
        except Exception as e:  # wrong exception type
            raised, msg = False, f"wrong exception {type(e).__name__}: {e}"
        cases.append({"case": cid, "scenario": name, "input": INPUTS[cid],
                      "raised_expected_exception": raised,
                      "exception": exc_type.__name__, "message": msg[:200], "pass": raised})
        print(f"F {cid} {name}: raised={raised} ({exc_type.__name__})", flush=True)

    good = [{"row_id": "a", "user_id": "U1", "target_item_id": 1, "ranked_item_ids": [1]},
            {"row_id": "b", "user_id": "U2", "target_item_id": 2, "ranked_item_ids": [2]}]
    dup = good + [{"row_id": "a", "user_id": "U1", "target_item_id": 1, "ranked_item_ids": [1]}]
    expect_raise("F1", "duplicate_row_id", lambda: validate_and_align(dup, ["a", "b", "c"], catalog),
                 RowIdentityError, "dup")
    expect_raise("F2", "missing_row_dropped", lambda: validate_and_align(good, ["a", "b", "c"], catalog),
                 RowIdentityError, "count")
    extra = good + [{"row_id": "zz", "user_id": "U9", "target_item_id": 1, "ranked_item_ids": [1]}]
    expect_raise("F3", "extra_row_misaligned", lambda: validate_and_align(extra, ["a", "b"], catalog),
                 RowIdentityError, "extra")
    badtgt = [{"row_id": "a", "user_id": "U1", "target_item_id": 999, "ranked_item_ids": [1]}]
    expect_raise("F4", "unknown_target", lambda: validate_and_align(badtgt, ["a"], catalog),
                 UnknownTargetError, "target")
    empty = [{"row_id": "a", "user_id": "U1", "target_item_id": 1, "ranked_item_ids": []}]
    expect_raise("F5", "empty_ranked_list", lambda: validate_and_align(empty, ["a"], catalog),
                 EmptyRankedListError, "empty")
    # Independent-reference identity checks (boss supplement 2026-11-11): row_id
    # correct but user/target tampered to ANOTHER LEGAL value must be rejected.
    REF = {"a": {"user_id": "U1", "target_item_id": 1},
           "b": {"user_id": "U2", "target_item_id": 2}}  # independent truth, NOT derived from predictions
    swap_user = [{"row_id": "a", "user_id": "U9", "target_item_id": 1, "ranked_item_ids": [1]},
                 {"row_id": "b", "user_id": "U2", "target_item_id": 2, "ranked_item_ids": [2]}]
    expect_raise("F6", "same_rowid_swapped_user_vs_reference",
                 lambda: validate_and_align(swap_user, ["a", "b"], catalog, REF),
                 RowIdentityError, "user")
    swap_tgt = [{"row_id": "a", "user_id": "U1", "target_item_id": 2, "ranked_item_ids": [2]},
                {"row_id": "b", "user_id": "U2", "target_item_id": 2, "ranked_item_ids": [1]}]
    expect_raise("F7", "same_rowid_swapped_legal_target_vs_reference",
                 lambda: validate_and_align(swap_tgt, ["a", "b"], catalog, REF),
                 RowIdentityError, "target")
    # Missing-label mapping lacking an aligned target -> reject, never default False
    partial_labels = {i: False for i in catalog}
    del partial_labels[2]  # row b's target has no label entry
    expect_raise("F8", "missing_label_entry_rejected",
                 lambda: evaluate_rows(validate_and_align(good, ["a", "b"], catalog, REF),
                                       catalog, labels=partial_labels),
                 LabelCoverageError, "label")

    all_pass = all(c["pass"] for c in cases)
    return {"schema": "phase5-metric-checks-1.0", "date": "2026-11-11",
            "tolerance": TOL, "n_cases": len(cases), "all_pass": all_pass, "cases": cases}


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="v2 unified item-ID ranking evaluator")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--predictions", type=str, default="", help="jsonl: row_id/user_id/target_item_id/ranked_item_ids")
    ap.add_argument("--catalog", type=str, default="", help="index.json with item_ids")
    ap.add_argument("--missing-labels", type=str, default="", help="quality_features.jsonl (item_id/original_description_missing)")
    ap.add_argument("--bind-row-ids", type=str, default="", help="txt/jsonl of bound row ids (one per line or json list)")
    ap.add_argument("--output", type=str, default="")
    ap.add_argument("--ks", type=str, default="3,5,10")
    args = ap.parse_args()
    ks = tuple(int(x) for x in args.ks.split(","))

    if args.self_test:
        res = run_self_test()
        if args.output:
            Path(args.output).write_text(json.dumps(res, ensure_ascii=False, indent=1))
            print("METRIC_CHECKS_SHA256:", sha256_file(args.output), flush=True)
        print("SELF_TEST_ALL_PASS:", res["all_pass"], flush=True)
        print("SELF_TEST_DONE", flush=True)
        return

    assert args.predictions and args.catalog and args.bind_row_ids and args.output, \
        "predictions/catalog/bind-row-ids/output required (or use --self-test)"
    catalog = set(int(x) for x in json.load(open(args.catalog))["item_ids"])
    bound, reference = [], {}
    for l in open(args.bind_row_ids):
        l = l.strip()
        if not l:
            continue
        if l.startswith("{"):
            d = json.loads(l)
            bound.append(d["row_id"])
            if "user_id" in d or "target_item_id" in d:
                reference[d["row_id"]] = {"user_id": d["user_id"], "target_item_id": d["target_item_id"]}
        else:
            bound.append(l)
    preds = [json.loads(l) for l in open(args.predictions) if l.strip()]
    labels = None
    if args.missing_labels:
        labels = {}
        for l in open(args.missing_labels):
            d = json.loads(l)
            labels[int(d["item_id"])] = bool(d["original_description_missing"])
    aligned = validate_and_align(preds, bound, catalog, reference or None)
    metrics = evaluate_rows(aligned, catalog, ks, labels)
    out = {"schema": "phase5-validation-metrics-1.0", "date": "2026-11-11",
           "config": {"catalog_sha256": sha256_file(args.catalog),
                      "predictions_sha256": sha256_file(args.predictions),
                      "bind_row_ids_sha256": sha256_file(args.bind_row_ids),
                      "reference_mode": bool(reference),
                      "missing_labels_sha256": sha256_file(args.missing_labels) if args.missing_labels else None,
                      "code_sha256": sha256_file(__file__), "ks": list(ks),
                      "n_bound_rows": len(bound)},
           "metrics": metrics}
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("EVALUATE_DONE n_rows=", metrics["n_rows"], flush=True)


if __name__ == "__main__":
    main()
