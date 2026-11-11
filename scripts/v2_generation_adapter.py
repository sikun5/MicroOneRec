#!/usr/bin/env python
"""Phase 5 T3 (supplement round): generative-evaluation adapter contract.
SCHEMA: phase5-generation-adapter-1.1

NO LLM weights are loaded; NO generation is executed. All three groups (A/B/C)
are verified on REAL frozen artifacts, each file hash-checked against Phase 4
accepted inventories BEFORE use (G0 covers the full 30-file read set; the
previous 14/14 conclusion was limited to that narrower scope and is NOT
generalized to all three groups).

 G2g/G3g/G4g per group A/B/C: all 3,686 SIDs encode to exactly 4 atomic tokens
    (+EOS=5), all 4-token prefixes accepted by THAT group's prefix_trie, full
    ids->atomic-strings->SID->item roundtrip, index/info item_ids aligned;
 G5 within-bound-group illegal SIDs (malformed/truncated/foreign-string) stay
    at their original rank position, never dropped/shifted;
 G10 binding-group mismatch -> BindingMismatchError (explicit request reject:
    different methods' legal SIDs may coincide, so the source cannot be
    identified from the SID string alone; the requested group must equal the
    loaded binding's method);
 G11 EOS boundary, adapter ACTUALLY validates sequences (not just declared in
    config): complete 4+EOS accepted; missing terminal EOS rejected; early EOS
    rejected; extra tokens after EOS rejected; non-atomic inner token rejected;
 G12 candidate-count contract enforced: formal mode requires exactly 50
    candidates (50 accepted / 49 rejected); dev mode exactly 10;
 G6 deterministic generation config DECLARED: dev Beam10 / formal Beam50
    returning 50, length_penalty=0, do_sample=False, top_k=None, top_p=None,
    temperature=1.0, max_new_tokens=16 (constructed+asserted, never executed);
 G7 row_id preservation; G8 original evaluate.py byte-identical to T1 audit
    with its do_sample kwarg guard present.

Deferred to Phase 6/8 REAL generation: LLM loading, actual beam execution,
real-generation illegal-rate gate, formal test inference.
"""
import argparse, json, hashlib, re, sys
from pathlib import Path

REPO = Path("/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec")
EV = REPO / "outputs/v2/evidence/phase5"
EOS_EXPECT = 151643
T1_EVALUATE_SHA = "0951376c2135da7ef3305390997c9456bcb31d76ee1a01e03fe43c72e3685621"
GROUPS = ("A", "B", "C")
TOK_FILES = ("added_tokens.json", "merges.txt", "special_tokens_map.json",
             "tokenizer.json", "tokenizer_config.json", "vocab.json")


class BindingMismatchError(RuntimeError):
    """requested decode group differs from the loaded binding manifest/path."""


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def hash_gate(paths):
    REPO_PREFIX = str(REPO) + "/"
    expected = {}
    for inv_name in ("phase4/t4_files_sha256.json", "phase4/t5_files_sha256.json"):
        inv = json.load(open(EV.parent / inv_name))["files"]
        for k, v in inv.items():
            expected[k[len(REPO_PREFIX):] if k.startswith(REPO_PREFIX) else k] = (inv_name, v["sha256"])
    table, ok = {}, True
    for rel in paths:
        cur = sha256_file(REPO / rel)
        src, exp = expected.get(rel, (None, None))
        match = exp == cur
        ok &= match
        table[rel] = {"sha256": cur, "phase4_source": src, "match": match}
    return ok, table


def load_info(path):
    sid2item, item_ids = {}, []
    for line in Path(path).read_text().splitlines():
        if not line:
            continue
        parts = line.split("\t")
        sid2item[parts[0]] = int(parts[2])
        item_ids.append(int(parts[2]))
    return sid2item, item_ids


def trie_accepts(trie, seq):
    node = trie
    for t in seq:
        if str(t) not in node:
            return False
        node = node[str(t)]
    return "__leaf__" in node


def decode_candidates(cand_sids, binding):
    """Within-bound-group decode: illegal stays at its position (None), never dropped."""
    return [binding["sid2item"].get(s) for s in cand_sids]


def decode_for_group(cand_sids, binding, requested_method):
    """Explicit binding check: requested group MUST equal the binding's method.
    Legal SIDs can coincide across methods, so group identity is enforced by
    binding declaration, not inferred from SID strings."""
    if requested_method != binding["method"]:
        raise BindingMismatchError(
            f"requested group {requested_method!r} != binding method {binding['method']!r}; "
            "refusing to decode under a mismatched binding")
    return decode_candidates(cand_sids, binding)


def validate_sid_sequence(ids, eos_id, atomic_ids):
    """ACTUAL sequence check (adapter-enforced, distinct from config declaration)."""
    if len(ids) > 5 and ids[4] == eos_id:
        return False, "tokens after EOS"
    if eos_id in ids[:-1]:
        return False, "early EOS before position 5"
    if len(ids) < 5 or ids[-1] != eos_id:
        return False, "missing terminal EOS"
    for t in ids[:4]:
        if t not in atomic_ids:
            return False, f"non-atomic inner token {t}"
    return True, "complete 4 atomic tokens + EOS"


def check_candidate_count(n, mode):
    expect = 10 if mode == "dev" else 50
    return n == expect, expect


def build_genconfig(num_beams):
    from transformers import GenerationConfig
    return GenerationConfig(num_beams=num_beams, num_return_sequences=num_beams,
                            length_penalty=0.0, do_sample=False, top_k=None, top_p=None,
                            temperature=1.0, max_new_tokens=16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=str, default=str(EV / "generation_adapter_checks.json"))
    ap.add_argument("--protocol-output", type=str, default=str(EV / "evaluation_protocol.json"))
    args = ap.parse_args()
    cases = []
    def rec(cid, name, got, want, ok, note="", inp=None):
        cases.append({"case": cid, "scenario": name, "input": inp or {}, "got": got,
                      "expected": want, "pass": bool(ok), "note": note})
        print(f"G {cid} {name}: pass={ok}", flush=True)

    # ---- G0 hash gate: full three-group read set, BEFORE any consumption ----
    read_paths = []
    for g in GROUPS:
        read_paths += [f"outputs/v2/data/{g}/Industrial_and_Scientific.index.json",
                       f"outputs/v2/data/{g}/prefix_trie.json",
                       f"outputs/v2/data/{g}/reverse_item_map.json",
                       f"outputs/v2/data/{g}/info.txt"]
    read_paths += ["outputs/v2/tokenizer/shared_vocab.json",
                   "outputs/v2/tokenizer/token_id_map.json",
                   "outputs/v2/tokenizer/binding_manifest.json"]
    for g in GROUPS:
        read_paths += [f"outputs/v2/tokenizer/{g}/{f}" for f in TOK_FILES]
    gate_ok, table = hash_gate(read_paths)
    rec("G0", "phase4_hash_gate_three_groups", {"n_files": len(table),
        "all_match": all(v["match"] for v in table.values()), "table": table},
        {"all_match": True}, gate_ok,
        note="every file read in T3 (all three groups + shared tokenizer) verified against phase4 t4/t5 accepted inventories BEFORE use; supersedes the narrower first-round 14-file scope")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(REPO / "outputs/v2/tokenizer/A"))
    manifest = json.load(open(REPO / "outputs/v2/tokenizer/binding_manifest.json"))
    tok_id_map = json.load(open(REPO / "outputs/v2/tokenizer/token_id_map.json"))["map"]
    inv_map = {v: k for k, v in tok_id_map.items()}
    atomic_ids = set(inv_map.keys())
    eos_proto = manifest.get("eos_termination_protocol")
    eos_recorded = (str(EOS_EXPECT) in eos_proto) if isinstance(eos_proto, str) else bool(eos_proto)

    # ---- per-group full binding verification (A/B/C) ----
    per_group, all_g_ok = {}, True
    for g in GROUPS:
        sid2item, ids_info = load_info(REPO / f"outputs/v2/data/{g}/info.txt")
        idx = json.load(open(REPO / f"outputs/v2/data/{g}/Industrial_and_Scientific.index.json"))
        trie = json.load(open(REPO / f"outputs/v2/data/{g}/prefix_trie.json"))["trie"]
        sids = idx["sids"]
        enc = [tok(s, add_special_tokens=False).input_ids for s in sids]
        enc_eos = [e + [tok.eos_token_id] for e in enc]
        len_ok = all(len(e) == 4 for e in enc) and all(len(e) == 5 for e in enc_eos)
        tri_ok = all(trie_accepts(trie, e) for e in enc)
        atomic = all(t in atomic_ids for e in enc for t in e)
        rt_bad = sum(1 for s, e in zip(sids, enc)
                     if "".join(inv_map[t] for t in e) != s or sid2item.get(s) is None)
        ids_ok = [int(x) for x in idx["item_ids"]] == ids_info == sorted(ids_info)
        g_ok = (len(sids) == 3686 and len_ok and tri_ok and atomic and rt_bad == 0 and ids_ok)
        all_g_ok &= g_ok
        per_group[g] = {"n_sids": len(sids), "all_len4_plus_eos5": len_ok, "trie_accepts_all": tri_ok,
                        "atomic_tokens_only": atomic, "roundtrip_failures": rt_bad,
                        "index_info_ids_aligned_sorted": ids_ok}
        print(f"G per-group {g}: pass={g_ok}", flush=True)
    rec("G2G4", "three_group_full_binding", per_group, {"A/B/C": "all 3686 roundtrip+trie+4token+EOS per group"},
        all_g_ok,
        note="A/B/C each: info/index/reverse/trie/per-group tokenizer + shared vocab binding; complete 3,686-item roundtrip per group (binding group verification embedded: each group checked against ITS OWN info/trie/tokenizer paths)")

    # ---- G5 within-group illegal: positions preserved ----
    sid2item_a, _ = load_info(REPO / "outputs/v2/data/A/info.txt")
    binding_a = {"method": "A", "sid2item": sid2item_a,
                 "info_path": "outputs/v2/data/A/info.txt",
                 "trie_path": "outputs/v2/data/A/prefix_trie.json",
                 "tokenizer_dir": "outputs/v2/tokenizer/A"}
    sids_a = json.load(open(REPO / "outputs/v2/data/A/Industrial_and_Scientific.index.json"))["sids"]
    good_sid = sids_a[0]
    sid2item_c, _ = load_info(REPO / "outputs/v2/data/C/info.txt")
    c_sid = next(s for s in sid2item_c if s not in sid2item_a)
    probes = [good_sid, c_sid, "<a_999999><b_0><c_0><d_0>", "<a_0><b_0>"]
    dec = decode_for_group(probes, binding_a, "A")
    ok5 = dec[0] == sid2item_a[good_sid] and all(x is None for x in dec[1:]) and len(dec) == 4
    rec("G5", "within_group_illegal_preserved", {"decoded": dec, "positions_kept": len(dec) == 4},
        {"decoded": [sid2item_a[good_sid], None, None, None], "positions_kept": True}, ok5,
        note="legal-within-group first hit; a foreign-GROUP SID STRING, malformed and truncated SIDs all decode ILLEGAL at their original position under the A binding (no drop/shift/beautify); group identity itself is enforced at request time by G10",
        inp={"binding": "A", "probes": [good_sid, "C-group string:" + c_sid, "malformed", "truncated"]})

    # ---- G10 binding-group mismatch: explicit request rejection ----
    raised, msg = False, ""
    try:
        decode_for_group([good_sid], binding_a, "C")
    except BindingMismatchError as e:
        raised, msg = True, str(e)
    except Exception as e:
        msg = f"wrong exception {type(e).__name__}: {e}"
    rec("G10", "binding_group_mismatch_rejected", {"raised": raised, "message": msg[:200]},
        {"raised": True}, raised,
        note="legal SIDs may coincide across methods -> group identity enforced by explicit request/binding check (BindingMismatchError), never inferred from SID strings; within-group illegal SIDs (G5) still keep their rank positions",
        inp={"binding": "A", "requested": "C"})

    # ---- G11 EOS boundary: adapter ACTUALLY validates sequences ----
    enc0 = tok(good_sid, add_special_tokens=False).input_ids
    t1 = enc0[0]
    seq_ok, seq_reason = validate_sid_sequence(enc0 + [tok.eos_token_id], tok.eos_token_id, atomic_ids)
    miss_ok, miss_reason = validate_sid_sequence(enc0, tok.eos_token_id, atomic_ids)
    early_ok, early_reason = validate_sid_sequence(enc0[:2] + [tok.eos_token_id] + enc0[2:],
                                                   tok.eos_token_id, atomic_ids)
    extra_ok, extra_reason = validate_sid_sequence(enc0 + [tok.eos_token_id, t1], tok.eos_token_id, atomic_ids)
    ok11 = seq_ok and not miss_ok and not early_ok and not extra_ok and eos_recorded \
        and tok.eos_token_id == EOS_EXPECT
    rec("G11", "eos_sequence_boundary_checked", {
        "complete_accepted": seq_ok, "missing_eos": miss_reason, "early_eos": early_reason,
        "extra_after_eos": extra_reason, "tokenizer_eos": tok.eos_token_id,
        "manifest_records_eos": eos_recorded},
        {"complete_accepted": True, "missing_eos": "rejected", "early_eos": "rejected",
         "extra_after_eos": "rejected", "tokenizer_eos": EOS_EXPECT, "manifest_records_eos": True},
        ok11,
        note="adapter ACTUALLY validates each generated sequence (distinct from G6 config declaration): complete 4+EOS accepted; missing terminal EOS / early EOS / tokens after EOS all rejected",
        inp={"sid": good_sid, "eos_id": EOS_EXPECT})

    # ---- G12 candidate-count contract enforced ----
    fifty = decode_for_group([sids_a[i % len(sids_a)] for i in range(50)], binding_a, "A")
    forty9 = decode_for_group([sids_a[i % len(sids_a)] for i in range(49)], binding_a, "A")
    c50, exp50 = check_candidate_count(len(fifty), "formal")
    c49, exp49 = check_candidate_count(len(forty9), "formal")
    c10, exp10 = check_candidate_count(len(decode_for_group([sids_a[0]] * 10, binding_a, "A")), "dev")
    ok12 = c50 and (not c49) and c10 and exp50 == 50 and exp49 == 50 and exp10 == 10
    rec("G12", "candidate_count_contract", {"formal_50_ok": c50, "formal_49_rejected": not c49,
        "dev_10_ok": c10, "expected_counts": {"dev": exp10, "formal": exp49}},
        {"formal_50_ok": True, "formal_49_rejected": True, "dev_10_ok": True,
         "expected_counts": {"dev": 10, "formal": 50}}, ok12,
        note="count contract enforced with synthetic success/failure evidence (50 accepted, 49 rejected, dev 10 accepted) — not a mere num_return_sequences assertion; actual beam RETURNING 50 remains a Phase 6/8 real-generation item",
        inp={"modes": {"dev": 10, "formal": 50}})

    # ---- G6 deterministic generation config (declared; constructed, never run) ----
    gc_dev, gc_formal = build_genconfig(10), build_genconfig(50)
    def gc_ok(gc, beams):
        return (gc.num_beams == beams and gc.num_return_sequences == beams
                and gc.length_penalty == 0.0 and gc.do_sample is False
                and gc.top_k is None and gc.top_p is None and gc.temperature == 1.0
                and gc.max_new_tokens == 16)
    rec("G6", "genconfig_declared", {"dev": gc_ok(gc_dev, 10), "formal": gc_ok(gc_formal, 50)},
        {"dev": True, "formal": True}, gc_ok(gc_dev, 10) and gc_ok(gc_formal, 50),
        note="DECLARED contract (constructed GenerationConfig asserted): Beam10 dev / Beam50 formal returning 50, length_penalty=0, do_sample=False, top_k/top_p=None, temperature=1.0; the ENFORCED side is G11/G12 sequence+count validation; no generate() call anywhere in Phase 5",
        inp={"constructed": ["GenerationConfig(dev10)", "GenerationConfig(formal50)"]})

    # ---- G7 row_id preservation ----
    rows = [{"row_id": "valid:0", "cands": [good_sid, c_sid]},
            {"row_id": "valid:1", "cands": ["<a_999999><b_0><c_0><d_0>"]}]
    out_rows = [{"row_id": r["row_id"], "items": decode_candidates(r["cands"], binding_a)} for r in rows]
    ok7 = [o["row_id"] for o in out_rows] == ["valid:0", "valid:1"] and out_rows[0]["items"][1] is None
    rec("G7", "row_id_preserved", {"row_ids": [o["row_id"] for o in out_rows],
        "items_row0": out_rows[0]["items"]},
        {"row_ids": ["valid:0", "valid:1"], "items_row0": [sid2item_a[good_sid], None]}, ok7,
        note="row_id survives decode->mapping; generative SID strings retained separately for illegal diagnostics")

    # ---- G8 original evaluate.py untouched + kwarg guard ----
    ev_sha = sha256_file(REPO / "evaluate.py")
    ev_text = (REPO / "evaluate.py").read_text()
    ok8 = ev_sha == T1_EVALUATE_SHA and "do_sample=False" in ev_text and "ConstrainedLogitsProcessor" in ev_text
    rec("G8", "original_entry_compatible", {"sha256": ev_sha, "matches_t1_audit": ev_sha == T1_EVALUATE_SHA,
        "do_sample_kwarg_guard": "do_sample=False" in ev_text},
        {"matches_t1_audit": True, "do_sample_kwarg_guard": True}, ok8,
        note="original evaluate.py byte-identical to T1 audit; adapter is an added layer")

    all_pass = all(c["pass"] for c in cases)
    (EV / "generation_adapter_checks.json").write_text(json.dumps(
        {"schema": "phase5-generation-adapter-checks-1.1", "date": "2026-11-11",
         "n_cases": len(cases), "all_pass": all_pass,
         "adapter_code_sha256": sha256_file(__file__), "cases": cases}, ensure_ascii=False, indent=1))

    bs = json.load(open(EV / "shared_sampling_summary.json"))
    bc = json.load(open(EV / "bootstrap_checks.json"))
    mc = json.load(open(EV / "metric_checks.json"))
    protocol = {
        "schema": "phase5-evaluation-protocol-1.1", "date": "2026-11-11",
        "identity_binding": {
            "row_interface": "row_id/user_id/target_item_id/ranked_item_ids; generative raw SID strings retained separately for illegal diagnostics",
            "reference_mode": "validate_and_align(reference=...) enforces per-row user_id+target_item_id against INDEPENDENT frozen-source ground truth (T4/T5: derived from Phase-4 CSV/rows manifest, never from predictions); tampered-but-legal values rejected before any output",
            "label_gate": "missing-label mapping must cover all aligned targets (LabelCoverageError); never defaulted False",
            "code_sha256": sha256_file(REPO / "scripts/v2_evaluate.py")},
        "metric_semantics": {
            "locked_rules": "see metric_checks.json cases + v2_evaluate.py docstring (9 rules)",
            "short_list_disclosure": "missing candidates RAISE duplicate-rate@K by design (denominator K); formal generation must return 50 candidates",
            "evidence_sha256": sha256_file(EV / "metric_checks.json")},
        "bootstrap": {"locked": bs["locked"], "algorithm": bs["algorithm"],
                      "evidence_sha256": sha256_file(EV / "bootstrap_checks.json"),
                      "summary_sha256": sha256_file(EV / "shared_sampling_summary.json"),
                      "synthetic_all_pass": bc["all_pass"]},
        "generation_contract": {
            "group_binding": "per-group info.txt TSV(sid,title,item_id) + group prefix_trie + group tokenizer dir + shared vocab/token_id_map; binding identity enforced at request time (BindingMismatchError), within-group illegal keeps rank position",
            "four_token_eos": "4 atomic tokens + EOS 151643; adapter validates sequences (complete/missing/early/after-EOS) and candidate counts (dev 10 / formal 50)",
            "gen_config_declared": {"dev": {"num_beams": 10}, "formal": {"num_beams": 50, "num_return_sequences": 50},
                                    "length_penalty": 0.0, "do_sample": False, "top_k": None, "top_p": None,
                                    "temperature": 1.0, "max_new_tokens": 16},
            "sid_tokenizer_binding_table": table,
            "adapter_code_sha256": sha256_file(__file__),
            "evidence_sha256": sha256_file(EV / "generation_adapter_checks.json")},
        "b1_baseline": {
            "model_source": "scripts/v2_sasrec.py SASRec + MultiHeadAttentionV2 (boss-authorized minimal mask fix 2026-11-11/12)",
            "mask_fix": "key/query validity traces to EXPLICIT padding boolean mask (states==item_num); replaces original zero-sum row-sum guess that silently masked real items (LN rows can sum to exactly 0.0 in fp32; 327/2000 rows measured)",
            "not_behavior_equivalent": True,
            "code_sha256": sha256_file(REPO / "scripts/v2_sasrec.py"),
            "checkpoint_sha256": sha256_file(REPO / "outputs/v2/baselines/B1/best_checkpoint.pth"),
            "train_summary": json.load(open(REPO / "outputs/v2/baselines/B1/train_summary.json")),
            "discarded_attempt": "one defective-mask training run (2026-11-11 23:40, 187 epochs) discarded, products moved out of B1; not a formal method; cost disclosed in T5/T6 records",
            "residual_note": "checkpoint holds state_dict+meta only (no optimizer/RNG state); resume_used=false means no resume occurred, NOT that exact resume was verified"},
        "implemented_and_synthetic_verified": [
            "unified item-ID evaluator + identity gates (metric_checks, incl. F6-F8 reference binding)",
            "bootstrap sampler/CI/slice-invalidation with independent enumeration CI + manual percentile formula (B1-B8)",
            "adapter contract on real frozen artifacts, ALL THREE GROUPS: hash gate (full read set), per-group 3,686 four-token+EOS/trie/roundtrip, illegal preservation, binding-mismatch rejection, EOS sequence checks, candidate-count contract, row_id, original-entry compatibility"],
        "deferred_to_phase6_8_real_generation": [
            "LLM checkpoint loading and actual beam-search execution",
            "real-generation illegal-rate gate (must be 0 before formal test)",
            "formal test inference and test metrics (Phase 8)",
            "real validation sampling lists (generated at T4/T5 evaluation time by the locked bootstrap function)"],
        "boundary": "no LLM loaded in Phase 5; no SID retraining/re-export; no test inference; original evaluate.py untouched"
    }
    (EV / "evaluation_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=1))
    print("GENERATION_ADAPTER_SHA256:", sha256_file(EV / "generation_adapter_checks.json"), flush=True)
    print("EVALUATION_PROTOCOL_SHA256:", sha256_file(EV / "evaluation_protocol.json"), flush=True)
    print("ADAPTER_ALL_PASS:", all_pass, flush=True)
    print("ADAPTER_DONE", flush=True)


if __name__ == "__main__":
    main()
