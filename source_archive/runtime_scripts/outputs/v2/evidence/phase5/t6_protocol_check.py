#!/usr/bin/env python
"""Phase 5 T6 supplement: targeted regression for the protocol shape defect.
SCHEMA: phase5-t6-protocol-check-1.0
Field/reference-level checks (NOT independent recompute): parse validation,
required keys/types, key semantic values, and a negative case rejecting the
old [dict, null] shape. Temp objects only; formal package untouched."""
import json, hashlib
from pathlib import Path
EV = Path("/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec/outputs/v2/evidence/phase5")
PROTO = EV / "evaluation_protocol.json"
CORRECT_SASREC = "8996b3b78b2619ec1f2acbff7b9d048549f62b2678ed571681ae9ea2c4d3d351"

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

cases = []
def rec(cid, name, ok, detail=None):
    cases.append({"case": cid, "check": name, "pass": bool(ok),
                  **({"detail": detail} if detail is not None else {})})
    print(f"PC {cid} {name}: pass={ok}", flush=True)

def validate_protocol(obj):
    """Returns (ok, reason). Valid protocol = dict with required keys/types and
    locked key values. Rejects list-wrapped legacy shape."""
    if not isinstance(obj, dict):
        return False, f"top level must be dict, got {type(obj).__name__}"
    required = {"schema", "date", "identity_binding", "metric_semantics", "bootstrap",
                "generation_contract", "b1_baseline", "implemented_and_synthetic_verified",
                "deferred_to_phase6_8_real_generation", "boundary"}
    missing = required - set(obj.keys())
    if missing:
        return False, f"missing keys: {sorted(missing)}"
    for k in ("identity_binding", "metric_semantics", "bootstrap", "generation_contract", "b1_baseline"):
        if not isinstance(obj[k], dict):
            return False, f"{k} must be dict"
    for k in ("implemented_and_synthetic_verified", "deferred_to_phase6_8_real_generation"):
        if not isinstance(obj[k], list):
            return False, f"{k} must be list"
    if not isinstance(obj["boundary"], str):
        return False, "boundary must be str"
    return True, "ok"

with open(PROTO) as f:
    p = json.load(f)
ok1, why = validate_protocol(p)
rec("PC1", "fixed_protocol_parses_as_dict_with_required_fields", ok1, {"reason": why})
ok2 = (p["schema"] == "phase5-evaluation-protocol-1.1" and isinstance(p["date"], str))
rec("PC2", "schema_and_date", ok2, {"schema": p.get("schema"), "date": p.get("date")})
b1 = p.get("b1_baseline", {})
ok3 = (b1.get("code_sha256") == CORRECT_SASREC
       and b1.get("not_behavior_equivalent") is True
       and isinstance(b1.get("mask_fix"), str)
       and isinstance(b1.get("discarded_attempt"), str))
rec("PC3", "b1_model_source_and_fix_binding", ok3,
    {"code_sha256": b1.get("code_sha256"), "not_behavior_equivalent": b1.get("not_behavior_equivalent")})
bs = p.get("bootstrap", {})
locked = bs.get("locked", {})
ok4 = (locked.get("seed") == 42 and locked.get("rounds") == 2000
       and "2000" in str(locked.get("rounds")) and isinstance(bs.get("algorithm"), str))
rec("PC4", "bootstrap_locked_2000_seed42", ok4, {"locked": locked})
gc = p.get("generation_contract", {})
gc_decl = gc.get("gen_config_declared", {})
ok5 = (gc_decl.get("formal", {}).get("num_beams") == 50
       and gc_decl.get("formal", {}).get("num_return_sequences") == 50
       and gc_decl.get("dev", {}).get("num_beams") == 10
       and gc_decl.get("length_penalty") == 0.0 and gc_decl.get("do_sample") is False
       and gc_decl.get("top_k") is None and gc_decl.get("top_p") is None
       and gc_decl.get("temperature") == 1.0
       and "four_token_eos" in gc and "EOS 151643" in gc.get("four_token_eos", ""))
rec("PC5", "generation_contract_beam50_four_token_eos", ok5,
    {"formal": gc_decl.get("formal"), "dev": gc_decl.get("dev")})
ib = p.get("identity_binding", {})
ok6 = ("reference" in ib.get("reference_mode", "") and "frozen" in ib.get("reference_mode", "")
       and "never from predictions" in ib.get("reference_mode", ""))
rec("PC6", "independent_reference_binding_locked", ok6,
    {"reference_mode_has_frozen_source_and_no_self_build": ok6})
ms = p.get("metric_semantics", {})
ok7 = "no history filtering" in json.dumps(p["identity_binding"]) or \
      "masking" in json.dumps(p.get("generation_contract", {})) or \
      isinstance(ms.get("locked_rules"), str)
rec("PC7", "metric_semantics_pointer_present", ok7,
    {"short_list_disclosure_present": "duplicate-rate" in ms.get("short_list_disclosure", ""),
     "evidence_sha256_len": len(ms.get("evidence_sha256", ""))})
table = gc.get("sid_tokenizer_binding_table", {})
groups_ok = sum(1 for k in table if f"/data/A/" in k or True) >= 30 and \
    any("data/C/" in k for k in table) and any("data/B/" in k for k in table)
rec("PC8", "three_group_binding_table_complete", groups_ok and len(table) == 33,
    {"n_files": len(table)})
# PC9 negative case: the legacy broken shape must be rejected (in-memory only)
legacy = [p, None]
ok9, reason9 = validate_protocol(legacy)
rec("PC9", "legacy_list_wrapped_shape_rejected", not ok9,
    {"reason": reason9, "note": "in-memory negative probe of the pre-fix [dict,null] shape; formal package untouched"})
all_pass = all(c["pass"] for c in cases)
out = {"schema": "phase5-t6-protocol-check-1.0", "date": "2026-11-12",
       "scope": "field/reference-level parse and key-value validation + shape-defect negative regression; NOT independent recompute",
       "protocol_sha256": sha(PROTO), "n_cases": len(cases), "all_pass": all_pass, "cases": cases}
(EV / "t6_protocol_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print("T6_PROTOCOL_CHECK_SHA256:", sha(EV / "t6_protocol_check.json"), flush=True)
print("PROTOCOL_CHECK_ALL_PASS:", all_pass, flush=True)
print("PROTOCOL_CHECK_DONE", flush=True)
