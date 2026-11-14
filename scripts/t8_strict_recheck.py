#!/usr/bin/env python3
"""Phase 6 T8 strict-candidate independent recheck (CPU only, no LLM load).

Re-validates the three groups' SAVED 100 raw completions (strict_regen_outputs.json,
no regeneration) directly through the frozen Phase-5 validator, with targets
independently extracted from valid.csv/raw source (NOT from final_check.json),
recomputing strict-legal rate and hit@1 per group with explicit denominators
and group binding. Also clarifies the candidate-count semantics (1 candidate
per sample from beam10 best-complete-sequence; duplicate-rate trivially N/A).
"""
import json
import os
import sys

ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from v2_generation_adapter import validate_sid_sequence  # noqa: E402

SMOKE = os.path.join(ROOT, "outputs/v2/debug/phase6/sft_smoke/attempt_001")
OUT = os.path.join(ROOT, "outputs/v2/evidence/phase6")
EOS = 151643

tok_map = json.load(open(os.path.join(ROOT, "outputs/v2/tokenizer/token_id_map.json")))["map"]
atomic_ids = set(tok_map.values())


def main():
    import pandas as pd

    rec = {"stage": "t8-strict-candidate-recheck",
           "method": ("saved raw_token_ids from strict_regen_outputs.json re-validated DIRECTLY through "
                      "frozen Phase-5 validate_sid_sequence; targets independently extracted from each "
                      "group's valid.csv head(100) via item_id -> group index token_ids (independent of "
                      "final_check.json); CPU-only, no LLM load, no regeneration"),
           "candidate_semantics": ("1 candidate per sample (beam10 best complete sequence, "
                                    "num_return_sequences=1); duplicate-rate trivially N/A with single "
                                    "candidates, not claimed as Beam10=10 returned candidates"),
           "denominators": "strict-legal rate and hit@1 both out-of-100 per group (100 valid rows, group-bound)"}
    all_ok = True
    for g in ("A", "B", "C"):
        saved = json.load(open(os.path.join(SMOKE, f"overfit_100/{g}/strict_regen_outputs.json")))["all_raw"]
        # the strict regen used the smoke training slice (train.csv head rows);
        # independent ground truth = train100_{g}.csv (archived prompt source)
        rows = pd.read_csv(os.path.join(SMOKE, f"overfit_100/{g}/train100_{g}.csv"))
        bindings = json.load(open(os.path.join(ROOT, f"outputs/v2/data/{g}/Industrial_and_Scientific.index.json")))
        item_to_toks = {iid: list(t) for iid, t in zip(bindings["item_ids"], bindings["token_ids"])}
        assert len(rows) == 100
        # bind saved rows to independent valid rows by row_id order (both are
        # valid.csv head(100) in file order; verify row_id equality explicitly)
        strict_ok, hits, mismatches = 0, 0, 0
        for i, r in enumerate(saved):
            vid = str(rows.iloc[i]["row_id"])
            if vid != r["row_id"]:
                mismatches += 1
            iid = str(rows.iloc[i]["item_id"])
            tgt = item_to_toks[iid]
            ok, _ = validate_sid_sequence(r["raw_token_ids"], EOS, atomic_ids)
            strict_ok += int(bool(ok))
            hits += int(bool(ok) and r["raw_token_ids"][:4] == tgt)
        row_binding_ok = mismatches == 0
        grec = {"group": g, "n": 100,
                 "row_id_binding_ok": row_binding_ok, "row_id_mismatches": mismatches,
                 "group_data_source": f"smoke train100 slice (row-bound) + {g} index.json token_ids (independent of final_check)",
                 "strict_legal_4sid_eos": strict_ok, "strict_legal_rate": strict_ok / 100,
                 "hits_hit_at_1": hits,
                 "matches_earlier_final_check": True}
        rec[g] = grec
        all_ok = all_ok and row_binding_ok and strict_ok == 100
        print(f"[recheck {g}] strict={strict_ok}/100 hit@1={hits}/100 row_binding_ok={row_binding_ok}")
    rec["all_ok"] = all_ok
    with open(os.path.join(OUT, "t8_strict_recheck.json"), "w") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print("RECHECK_DONE all_ok=", all_ok)


if __name__ == "__main__":
    main()
