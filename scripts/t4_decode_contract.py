#!/usr/bin/env python3
"""Phase 6 T4 decode-contract supplement (boss ruling 2026-11-12).

1. Token-by-token annotation of the raw smoke completions (SID atoms / newline
   / EOS) with hit and non-hit examples per group.
2. Re-validation of ALL 100 completions x 3 groups through the Phase-5 accepted
   validator (v2_generation_adapter.validate_sid_sequence) after the disclosed
   decoder adaptation: strip the trailing newline that the SFT target template
   (sid + "\n" + EOS) teaches, then append EOS -> strict 4-atomic-tokens + EOS.
3. Frozen formal-config disclosure with per-run commands/exit codes and wall
   accounting (measured wall vs unknown GPU-busy).
"""
import json
import os
import sys

ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from v2_generation_adapter import validate_sid_sequence  # noqa: E402  (Phase-5 accepted path)

OUT = os.path.join(ROOT, "outputs/v2/evidence/phase6")
SMOKE = os.path.join(ROOT, "outputs/v2/debug/phase6/sft_smoke/attempt_001")
EOS = 151643
NL = 198

tok_map = json.load(open(os.path.join(ROOT, "outputs/v2/tokenizer/token_id_map.json")))["map"]
atomic_ids = set(tok_map.values())
assert len(atomic_ids) == 3308

from transformers import AutoTokenizer  # noqa: E402
_tok = AutoTokenizer.from_pretrained(
    "/hpc2hdd/home/swu073/sikun/dsaa5020/models/Qwen3-0.6B-Base", local_files_only=True)


def annotate(ids):
    parts = []
    for i, t in enumerate(ids):
        if t == EOS:
            kind = "EOS"
        elif t == NL:
            kind = "newline"
        elif t in atomic_ids:
            kind = f"SID-atom({i if i < 4 else 'extra'})"
        else:
            kind = "base-token"
        parts.append(f"[{i}]={t}:{_tok.convert_ids_to_tokens(t)}({kind})")
    return " ".join(parts)


def main():
    rec = {"stage": "t4-decode-contract-supplement",
           "phase5_validator": "v2_generation_adapter.validate_sid_sequence (module SHA 67be20d4..., imported without side effects)",
           "newline_finding": {
               "raw_completion_structure": "[SID_a, SID_b, SID_c, SID_d, newline(198), EOS(151643)] = 6 tokens",
               "newline_source": "SFT training target template encodes target as sid+\\n with EOS (data.py golden = encode(target+\\n, eos=True)); the model learned the newline as a trailing separator",
               "phase5_contract": "strict 4 atomic tokens + EOS, extra tokens rejected (5-token candidates)",
               "resolution": "decoder adaptation = strip the single trailing newline, keep 4 SID atoms + EOS; adaptation disclosed here and applies only to the debug smoke decoding, NOT a relaxation of the Phase-5 validator",
               "training_template_separators": "newline is a legal separator inside the training template only; the final candidate interface is 4 atomic tokens + EOS"},
           "statistics_unit": "per group: 100 samples x 1 candidate each (beam10 best complete sequence, num_return_sequences=1); hits are hit@1 out of 100",
           "per_group": {}}
    all_ok = True
    for g in ("A", "B", "C"):
        gen = json.load(open(os.path.join(SMOKE, f"overfit_100/{g}/generation_outputs.json")))
        n = len(gen)
        strict_ok, mismatch_with_old, annotated_hit, annotated_miss = 0, 0, None, None
        for r in gen:
            raw = r["raw_token_ids"]
            dev = None
            if len(raw) == 6 and raw[4] == NL and raw[5] == EOS:
                adapted = raw[:4] + [EOS]
            elif len(raw) == 5 and raw[4] == EOS:
                adapted = raw
                dev = "already 5-token (no newline)"
            else:
                adapted = None
                dev = f"unexpected structure len={len(raw)}"
            ok, msg = (False, dev) if adapted is None else validate_sid_sequence(adapted, EOS, atomic_ids)
            strict_ok += int(bool(ok))
            if not ok:
                mismatch_with_old += 1
            tgt = r["target_token_ids"]
            hit = adapted is not None and adapted[:4] == list(tgt)
            if hit != r["match"]:
                mismatch_with_old += 1
            if hit and annotated_hit is None:
                annotated_hit = {"row_id": r["row_id"], "annotated": annotate(raw),
                                 "adapted_5token": adapted, "phase5_validate": msg,
                                 "match": hit}
            if (not hit) and annotated_miss is None:
                annotated_miss = {"row_id": r["row_id"], "annotated": annotate(raw),
                                  "adapted_5token": adapted, "phase5_validate": msg,
                                  "target_annotated": annotate(list(tgt) + [EOS]),
                                  "match": hit}
        group_ok = (strict_ok == n and mismatch_with_old == 0)
        rec["per_group"][g] = {
            "n_samples": n, "strict_legal_after_adaptation": strict_ok,
            "strict_legal_rate": strict_ok / n,
            "hits_hit_at_1": sum(1 for r in gen if r["match"]),
            "mismatch_with_earlier_report": mismatch_with_old,
            "annotated_hit_example": annotated_hit,
            "annotated_non_hit_example": annotated_miss,
            "gate": group_ok}
        all_ok = all_ok and group_ok
        print(f"[contract {g}] strict_legal={strict_ok}/{n} hit@1={sum(1 for r in gen if r['match'])}/{n} ok={group_ok}")

    rec["formal_config_disclosure"] = {
        "micro_train": 2, "formal_accum_for_1024": 512, "smoke_accum": 1,
        "eval_micro": 4, "gradient_checkpointing": False,
        "allocator": {"probe_512_run": "PYTORCH_CUDA_ALLOC_CONF unset",
                       "smoke_runs": "expandable_segments:True"},
        "pressure_and_smoke_micro_same": True,
        "probe_input_shape": "[micro, 512] (max_length padding; attention length 512)",
        "probe_optimizer_steps": 2,
        "accumulation_probe_per_group": "512 micro fwd/bwd -> 1 optimizer.step (A/B/C identical)",
        "smoke_epochs_and_updates": {"A": {"epochs": 60, "updates": 3000}, "B": {"epochs": 60, "updates": 3000},
                                      "C": {"epochs": 52, "updates": 2600, "stop": "memorized(eval<=0.5)"}},
        "commands_exit_codes": {
            "probes": "configs/t4_probe_run2.sh -> reports/t4_probe_run512.log: FEAS512 A=2 B=2 C=2 COMMON=2; T4_PROBES512_ALL_DONE (exit 0)",
            "smoke": "configs/t4_smoke_run6.sh -> reports/t4_smoke_run6.log: SMOKE_RC A 0 / B 0 / C 0; T4_SMOKE_ALL_DONE (exit 0)",
            "probe_reload": "RELOAD_OK A/B/C micro2",
            "smoke_reload": "configs/t4_reload_reports.sh -> reports/t4_reload_run.log: SMOKE_RELOAD_OK A/B/C (abs_diff=0.0); T4_RELOADS_DONE"},
        "wall_accounting": {
            "measured_wall": {"smoke_v6_seconds": {"A": 933.3, "B": 937.6, "C": 808.0}},
            "unmeasured": "probe/reload wall per run and total GPU-busy time not collected (unknown); the ~1.5h figure in the worker COMMENTS was a mixed wall-clock estimate, not GPU-busy time"}
    }
    rec["all_ok"] = all_ok
    with open(os.path.join(OUT, "t4_decode_contract.json"), "w") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print("CONTRACT_DONE all_ok=", all_ok)


if __name__ == "__main__":
    main()
