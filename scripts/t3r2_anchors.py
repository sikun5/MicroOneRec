#!/usr/bin/env python3
"""Phase 7 T3-R2(1)+(3): full artifact anchors, control summary with per-candidate
error vectors, reward source attribution, device-labeling table, config-lock
hash chain, P1-1 scope statements -> t3r2_anchors.json."""
import hashlib
import json
import math
import os
import sys

ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
EV = "outputs/v2/evidence/phase7"

from v2_train_grpo import rule_reward, make_ndcg_rule_reward  # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


G = 16
ndcg = make_ndcg_rule_reward(G)
S = sum(1.0 / math.log2(m + 2) for m in range(G))
NEG = [-(1.0 / math.log2(j + 2)) / S for j in range(G)]

# --- one-hit j=3 full expected/actual vectors ---
j = 3
SID = "<a_1><b_2><c_3><d_4>"
SID_Y = "<a_5><b_6><c_7><d_8>"
cands = [SID_Y + f"<v{k}>" for k in range(G)]
cands[j] = SID
exp_rule = [1.0 if i == j else 0.0 for i in range(G)]
exp_rank = [0.0 if i == j else NEG[i] for i in range(G)]
act_rule = rule_reward(prompts=["p"] * G, completions=cands, target_sid=[SID] * G)
act_rank = ndcg(prompts=["p"] * G, completions=cands, target_sid=[SID] * G)
j3 = {"target_position": j,
      "expected_rule": exp_rule, "actual_rule": act_rule,
      "expected_rank": [round(v, 12) for v in exp_rank],
      "actual_rank": [round(v, 12) for v in act_rank],
      "rule_abs_err_max": max(abs(a - e) for a, e in zip(act_rule, exp_rule)),
      "rank_abs_err_max": max(abs(a - e) for a, e in zip(act_rank, exp_rank))}

# --- 16-control summary with max errors (recomputed deterministically) ---
def case_one_hit(jj):
    cc = [SID_Y + f"<v{k}>" for k in range(G)]
    cc[jj] = SID
    ar = rule_reward(prompts=["p"] * G, completions=cc, target_sid=[SID] * G)
    an = ndcg(prompts=["p"] * G, completions=cc, target_sid=[SID] * G)
    er = [1.0 if i == jj else 0.0 for i in range(G)]
    en = [0.0 if i == jj else NEG[i] for i in range(G)]
    return max(max(abs(x - y) for x, y in zip(ar, er)),
               max(abs(x - y) for x, y in zip(an, en)))


rt = json.load(open(f"{EV}/reward_target_checks.json"))
summary = [{"control": r["control"], "ok": r["ok"]} for r in rt["results"]]
summary.append({"control": "C1_one_hit_j3_max_abs_err", "value": case_one_hit(3),
                "threshold": 1e-12, "ok": case_one_hit(3) < 1e-12})

anchors = {
    "task": "Phase7 T3-R2(1)+(3) artifact anchors and scope clarifications",
    "date": "2026-11-15",
    "full_sha256": {
        "scripts/v2_train_grpo.py": sha("scripts/v2_train_grpo.py"),
        "scripts/v2_grpo_stream.py": sha("scripts/v2_grpo_stream.py"),
        "scripts/t3_reward_checks.py": sha("scripts/t3_reward_checks.py"),
        "scripts/t3_tech_checks.py": sha("scripts/t3_tech_checks.py"),
        "scripts/t3r2_resume_sync.py": sha("scripts/t3r2_resume_sync.py"),
        "outputs/v2/manifests/rl_stream_20000_A.jsonl": sha("outputs/v2/manifests/rl_stream_20000_A.jsonl"),
        "outputs/v2/manifests/rl_stream_20000_B.jsonl": sha("outputs/v2/manifests/rl_stream_20000_B.jsonl"),
        "outputs/v2/manifests/rl_stream_20000_C.jsonl": sha("outputs/v2/manifests/rl_stream_20000_C.jsonl"),
        "outputs/v2/evidence/phase7/t3_stream_report.json": sha(f"{EV}/t3_stream_report.json"),
        "outputs/v2/evidence/phase7/reward_target_checks.json": sha(f"{EV}/reward_target_checks.json"),
        "outputs/v2/evidence/phase7/rollout_gradient_checks.json": sha(f"{EV}/rollout_gradient_checks.json"),
        "outputs/v2/evidence/phase7/resume_check.json": sha(f"{EV}/resume_check.json"),
        "outputs/v2/evidence/phase7/grpo_config_lock.json": sha(f"{EV}/grpo_config_lock.json"),
        "outputs/v2/evidence/phase7/tech_checks_part_a.json": sha(f"{EV}/tech_checks_part_a.json"),
        "outputs/v2/evidence/phase7/tech_checks_part_b.json": sha(f"{EV}/tech_checks_part_b.json"),
    },
    "reward_source_attribution": {
        "file": "scripts/v2_train_grpo.py", "sha256": sha("scripts/v2_train_grpo.py"),
        "rule_fn": "rule_reward(prompts, completions, target_sid) - exact-match 1/0, strip(\"\\n\\\" \")",
        "rank_fn": "make_ndcg_rule_reward(G) -> ndcg_rule_reward - a_j=1/log2(j+2), "
                   "hit-group: target 0 / others -a_j/S; no-hit group: all 0",
        "adaptation_note": "both imported from the entry, not re-implemented in checkers; targets "
                           "delivered via target_sid column (identity wiring) replacing prompt-keyed dicts",
        "normalization_vs_raw_legality": ("reward compares candidate TEXT vs target after strip(\"\\n\\\" \") "
                                          "identical to original rl.py - this is reward normalization ONLY; "
                                          "Phase5 raw-sequence strict legality (validate_sid_sequence over "
                                          "raw token ids, no-newline/no-illegal-token) is a separate check "
                                          "gated in T4/T5/T8 evidence and is NOT implied by reward hits; "
                                          "stripping a newline never upgrades an illegal raw sequence"),
    },
    "one_hit_j3_vectors": j3,
    "sixteen_control_summary": summary,
    "device_labeling": {
        "part_a_construction_invariants": {"device": "CPU (T3_PIN_CPU=1 -> CUDA_VISIBLE_DEVICES='')",
                                           "model": "~10M fp32 random Qwen3", "dtype": "float32",
                                           "mocks": "generate mocked; adamw_torch substituted"},
        "part_b_counting_regression": {"device": "idle MIG slice (CUDA visible)",
                                       "model": "~10M fp32 random Qwen3", "dtype": "float32",
                                       "probe_only_max_prompt_length": 64,
                                       "mocks": "generate mocked; rewards synthetic wrappers around real fns; "
                                                "adamw_torch substituted; counting only"},
        "part_c_resume_compare": {"device": "CPU max_steps=8 run used by T3 first round; see resume_check.json",
                                  "model": "~10M fp32 random Qwen3", "dtype": "float32",
                                  "mocks": "generate mocked"},
        "note": "labels corrected once here; no computation rerun for relabeling; "
                "tiny probes do NOT evidence real 599M/G16/paged-optimizer feasibility (that is T4)",
    },
    "config_lock_hash_chain": {
        "v1_t2_initial": "a8fab56c5e3f90a12214f95e85dbdb6dc1d1d0715bbb49be128b66a0ab54e6a6",
        "v2_t2r1_addendum": "36500e8acf382c9b030b89395d38367324e3bd0812a50997263dd139825a7620",
        "v3_t3_implementation": "39b2475c1a00d3401d2e27065f81181d4aa852407647058339f91f1cbf7a4226",
        "note": "each addendum carries prev_file_sha256 internally; chain verifiable from file history",
    },
    "step512_scope": "the 512-sync observation comes only from the tiny counting probes; it does not "
                     "evidence that real 599M-policy + full ref + paged optimizer has run (T4 gate)",
}
with open(f"{EV}/t3r2_anchors.json", "w") as f:
    json.dump(anchors, f, ensure_ascii=False, indent=1)
print("ANCHORS_OK")
print("j3_err", j3["rule_abs_err_max"], j3["rank_abs_err_max"])
print("t3r2_anchors.json", sha(f"{EV}/t3r2_anchors.json"))
