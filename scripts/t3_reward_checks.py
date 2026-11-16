#!/usr/bin/env python3
"""Phase 7 T3 manual reward controls (boss acceptance list) -> reward_target_checks.json.

Six control families on G=16 candidate groups, using the EXACT reward functions
imported from scripts/v2_train_grpo.py (no re-implementation), verified against
closed-form manual expectations:
  a_j = 1/log2(j+2), j=0..15, S = sum(a_j)
  group with a true hit: target candidate rank term = 0, non-target at rank m
  gets -a_m/S; rule = 1 iff candidate equals target after strip("\n\" ");
  group with no hit: both rewards all 0.
Controls: one-hit (several j), all-miss, duplicate candidates, same-history
different-task targets (identity-column wiring vs dict overwrite), first-3-layers
same / 4th differs, EOS/newline/illegal-token boundaries (full G=16 groups:
the ranking window closes on multiples of G, exactly like the trainer's
16-rows-per-group micro-batches).
"""
import hashlib
import json
import math
import os
import sys

ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

from v2_train_grpo import rule_reward, make_ndcg_rule_reward  # noqa: E402

G = 16
ndcg = make_ndcg_rule_reward(G)
NEG = [-(1.0 / math.log2(j + 2)) / sum(1.0 / math.log2(m + 2) for m in range(G))
       for j in range(G)]

results = []


def check(name, ok, detail):
    results.append({"control": name, "ok": bool(ok), "detail": detail})


def run_case(name, targets, completions, expect_rule, expect_rank):
    got_rule = rule_reward(prompts=["p"] * len(completions), completions=completions,
                           target_sid=targets)
    got_rank = ndcg(prompts=["p"] * len(completions), completions=completions,
                    target_sid=targets)
    ok = got_rule == expect_rule and got_rank == expect_rank
    check(name, ok, {"n_candidates": len(completions),
                     "rule_match": got_rule == expect_rule,
                     "rank_match": got_rank == expect_rank,
                     "got_rank": [round(v, 10) for v in got_rank]})
    return ok


SID = "<a_1><b_2><c_3><d_4>"
SID_X = "<a_1><b_2><c_3><d_9>"    # first 3 layers same, 4th differs
SID_Y = "<a_5><b_6><c_7><d_8>"

# C1: one exact hit at rank j
for j in (0, 3, 15):
    cands = [SID_Y + f"<v{k}>" for k in range(G)]
    cands[j] = SID
    exp_rule = [1.0 if i == j else 0.0 for i in range(G)]
    exp_rank = [0.0 if i == j else NEG[i] for i in range(G)]
    run_case(f"C1_one_hit_j{j}", [SID] * G, cands, exp_rule, exp_rank)

# C2: all miss
run_case("C2_all_miss", [SID] * G, [SID_Y + f"<v{k}>" for k in range(G)],
         [0.0] * G, [0.0] * G)

# C3: duplicate candidates -> identical rewards, window intact, no dedup variance
cands = [SID_X + "\n"] * 8 + [SID_Y + "\n"] * 8
got_rank = ndcg(prompts=["p"] * G, completions=cands, target_sid=[SID] * G)
uniq = sorted(set(got_rank))
check("C3_duplicate_candidates", len(uniq) <= 2 and all(r <= 0 for r in got_rank),
      {"n_unique_rewards": len(uniq), "rewards": uniq,
       "note": "duplicate candidates receive identical rewards; candidate uniqueness is "
               "reported by trainer metrics (candidate_unique_rate), never by reward-side dedup"})

# C4: same history, different task targets -> identity-column wiring
prompt = ("### User Input: \nThe user has interacted with items H in chronological order.\n\n"
          "### Response:\n")
got_A = rule_reward(prompts=[prompt] * G, completions=[SID + "\n"] * G, target_sid=[SID] * G)
got_B = rule_reward(prompts=[prompt] * G, completions=[SID_X + "\n"] * G,
                    target_sid=[SID_X] * G)
check("C4_same_history_diff_targets_identity",
      got_A == [1.0] * G and got_B == [1.0] * G,
      {"rowA_rule_sum": sum(got_A), "rowB_rule_sum": sum(got_B),
       "dict_would_give": "history->target overwrite scores one row's true target as miss (0)",
       "identity_wiring": "target_sid column rides each row; no prompt-keyed dict anywhere"})

# C4b: mixed batch (one call, per-position identity targets) — same code path the
# trainer uses inside one micro-batch worth of rows
mixed_comp = [SID + "\n"] * 8 + [SID_X + "\n"] * 8
mixed_tgt = [SID] * 8 + [SID_X] * 8
got_mixed = rule_reward(prompts=["p"] * G, completions=mixed_comp, target_sid=mixed_tgt)
check("C4b_mixed_batch_identity", got_mixed == [1.0] * G,
      {"note": "each row scored against its own target_sid at its own position; no cross-row dict"})

# C5: first-3-layers same / 4th differs is NOT a hit
run_case("C5_prefix3_same_4th_diff", [SID] * G, [SID_X + "\n"] * G, [0.0] * G, [0.0] * G)

# C6: EOS / newline / illegal-token boundaries (full G=16 groups)
c6_cases = [
    ("target_with_trailing_newline", SID + "\n", True),
    ("target_without_newline", SID, True),
    ("quoted_target", '"' + SID + '"', True),
    ("trailing_whitespace", SID + " ", True),
    ("extra_token_after_sid", SID + "<extra>", False),
    ("newline_inside", SID + "\n" + SID_X, False),
]
ok_c6 = True
for nm, comp, is_hit in c6_cases:
    exp_rule = [1.0] * G if is_hit else [0.0] * G
    ok_c6 &= run_case(f"C6_{nm}", [SID] * G, [comp] * G, exp_rule, [0.0] * G)
check("C6_summary", ok_c6, {"cases": len(c6_cases),
                            "note": "strip rules identical to original rl.py; EOS never appears in "
                                    "decoded text (skip_special_tokens); windows close on G=16 boundaries "
                                    "exactly as trainer batches 16 rows = one prompt group"})

# closed-form cross-check
manual_mid = -(1.0 / math.log2(7 + 2)) / sum(1.0 / math.log2(m + 2) for m in range(G))
check("rank_formula_closed_form", abs(NEG[7] - manual_mid) < 1e-15,
      {"NEG[7]": NEG[7], "manual": manual_mid})

doc = {"task": "Phase7 T3 manual reward controls", "date": "2026-11-15", "G": G,
       "functions_source": "scripts/v2_train_grpo.py (imported, not re-implemented)",
       "formula": {"a_j": "1/log2(j+2)", "rank_term_non_target": "-a_j/S",
                   "target_rank_term": 0.0,
                   "rule": "1 iff strip(\"\\n\\\" \")-match else 0; no-hit group -> rank all 0"},
       "results": results,
       "summary": {"checks": len(results),
                   "passed": sum(1 for r in results if r["ok"]),
                   "all_ok": all(r["ok"] for r in results)}}
out = "outputs/v2/evidence/phase7/reward_target_checks.json"
with open(out, "w") as f:
    json.dump(doc, f, ensure_ascii=False, indent=1)
print(f"REWARD_CHECKS {doc['summary']['passed']}/{len(results)}")
for r in results:
    print(("PASS" if r["ok"] else "FAIL"), r["control"])
print("reward_target_checks.json", hashlib.sha256(open(out, "rb").read()).hexdigest())
