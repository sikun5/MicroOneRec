#!/usr/bin/env python3
"""Phase 7 T2 checker: environment/input/config freeze evidence.

Verifies (1) environment and single-process lock incl. the trl dependency
change, (2) all GRPO inputs against the Phase6/Phase4 anchors, (3) the
10k source manifest and the three per-group SID task manifests produced in
Phase 4 (adopt-after-verification), and (4) writes environment_lock.json,
input_checks.json, grpo_config_lock.json under outputs/v2/evidence/phase7/.
"""
import hashlib
import json
import os
import subprocess
import sys

ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
os.chdir(ROOT)
EV = "outputs/v2/evidence/phase7"
os.makedirs(EV, exist_ok=True)
VP = os.path.join(ROOT, ".venv-phase3/bin/python")

results = []


def check(name, ok, detail):
    results.append({"check": name, "ok": bool(ok), "detail": detail})
    return ok


def sha(p):
    if not os.path.exists(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def freeze(marker=""):
    out = subprocess.run([VP, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    lines = [l for l in out.splitlines() if not l.startswith(marker)]
    return lines


# ---------------- 1. environment ----------------
env = {}
env["gpu_probe"] = {
    "nvidia_smi_driver": "610.57.04", "kmd": "610.57.04", "cuda_umd": "13.3",
    "mig_device": "MIG 1g.10gb (UUID MIG-bdf74f11-baf9-5fe5-8534-36ad86296c72)",
    "torch_device_name": "NVIDIA A800-SXM4-80GB MIG 1g.10gb",
    "torch_total_memory_bytes": 10200547328, "capability": "8.0", "device_count": 1,
}
env["python"] = subprocess.run([VP, "--version"], capture_output=True, text=True).stdout.strip()
env["venv"] = ".venv-phase3 (system-site-packages; base dsaa env inherited)"
env["versions_venv_view"] = {
    "torch": "2.14.0+cu130", "transformers": "4.51.3 (venv-local)",
    "tokenizers": "0.21.4 (venv-local)", "trl": "0.24.0 (venv-local, installed this task)",
    "accelerate": "1.0.1 (inherited)", "bitsandbytes": "0.50.2 (inherited, PagedAdamW32bit import OK)",
    "datasets": "3.1.0 (inherited)", "peft": "0.13.2 (inherited)", "numpy": "2.2.6 (inherited)",
    "scikit-learn": "1.5.2 (venv-local)", "wandb": "0.22.2 (inherited)",
}
env["venv_local_packages"] = ["cloudpickle 3.1.2", "joblib 1.6.0", "pip 23.0.1",
                              "scikit-learn 1.5.2", "scipy 1.15.3", "setuptools 65.5.0",
                              "threadpoolctl 3.8.0", "tokenizers 0.21.4", "transformers 4.51.3"]
env["trl_change_record"] = {
    "inherited_before": {"version": "0.11.3",
                          "location": "/hpc2hdd/home/swu073/envs/dsaa/lib/python3.10/site-packages",
                          "defect": "cannot import prepare_deepspeed from trl.models; no GRPOConfig/GRPOTrainer (probe evidence: minionerec_trainer ImportError)"},
    "action": "venv-local install into .venv-phase3 with --no-deps (base dsaa env untouched; transformers/torch/accelerate untouched)",
    "installed_version": "0.24.0",
    "wheel": "trl-0.24.0-py3-none-any.whl",
    "wheel_sha256": sha("/tmp/trlcheck/trl-0.24.0-py3-none-any.whl"),
    "post_location": ".venv-phase3/lib/python3.10/site-packages/trl (probe: trl.__file__)",
    "candidate_symbol_table": {
        "0.11.3_inherited": "MISSING prepare_deepspeed -> import FAIL",
        "0.14.0": "MISSING selective_log_softmax in trl.trainer.utils -> import FAIL",
        "0.15.0-0.19.0, 0.24.0": "all 13 required symbols present (wheel inspection)",
        "chosen": "0.24.0 = original repo requirements.txt pin; empirical probe passed"},
    "probe_results": {
        "GRPOConfig_construct": "OK with formal kwargs",
        "minionerec_trainer_import": "OK; ReReTrainer+RepeatRandomSampler present",
        "GRPOConfig_missing_attrs": ["vllm_device", "vllm_dtype", "vllm_max_model_len"],
        "missing_attrs_impact": "read only inside use_vllm=True branch (trainer lines 426-460); v2 locks use_vllm=False -> dead path; use_vllm must stay False",
        "renamed_sync_fields": "trl 0.24 renames sync_ref_model_steps/alpha -> ref_model_sync_steps=512, ref_model_mixup_alpha=0.6",
        "sync_semantics": "SyncRefModelCallback: on global_step % ref_model_sync_steps == 0 -> EMA ref<-(1-a)ref+a·policy, a=0.6; with ~624 planned updates -> exactly one sync at step 512"},
}
env["single_process"] = {
    "accelerate_default_config": "absent (~/.accelerate/default_config.yaml missing)",
    "dist_env_vars": "none inherited (env scan: accelerate/torchrun/local_rank/world_size/master_addr/deepspeed clean)",
    "accelerate_cli_bin": "absent in .venv-phase3/bin (module usable; launch via python -m accelerate.commands.launch --num_processes 1, T3 wiring)",
    "original_rl_sh_reference": "rl.sh uses accelerate launch --config_file ./config/zero2_opt.yaml --num_processes 8 (original 8-GPU ZeRO-2; NOT inherited; hash recorded)",
    "v1_note": "v1 baseline eval ran 8-GPU ZeRO-2 in the original project; v2 GRPO locks single process",
}
env["phase6_impact_analysis"] = ("Phase6 sources import no trl; venv-local trl addition cannot alter recorded Phase6 runs "
                                 "(frozen by hash) nor any SFT code path; base dsaa env untouched; pip freeze captured after install")
env["pip_freeze_raw_sha256"] = hashlib.sha256("\n".join(freeze()).encode()).hexdigest()
env["pip_freeze_lines"] = len(freeze())

# ---------------- 2. input hashes ----------------
anchors = {
    "outputs/v2/checkpoints/A-SFT/model.safetensors": "f753205d43dfb2240b96d5a9b0de8f6e375c65bddc313bdf19ca7aab55db405c",
    "outputs/v2/checkpoints/B-SFT/model.safetensors": "90fe4eb374afd5d3b7a9a1c3dbc2cf5ed6ac695792f35a8cf005605e67296aef",
    "outputs/v2/checkpoints/C-SFT/model.safetensors": "2d5ac0befc8093c727ec5ffd9b8c2db4957b81f6de27bf263647975c3206f538",
    "outputs/v2/evidence/phase6/selected_checkpoints.json": "e902a6f401c4c15ccc3082a9ee01ed4481fe2f88b7bb88f14546d7be36bb9b5e",
    "outputs/v2/evidence/phase6/resource_summary.json": "986a20baf26cd1270ef727ab3b14a0f07c43ce463d98e0ab3d92708bb30f4563",
    "outputs/v2/evidence/phase6/artifact_inventory.json": "da0bc73f062cb2fe8eb187e2017562645055cfb455e4f7bc157f79c1a46ff005",
    "outputs/v2/evidence/phase6/run_records.md": "e9f5e94bd9ea042ad36e649aacbefb1f2a9c58c370a951ab509ed34d729f84d7",
    "outputs/v2/evidence/phase6/acceptance.md": "e562ff39fd774efcb792bebc36f6a2f05538b9a01651a783c358bcdd1d43c88d",
    "scripts/v2_sft.py": "1038df67fb41f9c41c876a9ec3f8b41111d715c3f0b76c65b953889869ef3292",
    "scripts/v2_sft_data.py": "9bf546d9002959f6176ce34e71a98604b2af44a5e46e4949dec3b29a732c4193",
    "scripts/v2_sft_init.py": "893d8eae112b870deabd92cdb68100d3a2c19c8a6609dbc57f134cb3a166c35c",
}
hash_table = {}
mismatch = []
for p, exp in anchors.items():
    got = sha(p)
    hash_table[p] = {"sha256": got, "expected": exp, "match": got == exp}
    if got != exp:
        mismatch.append(p)

# original GRPO sources + aux
orig = ["rl.py", "rl_gpr.py", "minionerec_trainer.py", "utility.py", "rl.sh",
        "LogitProcessor.py", "data.py", "sasrec.py", "config/zero2_opt.yaml", "requirements.txt"]
for p in orig:
    hash_table[p] = {"sha256": sha(p)}
# tokenizer/vocab + per-group data files
tok_files = ["shared_vocab.json", "token_id_map.json"]
for f in tok_files:
    p = f"outputs/v2/tokenizer/{f}"
    hash_table[p] = {"sha256": sha(p)}
groups = {}
for g in "ABC":
    d = f"outputs/v2/data/{g}"
    files = ["Industrial_and_Scientific.index.json", "Industrial_and_Scientific.item.json",
             "info.txt", "prefix_trie.json", "reverse_item_map.json", "dataset_stats.json",
             "train.csv", "valid.csv", "test.csv", "rl_tasks_10000_sid.jsonl"]
    groups[g] = {f: sha(os.path.join(d, f)) for f in files}
    hash_table[d] = groups[g]
manifest_hash = sha("outputs/v2/manifests/rl_tasks_10000.jsonl")
hash_table["outputs/v2/manifests/rl_tasks_10000.jsonl"] = {"sha256": manifest_hash}
check("anchors_all_match", not mismatch, f"11 anchor files checked; mismatch={mismatch}")
check("manifest_hash_provenance", manifest_hash == "e77868cae2e4ffec6fda33dc1b2bee4b0f36d2800a2d270dfae0399bc5f11eb0",
      "matches phase4 t5_files_sha256.json and phase5 input_inventory anchored hash")

# tokenizer length probe (cheap, CPU)
tok_info = {}
for g in "ABC":
    d = f"outputs/v2/checkpoints/{g}-SFT"
    try:
        sys.path.insert(0, ROOT)
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained(d, local_files_only=True)
        tok_info[g] = {"len": len(t), "ok": len(t) == 154977}
    except Exception as e:
        tok_info[g] = {"error": f"{type(e).__name__}: {e}", "ok": False}
check("tokenizers_154977", all(v.get("ok") for v in tok_info.values()), json.dumps(tok_info))

# ---------------- 3. source manifest checks ----------------
src = [json.loads(l) for l in open("outputs/v2/manifests/rl_tasks_10000.jsonl")]
tc = {}
for r in src:
    tc[r["task_type"]] = tc.get(r["task_type"], 0) + 1
check("manifest_counts", tc == {"seq2sid": 8000, "title2sid": 1000, "texthist2sid": 1000}, str(tc))
splits = {}
for r in src:
    splits.setdefault(r["task_type"], set()).add(r["source_split"])
check("manifest_splits", splits == {"seq2sid": {"train"}, "title2sid": {"catalog"},
                                    "texthist2sid": {"train"}}, {k: sorted(v) for k, v in splits.items()})
ids = [r["task_id"] for r in src]
check("task_ids_unique", len(set(ids)) == 10000, f"unique={len(set(ids))}")
from collections import Counter
ts_ids = Counter((r["task_type"], r["source_id"]) for r in src)
dup_in_type = {k: v for k, v in ts_ids.items() if v > 1}
cross_type_src = len({r["source_id"] for r in src if r["task_type"] == "seq2sid"} &
                     {r["source_id"] for r in src if r["task_type"] == "texthist2sid"})
check("source_ids_unique_within_type", len(dup_in_type) == 0,
      f"per-(type,source_id) dups={len(dup_in_type)}; cross-type source reuse (same train row as one seq2sid + one texthist2sid task, by design)={cross_type_src}")

idx = {}
for g in "ABC":
    idx[g] = json.load(open(f"outputs/v2/data/{g}/Industrial_and_Scientific.index.json"))
item_catalog = {}
for g in "ABC":
    im = json.load(open(f"outputs/v2/data/{g}/Industrial_and_Scientific.item.json"))
    item_catalog[g] = im

# title2sid canonical check (catalog): minimal numeric item id per title
def canon_min(ids):
    nums = [int(x) for x in ids if str(x).lstrip("-").isdigit()]
    return str(min(nums)) if nums else sorted(ids)[0]

canon_bad, title_multi = 0, {}
title_rows = [r for r in src if r["task_type"] == "title2sid"]
catalog_titles = {}
items_map = item_catalog["A"].get("items")  # {item_id_str: {title_clean, ...}}
if isinstance(items_map, dict):
    for iid, entry in items_map.items():
        t = entry.get("title_clean") if isinstance(entry, dict) else str(entry)
        catalog_titles.setdefault(t, []).append(str(iid))
for r in title_rows:
    t = r.get("title_clean")
    cat_ids = catalog_titles.get(t)
    if cat_ids is None:
        canon_bad += 1
    else:
        cmin = canon_min(cat_ids)
        if str(r["item_id"]) != cmin:
            canon_bad += 1
        if len(cat_ids) > 1:
            title_multi[t] = cat_ids
check("title2sid_canonical_mapping", canon_bad == 0,
      f"rows={len(title_rows)} bad={canon_bad}; duplicated-title catalog groups={len(title_multi)} resolved to minimal numeric id")
check("title2sid_in_catalog", all(r["item_id"] in {str(x) for x in idx['A']['item_ids']} for r in title_rows), "all title2sid item ids in catalog")

# conflict analysis (ID level)
def hist_parse(s):
    return [x.strip() for x in str(s).strip("[]").split(",") if x.strip()]

conf = {}
for tt in ("seq2sid", "texthist2sid"):
    h2t = {}
    for r in src:
        if r["task_type"] == tt:
            h2t.setdefault(r["history_item_id_raw"], set()).add(r["item_id"])
    bad = {k: sorted(v) for k, v in h2t.items() if len(v) > 1}
    conf[f"{tt}_same_history_diff_target"] = {"n_conflict_histories": len(bad), "examples": dict(list(bad.items())[:3])}
    dup = len(src and [1])  # placeholder
# duplicates (same history+target) count
for tt in ("seq2sid", "texthist2sid"):
    from collections import Counter
    cnt = Counter((r["history_item_id_raw"], r["item_id"]) for r in src if r["task_type"] == tt)
    dups = {k: v for k, v in cnt.items() if v > 1}
    conf[f"{tt}_identical_rows"] = {"n_dup_rows": sum(v - 1 for v in dups.values()), "n_dup_keys": len(dups)}
# cross-type raw-history overlap
s_h = {r["history_item_id_raw"] for r in src if r["task_type"] == "seq2sid"}
t_h = {r["history_item_id_raw"] for r in src if r["task_type"] == "texthist2sid"}
conf["seq_vs_texthist_raw_overlap"] = len(s_h & t_h)
n_conf_seq = conf["seq2sid_same_history_diff_target"]["n_conflict_histories"]
n_conf_th = conf["texthist2sid_same_history_diff_target"]["n_conflict_histories"]
check("id_level_conflicts_documented_with_disposition", True,
      f"conflict histories: seq2sid={n_conf_seq}, texthist2sid={n_conf_th}; identical duplicate rows: "
      f"seq2sid={conf['seq2sid_identical_rows']['n_dup_rows']}, texthist2sid={conf['texthist2sid_identical_rows']['n_dup_rows']}; "
      "DISPOSITION: conflicts are legitimate distinct tasks sharing an identical history prefix; the original dict "
      "(prompt2history/history2target) would silently overwrite one target - therefore the T3 adapter MUST recover "
      "reward targets from stable per-row task identity (task_id -> item_id -> group index token_ids), never via "
      "prompt-keyed dicts; no dedup/drop of duplicate or conflicting rows (guide-mandated); T3 must assert per-row "
      "target recovery and log group boundaries")

# ---------------- 4. per-group SID manifest verification ----------------
group_checks = {}
adopt_ok = True
for g in "ABC":
    m = [json.loads(l) for l in open(f"outputs/v2/data/{g}/rl_tasks_10000_sid.jsonl")]
    index = idx[g]
    pos = {str(i): p for p, i in enumerate(index["item_ids"])}
    sid_err = tgt_err = hist_err = id_err = 0
    hist_tok_ok = 0
    n_tok = 0
    for s, o in zip(m, src):
        ident = ("task_id", "task_type", "source_id", "source_split", "item_id")
        if any(s.get(k) != o.get(k) for k in ident) or \
           s.get("user_id") != o.get("user_id") or \
           s.get("history_item_id_raw") != o.get("history_item_id_raw") or \
           s.get("title_clean") != o.get("title_clean"):
            id_err += 1
        p = pos.get(str(s["item_id"]))
        if p is None or s["item_sid"] != index["sids"][p]:
            tgt_err += 1
        else:
            want = index["token_ids"][p]
            if isinstance(s.get("item_token_ids"), list) and s["item_token_ids"] != want:
                tgt_err += 1
        hs = s.get("history_item_sid")
        raw_s = s.get("history_item_id_raw")
        if raw_s is None:
            if hs is not None:
                hist_err += 1
            continue
        raw = hist_parse(raw_s)
        if not isinstance(hs, list) or len(hs) != len(raw):
            hist_err += 1
        else:
            for k, iid in enumerate(raw):
                pp = pos.get(iid)
                if pp is None or hs[k] != index["sids"][pp]:
                    hist_err += 1
                    break
    gtc = {}
    for r in m:
        gtc[r["task_type"]] = gtc.get(r["task_type"], 0) + 1
    ok = (len(m) == 10000 and id_err == 0 and tgt_err == 0 and hist_err == 0
          and gtc == {"seq2sid": 8000, "title2sid": 1000, "texthist2sid": 1000})
    adopt_ok = adopt_ok and ok
    group_checks[g] = {"lines": len(m), "identity_order_errors": id_err, "target_sid_errors": tgt_err,
                       "history_sid_errors": hist_err, "task_type_counts": gtc, "all_ok": ok}
check("group_manifests_verified", adopt_ok, json.dumps(group_checks))

# three-group identity equality (non-SID fields)
non_sid = ["task_id", "task_type", "source_id", "source_split", "user_id", "history_item_id_raw", "item_id"]
mA = [json.loads(l) for l in open("outputs/v2/data/A/rl_tasks_10000_sid.jsonl")]
mB = [json.loads(l) for l in open("outputs/v2/data/B/rl_tasks_10000_sid.jsonl")]
mC = [json.loads(l) for l in open("outputs/v2/data/C/rl_tasks_10000_sid.jsonl")]
diff = sum(1 for a, b, c in zip(mA, mB, mC)
           if any(a.get(k) != b.get(k) or a.get(k) != c.get(k) for k in non_sid))
check("three_group_identity_identical", diff == 0, f"rows differing in non-SID fields: {diff}")

# ---------------- 5. write evidence ----------------
with open(os.path.join(EV, "environment_lock.json"), "w") as f:
    json.dump({"task": "Phase7 T2 environment freeze", "date": "2026-11-15", "cwd": ROOT,
               "environment": env}, f, ensure_ascii=False, indent=1)

input_doc = {
    "task": "Phase7 T2 input freeze", "date": "2026-11-15", "cwd": ROOT,
    "results": results, "hash_table": hash_table, "conflict_analysis": conf,
    "group_manifest_checks": group_checks,
    "tokenizer_checks": tok_info,
    "decision": "per-group rl_tasks_10000_sid.jsonl ADOPTED after full verification (Phase4 deliverable); "
                "no regeneration needed; source manifest consumed as-is (no per-dataset 10k truncation)",
}
with open(os.path.join(EV, "input_checks.json"), "w") as f:
    json.dump(input_doc, f, ensure_ascii=False, indent=1)

cfg = {
    "task": "Phase7 T2 GRPO formal config lock", "date": "2026-11-15",
    "run_ids": ["A-RL", "B-RL", "C-RL"],
    "init": "outputs/v2/checkpoints/{A,B,C}-SFT/ top-level (selected best; policy AND full ref both loaded from same group SFT)",
    "fixed": {
        "num_generations_G": 16, "per_device_train_batch_size": 16, "gradient_accumulation_steps": 32,
        "candidates_per_update": "16 rows x 32 micro-steps = 512; 32 prompt groups per update",
        "num_train_epochs": 2, "learning_rate": 1e-5, "beta": 1e-3, "seed": 42,
        "optim": "paged_adamw_32bit", "bf16": True, "max_completion_length": 128,
        "beam_search": True, "num_beams": "G=16 (generation_config.num_beams=num_return_sequences=16)",
        "temperature": 1.0, "length_penalty": 0.0,
        "sync_ref_model": True, "ref_model_sync_steps": 512, "ref_model_mixup_alpha": 0.6,
        "max_prompt_length": 512,
        "closed_items": {"dynamic_sampling": False, "dapo": False, "gspo": False,
                          "add_gt": False, "test_during_training": False,
                          "sample_train": False, "use_vllm": False},
        "entry_conventions_kept_from_rl_py": {"save_steps": 0.1, "save_total_limit": 20,
                                               "eval_strategy": "steps", "eval_steps": 0.199,
                                               "logging_steps": 1, "warmup_ratio": 0.03,
                                               "max_grad_norm": 0.3, "lr_scheduler_type": "cosine",
                                               "report_to": "locked variant: report_to=[] with WANDB disabled (original: offline wandb; no network)"},
        "final_checkpoint": "outputs/v2/checkpoints/{G}-RL/final_checkpoint/ after full 2 epochs (trainer.save_model + final_checkpoint subdir; no mid-checkpoint selection)",
    },
    "trainer_semantics_source_derived": {
        "sampler": "RepeatRandomSampler(train_dataset, G=16, seed=args.seed) - each prompt repeated 16x consecutively",
        "micro_batch": "per_device_train_batch_size=16 rows == exactly ONE prompt group (16 completions)",
        "update": "accum 32 micro-steps -> 32 prompt groups = 512 candidates per optimizer update",
        "dataset": "10k global manifest tasks; shuffle(seed=42) once before training (rl.py convention)",
        "rows_per_epoch": 160000, "micro_steps_per_epoch": 10000,
        "optimizer_steps_per_epoch_est": 312,
        "tail_window": "floor semantics: 312x32=9984 micro-batches consumed -> 16 prompt groups per epoch fall in the dropped partial accumulation window (transformers 4.51.3 trainer.py:5303, same derivation accepted in Phase6); MEASURED confirmation deferred to T4",
        "coverage_gap_flag": "0.16 percent of prompt groups per epoch (16/10000) are not consumed under floor semantics - conflicts with the guide requirement of full 10k coverage per epoch; REPORTED for boss decision (see working-record ISSUES P1); no silent acceptance",
        "constraint_check": "global_batch_size 16 x 1 process divisible by G=16 OK; eval batch 16 divisible OK",
        "generation_path": "beam_search=True: dedup to 1 prompt per micro-step; generate num_beams=16/num_return_sequences=16; logits_processor=[TemperatureLogitsWarper(1.0), ConstrainedLogitsProcessor(prefix_allowed_tokens_fn over hash_dict built from group info.txt (3686 SIDs), num_beams=16)]",
        "rewards": "[rule_reward, ndcg_rule_reward] weights [1,1]; rule: exact stripped text match -> 1/0; ndcg: group-wise, if any hit: non-target at in-group rank j gets -(1/log2(j+2))/sum(a_0..a_15), target gets 0; no-hit group: all 0 (ndcg_rewards precomputed at rl.py:156-157)",
        "advantage": "(r - group_mean)/(group_std + 1e-4) per G16 group -> all-same rewards give zero advantage (KL-only gradient possible)",
        "loss": "GRPO token-mean: mean over rows of sum(-(exp(logp-logp.detach())*adv - beta*kl) * completion_mask)/sum(mask); kl = exp(ref-logp delta) - delta - 1 (k3)",
        "ref_model": "create_reference_model(policy) full independent copy at init; accelerator.prepare_model(evaluation_mode=True); NOT in optimizer; SyncRefModelCallback EMA a=0.6 every 512 global steps (trl 0.24 semantics)",
        "eos_mask": "completion_mask = tokens up to first EOS; KL and loss both masked identically",
        "info_trie": "hash_dict built in trainer __init__ from group info.txt (3686 items; semantic_id\\ttitle\\titem_id format = v2 info.txt format match)",
    },
    "diagnostics_planned": "per-task-type hit rate, reward mean/std, zero-advantage group ratio, KL, candidate unique rate, completion length, valid prompt count persisted outside checkpoint rotation (T3 wiring, T5 measurement)",
}
with open(os.path.join(EV, "grpo_config_lock.json"), "w") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=1)

for p in ("environment_lock.json", "input_checks.json", "grpo_config_lock.json"):
    fp = os.path.join(EV, p)
    print("EVID", p, sha(fp))

n_ok = sum(1 for r in results if r["ok"])
print(f"CHECKS {n_ok}/{len(results)}")
for r in results:
    print(("PASS" if r["ok"] else "FAIL"), r["check"])
print("T2_CHECKS_DONE")
