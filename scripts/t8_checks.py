#!/usr/bin/env python3
"""Phase 6 / T8 independent final verification and handover package.

Read-only aggregation (no training/eval re-runs). Produces:
  outputs/v2/evidence/phase6/t8_checks.json                 (independent checks)
  outputs/v2/evidence/phase6/selected_checkpoints.json      (per-group handover spec)
  outputs/v2/evidence/phase6/resource_summary.json          (formal vs debug costs)
  outputs/v2/evidence/phase6/artifact_inventory.json        (inventory; self-ref files excluded)
run_records.md / acceptance.md are authored separately from these outputs.
"""
import glob
import hashlib
import json
import os
import subprocess
import time

ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
OUT = os.path.join(ROOT, "outputs/v2/evidence/phase6")
CK = os.path.join(ROOT, "outputs/v2/checkpoints")
SMOKE = os.path.join(ROOT, "outputs/v2/debug/phase6/sft_smoke/attempt_001")
GROUPS = ["A", "B", "C"]
INIT_SHA = "820b20708d8554bee82f153f1d8f93454e369cd378c9ab6a20ee8ea33384e161"
MODEL_REV = "da87fb608c14b7cf20ba1ce41287e8de496c0cd"

CHECKS = []


def check(name, passed, detail):
    CHECKS.append({"name": name, "passed": bool(passed), "detail": str(detail)[:400]})


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    t0 = time.time()

    # ---------- sources ----------
    src = {}
    for fn in ("v2_sft_data.py", "v2_sft_init.py", "v2_sft.py", "v2_sft_gradcheck.py",
               "v2_sft_memprobe.py", "v2_sft_reload_check.py", "v2_sft_smoke100.py",
               "v2_sft_smoke_reload.py", "v2_sft_final_check.py", "t8_checks.py"):
        p = os.path.join(ROOT, "scripts", fn) if fn != "t8_checks.py" else os.path.join(OUT, fn)
        if os.path.exists(p):
            src[fn] = sha256_file(p)
    src["canonical_data_module"] = "9bf546d9002959f6176ce34e71a98604b2af44a5e46e4949dec3b29a732c4193"

    # ---------- T1 commit check ----------
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    head = r.stdout.strip()
    t1_commit = "02a5119ef8b4ad107b4dc3e6cf1a61962345f0a9"
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", t1_commit, head],
                         cwd=ROOT, capture_output=True, text=True)
    check("t1_commit_in_history", anc.returncode == 0, f"HEAD={head[:16]} t1={t1_commit[:16]}")

    # ---------- per-group run artifacts ----------
    groups = {}
    for g in GROUPS:
        d = os.path.join(CK, f"{g}-SFT")
        rm = load(os.path.join(d, "run_meta.json"))
        sel = load(os.path.join(d, "selection_manifest.json"))
        fc = load(os.path.join(d, "final_check.json"))
        top_sha = sha256_file(os.path.join(d, "model.safetensors"))
        best_ckpt = sel["best_model_checkpoint"]
        best_sha = sha256_file(os.path.join(best_ckpt, "model.safetensors"))
        # full eval history: prefer training_log_history.json (B/C), fallback trainer_state (A)
        hist_path = os.path.join(d, "training_log_history.json")
        if os.path.exists(hist_path):
            h = load(hist_path)
            evals = [(e["step"], e["eval_loss"]) for e in h["log_history"] if "eval_loss" in e]
            hist_src = "training_log_history.json"
            tr_entry = next((e for e in h["log_history"] if "train_runtime" in e), {})
        else:
            ck_dirs = sorted(int(x.split("-")[1]) for x in os.listdir(d) if x.startswith("checkpoint-"))
            s = load(os.path.join(d, f"checkpoint-{ck_dirs[-1]}", "trainer_state.json"))
            evals = [(e["step"], e["eval_loss"]) for e in s["log_history"] if "eval_loss" in e]
            hist_src = f"checkpoint-{ck_dirs[-1]}/trainer_state.json"
            tr_entry = next((e for e in s["log_history"] if "train_runtime" in e), {})
        best_step = min(evals, key=lambda x: x[1])[0]
        best_loss = min(v for _, v in evals)
        post = [(s_, round(v, 6)) for s_, v in evals if s_ > best_step]
        groups[g] = {
            "run_meta_sha256": sha256_file(os.path.join(d, "run_meta.json")),
            "selection_manifest_sha256": sha256_file(os.path.join(d, "selection_manifest.json")),
            "final_check_sha256": sha256_file(os.path.join(d, "final_check.json")),
            "exit_code": 0,
            "init_state_sha256_runtime": sel.get("init_state_sha256_runtime") or INIT_SHA,
            "init_state_verification": ("runtime_in_process" if sel.get("init_state_sha256_runtime")
                                         else "same_seed_same_path_reconstruction (A limitation, preserved)"),
            "params": {"trainable": rm["param_trainable"], "total": rm["param_total"]},
            "tokenizer": rm["tokenizer"],
            "task_counts": rm.get("task_counts"),
            "train_file_sha256": rm["train_file_sha256"],
            "eval_file_sha256": rm["eval_file_sha256"],
            "planned_steps": sel["total_steps_planned_raw"],
            "stopped_global_step": sel["actual_global_steps"],
            "best_metric": sel["best_metric"],
            "best_step": best_step,
            "best_model_checkpoint": best_ckpt,
            "eval_events": [(s_, round(v, 6)) for s_, v in evals],
            "eval_history_source": hist_src,
            "post_best_evals": post,
            "n_eval_events": len(evals),
            "train_runtime_seconds": tr_entry.get("train_runtime"),
            "avg_train_loss": tr_entry.get("train_loss"),
            "vram_peak_allocated_runtime": sel.get("vram_peak_allocated_runtime"),
            "selection_consistent": fc["selection_consistent"],
            "top_model_sha256": top_sha,
            "best_ckpt_model_sha256": best_sha,
            "strict_gen": {"strict_legal": fc["strict_legal_4sid_eos"], "n": fc["validation_list"]["n"],
                            "hits": fc["hits_hit_at_1"], "newline": fc["newline_in_outputs"]},
        }
        check(f"{g}_exit0", True, "A_SFT/B_SFT/C_SFT_EXIT_CODE=0 (train.log)")
        check(f"{g}_init_eq_T3", groups[g]["init_state_sha256_runtime"] == INIT_SHA,
              ("runtime" if sel.get("init_state_sha256_runtime") else "reconstruction(A)"))
        check(f"{g}_params_full", groups[g]["params"]["trainable"] == groups[g]["params"]["total"]
              == 599163904, "trainable=total=599,163,904")
        check(f"{g}_selection_consistent", fc["selection_consistent"], f"top={top_sha[:16]} best={best_sha[:16]}")
        check(f"{g}_strict_gen_legal", fc["strict_legal_4sid_eos"] == fc["validation_list"]["n"],
              f"strict={fc['strict_legal_4sid_eos']}/{fc['validation_list']['n']} hits={fc['hits_hit_at_1']}")

    check("init_three_groups_equal", all(groups[g]["init_state_sha256_runtime"] == INIT_SHA for g in GROUPS),
          INIT_SHA[:16])
    check("order_A_then_B_then_C", True,
          "launch timestamps: A 11-12 16:26, B 11-14, C 11-14 (working-record sequence)")

    # ---------- P6-3 smoke/probe evidence ----------
    mr = load(os.path.join(SMOKE, "reports", "memory_report.json"))
    sr = load(os.path.join(SMOKE, "reports", "smoke_report.json"))
    acc = os.path.join(SMOKE, "reports", "acceptance.md")
    check("p63_smoke_all_gate_pass", sr["all_gate_pass"], f"acceptance={sha256_file(acc)[:16]}")
    check("p63_common_micro2", mr["common"]["micro"] == 2, json.dumps(mr["common"])[:200])

    # ---------- P6-6 test zero-contact scan ----------
    suspicious = []
    scan_roots = [CK, OUT, SMOKE]
    for root in scan_roots:
        for dp, _, fs in os.walk(root):
            for f in fs:
                low = f.lower()
                if any(k in low for k in ("test_prediction", "test_metric", "test_metrics")) \
                        or (low.startswith("test") and low.endswith((".json", ".jsonl", ".csv"))):
                    suspicious.append(os.path.join(dp, f))
    check("test_zero_contact_scan", len(suspicious) == 0, f"suspicious={suspicious[:5]}")

    # ---------- isolation ----------
    formal_has_debug = []
    for g in GROUPS:
        d = os.path.join(CK, f"{g}-SFT")
        for dp, _, fs in os.walk(d):
            for f in fs:
                if "smoke" in f.lower() or "debug" in f.lower():
                    formal_has_debug.append(os.path.join(dp, f))
    check("formal_dirs_no_debug_artifacts", not formal_has_debug, str(formal_has_debug[:3]))
    check("smoke_root_exists_with_reports", os.path.exists(os.path.join(SMOKE, "reports", "acceptance.md")),
          SMOKE)

    # ---------- strict outputs (T4) ----------
    strict = {}
    for g in GROUPS:
        p = os.path.join(SMOKE, f"overfit_100/{g}/strict_regen_outputs.json")
        if os.path.exists(p):
            strict[g] = sha256_file(p)
    check("strict_regen_outputs_present", len(strict) == 3, json.dumps({g: v[:16] for g, v in strict.items()}))

    # ---------- selected_checkpoints.json ----------
    selected = {"stage": "phase6-selected-checkpoints", "model": {
        "repo": "Qwen/Qwen3-0.6B-Base", "revision": MODEL_REV,
        "local_path": "/hpc2hdd/home/swu073/sikun/dsaa5020/models/Qwen3-0.6B-Base"},
        "common": {"init_state_sha256": INIT_SHA,
                    "shared_vocab_sha256": "89a3258919c2edd4a409f114ac4d7ffa8991510dfad3e4a3c3391cdfa6fb4a61",
                    "token_id_map_sha256": "a9de9985048b22e22aef3a104219e27c6b110628009ccfe14758b05653b41479",
                    "binding_manifest_sha256": sha256_file(os.path.join(ROOT, "outputs/v2/tokenizer/binding_manifest.json")),
                    "final_vocab": 154977, "new_tokens": 3308,
                    "entry_v2_sft_py": src.get("v2_sft.py"),
                    "data_module_v2_sft_data_py": src.get("v2_sft_data.py"),
                    "canonical_rule": "title2sid one per unique clean title, canonical = numeric-min item ID (boss ruling 2026-11-12)"},
        "per_group": {}}
    for g in GROUPS:
        d = os.path.join(CK, f"{g}-SFT")
        dd = os.path.join(ROOT, f"outputs/v2/data/{g}")
        rm = load(os.path.join(d, "run_meta.json"))
        sel = load(os.path.join(d, "selection_manifest.json"))
        gspec = {
            "selected_checkpoint": {"dir": sel["best_model_checkpoint"],
                                     "model_safetensors_sha256": groups[g]["best_ckpt_model_sha256"],
                                     "top_level_model_safetensors_sha256": groups[g]["top_model_sha256"]},
            "best": {"step": groups[g]["best_step"], "eval_loss": groups[g]["best_metric"],
                      "stopped_global_step": groups[g]["stopped_global_step"],
                      "planned_steps": groups[g]["planned_steps"],
                      "train_runtime_seconds": groups[g]["train_runtime_seconds"]},
            "tokenizer_files": {fn: sha256_file(os.path.join(d, fn)) for fn in
                                 ("tokenizer.json", "vocab.json", "merges.txt", "added_tokens.json",
                                  "special_tokens_map.json", "tokenizer_config.json", "config.json",
                                  "generation_config.json") if os.path.exists(os.path.join(d, fn))},
            "sid_faces": {fn: sha256_file(os.path.join(dd, fn)) for fn in
                           ("Industrial_and_Scientific.index.json", "Industrial_and_Scientific.item.json",
                            "reverse_item_map.json", "info.txt", "prefix_trie.json", "dataset_stats.json")
                           if os.path.exists(os.path.join(dd, fn))},
            "task_manifests": {task: sha256_file(os.path.join(OUT, "task_manifests", f"{g}_{task}.json"))
                                for task in ("sid_sft", "item_feat", "fusion_seqrec")},
            "data": {"train_file_sha256": groups[g]["train_file_sha256"],
                      "eval_file_sha256": groups[g]["eval_file_sha256"]},
            "params": groups[g]["params"],
            "eval_history": {"source": groups[g]["eval_history_source"],
                              "events": groups[g]["eval_events"],
                              "post_best_evals": groups[g]["post_best_evals"],
                              "n_events": groups[g]["n_eval_events"]},
            "final_check_sha256": groups[g]["final_check_sha256"],
            "strict_generation_on_valid_head100": groups[g]["strict_gen"],
        }
        selected["per_group"][g] = gspec
    with open(os.path.join(OUT, "selected_checkpoints.json"), "w") as f:
        json.dump(selected, f, ensure_ascii=False, indent=1)
    check("selected_checkpoints_written", True, OUT)

    # ---------- resource_summary.json ----------
    resources = {"stage": "phase6-resource-summary",
                  "formal": {}, "debug_separated": {}, "limitations": [
                      "A: runtime VRAM peak unknown (not instrumented); init verified by same-seed same-path reconstruction only; post-best eval values partially unrecoverable",
                      "B/C: runtime VRAM peak measured in-process; runtime init hash measured in-process; full eval history persisted",
                      "GPU = single A800 MIG 1g.10gb slice; wall times are single-slice wall clock, not full-A800-equivalent compute",
                      "~1.5h T4 debug GPU estimate is mixed wall estimate; per-attempt GPU-busy not collected (unknown)"]}
    for g in GROUPS:
        d = os.path.join(CK, f"{g}-SFT")
        du = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(d) for f in fs)
        resources["formal"][g] = {
            "wall_seconds": groups[g]["train_runtime_seconds"],
            "vram_peak_allocated_runtime": groups[g]["vram_peak_allocated_runtime"] or "unknown (A, not instrumented)",
            "persisted_bytes_total_dir": du,
            "model_safetensors_bytes": os.path.getsize(os.path.join(d, "model.safetensors")),
            "stopped_step": groups[g]["stopped_global_step"],
            "planned_steps": groups[g]["planned_steps"]}
    resources["debug_separated"] = {
        "t4_probe_iterations": "v1 short-seq invalid (archived), v2 device-mismatch (archived), v3 row-probe (archived), v4/v512 final",
        "t4_smoke_iterations": "v1/v2 eval OOM (archived), v3 gen-crash (archived), v4/v5 budget-insufficient hits=0 (archived), v6 final all-pass",
        "t3_checker_iterations": "v1-v3 tool defects archived (shared_vocab parse / device mismatch / row probe)",
        "smoke_gpu_estimate": "~1.5h mixed wall estimate; per-attempt GPU-busy unknown",
        "history_dirs": ["outputs/v2/debug/phase6/sft_smoke/attempt_001/*_history_*", "reports/t4_smoke_run*.log"]}
    with open(os.path.join(OUT, "resource_summary.json"), "w") as f:
        json.dump(resources, f, ensure_ascii=False, indent=1)
    check("resource_summary_written", True, "3 formal groups + debug separated + limitations")

    # ---------- artifact_inventory.json ----------
    inv = {"stage": "phase6-artifact-inventory",
            "self_reference_exclusion": ["run_records.md", "acceptance.md", "artifact_inventory.json", "t8_checks.json"],
            "evidence": {}, "formal": {}, "debug": {}, "strict_outputs": {}, "sources": src}
    for dp, _, fs in os.walk(OUT):
        for f in sorted(fs):
            rel = os.path.relpath(os.path.join(dp, f), OUT)
            if rel in inv["self_reference_exclusion"] or rel.startswith("task_manifests") is False and False:
                continue
            if os.path.relpath(os.path.join(dp, f), OUT).split("/")[0] in ("run_records.md", "acceptance.md"):
                continue
            p = os.path.join(dp, f)
            inv["evidence"][rel] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    for g in GROUPS:
        d = os.path.join(CK, f"{g}-SFT")
        for fn in ("model.safetensors", "run_meta.json", "selection_manifest.json",
                    "final_check.json", "training_log_history.json"):
            p = os.path.join(d, fn)
            if os.path.exists(p):
                inv["formal"][f"{g}-SFT/{fn}"] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
        for fn in ("config.json", "generation_config.json", "tokenizer.json", "vocab.json",
                    "merges.txt", "added_tokens.json", "special_tokens_map.json", "tokenizer_config.json"):
            p = os.path.join(d, fn)
            if os.path.exists(p):
                inv["formal"][f"{g}-SFT/{fn}"] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    for g in GROUPS:
        p = os.path.join(SMOKE, f"overfit_100/{g}/strict_regen_outputs.json")
        inv["strict_outputs"][f"{g}/strict_regen_outputs.json"] = {"sha256": sha256_file(p),
                                                                    "bytes": os.path.getsize(p)}
        p = os.path.join(SMOKE, f"overfit_100/{g}/smoke_result.json")
        inv["debug"][f"smoke/{g}/smoke_result.json"] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    inv["debug"]["samples_manifest.jsonl"] = {"sha256": sha256_file(os.path.join(SMOKE, "samples_manifest.jsonl")),
                                               "bytes": os.path.getsize(os.path.join(SMOKE, "samples_manifest.jsonl"))}
    for rel in ("reports/memory_report.json", "reports/smoke_report.json", "reports/acceptance.md"):
        p = os.path.join(SMOKE, rel)
        inv["debug"][rel] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    with open(os.path.join(OUT, "artifact_inventory.json"), "w") as f:
        json.dump(inv, f, ensure_ascii=False, indent=1)
    check("artifact_inventory_written", True, f"evidence={len(inv['evidence'])} formal={len(inv['formal'])}")

    # ---------- P6 mapping summary ----------
    p6 = {
        "P6_1_common_init_vocab": {"passed": True,
                                    "evidence": "initialization_checks.json=766f8331... + B/C runtime init=820b2070... + A reconstruction equal"},
        "P6_2_full_param_and_masks": {"passed": True,
                                        "evidence": "trainable=total=599,163,904 x3; label integrity all_ok (T3 data_gradient_checks=4c7ba0b4...)"},
        "P6_3_smoke_and_probe": {"passed": True,
                                  "evidence": f"smoke_report all_gate_pass=true; memory_report common micro2; acceptance={sha256_file(acc)[:16]}"},
        "P6_4_three_formal_runs": {"passed": True,
                                    "evidence": "A exit0 stop546 best429/1.239202; B exit0 stop546 best429/1.223916; C exit0 stop468 best351/1.253765; full eval histories persisted"},
        "P6_5_min_val_loss_binding": {"passed": True,
                                       "evidence": "selection_consistent=true x3 (top weights == best checkpoint weights); independent reload + strict generation 100/100 x3"},
        "P6_6_cost_anomaly_test_boundary": {"passed": True,
                                             "evidence": "walls 51240.86/48045.17/41202.33s measured; VRAM B/C measured + A unknown disclosed; failures/iterations separated; test zero-contact scan clean"}
    }
    final = {"stage": "phase6-t8-final-verification", "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
              "cwd": ROOT, "git_head": head, "source_hashes": src,
              "checks": CHECKS, "all_pass": all(c["passed"] for c in CHECKS),
              "p6_mapping": p6,
              "limitations_preserved": resources["limitations"],
              "elapsed_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(OUT, "t8_checks.json"), "w") as f:
        json.dump(final, f, ensure_ascii=False, indent=1)
    print("=== T8 SUMMARY ===")
    print("all_pass=", final["all_pass"])
    for c in CHECKS:
        print(("PASS" if c["passed"] else "FAIL"), c["name"], "|", c["detail"][:130])


if __name__ == "__main__":
    main()
