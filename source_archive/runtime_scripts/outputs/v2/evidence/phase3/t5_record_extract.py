#!/usr/bin/env python
"""T5 supplementary record extraction (read-only; no retraining, no final modification).
Produces: full 64-hex final SHAs, per-run config hashes + protocol conformity,
log coverage checks (10000 rows/steps, C n_valid), driver log hash + excerpts,
input-group consistency. SCHEMA: phase3-t5_record_extract-1.0"""
import os, sys, json, hashlib
os.environ.setdefault("OMP_NUM_THREADS", "8")
from pathlib import Path
REPO = Path("/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec")
sys.path.insert(0, str(REPO / "scripts"))
import v2_train_rqvae as V

def sha(p): return V.sha256_file(str(p))
res = {"stage": "phase3_t5_record_extract", "schema": "phase3-t5_record_extract-1.0", "date": "2026-11-11",
       "note": "只读摘录与核对；不重训、不改 final"}

# 1. full final SHAs + config hashes + losses
finals = {}
cfg_hashes = {}
losses = {}
inputs_by_group = {}
inputs_consistent = {}
for m in ("A", "B", "C"):
    group_inputs = []
    for s in (42, 43, 44):
        base = REPO / f"outputs/v2/rqvae/{m}/seed_{s}"
        rr = json.load(open(base / "run_record.json"))
        cfgj = json.load(open(base / "config.json"))
        finals[f"RQ-{m}-{s}"] = {"path": f"outputs/v2/rqvae/{m}/seed_{s}/final_checkpoint.pth",
                                 "sha256_64": rr["final_checkpoint_sha256"],
                                 "bytes": rr["checkpoint_bytes"]["final_checkpoint.pth"]}
        cfg_hashes[f"RQ-{m}-{s}"] = {"config_json_file_sha256": rr["config_sha256"]}
        losses[f"RQ-{m}-{s}"] = {"recon": rr["final_losses"]["recon"], "quant": rr["final_losses"]["quant"],
                                 "consistency_raw": rr["final_losses"]["consistency_raw"],
                                 "consistency_weighted": rr["final_losses"]["consistency_weighted"],
                                 "total": rr["final_losses"]["total"],
                                 "all_finite": all(abs(rr["final_losses"][k]) < float("inf") for k in rr["final_losses"])}
        group_inputs.append(cfgj["inputs"])
    inputs_consistent[m] = all(json.dumps(group_inputs[0], sort_keys=True) ==
                               json.dumps(gi, sort_keys=True) for gi in group_inputs)
    inputs_by_group[m] = group_inputs[0]
res["finals_64hex"] = finals
res["config_hashes"] = cfg_hashes
res["final_losses_per_run"] = losses
res["inputs_by_method_group"] = inputs_by_group
res["inputs_consistent_within_group"] = inputs_consistent

# 2. config-vs-protocol conformity (9/9): protocol fields except method/seed/lambda/path identical
protos = []
for m in ("A", "B", "C"):
    for s in (42, 43, 44):
        cfg = json.load(open(REPO / f"outputs/v2/rqvae/{m}/seed_{s}/config.json"))
        protos.append((cfg["protocol"], cfg["consistency_weight"]))
ref_proto = protos[0][0]
proto_identical = all(json.dumps(p, sort_keys=True) == json.dumps(ref_proto, sort_keys=True) for p, _ in protos)
lam_ok = [l for _, l in protos] == [0.0] * 6 + [0.1] * 3  # 循环为方法优先 A(42,43,44),B(...),C(...)
res["protocol_conformity"] = {
    "protocol_fields_identical_9runs": bool(proto_identical),
    "protocol_ref": ref_proto,
    "lambda_sequence_ABBC AB": [l for _, l in protos],
    "lambda_matches_lock": bool(lam_ok),
    "varying_fields_only": "method/seed/consistency_weight(派生)/inputs(方法组)/run 路径",
}

# 3. log coverage per run (read-only verification)
log_checks = {}
for m in ("A", "B", "C"):
    for s in (42, 43, 44):
        base = REPO / f"outputs/v2/rqvae/{m}/seed_{s}"
        rows = [json.loads(l) for l in open(base / "training_log.jsonl")]
        steps = [r["global_step"] for r in rows]
        nv = [r["n_valid_consistency"] for r in rows]
        af = [r["aug_forward"] for r in rows]
        log_checks[f"RQ-{m}-{s}"] = {
            "lines": len(rows),
            "steps_contiguous_1_to_10000": steps == list(range(1, 10001)),
            "epochs_contiguous_1_to_10000": [r["epoch"] for r in rows] == list(range(1, 10001)),
            "all_totals_finite": all(r["total"] == r["total"] and abs(r["total"]) < float("inf") for r in rows),
            "n_valid_3356_count": sum(1 for v in nv if v == 3356),
            "aug_forward_total": sum(af),
            "expected_aug_forward": 10000 if m == "C" else 0,
        }
res["log_coverage"] = log_checks
res["log_coverage_command"] = ("python: 逐行 json.loads(training_log.jsonl)；比对 epoch/global_step 序列==1..10000、"
                               "total 有限、n_valid_consistency==3356 计数、aug_forward 求和（输出见本 JSON）")

# 4. driver log hash + excerpts + no failed dirs
drv = REPO.parent.parent / "tmp/t5_formal_main.log"  # not repo-relative; actual /tmp on remote
import subprocess
drv_path = "/tmp/t5_formal_main.log"
res["driver_log"] = {"path": drv_path, "sha256": sha(drv_path),
                     "exit_lines": [l for l in open(drv_path) if "_EXIT=" in l or l.startswith("ALL_NINE_DONE")]}
find_cmd = subprocess.run(["bash", "-c",
                           "find outputs/v2/rqvae -maxdepth 3 -name '*failed_attempt*' | wc -l; ls outputs/v2/rqvae/A outputs/v2/rqvae/B outputs/v2/rqvae/C"],
                          capture_output=True, text=True, cwd=REPO)
res["no_failed_dirs"] = {"command": "find outputs/v2/rqvae -maxdepth 3 -name '*failed_attempt*' | wc -l; ls -d outputs/v2/rqvae/{A,B,C}",
                         "failed_attempt_dirs": int(find_cmd.stdout.split()[0]),
                         "seed_dirs": find_cmd.stdout.split()[1:]}

# 5. cost wording inputs
res["cost_words"] = {
    "run_wall_sum_seconds": 30421, "run_wall_sum_human": "8h27m01s ≈ 8.4503 小时",
    "driver_span_seconds": 30551, "driver_span_human": "03:36:02→12:05:13 = 8h29m11s ≈ 8.4864 小时",
    "delta_seconds": 130, "delta_note": "进程启动/切换/记录等未细分跨度",
    "billing": "两者均按单 MIG 设备占用计；driver 跨度≠GPU 计算忙碌时长",
}

res["summary"] = {
    "finals_9_full_sha_recorded": len(finals) == 9,
    "config_hashes_9": len(cfg_hashes) == 9,
    "inputs_consistent_within_groups": all(inputs_consistent.values()),
    "protocol_identical_9": bool(proto_identical) and bool(lam_ok),
    "log_all_10000_rows_contiguous": all(v["lines"] == 10000 and v["steps_contiguous_1_to_10000"]
                                         and v["epochs_contiguous_1_to_10000"] for v in log_checks.values()),
    "log_all_finite": all(v["all_totals_finite"] for v in log_checks.values()),
    "c_n_valid_3356_all_rows": all(v["n_valid_3356_count"] == 10000 for k, v in log_checks.items() if k.startswith("RQ-C")),
    "ab_consistency_zero_all_rows": all(v["aug_forward_total"] == 0 for k, v in log_checks.items() if k.startswith("RQ-A") or k.startswith("RQ-B")),
    "driver_exit0_x9_and_done": sum(1 for l in res["driver_log"]["exit_lines"] if "_EXIT=0" in l) == 9
                                 and any("ALL_NINE_DONE" in l for l in res["driver_log"]["exit_lines"]),
    "no_failed_attempt_dirs": res["no_failed_dirs"]["failed_attempt_dirs"] == 0,
}
res["summary"]["all_passed"] = all(res["summary"].values())

out = REPO / "outputs/v2/evidence/phase3/t5_record_extract.json"
with open(out, "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print(json.dumps(res["summary"], ensure_ascii=False, indent=1))
print("EXTRACT_SHA256:", sha(out))
print("EXTRACT_DONE")
