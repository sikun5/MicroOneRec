#!/usr/bin/env python
"""T6 independent checkpoint checks: reload all 9 finals from disk (no training,
eval-only sanity forward, no param update, no K-means re-run).
SCHEMA: phase3-t6_checkpoint_checks-1.0"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys, json, hashlib
from pathlib import Path
import numpy as np
import torch

REPO = Path("/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec")
sys.path.insert(0, str(REPO / "rq"))
sys.path.insert(0, str(REPO / "scripts"))
import v2_train_rqvae as V

DEVICE = torch.device("cuda:0")
inv = json.load(open(REPO / "outputs/v2/evidence/phase3/run_inventory.json"))
CAND = inv["candidate_script_v13"]
res = {"stage": "phase3_t6_checkpoint_checks", "schema": "phase3-t6_checkpoint_checks-1.0",
       "date": "2026-11-12", "candidate_script_v13": CAND,
       "note": "独立从磁盘重载九个 final；eval-only sanity；全程无参数更新/无 K-means 重跑"}

def param_checksum(model):
    h = hashlib.sha256()
    for k in sorted(model.state_dict()):
        h.update(k.encode())
        h.update(model.state_dict()[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]

checks = {}
for m in ("A", "B", "C"):
    for s in (42, 43, 44):
        rid = f"RQ-{m}-{s}"
        base = REPO / f"outputs/v2/rqvae/{m}/seed_{s}"
        fp = base / "final_checkpoint.pth"
        c = {"final_path": f"outputs/v2/rqvae/{m}/seed_{s}/final_checkpoint.pth"}
        # 1. file hash vs inventory
        c["file_sha256_64"] = V.sha256_file(str(fp))
        inv_row = next(r for r in inv["runs"] if r["run_id"] == rid)
        c["file_sha_matches_inventory"] = c["file_sha256_64"] == inv_row["final_sha256"]
        # 2. load
        st = torch.load(fp, map_location="cpu", weights_only=False)
        c["method_seed"] = {"method": st["config"]["method"], "seed": st["config"]["seed"],
                            "matches": st["config"]["method"] == m and st["config"]["seed"] == s}
        c["progress"] = {"epoch_completed": st["epoch_completed"], "global_step": st["global_step"],
                         "is_10000": st["epoch_completed"] == 10000 and st["global_step"] == 10000}
        c["state_fields_present"] = {
            "optimizer": "state" in st["optimizer"] and len(st["optimizer"]["state"]) == 31,  # 31 个 nn.Parameter（encoder 14+decoder 14+码本 3）
            "optimizer_steps": sorted({float(v["step"]) for v in st["optimizer"]["state"].values()}) == [10000.0],
            "scheduler_last_epoch": st["scheduler"].get("last_epoch"),
            "rng_keys": sorted(st["rng"].keys()),
            "init_state_vq_initted": st["init_state"]["vq_initted"],
            "config_script_sha256": st["config"]["script_sha256"],
            "config_script_is_v13": st["config"]["script_sha256"] == CAND,
            "config_inputs_bound": {k: v["sha256"][:16] for k, v in st["config"]["inputs"].items() if isinstance(v, dict)},
            "schema": st["schema"],
        }
        # 3. instantiate + load + params finite
        model = V.V2RQVAE(in_dim=st["init_state"]["in_dim"], num_emb_list=[256, 256, 256], e_dim=32,
                          layers=[2048, 1024, 512, 256, 128, 64], dropout_prob=0.0, bn=False,
                          loss_type="mse", quant_loss_weight=1.0, beta=0.25, kmeans_init=True,
                          kmeans_iters=100, sk_epsilons=[0.0, 0.0, 0.0], sk_iters=50).to(DEVICE)
        model.load_state_dict(st["state_dict"], strict=True)
        V.set_vq_initted(model, st["init_state"]["vq_initted"])  # 生产 resume 语义：恢复初始化标志
        model.eval()
        c["load_strict_ok"] = True
        c["params_finite"] = all(bool(torch.isfinite(p).all()) for p in model.parameters())
        c["param_elements"] = int(sum(p.numel() for p in model.parameters()))
        pre = param_checksum(model)
        # init guard active (initted True -> init branch unreachable)
        c["vq_initted_after_load"] = V.vq_initted_flags(model)
        c["vq_initted_matches_checkpoint"] = V.vq_initted_flags(model) == st["init_state"]["vq_initted"]
        # 4. eval-only sanity: fixed small input (first 8 rows of the method's clean), no_grad
        x8 = torch.from_numpy(np.load(REPO / f"outputs/v2/embeddings/{m}/clean.npy")[:8]).to(DEVICE)
        with torch.no_grad():
            out, rq_loss, indices, zq = model(x8, use_sk=True)
            idx_only = model.get_indices(x8, use_sk=False)
        c["eval_forward"] = {
            "input_rows": 8, "out_shape": list(out.shape), "out_finite": bool(torch.isfinite(out).all()),
            "zq_shape": list(zq.shape), "zq_finite": bool(torch.isfinite(zq).all()),
            "indices_shape": list(indices.shape),
            "get_indices_matches_forward": bool(torch.equal(indices, idx_only)),
            "rq_loss_finite": bool(torch.isfinite(rq_loss.detach())),
            "no_grad": not out.requires_grad,
        }
        post = param_checksum(model)
        c["params_unchanged_by_forward"] = pre == post
        c["initted_unchanged_by_forward"] = V.vq_initted_flags(model) == st["init_state"]["vq_initted"]
        # 5. log per-component finite (reference extract) + C extra forward
        rows = [json.loads(l) for l in open(base / "training_log.jsonl")]
        c["log_10000_rows"] = len(rows) == 10000
        c["log_all_components_finite"] = all(
            all(r[k] == r[k] and abs(r[k]) < float("inf")
                for k in ("lr", "recon", "quant", "consistency_raw", "consistency_weighted", "total"))
            for r in rows)
        c["log_aug_forward_total"] = sum(r["aug_forward"] for r in rows)
        c["final_not_best_substitute"] = {
            "final_epoch_is_10000": st["epoch_completed"] == 10000,
            "best_loss_diagnostic_value": st.get("best_loss_diagnostic"),
            "note": "best_loss/best_collision 为历史诊断文件；正式 final 独立落盘于第 10000 epoch",
        }
        ok = (c["file_sha_matches_inventory"] and c["method_seed"]["matches"] and c["progress"]["is_10000"]
              and c["state_fields_present"]["optimizer"] and c["state_fields_present"]["optimizer_steps"]
              and c["state_fields_present"]["config_script_is_v13"]
              and c["load_strict_ok"] and c["params_finite"]
              and c["eval_forward"]["out_finite"] and c["eval_forward"]["zq_finite"]
              and c["eval_forward"]["get_indices_matches_forward"] and c["eval_forward"]["no_grad"]
              and c["params_unchanged_by_forward"] and c["initted_unchanged_by_forward"]
              and c["vq_initted_matches_checkpoint"]
              and c["log_10000_rows"] and c["log_all_components_finite"]
              and c["log_aug_forward_total"] == (10000 if m == "C" else 0))
        c["run_passed"] = bool(ok)
        checks[rid] = c
        print(rid, "PASS" if ok else "FAIL", flush=True)

res["checks"] = checks
res["summary"] = {"all_9_passed": all(c["run_passed"] for c in checks.values())}
out = REPO / "outputs/v2/evidence/phase3/checkpoint_checks.json"
with open(out, "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("SUMMARY:", json.dumps(res["summary"]))
print("CKPT_SHA256:", V.sha256_file(str(out)))
print("T6_CKPT_DONE")
