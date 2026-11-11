#!/usr/bin/env python
"""T4 resume equivalence: continuous 60ep vs interrupted(kill@30+resume@snapshot20) 60ep.
Pre-declared tolerance: bitwise (0) for state_dict/optimizer/scheduler/RNG/log rows.
SCHEMA: phase3-t4_resume_checks-1.0"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
import sys, json
from pathlib import Path
import numpy as np
import torch

REPO = Path("/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec")
sys.path.insert(0, str(REPO / "rq"))
sys.path.insert(0, str(REPO / "scripts"))
import v2_train_rqvae as V

CONT = REPO / "outputs/v2/smoke/phase3/resume-cont/C/seed_42"
INTR = REPO / "outputs/v2/smoke/phase3/resume-test/C/seed_42"
res = {"stage": "phase3_t4_resume_checks", "schema": "phase3-t4_resume_checks-1.0",
       "date": "2026-11-11", "pre_declared_tolerance": "bitwise (0) — same device/version/deterministic algorithms",
       "budget_epochs": 120, "snapshot_every": 40, "interrupt_sim": "kill -9 at ~35s (≈epoch 50; >snapshot 40)",
       "candidate_script": V.sha256_file(str(REPO / "scripts" / "v2_train_rqvae.py"))}

cont = torch.load(CONT / "final_checkpoint.pth", map_location="cpu", weights_only=False)
intr = torch.load(INTR / "final_checkpoint.pth", map_location="cpu", weights_only=False)

# 1. epoch/step
res["progress"] = {"cont_epoch_completed": cont["epoch_completed"], "intr_epoch_completed": intr["epoch_completed"],
                   "cont_global_step": cont["global_step"], "intr_global_step": intr["global_step"],
                   "epoch_step_equal": (cont["epoch_completed"] == intr["epoch_completed"] == 120
                                        and cont["global_step"] == intr["global_step"] == 120)}

# 2. state_dict bitwise
sd_equal = all(torch.equal(cont["state_dict"][k], intr["state_dict"][k]) for k in cont["state_dict"])
max_sd_diff = max(float((cont["state_dict"][k] - intr["state_dict"][k]).abs().max()) for k in cont["state_dict"])
n_kmeans_rerun_proof_keys = [k for k in cont["state_dict"] if k.startswith("rq.vq_layers")]
res["state_dict"] = {"all_keys_equal": set(cont["state_dict"]) == set(intr["state_dict"]),
                     "bitwise_equal": sd_equal, "max_abs_diff": max_sd_diff,
                     "codebook_keys_checked": n_kmeans_rerun_proof_keys,
                     "kmeans_not_rerun_bitwise_proof": all(
                         torch.equal(cont["state_dict"][k], intr["state_dict"][k])
                         for k in n_kmeans_rerun_proof_keys if k.endswith("embedding.weight"))}

# 3. optimizer state bitwise
def opt_equal(a, b):
    sa, sb = a["state"], b["state"]
    if set(sa) != set(sb) or a["param_groups"] != b["param_groups"]:
        return False, 0.0
    maxd = 0.0
    for k in sa:
        for field in sa[k]:
            va, vb = sa[k][field], sb[k][field]
            if torch.is_tensor(va):
                if not torch.equal(va, vb):
                    return False, max(maxd, float((va - vb).abs().max()))
            elif va != vb:
                return False, -1.0
    return True, maxd
opt_ok, opt_maxd = opt_equal(cont["optimizer"], intr["optimizer"])
res["optimizer"] = {"bitwise_equal": opt_ok, "max_abs_diff": opt_maxd,
                    "param_groups_equal": cont["optimizer"]["param_groups"] == intr["optimizer"]["param_groups"]}

# 4. scheduler
res["scheduler"] = {"cont_last_epoch": cont["scheduler"]["last_epoch"],
                    "intr_last_epoch": intr["scheduler"]["last_epoch"],
                    "bitwise_equal": cont["scheduler"] == intr["scheduler"]}

# 5. RNG bitwise
rc, ri = cont["rng"], intr["rng"]
res["rng"] = {"torch_bitwise": bool(torch.equal(rc["torch"], ri["torch"])),
              "python_equal": rc["python"] == ri["python"],
              "numpy_equal": all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
                                 for a, b in zip(rc["numpy"], ri["numpy"])),
              "cuda_count_equal": len(rc["torch_cuda"]) == len(ri["torch_cuda"]),
              "cuda_bitwise": all(torch.equal(a, b) for a, b in zip(rc["torch_cuda"], ri["torch_cuda"]))}

# 6. init_state
res["init_state"] = {"cont": cont["init_state"], "intr": intr["init_state"],
                     "equal": cont["init_state"]["vq_initted"] == intr["init_state"]["vq_initted"]
                              and cont["init_state"]["in_dim"] == intr["init_state"]["in_dim"]}

# 7. final losses bitwise
res["final_losses_equal"] = cont["final_losses"] == intr["final_losses"]
res["final_losses"] = {"cont": cont["final_losses"], "intr": intr["final_losses"]}

# 8. config semantic equality (full 10-field verify)
try:
    V.verify_resume_config(cont["config"], intr["config"])
    res["config_verify_equal"] = True
except ValueError as e:
    res["config_verify_equal"] = False
    res["config_verify_error"] = str(e)

# 9. training_log rows: fresh part (1..P) from interrupted attempt == cont; resumed part (P+1..N) == cont
#    resume point P read dynamically from the resumed run's attempts record
rr0 = json.load(open(INTR / "run_record.json"))
resume_at = next((a.get("resumed_at_epoch_completed") for a in rr0["attempts"]
                  if a.get("type") == "resume"), None)
cont_rows = [json.loads(l) for l in open(CONT / "training_log.jsonl")]
intr_rows = [json.loads(l) for l in open(INTR / "training_log.jsonl")]
P = resume_at if resume_at is not None else 0
res["log_rows"] = {"cont_lines": len(cont_rows), "intr_lines": len(intr_rows),
                   "resume_point_P": P,
                   "fresh_prefix_equal": cont_rows[:P] == intr_rows[:P],
                   "resumed_suffix_equal": cont_rows[P:] == intr_rows[P:],
                   "all_equal": cont_rows == intr_rows}

# 10. loadability: final + snapshot of BOTH runs load into fresh V2RQVAE
load_ok, load_detail = {}, {}
for tag, base in (("cont", CONT), ("intr", INTR)):
    for fname in ("final_checkpoint.pth", "resume_snapshot.pth"):
        p = base / fname
        st = torch.load(p, map_location="cpu", weights_only=False)
        m = V.V2RQVAE(in_dim=st["init_state"]["in_dim"], num_emb_list=[256, 256, 256], e_dim=32,
                      layers=[2048, 1024, 512, 256, 128, 64], dropout_prob=0.0, bn=False,
                      loss_type="mse", quant_loss_weight=1.0, beta=0.25, kmeans_init=True,
                      kmeans_iters=100, sk_epsilons=[0.0, 0.0, 0.0], sk_iters=50)
        m.load_state_dict(st["state_dict"])
        load_detail[f"{tag}/{fname}"] = {"loaded": True, "epoch_completed": st["epoch_completed"],
                                         "snapshot_flag": st.get("snapshot")}
res["loadability"] = {"all_loaded": all(v["loaded"] for v in load_detail.values()), "detail": load_detail}

# 11. beta refusal CLI log evidence
ref_log = (REPO / "outputs/v2/smoke/phase3/resume_beta_refusal.log").read_text()
res["beta_refusal"] = {"exit_nonzero_evidence": "resume refused — config semantic mismatch" in ref_log
                       or "config semantic mismatch" in ref_log,
                       "log_tail": ref_log.strip().splitlines()[-1] if ref_log.strip() else ""}

# 12. attempts record of resumed run
rr = json.load(open(INTR / "run_record.json"))
res["attempts"] = rr["attempts"]

res["summary"] = {
    "epoch_step_equal": res["progress"]["epoch_step_equal"],
    "state_dict_bitwise": sd_equal, "optimizer_bitwise": opt_ok,
    "scheduler_bitwise": res["scheduler"]["bitwise_equal"],
    "rng_bitwise": all(res["rng"].values()),
    "init_state_equal": res["init_state"]["equal"],
    "final_losses_bitwise": res["final_losses_equal"],
    "config_verify_equal": res["config_verify_equal"],
    "log_all_equal": res["log_rows"]["all_equal"],
    "resume_point_P": res["log_rows"]["resume_point_P"],
    "loadability_ok": res["loadability"]["all_loaded"],
    "beta_refusal_ok": res["beta_refusal"]["exit_nonzero_evidence"],
    "kmeans_not_rerun": res["state_dict"]["kmeans_not_rerun_bitwise_proof"],
}
res["summary"]["all_passed"] = all(res["summary"].values())

out = REPO / "outputs/v2/evidence/phase3/resume_checks.json"
with open(out, "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print(json.dumps(res["summary"], ensure_ascii=False))
print("max_sd_diff:", max_sd_diff, "| opt_maxd:", opt_maxd)
print("OUT_SHA256:", V.sha256_file(str(out)))
print("RESUME_CHECKS_DONE")
