#!/usr/bin/env python
"""T4 E1: same-process snapshot/restore exactness — one process:
  (a) train fresh C model epochs 1..40 (formal loop replica incl. eval/scheduler/RNG semantics)
  (b) continue SAME objects 41..120 -> F_cont
  (c) restore snapshot(a) into the SAME model/optimizer objects + RNG -> continue 41..120 -> F_res
  compare F_cont vs F_res bitwise (pre-declared: bitwise 0 — same process/addresses).
SCHEMA: phase3-t4_resume_inprocess-1.0"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path("/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec")
sys.path.insert(0, str(REPO / "rq"))
sys.path.insert(0, str(REPO / "scripts"))
import v2_train_rqvae as V

SEED = 42
DEVICE = torch.device("cuda:0")
res = {"stage": "phase3_t4_resume_inprocess_exact", "schema": "phase3-t4_resume_inprocess-1.0",
       "date": "2026-11-11", "seed": SEED, "budget_epochs": 120, "snapshot_at": 40,
       "pre_declared_tolerance": "bitwise (0) — same process, same tensor addresses"}

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)

C_CLEAN = np.load(REPO / "outputs/v2/embeddings/C/clean.npy")
C_AUG = np.load(REPO / "outputs/v2/embeddings/C/train_aug.npy")
C_VALID = np.load(REPO / "outputs/v2/embeddings/C/valid_aug.npy")

class PairDS(torch.utils.data.Dataset):
    def __init__(self):
        self.clean, self.aug, self.valid = C_CLEAN, C_AUG, C_VALID
        self.dim = C_CLEAN.shape[-1]
    def __len__(self):
        return len(self.clean)
    def __getitem__(self, i):
        return (torch.FloatTensor(self.clean[i]), torch.FloatTensor(self.aug[i]),
                bool(self.valid[i]))

model = V.V2RQVAE(in_dim=PairDS().dim, num_emb_list=[256, 256, 256], e_dim=32,
                  layers=[2048, 1024, 512, 256, 128, 64], dropout_prob=0.0, bn=False,
                  loss_type="mse", quant_loss_weight=1.0, beta=0.25, kmeans_init=True,
                  kmeans_iters=100, sk_epsilons=[0.0, 0.0, 0.0], sk_iters=50).to(DEVICE)
loader = DataLoader(PairDS(), num_workers=0, batch_size=20480, shuffle=True, pin_memory=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
scheduler = V.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=50)
LAM = 0.1

def train_epochs(start_ep, end_ep):
    rows = []
    for epoch_idx in range(start_ep - 1, end_ep):
        te0 = time.time()
        for batch in loader:
            x, aug, valid = batch
            if not torch.is_tensor(valid):
                valid = torch.tensor(valid, dtype=torch.bool)
            aug = aug.to(DEVICE); valid = valid.to(DEVICE); x = x.to(DEVICE)
            optimizer.zero_grad()
            out, rq_loss, _, zq = model(x, use_sk=True)
            loss_recon = F.mse_loss(out, x)
            cons = torch.zeros((), device=DEVICE)
            if LAM > 0:
                av = aug[valid]
                _, _, _, zq_a = model(av, use_sk=True)
                cons, _ = V.consistency_loss(zq, zq_a, valid)
            total = loss_recon + 1.0 * rq_loss + LAM * cons
            if torch.isnan(total) or torch.isinf(total):
                raise ValueError("non-finite")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        rows.append({"epoch": epoch_idx + 1, "total": float(total.detach()),
                     "lr": scheduler.get_last_lr()[0],
                     "recon": float(loss_recon.detach()), "quant": float(rq_loss.detach()),
                     "cons": float(cons.detach())})
        # locked eval behavior at epochs divisible by 50 (consumes loader RNG identically)
        if (epoch_idx + 1) % 50 == 0:
            V.collision_eval(model, loader, DEVICE)
    return rows

import time
t0 = time.time()
rows_1_40 = train_epochs(1, 40)
SNAP = {"state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "rng": V.get_rng_state(),
        "initted": V.vq_initted_flags(model)}
t_snap = time.time()

rows_cont = train_epochs(41, 120)
t_cont = time.time()
F_cont = {k: v.detach().clone() for k, v in model.state_dict().items()}
opt_cont = {k: {kk: {f: (vv.clone() if torch.is_tensor(vv) else vv) for f, vv in d.items()}
                for kk, d in optimizer.state_dict()["state"].items()}
            for k in [0]}  # deep clone below
opt_cont = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
opt_cont.load_state_dict(optimizer.state_dict())
opt_cont_sd = {k: {f: (v.clone() if torch.is_tensor(v) else v) for f, v in s.items()}
               for k, s in optimizer.state_dict()["state"].items()}
pg_cont = json.loads(json.dumps(optimizer.state_dict()["param_groups"]))
sch_cont = json.loads(json.dumps(scheduler.state_dict()))
rng_cont = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in
            [("torch", torch.get_rng_state())]}
rng_cont_torch = torch.get_rng_state().clone()

# restore into the SAME objects
model.load_state_dict(SNAP["state_dict"])
optimizer.load_state_dict(SNAP["optimizer"])
scheduler.load_state_dict(SNAP["scheduler"])
V.set_vq_initted(model, SNAP["initted"])
V.set_rng_state(SNAP["rng"])
rows_res = train_epochs(41, 120)
t_res = time.time()
F_res = {k: v.detach().clone() for k, v in model.state_dict().items()}
opt_res_sd = {k: {f: (v.clone() if torch.is_tensor(v) else v) for f, v in s.items()}
              for k, s in optimizer.state_dict()["state"].items()}
F_res2 = None
rows_res2 = None

# restore AGAIN and rerun the same 41..120 — res1 vs res2 tests intra-process run-to-run determinism
model.load_state_dict(SNAP["state_dict"])
optimizer.load_state_dict(SNAP["optimizer"])
scheduler.load_state_dict(SNAP["scheduler"])
V.set_vq_initted(model, SNAP["initted"])
V.set_rng_state(SNAP["rng"])
rows_res2 = train_epochs(41, 120)
F_res2 = {k: v.detach().clone() for k, v in model.state_dict().items()}

def sd_max_diff(a, b):
    return max(float((a[k] - b[k]).abs().max()) for k in a)

sd_eq = all(torch.equal(F_cont[k], F_res[k]) for k in F_cont)
res_eq_eq = all(torch.equal(F_res[k], F_res2[k]) for k in F_res)
opt_eq = all(torch.equal(opt_cont_sd[k][f], opt_res_sd[k][f])
             for k in opt_cont_sd for f in opt_cont_sd[k] if torch.is_tensor(opt_cont_sd[k][f]))
sch_eq = sch_cont == json.loads(json.dumps(scheduler.state_dict()))
rng_eq = bool(torch.equal(rng_cont_torch, torch.get_rng_state()))
rows_eq = rows_cont == rows_res
first_div = next((i for i in range(len(rows_cont)) if rows_cont[i] != rows_res[i]), None)
res["bitwise"] = {"state_dict_cont_vs_res": sd_eq, "res1_vs_res2": res_eq_eq,
                  "optimizer_cont_vs_res": opt_eq, "scheduler": sch_eq,
                  "rng": rng_eq, "rows_41_120_equal": rows_eq}
res["first_divergence_epoch"] = (first_div + 41) if first_div is not None else None
if first_div is not None:
    res["first_div_row_cont"] = rows_cont[first_div]
    res["first_div_row_res"] = rows_res[first_div]
    res["first_div_total_diff"] = abs(rows_cont[first_div]["total"] - rows_res[first_div]["total"])
res["max_state_diff"] = {"cont_vs_res": sd_max_diff(F_cont, F_res),
                         "res1_vs_res2": sd_max_diff(F_res, F_res2)}
res["elapsed_s"] = {"train_1_40": t_snap - t0, "cont_41_120": t_cont - t_snap,
                    "restore_and_resumed": t_res - t_cont}
res["kmeans_initted_at_snapshot"] = SNAP["initted"]
res["summary"] = {"restore_exact_bitwise": all([sd_eq, opt_eq, sch_eq, rng_eq, rows_eq]),
                  "run_to_run_deterministic": res_eq_eq}

out = REPO / "outputs/v2/evidence/phase3/resume_inprocess_exact.json"
with open(out, "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("bitwise:", json.dumps(res["bitwise"]))
print("first_div_epoch:", res.get("first_divergence_epoch"), "| first_div_total_diff:", res.get("first_div_total_diff"))
print("max_state_diff:", json.dumps(res["max_state_diff"]))
print("summary:", json.dumps(res["summary"]))
print("OUT_SHA256:", V.sha256_file(str(out)))
print("E1_DONE")
