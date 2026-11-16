#!/usr/bin/env python
"""T4 E4: L5 pair test with torch.backends.cuda.preferred_blas_library("cublas")
(candidate fix for allocator-address-dependent cuBLASLt nondeterminism)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys, json, random, time
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
res = {"stage": "phase3_t4_e4_cublas_force", "date": "2026-11-11"}
try:
    torch.backends.cuda.preferred_blas_library("cublas")
    res["blas_lib"] = str(torch.backends.cuda.preferred_blas_library())
except Exception as e:
    res["blas_lib"] = f"UNAVAILABLE: {e}"

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

def fresh_setup():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = V.V2RQVAE(in_dim=2560, num_emb_list=[256, 256, 256], e_dim=32,
                      layers=[2048, 1024, 512, 256, 128, 64], dropout_prob=0.0, bn=False,
                      loss_type="mse", quant_loss_weight=1.0, beta=0.25, kmeans_init=True,
                      kmeans_iters=100, sk_epsilons=[0.0, 0.0, 0.0], sk_iters=50).to(DEVICE)
    loader = DataLoader(PairDS(), num_workers=0, batch_size=20480, shuffle=True, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    sch = V.get_constant_schedule_with_warmup(opt, num_warmup_steps=50)
    return model, loader, opt, sch

def epoch_once(model, loader, opt, sch):
    model.train()
    for batch in loader:
        x, aug, valid = batch
        if not torch.is_tensor(valid):
            valid = torch.tensor(valid, dtype=torch.bool)
        aug = aug.to(DEVICE); valid = valid.to(DEVICE); x = x.to(DEVICE)
        opt.zero_grad()
        out, rq_loss, _, zq = model(x, use_sk=True)
        lr_ = F.mse_loss(out, x)
        av = aug[valid]
        _, _, _, zq_a = model(av, use_sk=True)
        cons, _ = V.consistency_loss(zq, zq_a, valid)
        total = lr_ + 1.0 * rq_loss + 0.1 * cons
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
    return float(total.detach())

model, loader, opt, sch = fresh_setup()
for _ in range(40):
    epoch_once(model, loader, opt, sch)
SNAP = {k: v.detach().clone() for k, v in model.state_dict().items()}
SNAP_OPT = {k: {f: (v.clone() if torch.is_tensor(v) else v) for f, v in s.items()}
            for k, s in opt.state_dict()["state"].items()}
SNAP_SCH = json.loads(json.dumps(sch.state_dict()))
SNAP_RNG_T = torch.get_rng_state().clone()

def restore():
    model.load_state_dict(SNAP)
    opt.load_state_dict({"state": SNAP_OPT, "param_groups": opt.state_dict()["param_groups"]})
    sch.load_state_dict(SNAP_SCH)
    torch.set_rng_state(SNAP_RNG_T)
    model.train()

def two_epochs():
    restore()
    r = []
    for _ in range(2):
        r.append(epoch_once(model, loader, opt, sch))
    return r

p1 = two_epochs()
p2 = two_epochs()
p3 = two_epochs()
res["pairs"] = {"p1": p1, "p2": p2, "p3": p3}
res["bitwise_p1_p2"] = p1 == p2
res["bitwise_p1_p3"] = p1 == p3
res["summary"] = {"all_pairs_bitwise": p1 == p2 == p3}
out = REPO / "outputs/v2/evidence/phase3/e4_cublas_force.json"
with open(out, "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
print("E4_DONE")
