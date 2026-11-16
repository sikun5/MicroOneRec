#!/usr/bin/env python
"""T4 E3: layered determinism bisection on the MIG slice.
Layer 1: raw matmul repeated. Layer 2: model forward repeated. Layer 3: forward+backward grads.
Layer 4: full single epoch (fwd+bwd+step) from restored state. Layer 5: two consecutive epochs.
SCHEMA: phase3-t4_e3_bisect-1.0"""
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
res = {"stage": "phase3_t4_e3_bisect", "date": "2026-11-11",
       "env": {"cublas_ws": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
               "omp": os.environ.get("OMP_NUM_THREADS"),
               "det_flag": None}}

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)
res["env"]["det_flag"] = bool(torch.are_deterministic_algorithms_enabled())

# Layer 1: raw matmul, same tensors, 3 calls
a = torch.randn(4096, 4096, device=DEVICE)
b = torch.randn(4096, 4096, device=DEVICE)
r1, r2, r3 = a @ b, a @ b, a @ b
res["L1_matmul"] = {"repro_bitwise": bool(torch.equal(r1, r2) and torch.equal(r2, r3)),
                    "max_diff_12": float((r1 - r2).abs().max())}
# Layer 1b: embedding forward/backward repeated (grad compare)
emb = torch.nn.Embedding(256, 32).to(DEVICE)
idx = torch.randint(0, 256, (3686,), device=DEVICE)
for p in emb.parameters():
    p.grad = None
out1 = emb(idx); out1.sum().backward(); g1 = emb.weight.grad.clone()
for p in emb.parameters():
    p.grad = None
out2 = emb(idx); out2.sum().backward(); g2 = emb.weight.grad.clone()
res["L1b_embedding_bwd"] = {"repro_bitwise": bool(torch.equal(out1, out2) and torch.equal(g1, g2)),
                            "grad_max_diff": float((g1 - g2).abs().max())}

# real C state at epoch ~40
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
# quick 40 epochs to a realistic state (untracked drift irrelevant for repro tests)
t0 = time.time()
for _ in range(40):
    epoch_once(model, loader, opt, sch)
res["warmup_40ep_s"] = time.time() - t0
SNAP = {k: v.detach().clone() for k, v in model.state_dict().items()}
SNAP_OPT = {k: {f: (v.clone() if torch.is_tensor(v) else v) for f, v in s.items()}
            for k, s in opt.state_dict()["state"].items()}
SNAP_SCH = json.loads(json.dumps(sch.state_dict()))
SNAP_RNG = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in
            {"torch": torch.get_rng_state()}.items()}
SNAP_RNG_T = torch.get_rng_state().clone()

def restore():
    model.load_state_dict(SNAP)
    opt.load_state_dict({"state": SNAP_OPT, "param_groups": opt.state_dict()["param_groups"]})
    sch.load_state_dict(SNAP_SCH)
    torch.set_rng_state(SNAP_RNG_T)
    model.train()

# Layer 2: forward-only repeated (3x) on the fixed first batch
restore()
batches = []
for batch in loader:
    batches.append(batch)
    break  # epoch-42 batch (post-restore first perm draw)
x, aug, valid = batches[0]
if not torch.is_tensor(valid):
    valid = torch.tensor(valid, dtype=torch.bool)
x, aug, valid = x.to(DEVICE), aug.to(DEVICE), valid.to(DEVICE)
with torch.no_grad():
    f_out = [model(x, use_sk=True)[0] for _ in range(3)]
res["L2_forward_3x"] = {"repro_bitwise": bool(torch.equal(f_out[0], f_out[1]) and torch.equal(f_out[1], f_out[2])),
                        "max_diff_12": float((f_out[0] - f_out[1]).abs().max())}

# Layer 3: forward+backward grads repeated
for p in model.parameters():
    p.grad = None
o1 = model(x, use_sk=True)
l1 = F.mse_loss(o1[0], x) + o1[1] + 0.1 * V.consistency_loss(o1[3], model(aug[valid], use_sk=True)[3], valid)[0]
l1.backward()
g1 = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
for p in model.parameters():
    p.grad = None
o2 = model(x, use_sk=True)
l2 = F.mse_loss(o2[0], x) + o2[1] + 0.1 * V.consistency_loss(o2[3], model(aug[valid], use_sk=True)[3], valid)[0]
l2.backward()
g2 = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
res["L3_fwdbwd_grads"] = {"loss_bitwise": bool(torch.equal(l1.detach(), l2.detach())),
                          "grads_bitwise": all(torch.equal(g1[n], g2[n]) for n in g1),
                          "grad_max_diff": max(float((g1[n] - g2[n]).abs().max()) for n in g1)}

# Layer 4: full epoch (fwd+bwd+step) from restored state, 2x
def full_epoch():
    restore()
    return epoch_once(model, loader, opt, sch)
e1 = full_epoch()
e2 = full_epoch()
res["L4_full_epoch"] = {"loss_e1": e1, "loss_e2": e2,
                        "bitwise": bool(e1 == e2)}

# Layer 5: two consecutive epochs from restored state
def two_epochs():
    restore()
    r = []
    for _ in range(2):
        r.append(epoch_once(model, loader, opt, sch))
    return r
p1 = two_epochs()
p2 = two_epochs()
res["L5_two_epochs"] = {"pair1": p1, "pair2": p2, "bitwise": p1 == p2}

res["summary"] = {"any_layer_nondeterministic": not all([
    res["L1_matmul"]["repro_bitwise"], res["L1b_embedding_bwd"]["repro_bitwise"],
    res["L2_forward_3x"]["repro_bitwise"], res["L3_fwdbwd_grads"]["grads_bitwise"],
    res["L4_full_epoch"]["bitwise"], res["L5_two_epochs"]["bitwise"]])}
out = REPO / "outputs/v2/evidence/phase3/e3_bisect_checks.json"
with open(out, "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print(json.dumps({k: res[k] for k in ("L1_matmul", "L1b_embedding_bwd", "L2_forward_3x",
                                      "L3_fwdbwd_grads", "L4_full_epoch", "L5_two_epochs",
                                      "summary")}, ensure_ascii=False, indent=1))
print("OUT_SHA256:", V.sha256_file(str(out)))
print("E3_DONE")
