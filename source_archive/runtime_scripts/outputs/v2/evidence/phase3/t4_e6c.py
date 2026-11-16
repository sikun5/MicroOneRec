import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys, json, random, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
REPO = Path("/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec")
sys.path.insert(0, str(REPO / "rq")); sys.path.insert(0, str(REPO / "scripts"))
import v2_train_rqvae as V
SEED = 42; DEVICE = torch.device("cuda:0")
def wsum(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode()); h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]
def gradsum(model):
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        if p.grad is not None:
            h.update(n.encode()); h.update(p.grad.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)
C_CLEAN = np.load(REPO/"outputs/v2/embeddings/C/clean.npy")
C_AUG = np.load(REPO/"outputs/v2/embeddings/C/train_aug.npy")
C_VALID = np.load(REPO/"outputs/v2/embeddings/C/valid_aug.npy")
class PairDS(torch.utils.data.Dataset):
    def __init__(self):
        self.clean, self.aug, self.valid = C_CLEAN, C_AUG, C_VALID; self.dim = C_CLEAN.shape[-1]
    def __len__(self): return len(self.clean)
    def __getitem__(self, i):
        return (torch.FloatTensor(self.clean[i]), torch.FloatTensor(self.aug[i]), bool(self.valid[i]))
m0, ld, o, s = None, None, None, None
def fresh():
    global m0
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    m0 = V.V2RQVAE(in_dim=2560, num_emb_list=[256,256,256], e_dim=32,
                   layers=[2048,1024,512,256,128,64], dropout_prob=0.0, bn=False,
                   loss_type="mse", quant_loss_weight=1.0, beta=0.25, kmeans_init=True,
                   kmeans_iters=100, sk_epsilons=[0.,0.,0.], sk_iters=50).to(DEVICE)
    ld = DataLoader(PairDS(), num_workers=0, batch_size=20480, shuffle=True, pin_memory=True)
    o = torch.optim.AdamW(m0.parameters(), lr=1e-3, weight_decay=0.0)
    s = V.get_constant_schedule_with_warmup(o, num_warmup_steps=50)
    return m0, ld, o, s
def epoch_once(m, ld, o, s):
    m.train(); rec = {}
    for bi, batch in enumerate(ld):
        x, aug, valid = batch
        if not torch.is_tensor(valid): valid = torch.tensor(valid, dtype=torch.bool)
        aug = aug.to(DEVICE); valid = valid.to(DEVICE); x = x.to(DEVICE)
        rec["batch_sha"] = hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()[:16]
        o.zero_grad()
        out_, rq, _, zq = m(x, use_sk=True)
        av = aug[valid]
        _,_,_, zq_a = m(av, use_sk=True)
        cons, _ = V.consistency_loss(zq, zq_a, valid)
        total = F.mse_loss(out_, x) + rq + 0.1*cons
        rec["loss"] = float(total.detach())
        total.backward()
        rec["grads_sha"] = gradsum(m)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        o.step(); s.step()
    rec["wsum"] = wsum(m.state_dict())
    rec["lr"] = float(o.state_dict()["param_groups"][0]["lr"])
    return rec
m, ld, o, s = fresh()
for _ in range(40): epoch_once(m, ld, o, s)
SNAP_OPT_FULL = json.loads(json.dumps(o.state_dict())) if False else o.state_dict()
SNAP_OPT_SD = {"state": {k: {f: (v.clone() if torch.is_tensor(v) else v) for f, v in st.items()}
                         for k, st in o.state_dict()["state"].items()},
               "param_groups": json.loads(json.dumps(o.state_dict()["param_groups"]))}
SNAP_SCH = json.loads(json.dumps(s.state_dict())); SNAP_RNG_T = torch.get_rng_state().clone()
def restore():
    m.load_state_dict({k: v.clone() for k, v in
                       {kk: vv for kk, vv in SNAP_OPT_SD.items()} and m.state_dict().items()})
m_failed = None
def restore_v2():
    m.load_state_dict(SNAP_W)
    o.load_state_dict(SNAP_OPT_SD)
    s.load_state_dict(SNAP_SCH); torch.set_rng_state(SNAP_RNG_T); m.train()
SNAP_W = {k: v.detach().clone() for k, v in m.state_dict().items()}
runs = []
for i in range(3):
    restore_v2()
    runs.append(epoch_once(m, ld, o, s))
keys = ["batch_sha", "loss", "grads_sha", "wsum", "lr"]
out = {}
for k in keys:
    vals = [r[k] for r in runs]
    out[k] = {"values": vals if k in ("lr",) else vals, "all_equal": len(set(vals)) == 1}
print(json.dumps(out, indent=1))
print("E6C_DONE")
