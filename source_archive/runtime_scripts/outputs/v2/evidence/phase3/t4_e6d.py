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
def hsh(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]
def opt_state_sum(o):
    ea = hashlib.sha256(); eas = hashlib.sha256(); steps = []
    for k, st in o.state.items():
        ea.update(st["exp_avg"].detach().cpu().numpy().tobytes())
        eas.update(st["exp_avg_sq"].detach().cpu().numpy().tobytes())
        steps.append(float(st["step"]))
    return ea.hexdigest()[:12], eas.hexdigest()[:12], steps
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
def build(foreach):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    m = V.V2RQVAE(in_dim=2560, num_emb_list=[256,256,256], e_dim=32,
                  layers=[2048,1024,512,256,128,64], dropout_prob=0.0, bn=False,
                  loss_type="mse", quant_loss_weight=1.0, beta=0.25, kmeans_init=True,
                  kmeans_iters=100, sk_epsilons=[0.,0.,0.], sk_iters=50).to(DEVICE)
    ld = DataLoader(PairDS(), num_workers=0, batch_size=20480, shuffle=True, pin_memory=True)
    o = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0, foreach=foreach)
    s = V.get_constant_schedule_with_warmup(o, num_warmup_steps=50)
    return m, ld, o, s
def epoch_once(m, ld, o, s):
    m.train()
    for batch in ld:
        x, aug, valid = batch
        if not torch.is_tensor(valid): valid = torch.tensor(valid, dtype=torch.bool)
        aug = aug.to(DEVICE); valid = valid.to(DEVICE); x = x.to(DEVICE)
        o.zero_grad()
        out_, rq, _, zq = m(x, use_sk=True)
        av = aug[valid]
        _,_,_, zq_a = m(av, use_sk=True)
        cons, _ = V.consistency_loss(zq, zq_a, valid)
        total = F.mse_loss(out_, x) + rq + 0.1*cons
        total.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        o.step(); s.step()
    return float(total.detach())
out = {}
for foreach in (True, False):
    m, ld, o, s = build(foreach)
    for _ in range(40): epoch_once(m, ld, o, s)
    SNAP_W = {k: v.detach().clone() for k, v in m.state_dict().items()}
    SNAP_O = json.loads(json.dumps({"state": {k: {f: (v.tolist() if f == "step" else v.tolist()) for f, v in st.items()} for k, st in o.state_dict()["state"].items()}, "param_groups": o.state_dict()["param_groups"]})) if False else None
    SNAP_ST = {k: {f: (v.clone() if torch.is_tensor(v) else v) for f, v in st.items()} for k, st in o.state_dict()["state"].items()}
    SNAP_PG = json.loads(json.dumps(o.state_dict()["param_groups"]))
    SNAP_RNG = torch.get_rng_state().clone()
    def restore():
        m.load_state_dict(SNAP_W)
        o.load_state_dict({"state": SNAP_ST, "param_groups": SNAP_PG})
        s.load_state_dict(SNAP_SCH := SNAP_SCH) if False else None
        torch.set_rng_state(SNAP_RNG); m.train()
    # scheduler restore: re-instantiate to reset last_epoch then advance
    SNAP_SCH_LAST = sch_last = s.state_dict()
    runs = []
    for i in range(3):
        restore()
        # capture pre-step checksums on the epoch's first batch
        for batch in ld:
            x, aug, valid = batch
            if not torch.is_tensor(valid): valid = torch.tensor(valid, dtype=torch.bool)
            aug = aug.to(DEVICE); valid = valid.to(DEVICE); x = x.to(DEVICE)
            o.zero_grad()
            out_, rq, _, zq = m(x, use_sk=True)
            av = aug[valid]
            _,_,_, zq_a = m(av, use_sk=True)
            cons, _ = V.consistency_loss(zq, zq_a, valid)
            total = F.mse_loss(out_, x) + rq + 0.1*cons
            total.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            pre = {"grad": hsh(list(m.parameters())[0].grad), "param": hsh(list(m.parameters())[0]),
                   "exp_avg": opt_state_sum(o)[0], "exp_avg_sq": opt_state_sum(o)[1], "steps": opt_state_sum(o)[2][:1]}
            o.step()
            post = {"param": hsh(list(m.parameters())[0]), "exp_avg": opt_state_sum(o)[0]}
            s.step()
            break
        # finish remaining epochs of the budget quickly (loop rest)
        runs.append({"pre": pre, "post": post})
    key_pre = json.dumps(runs[0]["pre"], sort_keys=True)
    out[f"foreach={foreach}"] = {
        "pre_step_identical_all3": all(json.dumps(r["pre"], sort_keys=True) == key_pre for r in runs),
        "post_step_identical_all3": all(r["post"] == runs[0]["post"] for r in runs),
        "pre_values": runs[0]["pre"], "post_values": [r["post"] for r in runs],
    }
print(json.dumps(out, indent=1))
print("E6D_DONE")
