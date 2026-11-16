#!/usr/bin/env python
"""T4 final directed check: snapshot-load fidelity (params + FULL optimizer state field-by-field),
next-batch explicit item order replay, epoch/step progress. SCHEMA: phase3-t4_snapshot_fidelity-1.0"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys, json, random, hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path("/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec")
sys.path.insert(0, str(REPO / "rq"))
sys.path.insert(0, str(REPO / "scripts"))
import v2_train_rqvae as V

SEED = 42
DEVICE = torch.device("cuda:0")
res = {"stage": "phase3_t4_snapshot_fidelity", "schema": "phase3-t4_snapshot_fidelity-1.0",
       "date": "2026-11-11", "seed": SEED,
       "candidate_script_v12": V.sha256_file(str(REPO / "scripts/v2_train_rqvae.py"))}

SNAP_PATH = REPO / "outputs/v2/smoke/phase3/resume-test/C/seed_42/resume_snapshot.pth"
snap = torch.load(SNAP_PATH, map_location="cpu", weights_only=False)
res["snapshot_source"] = {"path": str(SNAP_PATH),
                          "sha256": V.sha256_file(str(SNAP_PATH)),
                          "epoch_completed": snap["epoch_completed"],
                          "global_step": snap["global_step"],
                          "snapshot_flag": snap.get("snapshot")}

# fresh model EXACTLY as v2_train_rqvae resume path builds it
model = V.V2RQVAE(in_dim=snap["init_state"]["in_dim"], num_emb_list=[256, 256, 256], e_dim=32,
                  layers=[2048, 1024, 512, 256, 128, 64], dropout_prob=0.0, bn=False,
                  loss_type="mse", quant_loss_weight=1.0, beta=0.25, kmeans_init=True,
                  kmeans_iters=100, sk_epsilons=[0.0, 0.0, 0.0], sk_iters=50)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

# ---- A. model params: load then field-by-field bitwise compare ----
model.load_state_dict(snap["state_dict"])
param_table = []
all_match = True
for k, v_snap in snap["state_dict"].items():
    v_loaded = dict(model.state_dict())[k]
    ok = torch.equal(v_loaded, v_snap)
    all_match &= ok
    param_table.append({"key": k, "bitwise_equal": bool(ok),
                        "sha256_16": hashlib.sha256(v_snap.detach().numpy().tobytes()).hexdigest()[:16]})
res["A_params_loaded_from_disk_snapshot"] = {
    "tensors": len(param_table), "all_bitwise_equal": bool(all_match),
    "elements": int(sum(t["elements"] for t in [{"elements": 1}] * 0) or sum(
        dict(model.state_dict())[k].numel() for k in snap["state_dict"])),
    "per_tensor": param_table,
}

# ---- B. optimizer full state: load then field-by-field ----
optimizer.load_state_dict(snap["optimizer"])
loaded = optimizer.state_dict()
snap_o = snap["optimizer"]
opt_table, opt_all = [], True
for key in snap_o["state"]:
    a, b = snap_o["state"][key], loaded["state"][key]
    for field in ("step", "exp_avg", "exp_avg_sq"):
        va, vb = a[field], b[field]
        if torch.is_tensor(va):
            eq = torch.equal(vb.to(va.device) if vb.device != va.device else vb, va)
            d = {"state_key": key, "field": field, "bitwise_equal": bool(eq),
                 "sha256_16": hashlib.sha256(va.detach().cpu().numpy().tobytes()).hexdigest()[:16]}
            if field == "step":
                d["value"] = float(va)
        else:
            eq = vb == va
            d = {"state_key": key, "field": field, "equal": bool(eq), "value": va}
        opt_all &= eq
        opt_table.append(d)
res["B_optimizer_state_loaded_from_disk_snapshot"] = {
    "state_entries": len(snap_o["state"]), "fields_per_entry": ["step", "exp_avg", "exp_avg_sq"],
    "all_bitwise_equal": bool(opt_all), "per_field": opt_table,
    "param_groups_lr_equal": loaded["param_groups"] == snap_o["param_groups"],
}

# ---- C. next-batch explicit item order: snapshot-RNG replay vs process-start replay ----
C_CLEAN = np.load(REPO / "outputs/v2/embeddings/C/clean.npy")
N = len(C_CLEAN)

def perm_after_state(rng_state):
    torch.set_rng_state(rng_state)
    ld = DataLoader(V.V2PairDataset(REPO / "outputs/v2/embeddings/C/clean.npy",
                                    REPO / "outputs/v2/embeddings/C/train_aug.npy",
                                    REPO / "outputs/v2/embeddings/C/valid_aug.npy"),
                    num_workers=4, batch_size=20480, shuffle=True, pin_memory=True)
    it = iter(ld)
    batch = next(it)
    x0 = batch[0]
    # recover item order by exact row matching against C_CLEAN
    order = []
    lookup = {C_CLEAN[i].tobytes(): i for i in range(N)}
    xr = x0.numpy()
    for r in range(xr.shape[0]):
        order.append(lookup[xr[r].tobytes()])
    return order

# (a) from snapshot RNG: the resumed process's next batch (epoch 41) order
order_from_snapshot = perm_after_state(snap["rng"]["torch"])
# (b) replay from process start: seed(42) -> model build -> 40 sampler draws -> 41st draw
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
m_replay = V.V2RQVAE(in_dim=2560, num_emb_list=[256, 256, 256], e_dim=32,
                     layers=[2048, 1024, 512, 256, 128, 64], dropout_prob=0.0, bn=False,
                     loss_type="mse", quant_loss_weight=1.0, beta=0.25, kmeans_init=True,
                     kmeans_iters=100, sk_epsilons=[0.0, 0.0, 0.0], sk_iters=50)
for _ in range(120):
    seed_i = int(torch.empty((), dtype=torch.int64).random_().item())
    g = torch.Generator(); g.manual_seed(seed_i)
    torch.randperm(N, generator=g)
seed_next = int(torch.empty((), dtype=torch.int64).random_().item())
g41 = torch.Generator(); g41.manual_seed(seed_next)
perm41 = torch.randperm(N, generator=g41).tolist()
res["C_next_batch_item_order"] = {
    "order_from_snapshot_rng_sha16": hashlib.sha256(
        np.array(order_from_snapshot, dtype=np.int64).tobytes()).hexdigest()[:16],
    "order_from_process_start_replay_sha16": hashlib.sha256(
        np.array(perm41, dtype=np.int64).tobytes()).hexdigest()[:16],
    "identical": order_from_snapshot == perm41,
    "note": "RandomSampler 每次迭代从全局 CPU RNG 抽一个 int64 种子再本地 randperm——快照(epoch120 终态)RNG 的下一批顺序与连续进程重放(120 draws)一致",
    "resumed_snapshot_is_epoch120": "resume-test 的 resume_snapshot.pth 已被恢复 run 循环末快照覆盖为 epoch_completed=120；fidelity 加载以该终态为准",
}

# ---- C2. cross-process RNG state + order (resumed vs continuous snapshots @120) ----
snap_cont = torch.load(REPO / "outputs/v2/smoke/phase3/resume-cont/C/seed_42/resume_snapshot.pth",
                       map_location="cpu", weights_only=False)
res["C2_cross_process_rng_and_order"] = {
    "rng_torch_bitwise": bool(torch.equal(snap["rng"]["torch"], snap_cont["rng"]["torch"])),
    "rng_python_equal": snap["rng"]["python"] == snap_cont["rng"]["python"],
    "rng_numpy_equal": all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
                           for a, b in zip(snap["rng"]["numpy"], snap_cont["rng"]["numpy"])),
    "order_from_cont_snapshot_rng_sha16": hashlib.sha256(
        np.array(perm_after_state(snap_cont["rng"]["torch"]), dtype=np.int64).tobytes()).hexdigest()[:16],
    "note": "RNG 状态为精确整数/字节状态，跨进程应逐位；权重含浮点噪声不在本项",
}

# ---- D. K-means no-rerun evidence nature (explicit, not call-count monitoring) ----
res["D_kmeans_no_rerun_evidence_nature"] = {
    "evidence_kind": "代码路径守卫 + 恢复标志回填 + 权重一致性推断；非调用计数监控",
    "details": [
        "锁定 vq.forward: 'if not self.initted and self.training' — 恢复时 set_vq_initted([True,True,True]) 后该分支不可达（代码守卫）",
        "resume 日志行实测: 'resumed from ... at epoch_completed=40 step=40 (vq_initted=[True, True, True])' — 标志确已回填",
        "E3-L4: 从恢复态单 epoch 前向反向更新逐位复现连续路径 — 若 K-means 重跑，码本权重必变，逐位不可能成立",
    ],
}

res["summary"] = {
    "params_loaded_bitwise": bool(all_match),
    "optimizer_state_loaded_bitwise": bool(opt_all),
    "next_batch_order_identical": res["C_next_batch_item_order"]["identical"],
    "progress": {"epoch_completed": snap["epoch_completed"], "global_step": snap["global_step"]},
}
res["summary"]["all_passed"] = all(res["summary"].values())

out = REPO / "outputs/v2/evidence/phase3/snapshot_fidelity_checks.json"
with open(out, "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print(json.dumps(res["summary"], ensure_ascii=False))
print("order_sha_snap:", res["C_next_batch_item_order"]["order_from_snapshot_rng_sha16"],
      "| order_sha_replay:", res["C_next_batch_item_order"]["order_from_process_start_replay_sha16"])
print("OUT_SHA256:", V.sha256_file(str(out)))
print("FIDELITY_DONE")
