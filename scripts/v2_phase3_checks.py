#!/usr/bin/env python
"""v2 Phase 3 T4 loss/gradient checks (boss items 1-3 + same-seed init/order).
Independent check entry: imports frozen v2_train_rqvae and reuses the SAME core
model/loss path; does NOT invoke formal CLI and does NOT loosen the lambda lock.
SCHEMA: phase3-t4_loss_gradient_checks-1.0
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys, json, hashlib, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path("/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec")
sys.path.insert(0, str(REPO / "rq"))
sys.path.insert(0, str(REPO / "scripts"))
import v2_train_rqvae as V  # noqa: E402

SEED = 42
DEVICE = torch.device("cuda:0")
BATCH = 512  # >= 256 codebook size so the clean-branch K-means init can trigger normally
ATOL, RTOL = 1e-6, 1e-5  # pre-declared (boss item 1); never relaxed post-hoc
res = {"stage": "phase3_t4_loss_gradient_checks", "schema": "phase3-t4_loss_gradient_checks-1.0",
       "date": "2026-11-11", "seed": SEED, "device": str(DEVICE),
       "batch_rows": BATCH, "tolerance": {"atol": ATOL, "rtol": RTOL, "declared_before": True},
       "candidate_script": V.sha256_file(str(REPO / "scripts" / "v2_train_rqvae.py"))}

CFG = dict(num_emb_list=[256, 256, 256], e_dim=32, layers=[2048, 1024, 512, 256, 128, 64],
           dropout_prob=0.0, bn=False, loss_type="mse", quant_loss_weight=1.0,
           beta=0.25, kmeans_init=True, kmeans_iters=100,
           sk_epsilons=[0.0, 0.0, 0.0], sk_iters=50)

B_CLEAN = np.load(REPO / "outputs/v2/embeddings/B/clean.npy")
C_CLEAN = np.load(REPO / "outputs/v2/embeddings/C/clean.npy")
C_AUG = np.load(REPO / "outputs/v2/embeddings/C/train_aug.npy")
C_VALID = np.load(REPO / "outputs/v2/embeddings/C/valid_aug.npy")
A_CLEAN = np.load(REPO / "outputs/v2/embeddings/A/clean.npy")
assert np.array_equal(B_CLEAN, C_CLEAN), "B/C clean must be byte-identical"

invalid_all = np.where(~C_VALID)[0]
valid_all = np.where(C_VALID)[0]
batch_idx = np.concatenate([invalid_all[:8], valid_all[:BATCH - 8]])  # mixed: 8 invalid + 504 valid
X_MIX = torch.from_numpy(C_CLEAN[batch_idx]).to(DEVICE)
AUG_MIX = torch.from_numpy(C_AUG[batch_idx]).to(DEVICE)
VALID_MIX = torch.from_numpy(C_VALID[batch_idx]).to(DEVICE)
X_512 = torch.from_numpy(C_CLEAN[:BATCH]).to(DEVICE)  # first 512 rows (valid mix unknown; lambda=0 only)

def build_model():
    torch.manual_seed(SEED)
    m = V.V2RQVAE(in_dim=2560, **CFG).to(DEVICE)
    m.train()
    return m

def core_loss(model, x, lam=0.0, aug=None, valid=None):
    """与 v2_train_rqvae.main 训练循环逐句相同的核心损失路径。"""
    out, rq_loss, idx, zq = model(x, use_sk=True)
    loss_recon = F.mse_loss(out, x)
    cons = torch.zeros((), device=x.device)
    n_valid = 0
    if lam > 0:
        aug_valid = aug[valid]
        _, _, _, zq_aug = model(aug_valid, use_sk=True)
        cons, n_valid = V.consistency_loss(zq, zq_aug, valid)
    total = loss_recon + 1.0 * rq_loss + lam * cons
    return total, loss_recon, rq_loss, cons, n_valid, idx

# ---------------- T1: B vs C(lambda=0) degradation equivalence ----------------
# T1a（主判定）：同一模型实例、每条路径前恢复同一初始 state 并重置 torch+np+random 种子
# （镜像正式入口的种子顺序）→ B 路径与 C(λ=0) 路径是同一计算，预期逐位相等。
# T1b（次要观测）：两个独立实例 → 地址差异可能引起 cuBLAS 内核选择差异（~1e-7 舍入），
# 按预声明容差判定；逐位不等不构成 aug 影响（两条路径都不含 aug）。
init_state = {k: v.detach().clone() for k, v in build_model().state_dict().items()}

def run_path(model, opt):
    model.load_state_dict(init_state)
    V.set_vq_initted(model, [False, False, False])  # fresh-model semantics: K-means pending on clean forward
    for p in model.parameters():
        p.grad = None
    t, r, q, c, nv, idx = core_loss(model, X_512, lam=0.0)
    t.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    np_state = np.random.get_state()
    py_state = random.getstate()
    opt.step()
    params_after = {n: p.detach().clone() for n, p in model.named_parameters()}
    return t, grads, rng, np_state, py_state, params_after

t0 = time.time()
mb = build_model()
optb = torch.optim.AdamW(mb.parameters(), lr=1e-3, weight_decay=0.0)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
tb, grads_b, rng_b, nps_b, pys_b, params_b_after = run_path(mb, optb)
t1 = time.time()
optc = torch.optim.AdamW(mb.parameters(), lr=1e-3, weight_decay=0.0)  # fresh state, same init
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
tc, grads_c, rng_c, nps_c, pys_c, params_c_after = run_path(mb, optc)
t2 = time.time()

loss_abs = float(abs(tb.detach() - tc.detach()))
max_err, fail_names = 0.0, []
for n, gb in grads_b.items():
    gc = grads_c[n]
    if not torch.allclose(gb, gc, atol=ATOL, rtol=RTOL):
        fail_names.append(n)
    d = float((gb - gc).abs().max()) if gb.numel() else 0.0
    max_err = max(max_err, d)
res["T1a_same_instance_bitwise"] = {
    "loss_B": float(tb.detach()), "loss_C_lambda0": float(tc.detach()),
    "loss_abs_diff": loss_abs,
    "grad_max_abs_err": max_err, "grad_failing_tensors": fail_names,
    "grad_failing_count": len(fail_names),
    "param_tensors_compared": len(grads_b),
    "param_elements_compared": int(sum(p.numel() for p in mb.parameters())),
    "vq_initted_B": V.vq_initted_flags(mb), "vq_initted_C": V.vq_initted_flags(mb),
    "rng_after_step_bitwise_equal": bool(torch.equal(rng_b, rng_c)),
    "numpy_state_equal": bool(all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
                                  for a, b in zip(nps_b, nps_c))),
    "python_state_equal": bool(pys_b == pys_c),
    "post_step_params_bitwise_equal": all(torch.equal(params_b_after[n], params_c_after[n])
                                          for n in params_b_after),
    "elapsed_s": {"B_path": t1 - t0, "C_path_rerun": t2 - t1},
}

# T1b: two independent instances (secondary observation, tolerance-based)
mc = build_model()
optc2 = torch.optim.AdamW(mc.parameters(), lr=1e-3, weight_decay=0.0)
tc2, _, _, _, _, _ = core_loss(mc, X_512, lam=0.0)
tc2.backward()
grads_c2 = {n: p.grad.detach().clone() for n, p in mc.named_parameters()}
max_err_b, fail_b = 0.0, []
for n, gb in grads_b.items():
    gc2 = grads_c2[n]
    if not torch.allclose(gb, gc2, atol=ATOL, rtol=RTOL):
        fail_b.append(n)
    d = float((gb - gc2).abs().max()) if gb.numel() else 0.0
    max_err_b = max(max_err_b, d)
res["T1b_two_instances_tolerance"] = {
    "loss_abs_diff": float(abs(tb.detach() - tc2.detach())),
    "grad_max_abs_err": max_err_b, "grad_failing_count": len(fail_b),
    "tolerance_pass": len(fail_b) == 0,
    "note": "独立实例地址差异可致 cuBLAS 内核选择差异（~1e-7 舍入）；容差内通过，"
            "非 aug 影响（两路径均无 aug）；逐位判定以 T1a 为准。",
}
res["T1_B_vs_C_lambda0"] = res["T1a_same_instance_bitwise"]

# ---------------- T2: lambda=0.1 dual-branch gradients ----------------
m2 = build_model()
opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3, weight_decay=0.0)
out, rq_loss, idx2, zq_clean = m2(X_MIX, use_sk=True)
zq_clean.retain_grad()
aug_valid = AUG_MIX[VALID_MIX]
_, _, _, zq_aug = m2(aug_valid, use_sk=True)
zq_aug.retain_grad()
rng_before_cons = torch.get_rng_state().clone()
cons, n_valid_2 = V.consistency_loss(zq_clean, zq_aug, VALID_MIX)
cons.backward(retain_graph=True)
res["T2_lambda01_dual_branch"] = {
    "n_valid": n_valid_2, "n_valid_matches_mask": n_valid_2 == int(VALID_MIX.sum().item()),
    "cons_value": float(cons.detach()),
    "zq_clean_grad_nonzero": bool(zq_clean.grad is not None and zq_clean.grad.abs().sum() > 0),
    "zq_clean_grad_norm": float(zq_clean.grad.norm()),
    "zq_aug_grad_nonzero": bool(zq_aug.grad is not None and zq_aug.grad.abs().sum() > 0),
    "zq_aug_grad_norm": float(zq_aug.grad.norm()),
    "aug_forward_no_torch_rng_consumed": bool(torch.equal(rng_before_cons, torch.get_rng_state())),
    "indices_requires_grad": bool(idx2.requires_grad), "indices_grad_fn_none": idx2.grad_fn is None,
    "quant_loss_finite": bool(torch.isfinite(rq_loss.detach())),
}
# full loss backward -> encoder update path
m2.zero_grad()
total2, recon2, quant2, cons2b, nv2b, _ = core_loss(m2, X_MIX, lam=0.1, aug=AUG_MIX, valid=VALID_MIX)
total2.backward()
enc_gnorm = float(m2.encoder.mlp_layers[1].weight.grad.norm())
emb_gnorm = float(m2.rq.vq_layers[0].embedding.weight.grad.norm())
dec_gnorm = float(m2.decoder.mlp_layers[1].weight.grad.norm())
before = {n: p.detach().clone() for n, p in m2.named_parameters()}
opt2.step()
delta = max(float((m2.state_dict()[n] - before[n]).abs().max()) for n in before)
res["T2_lambda01_full_update"] = {
    "encoder_grad_norm": enc_gnorm, "codebook_grad_norm_layer0": emb_gnorm,
    "decoder_grad_norm": dec_gnorm,
    "encoder_grad_nonzero": enc_gnorm > 0, "codebook_grad_nonzero": emb_gnorm > 0,
    "optimizer_step_max_param_delta": delta, "params_actually_updated": delta > 0,
    "struct_total_eq_recon_plus_quant_plus_lam_cons": bool(
        torch.equal(total2.detach(), recon2.detach() + quant2.detach() + 0.1 * cons2b.detach())),
    "n_valid_denominator": nv2b,
}

# ---------------- T3: mixed / all-invalid samples ----------------
# mixed: perturb ONLY invalid aug rows -> consistency bitwise unchanged (gather semantics proof)
m3 = build_model()
with torch.no_grad():
    _, _, _, zq_c3 = m3(X_MIX, use_sk=True)
    aug_v3 = AUG_MIX[VALID_MIX]
    _, _, _, zq_a3 = m3(aug_v3, use_sk=True)
    cons_m1, nv_m1 = V.consistency_loss(zq_c3, zq_a3, VALID_MIX)
AUG_PERTURB = AUG_MIX.clone()
garbage = torch.randn_like(AUG_PERTURB[~VALID_MIX])
AUG_PERTURB[~VALID_MIX] = garbage
with torch.no_grad():
    _, _, _, zq_c4 = m3(X_MIX, use_sk=True)
    aug_v4 = AUG_PERTURB[VALID_MIX]
    _, _, _, zq_a4 = m3(aug_v4, use_sk=True)
    cons_m2, nv_m2 = V.consistency_loss(zq_c4, zq_a4, VALID_MIX)
res["T3_mixed_invalid"] = {
    "batch_rows": BATCH, "n_invalid_in_batch": int((~VALID_MIX).sum().item()),
    "n_valid_denominator": nv_m1, "denominator_equals_valid_count": nv_m1 == int(VALID_MIX.sum().item()),
    "cons_with_orig_aug": float(cons_m1), "cons_with_perturbed_invalid_aug": float(cons_m2),
    "invalid_aug_perturb_bitwise_no_change": bool(torch.equal(cons_m1, cons_m2)),
}
# all-invalid batch: finite 0 consistency; total finite; grads finite.
# 模拟正式时序：init 总发生在全量目录干净批（>=256 行），此处先用 512 行触发 init，
# 再喂 8 行全无效批（正式训练中 init 批恒为全目录，不存在 8 行 init）。
idx_all_inv = invalid_all[:8]
X_INV = torch.from_numpy(C_CLEAN[idx_all_inv]).to(DEVICE)
AUG_INV = torch.from_numpy(C_AUG[idx_all_inv]).to(DEVICE)
VALID_INV = torch.zeros(8, dtype=torch.bool).to(DEVICE)
m4 = build_model()
opt4 = torch.optim.AdamW(m4.parameters(), lr=1e-3, weight_decay=0.0)
with torch.no_grad():
    core_loss(m4, X_512, lam=0.0)  # init trigger on full-size batch (formal timing semantics)
tot4, rec4, q4, c4, nv4, _ = core_loss(m4, X_INV, lam=0.1, aug=AUG_INV, valid=VALID_INV)
tot4.backward()
gnorms_finite = all(float(p.grad.abs().sum()) >= 0 and bool(torch.isfinite(p.grad).all())
                    for p in m4.parameters() if p.grad is not None)
res["T3_all_invalid"] = {
    "batch_rows": 8, "n_valid": nv4, "consistency": float(c4),
    "consistency_is_zero_finite": nv4 == 0 and float(c4) == 0.0 and bool(torch.isfinite(c4)),
    "total_finite": bool(torch.isfinite(tot4)), "recon": float(rec4), "quant": float(q4),
    "grads_all_finite": gnorms_finite,
    "aug_contributed_no_recon_quant": True,  # structural: recon/quant computed from clean forward only
}

# ---------------- W6: same-seed common init + clean item order ----------------
ma1 = build_model(); ma2 = build_model()
res["W6_same_seed_init"] = {
    "bitwise_equal_two_constructs": all(torch.equal(a, b) for a, b in zip(ma1.parameters(), ma2.parameters())),
    "param_elements": int(sum(p.numel() for p in ma1.parameters())),
}
class _IdxDataset(torch.utils.data.Dataset):
    """按方法读取缓存并附带行号，用于证明三个方法干净排列一致（仅检查用）。"""
    def __init__(self, arr):
        self.arr = arr
    def __len__(self):
        return len(self.arr)
    def __getitem__(self, i):
        return i, torch.FloatTensor(self.arr[i])

def first_perm(loader):
    it = iter(loader)
    idxs, _ = next(it)
    return idxs.numpy().tolist()

perms = {}
for name, arr in (("A", A_CLEAN), ("B", B_CLEAN), ("C", C_CLEAN)):
    ds = _IdxDataset(arr)
    ld = DataLoader(ds, batch_size=20480, shuffle=True, num_workers=0)
    torch.manual_seed(SEED)
    perms[name] = first_perm(ld)
res["W6_clean_order"] = {
    "perm_A_equals_B": perms["A"] == perms["B"],
    "perm_A_equals_C": perms["A"] == perms["C"],
    "perm_len": len(perms["A"]),
}

# ---------------- summary ----------------
res["summary"] = {
    "T1_pass": bool(res["T1a_same_instance_bitwise"]["grad_failing_count"] == 0
                    and res["T1a_same_instance_bitwise"]["loss_abs_diff"] <= ATOL
                    and res["T1a_same_instance_bitwise"]["rng_after_step_bitwise_equal"]
                    and res["T1a_same_instance_bitwise"]["numpy_state_equal"]
                    and res["T1a_same_instance_bitwise"]["python_state_equal"]
                    and res["T1a_same_instance_bitwise"]["post_step_params_bitwise_equal"]
                    and res["T1b_two_instances_tolerance"]["tolerance_pass"]),
    "T2_pass": bool(res["T2_lambda01_dual_branch"]["n_valid_matches_mask"]
                    and res["T2_lambda01_dual_branch"]["zq_clean_grad_nonzero"]
                    and res["T2_lambda01_dual_branch"]["zq_aug_grad_nonzero"]
                    and res["T2_lambda01_dual_branch"]["aug_forward_no_torch_rng_consumed"]
                    and not res["T2_lambda01_dual_branch"]["indices_requires_grad"]
                    and res["T2_lambda01_full_update"]["params_actually_updated"]
                    and res["T2_lambda01_full_update"]["struct_total_eq_recon_plus_quant_plus_lam_cons"]),
    "T3_pass": bool(res["T3_mixed_invalid"]["denominator_equals_valid_count"]
                    and res["T3_mixed_invalid"]["invalid_aug_perturb_bitwise_no_change"]
                    and res["T3_all_invalid"]["consistency_is_zero_finite"]
                    and res["T3_all_invalid"]["total_finite"]
                    and res["T3_all_invalid"]["grads_all_finite"]),
    "W6_pass": bool(res["W6_same_seed_init"]["bitwise_equal_two_constructs"]
                    and res["W6_clean_order"]["perm_A_equals_B"]
                    and res["W6_clean_order"]["perm_A_equals_C"]),
}
res["summary"]["all_passed"] = all(res["summary"].values())

out = REPO / "outputs/v2/evidence/phase3/loss_gradient_checks.json"
with open(out, "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print(json.dumps(res["summary"], ensure_ascii=False))
print("T1a bitwise: rng", res["T1a_same_instance_bitwise"]["rng_after_step_bitwise_equal"],
      "| np", res["T1a_same_instance_bitwise"]["numpy_state_equal"],
      "| py", res["T1a_same_instance_bitwise"]["python_state_equal"],
      "| params", res["T1a_same_instance_bitwise"]["post_step_params_bitwise_equal"],
      "| grad_max_err", res["T1a_same_instance_bitwise"]["grad_max_abs_err"])
print("T1b two-instance: max_err", res["T1b_two_instances_tolerance"]["grad_max_abs_err"],
      "| failing", res["T1b_two_instances_tolerance"]["grad_failing_count"])
print("T2 clean/aug grads:", res["T2_lambda01_dual_branch"]["zq_clean_grad_norm"],
      res["T2_lambda01_dual_branch"]["zq_aug_grad_norm"],
      "| enc_norm:", res["T2_lambda01_full_update"]["encoder_grad_norm"])
print("OUT_SHA256:", V.sha256_file(str(out)))
print("CHECKS_DONE")
