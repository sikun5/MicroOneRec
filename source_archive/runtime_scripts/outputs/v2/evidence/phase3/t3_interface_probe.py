#!/usr/bin/env python
"""T3 CPU interface probe (synthetic data only; no formal caches, no training)."""
import os, sys, json, random
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec/scripts")
sys.path.insert(0, "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec/rq")
import numpy as np
import torch
import v2_train_rqvae as V
from models.rqvae import RQVAE
from transformers import get_constant_schedule_with_warmup

torch.use_deterministic_algorithms(True)
CFG = dict(in_dim=64, num_emb_list=[16, 16, 16], e_dim=8, layers=[32, 16],
           dropout_prob=0.0, bn=False, loss_type="mse", quant_loss_weight=1.0,
           beta=0.25, kmeans_init=True, kmeans_iters=100,
           sk_epsilons=[0.0, 0.0, 0.0], sk_iters=50)
res = {"stage": "t3_interface_probe", "device": "cpu"}

def build():
    return V.V2RQVAE(**CFG)

# 1. same-seed model init identity
torch.manual_seed(42); m1 = build()
torch.manual_seed(42); m2 = build()
res["same_seed_init_allclose"] = all(torch.equal(a, b) for a, b in
                                     zip(m1.parameters(), m2.parameters()))

# 2. state_dict keys identical to locked RQVAE
torch.manual_seed(42); m_locked = RQVAE(**CFG)
res["state_dict_keys_equal_locked"] = (list(m1.state_dict().keys()) ==
                                       list(m_locked.state_dict().keys()))

# 3. forward returns quantized_sum (ST), kmeans init triggers on first forward
torch.manual_seed(42); np.random.seed(42); random.seed(42)
ma = build()
x = torch.randn(64, 64)
out, rq_loss, indices, zq = ma(x, use_sk=True)
res["forward_shapes"] = {"out": list(out.shape), "zq": list(zq.shape), "indices": list(indices.shape)}
res["vq_initted_after_forward"] = V.vq_initted_flags(ma)
res["rq_loss_finite"] = bool(torch.isfinite(rq_loss.detach())) and float(rq_loss.detach()) >= 0

# 4. ST gradient routing: zq.sum().backward -> encoder grad yes, embedding grad none (detached code path)
ma.zero_grad()
zq.sum().backward()
enc_grad = ma.encoder.mlp_layers[1].weight.grad  # first Linear
emb_grad_after_zq_only = ma.rq.vq_layers[0].embedding.weight.grad
res["st_encoder_grad_nonzero"] = bool(enc_grad is not None and enc_grad.abs().sum() > 0)
res["st_embedding_grad_none_from_zq_only"] = bool(emb_grad_after_zq_only is None)

# 5. full loss backward on a FRESH forward (previous graph freed by zq backward):
#    embedding (codebook loss) + commitment both get grads; decoder too
ma.zero_grad()
out2, rq_loss2, _idx2, _zq2 = ma(x, use_sk=True)
loss_full = torch.nn.functional.mse_loss(out2, x) + rq_loss2
loss_full.backward()
res["full_loss_embedding_grad_nonzero"] = bool(ma.rq.vq_layers[0].embedding.weight.grad is not None and
                                               ma.rq.vq_layers[0].embedding.weight.grad.abs().sum() > 0)
res["full_loss_decoder_grad_nonzero"] = bool(ma.decoder.mlp_layers[1].weight.grad is not None and
                                             ma.decoder.mlp_layers[1].weight.grad.abs().sum() > 0)

# 6. consistency loss: zq_aug pre-gathered to valid rows (real training usage);
#    both branches get gradient; empty mask -> finite 0; mismatched rows refused
zc = torch.randn(16, 8, requires_grad=True)
za = torch.randn(12, 8, requires_grad=True)   # gathered: 12 valid rows
mask = torch.tensor([True]*12 + [False]*4)
raw, n_valid = V.consistency_loss(zc, za, mask)
raw.backward()
res["consistency"] = {"n_valid": n_valid, "raw": float(raw.detach()), "finite": bool(torch.isfinite(raw.detach())),
                      "clean_grad_nonzero": bool(zc.grad.abs().sum() > 0),
                      "aug_grad_nonzero": bool(za.grad.abs().sum() > 0)}
try:
    V.consistency_loss(torch.randn(16, 8), torch.randn(16, 8), mask)
    res["consistency_gather_mismatch_refused"] = False
except ValueError:
    res["consistency_gather_mismatch_refused"] = True
empty = torch.zeros(16, dtype=torch.bool)
raw0, n0 = V.consistency_loss(torch.randn(16, 8), torch.zeros(0, 8), empty)
res["consistency_empty_mask"] = {"raw": float(raw0), "n_valid": n0, "finite": bool(torch.isfinite(raw0))}

# 7. get_indices single-representation inference == forward indices (no weight change in between)
with torch.no_grad():
    idx2 = ma.get_indices(x, use_sk=False)
out2, rq2, idx_f2, zq2 = ma(x, use_sk=True)
res["get_indices_shape"] = list(idx2.shape)
res["get_indices_matches_forward"] = bool(torch.equal(idx2, idx_f2))

# 8. kmeans init determinism: same torch+np seed -> identical codebooks
torch.manual_seed(42); np.random.seed(42); random.seed(42)
mb = build()
_ = mb(x, use_sk=True)
res["kmeans_init_bitwise_equal"] = all(torch.equal(a.embedding.weight.data, b.embedding.weight.data) for a, b in
                                       zip(ma.rq.vq_layers, mb.rq.vq_layers))

# 9. checkpoint roundtrip: model/optimizer/scheduler/rng/init_state
optimizer = torch.optim.AdamW(ma.parameters(), lr=1e-3, weight_decay=0.0)
optimizer.step()  # populate state
scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=3)
for _ in range(5):
    optimizer.step(); scheduler.step()
state = {"state_dict": ma.state_dict(), "optimizer": optimizer.state_dict(),
         "scheduler": scheduler.state_dict(), "rng": V.get_rng_state(),
         "epoch_completed": 5, "global_step": 5,
         "init_state": {"vq_initted": V.vq_initted_flags(ma), "kmeans_triggered_on_clean": True},
         "config": {"schema": V.SCHEMA}}
torch.save(state, "/tmp/t3_roundtrip.pth", pickle_protocol=4)
s2 = torch.load("/tmp/t3_roundtrip.pth", map_location="cpu", weights_only=False)
mr = build()
mr.load_state_dict(s2["state_dict"])
res["ckpt_state_dict_bitwise_equal"] = all(torch.equal(a, b) for a, b in
                                           zip(ma.state_dict().values(), mr.state_dict().values()))
opt2 = torch.optim.AdamW(mr.parameters(), lr=1e-3, weight_decay=0.0)
opt2.load_state_dict(s2["optimizer"])
sch2 = get_constant_schedule_with_warmup(opt2, num_warmup_steps=3)
sch2.load_state_dict(s2["scheduler"])
res["ckpt_scheduler_last_epoch"] = sch2.last_epoch
V.set_vq_initted(mr, s2["init_state"]["vq_initted"])
res["ckpt_initted_restored"] = V.vq_initted_flags(mr)
rng_before = V.get_rng_state()  # current stream differs from saved
V.set_rng_state(s2["rng"])
rng_after = V.get_rng_state()
res["ckpt_rng_restored"] = {
    "torch_equal": bool(torch.equal(rng_after["torch"], s2["rng"]["torch"])),
    "numpy_equal": bool(all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
                            for a, b in zip(rng_after["numpy"], s2["rng"]["numpy"]))),
    "python_equal": bool(rng_after["python"] == s2["rng"]["python"]),
    "cuda_keys_present": "torch_cuda" in rng_after,
}

# 10. (T3 review item 1) consistency numerical definition: per-norm clamp, NOT product clamp
torch.manual_seed(7)
zca = torch.randn(4, 8); zaa = torch.randn(4, 8)
raw_a, n_a = V.consistency_loss(zca, zaa, torch.ones(4, dtype=torch.bool))
denom_manual = zca.norm(dim=1).clamp_min(1e-8) * zaa.norm(dim=1).clamp_min(1e-8)
cos_manual = (zca * zaa).sum(1) / denom_manual
res["cons_regular_matches_formula"] = bool(torch.allclose(raw_a.detach(), (1 - cos_manual).mean()))
# small norm same-direction: norms 1e-5 -> cos must be 1.0 (product clamp would give 0.01)
v1 = torch.tensor([[1e-5] + [0.0] * 7]); v2 = torch.tensor([[2e-5] + [0.0] * 7])
raw_s, _ = V.consistency_loss(v1, v2, torch.ones(1, dtype=torch.bool))
res["cons_smallnorm_cos_value"] = float(1.0 - raw_s.detach())
res["cons_smallnorm_is_one"] = bool(float(raw_s.detach()) < 1e-6)  # raw = 1-cos ≈ 0 ⇒ cos ≈ 1
# zero vector row: finite cos 0
z0 = torch.zeros(1, 8); zn = torch.randn(1, 8)
raw_z, _ = V.consistency_loss(z0, zn, torch.ones(1, dtype=torch.bool))
res["cons_zerovec_raw"] = float(raw_z.detach()); res["cons_zerovec_finite"] = bool(torch.isfinite(raw_z))
# mixed valid incl a zero-vector valid row: finite mean
zcm = torch.randn(3, 8); zcm[1] = 0.0
zam3 = torch.randn(3, 8)
raw_m, n_m = V.consistency_loss(zcm, zam3, torch.ones(3, dtype=torch.bool))
res["cons_mixed_valid"] = {"n_valid": n_m, "raw": float(raw_m.detach()),
                           "finite": bool(torch.isfinite(raw_m))}

# 11. (T3 review item 2) strict resume: full semantic config compare
import copy
base_cfg = {"schema": V.SCHEMA, "run_id": "RQ-C-42", "method": "C", "seed": 42,
            "consistency_weight": 0.1,
            "inputs": {"clean": {"path": "a.npy", "sha256": "h1"}},
            "locked_source": {"rq/rqvae.py": "s1"}, "script_sha256": "p1",
            "protocol": {"epochs": 10000, "batch_size": 20480, "learning_rate": 1e-3,
                         "warmup_epochs": 50, "num_workers": 4, "grad_clip_norm": 1.0,
                         "beta": 0.25, "quant_loss_weight": 1.0, "loss_type": "mse"},
            "environment": {"python": "3.10.13", "torch": "2.14.0+cu130",
                            "numpy": "2.2.6", "cuda": "13.0",
                            "sklearn": "1.5.2", "scipy": "1.15.3"}}
res["resume_accepts_identical"] = bool(V.verify_resume_config(copy.deepcopy(base_cfg),
                                                              copy.deepcopy(base_cfg)))
def resume_refuses(mod):
    c2 = copy.deepcopy(base_cfg); mod(c2)
    try:
        V.verify_resume_config(base_cfg, c2)
        return False
    except ValueError:
        return True
res["resume_refuses_lr_change"] = resume_refuses(lambda c: c["protocol"].__setitem__("learning_rate", 2e-3))
res["resume_refuses_warmup_change"] = resume_refuses(lambda c: c["protocol"].__setitem__("warmup_epochs", 100))
res["resume_refuses_batch_change"] = resume_refuses(lambda c: c["protocol"].__setitem__("batch_size", 4096))
res["resume_refuses_epochs_change"] = resume_refuses(lambda c: c["protocol"].__setitem__("epochs", 5000))
res["resume_refuses_input_hash_change"] = resume_refuses(lambda c: c["inputs"]["clean"].__setitem__("sha256", "h2"))
res["resume_refuses_script_hash_change"] = resume_refuses(lambda c: c.__setitem__("script_sha256", "p2"))
res["resume_refuses_torch_version_change"] = resume_refuses(lambda c: c["environment"].__setitem__("torch", "2.13.0"))
res["resume_refuses_beta_change"] = resume_refuses(lambda c: c["protocol"].__setitem__("beta", 0.30))
res["resume_refuses_sklearn_version_change"] = resume_refuses(lambda c: c["environment"].__setitem__("sklearn", "1.6.0"))

print(json.dumps(res, ensure_ascii=False, indent=1))
ok = (res["same_seed_init_allclose"] and res["state_dict_keys_equal_locked"]
      and res["rq_loss_finite"] and res["st_encoder_grad_nonzero"]
      and res["st_embedding_grad_none_from_zq_only"] and res["full_loss_embedding_grad_nonzero"]
      and res["full_loss_decoder_grad_nonzero"] and res["consistency"]["clean_grad_nonzero"]
      and res["consistency"]["aug_grad_nonzero"] and res["consistency_empty_mask"]["finite"]
      and res["consistency_gather_mismatch_refused"]
      and res["get_indices_matches_forward"] and res["kmeans_init_bitwise_equal"]
      and res["ckpt_state_dict_bitwise_equal"] and res["ckpt_rng_restored"]["torch_equal"]
      and res["ckpt_rng_restored"]["numpy_equal"] and res["ckpt_rng_restored"]["python_equal"]
      and res["cons_regular_matches_formula"] and res["cons_smallnorm_is_one"]
      and res["cons_zerovec_finite"] and res["cons_mixed_valid"]["finite"]
      and res["resume_accepts_identical"] and res["resume_refuses_lr_change"]
      and res["resume_refuses_warmup_change"] and res["resume_refuses_batch_change"]
      and res["resume_refuses_epochs_change"] and res["resume_refuses_input_hash_change"]
      and res["resume_refuses_script_hash_change"] and res["resume_refuses_torch_version_change"]
      and res["resume_refuses_beta_change"] and res["resume_refuses_sklearn_version_change"])
print("PROBE_ALL_OK:", ok)
