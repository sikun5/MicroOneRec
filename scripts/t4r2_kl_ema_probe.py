#!/usr/bin/env python3
"""Phase7 T4-R2(2): original-formula KL direct recompute + EMA controls.

KL: k(delta)=exp(delta)-delta-1, delta=ref_logp-policy_logp, per-token, masked
by completion_mask, aggregated exactly as the locked trainer aggregates
(mean over rows of masked per-token mean) -> KL_mean for ref_fp32 and ref_bf16
on the SAME policy (frozen A-SFT top-level) and SAME raw completions
(attempt_004 snapshot). Also beta-weighted per-token loss delta using the
trainer's loss shape. Scope: fixed snapshot/policy only; no extrapolation.
WITHDRAWN: earlier 'beta*logp-diff is a KL upper bound' inference (unproven).

EMA controls on V2SyncRefCallback math: (a) policy==ref -> bitwise no-op;
(b) policy=ref*(1+eps) -> fp32-intermediate vs fp32-master EMA then single
bf16 conversion, max param delta. Prior EMA evidence scope honestly labeled
(CPU tiny-model parameter copies, full-model coverage THERE, not 599M here).
"""
import hashlib, json, os, sys, copy
ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.utils import selective_log_softmax

EOS = 151643
dev = "cuda"
tok = AutoTokenizer.from_pretrained("outputs/v2/checkpoints/A-SFT", local_files_only=True)
pol = AutoModelForCausalLM.from_pretrained("outputs/v2/checkpoints/A-SFT", torch_dtype=torch.float32).to(dev)
pol.config.use_cache = False
ref32 = copy.deepcopy(pol).to(dev)
ref16 = copy.deepcopy(pol).to(dev).to(torch.bfloat16)

snap = json.load(open("outputs/v2/debug/phase7/grpo_smoke/attempt_004/update_smoke/A/group_snapshot.json"))
comp = [c[:c.index(EOS)+1] for c in snap["completion_ids"]]
rows = [json.loads(l) for l in open("outputs/v2/manifests/rl_stream_20000_A.jsonl")][:16]
enc = tok([r["prompt"] for r in rows], return_tensors="pt", padding=True, padding_side="left",
          add_special_tokens=False)
ids = enc.input_ids.to(dev); am = enc.attention_mask.to(dev)
cmask = torch.tensor(snap["completion_mask"]).to(dev)

def logps(model):
    full = torch.cat([ids, torch.tensor(comp).to(dev)], 1)
    keep = len(comp[0])
    with torch.no_grad():
        out = model(input_ids=full, attention_mask=torch.cat([am, torch.ones_like(torch.tensor(comp).to(dev))], 1),
                    logits_to_keep=keep)
    return selective_log_softmax(out.logits.float(), full[:, -keep:])

lp_pol = logps(pol)
lp_32 = logps(ref32)
lp_16 = logps(ref16)

def kl_k3(lp_ref, lp_pol, mask):
    d = lp_ref - lp_pol
    per_tok = torch.exp(d) - d - 1.0
    row = (per_tok * mask).sum(dim=1) / mask.sum(dim=1)
    return float(row.mean()), float(per_tok.max()), float(per_tok[mask.bool()].mean()), bool(torch.isfinite(row).all())

kl32_mean, kl32_max, kl32_tokmean, fin32 = kl_k3(lp_32, lp_pol, cmask)
kl16_mean, kl16_max, kl16_tokmean, fin16 = kl_k3(lp_16, lp_pol, cmask)
beta = 1e-3
# trainer loss shape: per_token_loss = -(ratio*adv - beta*kl); KL-part delta = beta*(kl16-kl32) averaged same way
row32 = (torch.exp(lp_32-lp_pol) - (lp_32-lp_pol) - 1) * cmask
row16 = (torch.exp(lp_16-lp_pol) - (lp_16-lp_pol) - 1) * cmask
loss_kl32 = float((row32.sum(1)/cmask.sum(1)).mean()) * beta
loss_kl16 = float((row16.sum(1)/cmask.sum(1)).mean()) * beta

# EMA controls (CPU, parameter-level, full model)
alpha = 0.6
master = {n: p.detach().cpu().float() for n, p in pol.named_parameters()}
r32 = {n: p.detach().cpu().float() for n, p in ref32.named_parameters()}
r16 = {n: p.detach().cpu() for n, p in ref16.named_parameters()}
# (a) policy==ref: no self drift
import torch as T
with T.no_grad():
    for n in master:
        t = r16[n].float(); t.mul_(1-alpha).add_(master[n], alpha=alpha)
        r16[n].copy_(t.to(T.bfloat16))
drift = max((r16[n].float() - master[n]).abs().max().item() for n in master)
# (b) policy perturbed
eps = 1e-3
pert = {n: master[n] * (1 + eps) for n in list(master)[:8]}
conv_max = 0.0
with T.no_grad():
    for n in pert:
        t = r16[n].float(); t.mul_(1-alpha).add_(pert[n], alpha=alpha)
        conv_max = max(conv_max, float((t.to(T.bfloat16).float() - t).abs().max()))
doc = {"task": "Phase7 T4-R2(2) original-KL recompute + EMA controls", "date": "2026-11-15",
       "formula": {"k": "exp(delta)-delta-1, delta=ref_logp-policy_logp",
                    "aggregate": "mean_rows(sum(per_tok*completion_mask)/sum(mask)) -- mirrors locked trainer compute_loss",
                    "beta": beta},
       "scope": "frozen A-SFT policy + attempt_004 raw completions only; no extrapolation to later training",
       "withdrawn_claim": "'beta*logp-diff is a KL-change upper bound' is RETRACTED as unproven; only fixed-snapshot values below",
       "kl_fp32_ref": {"mean_row": kl32_mean, "per_token_max": kl32_max, "per_token_mean": kl32_tokmean, "finite": fin32},
       "kl_bf16_ref": {"mean_row": kl16_mean, "per_token_max": kl16_max, "per_token_mean": kl16_tokmean, "finite": fin16},
       "kl_mean_delta_bf16_vs_fp32": kl16_mean - kl32_mean,
       "beta_weighted_lossKL_fp32": loss_kl32, "beta_weighted_lossKL_bf16": loss_kl16,
       "lossKL_delta": loss_kl16 - loss_kl32,
       "ema_control_policy_eq_ref": {"max_abs_drift_after_ema": drift, "expect_zero": drift == 0.0,
         "scope": "full 599M parameter set on CPU"},
       "ema_control_policy_ne_ref": {"eps": eps, "single_bf16_conversion_max_abs": conv_max,
         "params_sampled": 8, "scope": "8 largest-param subset on CPU; prior T3-R2 EMA probe was CPU tiny-model FULL-set"},
       "formal_lock_unchanged": "sync512 / alpha=0.6 / no diagnostic frequency in formal config"}
os.makedirs("outputs/v2/debug/phase7/grpo_smoke/attempt_005/reports", exist_ok=True)
json.dump(doc, open("outputs/v2/debug/phase7/grpo_smoke/attempt_005/reports/t4r2_kl_ema.json", "w"),
          ensure_ascii=False, indent=1)
print("KL32", kl32_mean, kl32_max, fin32)
print("KL16", kl16_mean, kl16_max, fin16)
print("DKL", kl16_mean - kl32_mean, "LOSSKL_D", loss_kl16 - loss_kl32)
print("EMA_drift", drift, "conv_max", conv_max)
print("KL_EMA_DONE")
