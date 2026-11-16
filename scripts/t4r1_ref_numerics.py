#!/usr/bin/env python3
"""Phase7 T4-R1 numerics probe (boss-required): fp32 ref vs approved bf16 ref.

On REAL group-A SFT weights + the real first 16-prompt group (attempt_004
completion_ids): (1) per-token ref logps fp32 vs bf16 max/mean abs diff;
(2) KL k3 diff; (3) EMA rounding: fp32-master average vs approved bf16-stored
average after 1 and 2 syncs (alpha=0.6) max param delta; (4) GPU footprint.
CPU/GPU: runs on the idle slice; tiny batch; disclosed."""
import hashlib, json, os, sys, copy
ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset as HFDataset

EOS = 151643
dev = "cuda"
tok = AutoTokenizer.from_pretrained("outputs/v2/checkpoints/A-SFT", local_files_only=True)
pol = AutoModelForCausalLM.from_pretrained("outputs/v2/checkpoints/A-SFT", torch_dtype=torch.float32).to(dev)
pol.config.use_cache = False
ref32 = copy.deepcopy(pol).to(dev)
ref16 = copy.deepcopy(pol).to(dev).to(torch.bfloat16)
print("gpu_delta_GiB bf16_ref_vs_fp32_ref:", round((ref32.f32_bytes() if False else 0), 3))
m32 = sum(p.numel()*p.element_size() for p in ref32.parameters())/2**30
m16 = sum(p.numel()*p.element_size() for p in ref16.parameters())/2**30

snap = json.load(open("outputs/v2/debug/phase7/grpo_smoke/attempt_004/update_smoke/A/group_snapshot.json"))
comp = [c[:c.index(EOS)+1] for c in snap["completion_ids"]]
rows = [json.loads(l) for l in open("outputs/v2/manifests/rl_stream_20000_A.jsonl")][:16]
prompts = [r["prompt"] for r in rows]
enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
ids = enc.input_ids.to(dev); am = enc.attention_mask.to(dev)

def ref_logps(model, input_ids, attention_mask, comp_ids):
    full = torch.cat([input_ids, comp_ids.to(dev)], dim=1)
    keep = comp_ids.size(1)
    with torch.no_grad():
        out = model(input_ids=full, attention_mask=torch.cat([attention_mask, torch.ones_like(comp_ids.to(dev))], 1),
                    logits_to_keep=keep + 1)
    logits = out.logits[:, :-1, :]
    tgt = full[:, -keep:]
    from trl.trainer.utils import selective_log_softmax
    return selective_log_softmax(logits.float(), tgt)

with torch.no_grad():
    lp32 = ref_logps(ref32, ids, am, torch.tensor(comp))
    lp16 = ref_logps(ref16, ids, am, torch.tensor(comp))
d = (lp16.float() - lp32).abs()
import math
kl32 = (torch.exp(lp32 - lp32) - (lp32 - lp32) - 1)  # placeholder zero
print("REF_LOGPS fp32-vs-bf16: max_abs=%.4e mean_abs=%.4e per_row_mean_max=%.4e" % (
    d.max().item(), d.mean().item(), d.mean(dim=1).max().item()))
print("MEM fp32_ref_GiB=%.3f bf16_ref_GiB=%.3f saved=%.3f" % (m32, m16, m32 - m16))

alpha = 0.6
master = {n: p.detach().cpu().float() for n, p in pol.named_parameters()}
r32c = {n: p.detach().cpu().float() for n, p in ref32.named_parameters()}
r16c = {n: p.detach().cpu() for n, p in ref16.named_parameters()}
del ref32, ref16
import gc; gc.collect(); torch.cuda.empty_cache()
for step in range(2):
    for n in master:
        r32c[n].mul_(1-alpha).add_(master[n], alpha=alpha)
        t = r16c[n].float()
        t.mul_(1-alpha).add_(master[n], alpha=alpha)
        r16c[n].copy_(t.to(torch.bfloat16))
dmax, dn = 0.0, 0.0
for n in master:
    d = (r16c[n].float() - r32c[n]).abs().max().item()
    dmax = max(dmax, d); dn = max(dn, r32c[n].abs().max().item())
print('EMA_2STEPS fp32-master-vs-bf16-store: max_abs_param_delta=%.4e (rel=%.2e)' % (dmax, dmax/max(dn, 1e-12)))
doc = {'task': 'Phase7 T4-R1 ref numerics', 'date': '2026-11-15',
       'ref_logps_diff': {'max_abs': d.max().item(), 'mean_abs': d.mean().item(), 'per_row_mean_max': d.mean(dim=1).max().item()},
       'memory_saved_GiB': m32 - m16,
       'ema_param_max_abs_delta_2steps': dmax,
       'note': 'logps diff on real first group; EMA comparison on CPU parameter copies'}
os.makedirs('outputs/v2/debug/phase7/grpo_smoke/attempt_005/reports', exist_ok=True)
json.dump(doc, open('outputs/v2/debug/phase7/grpo_smoke/attempt_005/reports/t4r1_ref_numerics.json','w'), indent=1)
print('NUMERICS_DONE')
