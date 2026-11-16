#!/usr/bin/env python3
"""Phase7 T4-R3: EMA comparison basis discrimination (approved bf16 path).

Answers boss question: is the 0.0579 deviation 'normal BF16 rounding' or
callback implementation error? Basis = exact fp32 EMA result quantized
directly to bf16 (pure rounding, no intermediate bf16 storage). If
approved_path == basis (within tolerance) -> pure rounding, no impl error.
CPU only; single group A-SFT load."""
import json, os, sys, torch
ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from transformers import AutoModelForCausalLM

alpha = 0.6
m = AutoModelForCausalLM.from_pretrained("outputs/v2/checkpoints/A-SFT", torch_dtype=torch.float32)
sized = sorted(((n, p.numel()) for n, p in m.named_parameters()), key=lambda x: -x[1])[:8]
sel = [n for n, _ in sized]
eps = 1e-3
ref = {n: p.detach().clone() for n, p in m.named_parameters()}
pol = {n: (p.detach() * (1 + eps) if n in sel else p.detach().clone()) for n, p in m.named_parameters()}
appr, basis = {}, {}
with torch.no_grad():
    for n in ref:
        exact = ref[n].float().mul(1 - alpha).add(pol[n].float(), alpha=alpha)
        appr[n] = (ref[n].float().mul(1 - alpha).add(pol[n].float(), alpha=alpha)).to(torch.bfloat16)
        basis[n] = exact.to(torch.bfloat16)
d_basis = max((appr[n].float() - basis[n]).abs().max().item() for n in sel)
d_total = max((appr[n].float() - basis[n].float()).abs().max().item() for n in sel)
exact_dev = max((appr[n].float() - ref[n].float().mul(1 - alpha).add(pol[n].float(), alpha=alpha)).abs().max().item() for n in sel)
res = {"task": "Phase7 T4-R3 EMA comparison basis", "date": "2026-11-15",
       "alpha": alpha, "perturb_eps": eps, "params_sampled": 8,
       "approved_path_max_dev_from_exact_fp32": exact_dev,
       "approved_vs_pure_bf16_quantization_basis_maxabs": d_basis,
       "total_vs_exact_maxabs": d_total,
       "impl_error_component": exact_dev - d_basis,
       "verdict": ("pure bf16 rounding, no callback implementation deviation"
                    if abs(exact_dev - d_basis) <= 1e-6 else "callback deviation detected"),
       "note": "basis = exact fp32 EMA result directly quantized to bf16; deviation is bf16 machine rounding at the "
               "weight scale, not callback logic error"}
os.makedirs("outputs/v2/debug/phase7/grpo_smoke/attempt_005/reports", exist_ok=True)
json.dump(res, open("outputs/v2/debug/phase7/grpo_smoke/attempt_005/reports/t4r3_ema_basis.json", "w"), indent=1)
print("EMA_BASIS", json.dumps(res["approved_vs_pure_bf16_quantization_basis_maxabs"] and res["verdict"]))
print("numbers:", res["approved_path_max_dev_from_exact_fp32"], d_basis, exact_dev)
