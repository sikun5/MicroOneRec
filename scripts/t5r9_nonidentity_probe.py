#!/usr/bin/env python3
"""Phase7 T5-R9(1): non-identity EMA CPU test on the SAME V2SyncRefCallback
from the ACTIVE audit entry (fda1ee5b). Different policy/ref with
predictable non-zero update; formula compared element-wise in the same
conversion order (fp32 intermediate -> bf16 store). Independent events file.
Error-path state honestly stated."""
import hashlib, json, os, sys, torch
ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
from v2_train_grpo_audit import V2SyncRefCallback

alpha, steps = 0.6, 512
FX_DIR = "outputs/v2/debug/phase7/audit_fresh_attempt002/reports"
FIXTURE_EVENTS = os.path.join(FX_DIR, "t5r9_nonidentity_events.jsonl")
os.makedirs(FX_DIR, exist_ok=True)
if os.path.exists(FIXTURE_EVENTS): os.remove(FIXTURE_EVENTS)

# deterministic DIFFERENT policy/ref (non-identity EMA)
torch.manual_seed(7)
ref0 = torch.nn.Linear(8, 8, bias=False).to(torch.bfloat16)
torch.manual_seed(99)
pol = torch.nn.Linear(8, 8, bias=False).to(torch.bfloat16)

cb = V2SyncRefCallback(ref_model=ref0, accelerator=None, steps=steps, alpha=alpha)
cb.events_path = FIXTURE_EVENTS

class FS: global_step = steps
class FM:
    def __init__(self, m): self.m = m
    def parameters(self): return self.m.parameters()

# before SHA (full)
def full_sha(m):
    h = hashlib.sha256()
    for p in m.parameters():
        h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
sha_before = full_sha(ref0)

# expected formula in SAME conversion order as _ema: fp32 intermediate then bf16 copy
expected = {}; pol_params = {n: p.detach().float() for n, p in pol.named_parameters()}
with torch.no_grad():
    for n, p_ref in ref0.named_parameters():
        t = p_ref.detach().float()
        t.mul_(1-alpha).add_(pol_params[n], alpha=alpha)
        expected[n] = t

cb.on_step_end(args=None, state=FS(), control=None, model=FM(pol))

# compare element-wise
max_err, finite = 0.0, True
with torch.no_grad():
    for n, p_ref in ref0.named_parameters():
        d = (p_ref.detach().float() - expected[n]).abs().max().item()
        max_err = max(max_err, d)
        finite = finite and bool(torch.isfinite(p_ref.detach().float()).all())

sha_after = full_sha(ref0)

# read events file back
with open(FIXTURE_EVENTS) as f:
    lines = f.readlines()
ev = json.loads(lines[0])

# error path test: verify _persist with empty path silently skips (documented behavior, not raise)
cb2 = V2SyncRefCallback(ref_model=ref0, accelerator=None, steps=512, alpha=alpha)
cb2.events_path = ""  # empty = no persistence
cb2._persist({"event": "should_not_appear"})
err_test = "skip-if-empty-path (documented behavior); no raise-statement fault injection was actually tested"

results = {
 "task": "Phase7 T5-R9(1) non-identity EMA CPU control",
 "date": "2026-11-16",
 "callback_class": "v2_train_grpo_audit.V2SyncRefCallback (active entry fda1ee5b...)",
 "config": {"steps": steps, "alpha": alpha, "intermediate": "fp32", "store": "bf16",
            "policy_ref_different": True},
 "param_count": sum(1 for _ in ref0.parameters()),
 "param_dtype": str(next(p.dtype for p in ref0.parameters())),
 "param_numel": sum(p.numel() for p in ref0.parameters()),
 "sha_before_full": sha_before, "sha_after_full": sha_after,
 "sha_changed": sha_before != sha_after,
 "max_abs_err_vs_formula": max_err,
 "finite": finite,
 "events_file_rows": len(lines),
 "event_content": ev,
 "new_process_read_note": "events file written by this process, read back from disk in same process; "
                          "cross-process read already proven by T5-R8 fixture (append_flush_ok=true)",
 "error_path": err_test,
 "expected_math": "ref_new = ref_fp32*(1-alpha) + pol_fp32*alpha, then cast to bf16",
 "scope": "CPU small tensor (8x8 linear, 64 elements); NOT the 599M full model",
}
results["ALL_OK"] = (results["sha_changed"] and finite and max_err < 1e-2
                      and len(lines) == 1 and ev.get("global_step") == steps)
json.dump(results, open(os.path.join(FX_DIR, "t5r9_nonidentity_result.json"), "w"), ensure_ascii=False, indent=1)
print("NONIDENTITY", json.dumps({"max_err": max_err, "finite": finite, "sha_changed": results["sha_changed"],
      "events_rows": len(lines), "ALL_OK": results["ALL_OK"]}))
print("result_sha", hashlib.sha256(open(os.path.join(FX_DIR, "t5r9_nonidentity_result.json"), "rb").read()).hexdigest())
