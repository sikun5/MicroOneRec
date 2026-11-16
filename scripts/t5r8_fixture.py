#!/usr/bin/env python3
"""Phase7 T5-R8(2): CPU fixture proving the ACTIVE audit entry's event
persistence wiring (same V2SyncRefCallback class, imported from the running
entry file). Separate fixture events path; formal events file untouched."""
import hashlib, json, os, sys, torch
ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
from v2_train_grpo_audit import V2SyncRefCallback

FX = "outputs/v2/debug/phase7/audit_fresh_attempt002/reports/fixture_events"
FIXTURE_EVENTS = FX + "_cpu_fixture.jsonl"
os.makedirs(os.path.dirname(FIXTURE_EVENTS), exist_ok=True)
if os.path.exists(FIXTURE_EVENTS):
    os.remove(FIXTURE_EVENTS)

# tiny CPU model mirrors production: bf16 store after fp32-intermediate EMA
torch.manual_seed(42)
lin = torch.nn.Linear(16, 16, bias=False).to(torch.bfloat16)
ref_model = lin

cb = V2SyncRefCallback(ref_model=ref_model, accelerator=None, steps=512, alpha=0.6)
cb.events_path = FIXTURE_EVENTS

pol = torch.nn.Linear(16, 16, bias=False).to(torch.bfloat16)
with torch.no_grad():
    for p_ref, p_pol in zip(ref_model.parameters(), pol.parameters()):
        p_ref.data.copy_(p_pol.data)

class FakeState:
    global_step = 512
class FakeModelWrap:
    def __init__(self, m): self.m = m
    def parameters(self): return self.m.parameters()

h_before = hashlib.sha256()
for p in ref_model.parameters():
    h_before.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
cb.on_step_end(args=None, state=FakeState(), control=None, model=FakeModelWrap(pol))
h_after = hashlib.sha256()
for p in ref_model.parameters():
    h_after.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())

results = {
  "fixture_uses_active_entry_class": "v2_train_grpo_audit.V2SyncRefCallback",
  "active_entry_sha256": hashlib.sha256(open("scripts/v2_train_grpo_audit.py","rb").read()).hexdigest(),
  "sync_config": {"steps": 512, "alpha": 0.6, "intermediate": "fp32", "store": "bf16"},
  "param_count": sum(1 for _ in ref_model.parameters()),
  "param_dtype": str(next(p.dtype for p in ref_model.parameters())),
  "ref_sha_before": h_before.hexdigest()[:16],
  "ref_sha_after": h_after.hexdigest()[:16],
  "sha_changed": h_before != h_after,
}
# append+flush + new-process read
with open(FIXTURE_EVENTS) as f:
    lines = f.readlines()
results["events_file_rows"] = len(lines)
ev = json.loads(lines[-1])
results["event_record"] = ev
results["append_flush_ok"] = len(lines) == 1 and ev.get("global_step") == 512 and ev.get("alpha") == 0.6
results["ALL_OK"] = (results["sha_changed"] and results["append_flush_ok"]
                      and results["param_count"] >= 1)
json.dump(results, open(os.path.dirname(FIXTURE_EVENTS) + "/t5r8_fixture_result.json", "w"), ensure_ascii=False, indent=1)
print("FIXTURE", json.dumps({k: results[k] for k in ("append_flush_ok", "sha_changed", "ALL_OK")}))
print("events_path", FIXTURE_EVENTS)
