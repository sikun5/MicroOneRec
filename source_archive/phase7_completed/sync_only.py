"""Read-only CPU verification of immutable native step512 snapshots; no CUDA."""
import sys,json,time
from pathlib import Path
import torch
from safetensors import safe_open
p=Path(sys.argv[1]);start=time.time();torch.set_num_threads(4);report=json.loads((p/'sync512/native_snapshot.json').read_text());different=0;max_abs=0.;count=0
with safe_open(str(p/'sync512/ref_before.safetensors'),framework='pt',device='cpu') as b,safe_open(str(p/'sync512/policy.safetensors'),framework='pt',device='cpu') as pol,safe_open(str(p/'sync512/ref_after.safetensors'),framework='pt',device='cpu') as aft:
 assert set(b.keys())==set(pol.keys())==set(aft.keys())
 for n in b.keys():
  before=b.get_tensor(n);policy=pol.get_tensor(n);after=aft.get_tensor(n)
  assert before.dtype==after.dtype==torch.bfloat16 and policy.dtype==torch.float32
  expected=before.float();expected.mul_(0.4).add_(policy,alpha=0.6);expected=expected.to(torch.bfloat16)
  different+=int((expected!=after).sum());max_abs=max(max_abs,float((expected.float()-after.float()).abs().max()));count+=1
r={'scope':'Independent CPU verification of existing immutable native snapshots while the formal process may finish; no model loading/CUDA/forward/training','tensors':count,'different_elements':different,'max_abs':max_abs,'exact':different==0,'native_step':report['global_step'],'native_callback_after':report['callback_event']['ref_bf16_sha256'],'start':start,'end':time.time()}
(p/'sync512/cpu_sync_check.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));assert count==310 and different==0
