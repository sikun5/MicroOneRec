import os,sys,json,time,hashlib
from pathlib import Path
from datetime import datetime
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
ROOT=Path('/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec');os.chdir(ROOT)
import torch
from transformers import AutoModelForCausalLM
from safetensors import safe_open
torch.set_num_threads(8)
out=ROOT/'outputs/v2/evidence/phase7/substitute_takeover_20261116T170236'
attempt=ROOT/'outputs/v2/debug/phase7/audit_fresh_attempt002'
report={'collected_at':datetime.now().astimezone().isoformat(),'device':'CPU only; CUDA_VISIBLE_DEVICES empty','scope':'Reconstructed frozen SFT reference baseline, not native live-before capture; optional expected EMA from saved policy512'}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
model=AutoModelForCausalLM.from_pretrained(ROOT/'outputs/v2/checkpoints/A-SFT',torch_dtype=torch.bfloat16,local_files_only=True,device_map='cpu')
params=list(model.named_parameters());h=hashlib.sha256()
for name,p in params:h.update(p.detach().contiguous().view(torch.uint8).numpy().tobytes())
report.update(n_ref_tensors=len(params),n_ref_numel=sum(p.numel() for _,p in params),baseline_bf16_sha256=h.hexdigest(),all_bf16=all(p.dtype==torch.bfloat16 for _,p in params),all_cpu=all(p.device.type=='cpu' for _,p in params),all_finite=all(torch.isfinite(p).all().item() for _,p in params))
events=[json.loads(x) for x in (attempt/'ref_sync_events.jsonl').read_text().splitlines() if x]
report['native_events']=events
report['baseline_differs_from_native_after']=len(events)==1 and report['baseline_bf16_sha256']!=events[0]['ref_bf16_sha256']
ck=attempt/'checkpoint-512/model.safetensors'
if ck.exists():
 report['policy512_sha256']=sha(ck)
 report['state512']=json.loads((attempt/'checkpoint-512/trainer_state.json').read_text())['global_step']
 hh=hashlib.sha256();missing=[]
 with safe_open(ck,framework='pt',device='cpu') as f:
  keys=set(f.keys())
  for name,p in params:
   if name not in keys:missing.append(name);continue
   q=f.get_tensor(name)
   t=p.detach().float();t.mul_(0.4).add_(q.detach().float(),alpha=0.6)
   expected=t.to(torch.bfloat16)
   hh.update(expected.contiguous().view(torch.uint8).numpy().tobytes())
 report['missing_policy_keys']=missing
 report['cpu_expected_ema_sha256']=hh.hexdigest()
 report['cpu_expected_matches_native_after']=not missing and report['cpu_expected_ema_sha256']==events[0]['ref_bf16_sha256']
 report['cpu_expected_limitation']='CPU and CUDA FP32 primitive rounding can differ; mismatch alone is not an implementation-error proof. No GPU replay performed.'
report['completed_at']=datetime.now().astimezone().isoformat()
p=out/'reference_baseline_and_expected.json';p.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({'path':str(p),'sha256':sha(p),'report':report},ensure_ascii=False,indent=2))
