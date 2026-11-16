"""Existing logs/config/source only; no model loading, forward or CUDA work."""
from pathlib import Path
import json,sys,inspect,hashlib,time,collections
p=Path(sys.argv[1]);root=Path('/data/user/swu073/dsaa5020/MicroOneRec');out=root/'outputs/v2/checkpoints/C-RL'
import torch
from transformers import Trainer
from accelerate import Accelerator
rows=[json.loads(x) for x in open(out/'train_metrics.jsonl') if x.strip()];m=[x for x in rows if 'kl' in x];assert len(m)==625
by={r['global_step']:r for r in m};top=sorted(m,key=lambda x:x['kl'],reverse=True)[:10];normtop=sorted(m,key=lambda x:x['grad_norm'],reverse=True)[:5]
selected=sorted(set([510,511,512,513,514,620,621,622,623,624,625]+[r['global_step'] for r in top]))
snippets={}
def collect(name,fn,keywords,pad=4):
 lines,start=inspect.getsourcelines(fn);include=set()
 for i,l in enumerate(lines):
  if any(k in l for k in keywords):include.update(range(max(0,i-pad),min(len(lines),i+pad+1)))
 text=''.join(f'{start+i}: {lines[i]}' for i in sorted(include))
 file=inspect.getsourcefile(fn);snippets[name]={'file':file,'file_sha256':hashlib.sha256(Path(file).read_bytes()).hexdigest(),'lines':text}
collect('Trainer._inner_training_loop',Trainer._inner_training_loop,['max_grad_norm','clip_grad_norm_','grad_norm =','on_pre_optimizer_step','self.optimizer.step()'],4)
collect('Accelerator.clip_grad_norm_',Accelerator.clip_grad_norm_,['def clip','unscale_gradients','clip_grad_norm_','return'],3)
collect('torch.nn.utils.clip_grad_norm_',torch.nn.utils.clip_grad_norm_,['total_norm','_clip_grads_with_norm_','return'],4)
init=json.loads((p/'initialization.json').read_text());cfg=init['config'];event=json.loads((p/'sync512/native_snapshot.json').read_text())
assert cfg['max_grad_norm']==0.3 and cfg['gradient_accumulation_steps']==32 and cfg['num_generations']==16
spike_steps={r['global_step'] for r in top};groups=collections.defaultdict(list)
for line in open(p/'group_loss.jsonl'):
 r=json.loads(line);step=r['global_step_before']+1
 if step in spike_steps:groups[step].append(r)
group_peaks={str(step):sorted(rs,key=lambda x:x['kl'],reverse=True)[:3] for step,rs in groups.items()}
r={'scope':'Read-only existing native logs/config and installed execution source; no GPU/model/forward/training/new rollout','max_kl_step':top[0]['global_step'],'max_kl':top[0]['kl'],'top_kl':[{k:x[k] for k in ['global_step','wall','kl','grad_norm','loss','learning_rate']} for x in top],'top_grad_norm':[{k:x[k] for k in ['global_step','kl','grad_norm','loss']} for x in normtop],'selected_steps':[{k:by[s][k] for k in ['global_step','wall','kl','grad_norm','loss','learning_rate']} for s in selected],'native_sync_step':event['global_step'],'native_sync_snapshot_wall':event['wall'],'event_count':sum(1 for _ in open(out/'ref_sync_events.jsonl')),'max_kl_updates_after_sync':top[0]['global_step']-512,'last_step':by[625],'max_grad_norm_actual_config':cfg['max_grad_norm'],'configuration_branch_evidence':{k:cfg.get(k) for k in ['max_grad_norm','fp16','bf16','half_precision_backend','deepspeed','fsdp','gradient_accumulation_steps']},'first_on_pre_optimizer_step_gradients':json.loads((p/'first_gradients.json').read_text()),'grad_norm_log_semantics':'For the normal non-DeepSpeed branch: _grad_norm is the value returned by Accelerator.clip_grad_norm_; torch.clip_grad_norm_ computes total_norm BEFORE clipping, clips using it, then returns that total_norm. Thus logged large grad_norm is pre-clip, not a measured post-clip norm.','post_clip_per_step_norm_measured':False,'post_clip_norm_value':None,'runtime_path_scope':'Actual frozen configuration and installed source establish the standard clip path; no per-step function-call trace or post-clip norm was captured. Do not claim numerical measurement <=0.3 at step623.','source_snippets':snippets,'largest_observed_groups_in_spike_updates':group_peaks,'diagnosis':'No target/group/reward/raw/mask/finite-weight/ref-EMA implementation error detected by the existing full independent audits. The finite spikes indicate optimization instability/outlier log-ratio sensitivity; temporal association with sync is not proved causal. Keep final625 and report the finding.','existing_full_audit_sha256':hashlib.sha256((p/'final_audit.json').read_bytes()).hexdigest(),'parent_inventory_sha256':hashlib.sha256((p/'C_artifact_inventory.json').read_bytes()).hexdigest(),'finished':time.time()}
(p/'C_S1_diag.json').write_text(json.dumps(r,indent=2)+'\n')
(p/'C_S1_sources.txt').write_text('\n\n'.join(k+'\n'+v['file']+'\n'+v['lines'] for k,v in snippets.items()))
print(json.dumps({k:v for k,v in r.items() if k not in ['source_snippets','largest_observed_groups_in_spike_updates','selected_steps','first_on_pre_optimizer_step_gradients']}))
