#!/usr/bin/env python3
"""Independent post-run audit of existing raw records and native tensors only.
No recommendation generation, test metrics, training or checkpoint selection.
"""
import argparse,collections,hashlib,itertools,json,math,os,sys,time
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--method',required=True,choices=['B','C']);ap.add_argument('--evidence',required=True);args=ap.parse_args()
root=Path('/data/user/swu073/dsaa5020/MicroOneRec');os.chdir(root);sys.path[:0]=[str(root),str(root/'scripts')]
e=Path(args.evidence);out=root/f'outputs/v2/checkpoints/{args.method}-RL';ck=out/'final_checkpoint';started=time.time()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def j(p):return json.loads(Path(p).read_text())
def jsonl(p):
 with open(p) as f:
  for line in f:yield json.loads(line)
assert j(e/'supervisor_exit.json')['exit_code']==0 and j(e/'in_process_exit.json')['exit_code']==0
assert j(out/'run_meta.json')['final_global_step']==625
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from v2_generation_adapter import validate_sid_sequence
torch.set_num_threads(4)
init=root/f'outputs/v2/checkpoints/{args.method}-SFT';tok=AutoTokenizer.from_pretrained(ck,local_files_only=True);initial_tok=AutoTokenizer.from_pretrained(init,local_files_only=True)
assert tok.get_vocab()==initial_tok.get_vocab() and len(tok)==154977
atom=j(root/'outputs/v2/tokenizer/token_id_map.json')['map'];assert len(atom)==3308 and all(tok.encode(t,add_special_tokens=False)==[int(i)] for t,i in atom.items());atomic=set(atom.values());eos=tok.eos_token_id
catalog={tuple(tok.encode(line.split('\t')[0].strip(),add_special_tokens=False)):line.split('\t')[0].strip() for line in open(root/f'outputs/v2/data/{args.method}/info.txt')}
assert len(catalog)==3686 and all(len(ts)==4 for ts in catalog)
stream=list(jsonl(root/f'outputs/v2/manifests/rl_stream_20000_{args.method}.jsonl'));assert len(stream)==20000
assert [r['stream_index'] for r in stream]==list(range(20000))
task_copies=collections.defaultdict(list)
for r in stream:task_copies[r['task_id']].append(r['copy_index'])
assert len(task_copies)==10000 and all(sorted(v)==[0,1] for v in task_copies.values())
metrics=list(jsonl(out/'train_metrics.jsonl'));training_metrics=[r for r in metrics if 'kl' in r]
assert [r['global_step'] for r in training_metrics]==list(range(1,626))
assert all(not isinstance(v,(int,float)) or math.isfinite(v) for r in metrics for v in r.values())
reward_it=iter(jsonl(out/'reward_records.jsonl'));loss_it=iter(jsonl(e/'group_loss.jsonl'));groups_it=iter(jsonl(e/'raw_groups.jsonl'))
stats=collections.defaultdict(lambda:collections.Counter());max_rank_error=0.;max_total_error=0.;max_adv_error=0.;nreward=0
rank=[1/math.log2(k+2) for k in range(16)];rank=[-a/sum(rank) for a in rank]
for idx,source in enumerate(stream):
 r=next(groups_it);loss=next(loss_it)
 for key in ['stream_index','task_id','copy_index','task_type']:assert r[key]==source[key],(idx,key)
 assert r['target_sid']==source['item_sid'] and loss['stream_index']==idx and r['global_step_before']==idx//32 and loss['global_step_before']==idx//32
 assert len(r['completion_ids'])==16 and len(r['completion_mask'])==16
 decoded=[]
 for seq,mask in zip(r['completion_ids'],r['completion_mask']):
  legal,why=validate_sid_sequence(seq,eos,atomic);assert legal and len(seq)==5 and tuple(seq[:4]) in catalog,(idx,seq,why)
  assert mask==[1]*5
  decoded.append(catalog[tuple(seq[:4])])
 expected_rule=[float(s.strip('\n\" ')==source['item_sid'].strip('\n\" ')) for s in decoded]
 anyhit=any(expected_rule);expected_rank=[0. if x else rank[k] for k,x in enumerate(expected_rule)] if anyhit else [0.]*16
 for fn,values in [('rule_reward',expected_rule),('ndcg_rule_reward',expected_rank)]:
  for val in values:
   saved=next(reward_it);nreward+=1
   for key in ['stream_index','task_id','copy_index','task_type']:assert saved[key]==source[key]
   assert saved['reward_fn']==fn
   err=abs(saved['reward']-val);max_rank_error=max(max_rank_error,err);assert err<1e-12
 expected_sum=torch.tensor(expected_rule,dtype=torch.float32)+torch.tensor(expected_rank,dtype=torch.float32)
 total_err=max(abs(float(a)-float(b)) for a,b in zip(expected_sum,r['summed_rewards']));max_total_error=max(max_total_error,total_err);assert total_err<2e-7
 adv=(expected_sum-expected_sum.mean())/(expected_sum.std()+1e-4)
 adv_err=max(abs(float(a)-float(b)) for a,b in zip(adv,r['advantages']));max_adv_error=max(max_adv_error,adv_err);assert adv_err<2e-5
 assert all(math.isfinite(loss[k]) for k in ['loss','kl','categorical_diversity','token_diversity','completion_length'])
 st=stats[source['task_type']];st['groups']+=1;st['candidates']+=16;st['hits']+=sum(expected_rule);st['hit_groups']+=int(anyhit);st['zero_advantage_groups']+=int(all(x==0. for x in r['advantages']))
 for z in r['summed_rewards']:st['reward_sum']+=z;st['reward_squared_sum']+=z*z
 for k in ['kl','loss','categorical_diversity','token_diversity','completion_length']:st[k+'_sum']+=loss[k]
for it in [reward_it,loss_it,groups_it]:
 try:next(it);raise AssertionError('Unexpected extra record')
 except StopIteration:pass
assert nreward==640000 and sorted(x['groups'] for x in stats.values())==[2000,2000,16000]
summary={}
for k,s in stats.items():
 n=s['candidates'];g=s['groups'];mean=s['reward_sum']/n
 summary[k]={**s,'candidate_hit_rate':s['hits']/n,'group_hit_rate':s['hit_groups']/g,'zero_advantage_ratio':s['zero_advantage_groups']/g,'reward_mean':mean,'reward_std_population':math.sqrt(max(0.,s['reward_squared_sum']/n-mean*mean)),**{q+'_mean':s[q+'_sum']/g for q in ['kl','loss','categorical_diversity','token_diversity','completion_length']}}
model_finite=[]
with safe_open(str(ck/'model.safetensors'),framework='pt',device='cpu') as f:
 for name in f.keys():
  t=f.get_tensor(name);assert t.dtype==torch.float32 and bool(torch.isfinite(t).all()),name;model_finite.append({'name':name,'shape':list(t.shape),'dtype':str(t.dtype),'numel':t.numel(),'finite':True})
assert sum(x['numel'] for x in model_finite)==599163904
sync=j(e/'sync512/native_snapshot.json');events=list(jsonl(out/'ref_sync_events.jsonl'));assert len(events)==1 and events[0]['global_step']==512 and events[0]['alpha']==0.6
errors=[];max_error=0.;different_elements=0
with safe_open(str(e/'sync512/ref_before.safetensors'),framework='pt',device='cpu') as b,safe_open(str(e/'sync512/policy.safetensors'),framework='pt',device='cpu') as p,safe_open(str(e/'sync512/ref_after.safetensors'),framework='pt',device='cpu') as a:
 assert set(b.keys())==set(p.keys())==set(a.keys())
 for n in b.keys():
  before=b.get_tensor(n);policy=p.get_tensor(n);after=a.get_tensor(n)
  expected=before.float();expected.mul_(0.4).add_(policy.float(),alpha=0.6);expected=expected.to(torch.bfloat16)
  err=float((expected.float()-after.float()).abs().max());different_elements+=int((expected!=after).sum());max_error=max(max_error,err)
  if err:errors.append({'parameter':n,'max_abs':err})
assert sync['after_native']['parameters_order_sha256']==events[0]['ref_bf16_sha256']
resource=j(e/'in_process_exit.json');assert resource['global_step']==625 and resource['groups_observed']==20000 and resource['candidates_observed']==320000
files=[*ck.iterdir(),out/'run_meta.json',out/'train_metrics.jsonl',out/'reward_records.jsonl',out/'ref_sync_events.jsonl',*e.glob('*.py'),e/'launch_config.json',e/'prepare_receipt.json',e/'supervisor_start.json',e/'supervisor_exit.json',e/'in_process_exit.json',e/'initialization.json',e/'first_gradients.json',e/'first_updates.jsonl',e/'raw_groups.jsonl',e/'group_loss.jsonl',*list((e/'sync512').iterdir())]
inventory={str(p.relative_to(root)):{'sha256':sha(p),'size_bytes':p.stat().st_size} for p in files if p.is_file()}
r={'method':args.method,'scope':'Independent raw replay and existing model/native tensor audit; no training or recommendation inference','training_steps':625,'groups':20000,'base_tasks':10000,'copies_per_task':[0,1],'candidate_rows':320000,'reward_rows':nreward,'type_stats':summary,'max_reward_error':max_rank_error,'max_summed_reward_fp32_error':max_total_error,'max_advantage_error':max_adv_error,'raw_catalog_legal':320000,'strict_raw_shape':'exactly 4 atomic tokens + EOS; no post-EOS tokens','model_tensors':len(model_finite),'model_elements':sum(x['numel'] for x in model_finite),'all_model_finite':True,'tokenizer_vocab_equal_to_initial':True,'atomic_tokens':3308,'catalog':3686,'native_sync_event':events[0],'native_before_captured':True,'cpu_ema_exact_match':different_elements==0,'cpu_ema_different_elements':different_elements,'cpu_ema_max_abs':max_error,'cpu_ema_errors':errors,'resources':resource,'inventory':inventory,'inventory_exclusions':'This report and its self-referential digest are excluded; raw intermediate checkpoints remain preserved but not rehashed here.','audit_wall_seconds':time.time()-started}
(e/'final_audit.json').write_text(json.dumps(r,indent=2)+'\n');(e/'model_finite.json').write_text(json.dumps(model_finite,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k not in ['inventory','type_stats','resources']}));assert different_elements==0,('CPU EMA mismatch needs diagnosis',different_elements,max_error)
