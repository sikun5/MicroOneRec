#!/usr/bin/env python3
"""Read-only final/prefix audit of already generated Phase7 GRPO records.
No training or generation. --final requires completed 625 and immutable final weights.
"""
import argparse,collections,json,math,os,hashlib,time
from pathlib import Path
from datetime import datetime
ROOT=Path('/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def audit_rewards(path,manifest,final=False):
 a=[1/math.log2(j+2) for j in range(16)];rank=[-x/sum(a) for x in a]
 stats={};bad=[];nlines=0;ng=0;max_rank_error=0.;stream_seen=collections.Counter();pair_seen=collections.Counter();buf=[]
 def fail(msg,details):
  if len(bad)<20:bad.append({'issue':msg,**details})
 for lineno,line in enumerate(open(path),1):
  nlines=lineno
  try:r=json.loads(line)
  except json.JSONDecodeError:
   fail('invalid_json',{'line':lineno});continue
  buf.append(r)
  if len(buf)<32:continue
  idx=ng;ng+=1
  if idx>=len(manifest):fail('extra_group',{'group':idx});buf=[];continue
  m=manifest[idx]
  for j,x in enumerate(buf):
   for key in ['stream_index','task_id','task_type','copy_index']:
    if x.get(key)!=m.get(key):fail('identity_mismatch',{'group':idx,'row':j,'field':key,'got':x.get(key),'expected':m.get(key)})
   expected_fn='rule_reward' if j<16 else 'ndcg_rule_reward'
   if x.get('reward_fn')!=expected_fn:fail('reward_fn_order',{'group':idx,'row':j,'got':x.get('reward_fn')})
   v=x.get('reward')
   if not isinstance(v,(int,float)) or not math.isfinite(v):fail('nonfinite_reward',{'group':idx,'row':j})
  rr=[x['reward'] for x in buf[:16]];nr=[x['reward'] for x in buf[16:]]
  if any(v not in [0.,1.] for v in rr):fail('nonbinary_rule',{'group':idx})
  hit=any(rr);expected=[0. if not hit or rr[j] else rank[j] for j in range(16)]
  er=max(abs(x-y) for x,y in zip(nr,expected));max_rank_error=max(max_rank_error,er)
  if er>1e-12:fail('ranking_math_mismatch',{'group':idx,'max_error':er})
  total=[x+y for x,y in zip(rr,nr)];mean=sum(total)/16;var=sum((x-mean)**2 for x in total)/15
  st=stats.setdefault(m['task_type'],{'groups':0,'candidates':0,'rule_hits':0,'groups_with_hit':0,'zero_advantage_groups':0,'rule_sum':0.,'rank_sum':0.,'reward_sum':0.,'reward_sq_sum':0.})
  st['groups']+=1;st['candidates']+=16;st['rule_hits']+=sum(rr);st['groups_with_hit']+=int(hit);st['zero_advantage_groups']+=int(var==0)
  st['rule_sum']+=sum(rr);st['rank_sum']+=sum(nr);st['reward_sum']+=sum(total);st['reward_sq_sum']+=sum(x*x for x in total)
  stream_seen[m['stream_index']]+=1;pair_seen[(m['task_id'],m['copy_index'])]+=1;buf=[]
 if buf:fail('partial_group',{'rows':len(buf)})
 if final and ng!=20000:fail('wrong_final_group_count',{'groups':ng})
 for st in stats.values():
  n=st['candidates'];g=st['groups'];mu=st['reward_sum']/n
  st.update(rule_candidate_hit_share=st['rule_hits']/n,training_group_hit_share=st['groups_with_hit']/g,zero_advantage_share=st['zero_advantage_groups']/g,reward_mean=mu,reward_population_std=math.sqrt(max(st['reward_sq_sum']/n-mu*mu,0.)))
 return {'rows':nlines,'complete_groups':ng,'trailing_rows':len(buf),'n_distinct_streams':len(stream_seen),'all_observed_streams_exactly_once':all(v==1 for v in stream_seen.values()),'n_task_copy_pairs':len(pair_seen),'all_observed_task_copy_pairs_exactly_once':all(v==1 for v in pair_seen.values()),'task_stats':stats,'max_rank_formula_abs_error':max_rank_error,'errors_head':bad,'all_ok':not bad}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--method',choices=['A','B','C'],required=True);ap.add_argument('--attempt',required=True);ap.add_argument('--out',required=True);ap.add_argument('--final',action='store_true');args=ap.parse_args();os.chdir(ROOT)
 t=time.time();attempt=Path(args.attempt);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 manifest=ROOT/f'outputs/v2/manifests/rl_stream_20000_{args.method}.jsonl';rows=[json.loads(x) for x in manifest.read_text().splitlines()]
 pair=collections.Counter((r['task_id'],r['copy_index']) for r in rows);tasks=collections.defaultdict(set)
 for r in rows:tasks[r['task_id']].add(r['copy_index'])
 mm={'n_rows':len(rows),'stream_index_exact_order':all(r['stream_index']==i for i,r in enumerate(rows)),'n_unique_task_copy':len(pair),'each_task_copy_once':all(v==1 for v in pair.values()),'n_base_tasks':len(tasks),'each_base_has_two_copies':all(v=={0,1} for v in tasks.values()),'task_counts':dict(collections.Counter(r['task_type'] for r in rows)),'sha256':sha(manifest)}
 metrics=[json.loads(x) for x in (attempt/'train_metrics.jsonl').read_text().splitlines() if x];train=[r for r in metrics if 'loss' in r];summary=[r for r in metrics if 'train_runtime' in r];steps=[r['global_step'] for r in train]
 report={'collected_at':datetime.now().astimezone().isoformat(),'method':args.method,'attempt':str(attempt),'final_required':args.final,'manifest':mm,'run_meta':json.loads((attempt/'run_meta.json').read_text()),'metrics':{'all_rows':len(metrics),'training_rows':len(train),'last_train':train[-1] if train else None,'last_summary':summary[-1] if summary else None,'steps_exact_1_to_n':steps==list(range(1,len(train)+1)),'finite_logged_numeric_values':all(not isinstance(v,float) or math.isfinite(v) for r in metrics for v in r.values()),'kl_first_max_last':[train[0].get('kl'),max(r.get('kl',-math.inf) for r in train),train[-1].get('kl')] if train else [],'loss_min_max':[min(r['loss'] for r in train),max(r['loss'] for r in train)] if train else []},'rewards':audit_rewards(attempt/'reward_records.jsonl',rows,args.final),'native_sync_events':[json.loads(x) for x in (attempt/'ref_sync_events.jsonl').read_text().splitlines() if x]}
 if args.final:
  import torch
  from safetensors import safe_open
  f=attempt/'final_checkpoint/model.safetensors';tensors=[];nonfinite=[]
  with safe_open(f,framework='pt',device='cpu') as sf:
   for name in sf.keys():
    x=sf.get_tensor(name);finite=torch.isfinite(x).all().item()
    tensors.append({'name':name,'numel':x.numel(),'dtype':str(x.dtype),'shape':list(x.shape),'finite':finite})
    if not finite:nonfinite.append(name)
  report['final_model']={'path':str(f),'sha256':sha(f),'size_bytes':f.stat().st_size,'n_tensor_keys':len(tensors),'n_stored_numel':sum(r['numel'] for r in tensors),'all_finite':not nonfinite,'nonfinite_names':nonfinite,'tensors':tensors}
  report['final_budget_checks']={'training_steps625':steps==list(range(1,626)),'run_meta_final625':report['run_meta'].get('final_global_step')==625,'reward_rows640000':report['rewards']['rows']==640000,'groups20000':report['rewards']['complete_groups']==20000,'unique_actual_task_copy20000':report['rewards']['n_task_copy_pairs']==20000,'one_sync_at512':len(report['native_sync_events'])==1 and report['native_sync_events'][0]['global_step']==512,'final_model_finite':not nonfinite}
 report['measurement_limits']={'reward_records':'Group candidate ordinal inferred from immutable per-function 16-row block; no raw completion strings in this stream. Does not establish per-candidate legality alone.','per_task_KL':'not collected; global per-update KL only, do not fabricate task attribution','peak_memory':'not collected for original A/fresh A; not replaced with T4 short-run peak','exit_status':'launch had no exit-code file; normal-end evidence must be stated separately from observed OS exit code','before_ref':'baseline reconstructed from frozen SFT, not native pre-EMA capture'}
 report['audit_wall_seconds']=time.time()-t;p=out/('final_audit.json' if args.final else 'prefix_audit.json');p.write_text(json.dumps(report,ensure_ascii=False,indent=2))
 brief={k:report[k] for k in ['collected_at','method','attempt','manifest','metrics','rewards','native_sync_events','audit_wall_seconds']}
 if args.final:brief['final_budget_checks']=report['final_budget_checks'];brief['final_model']={k:v for k,v in report['final_model'].items() if k!='tensors'}
 print(json.dumps({'path':str(p),'sha256':sha(p),'report':brief},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
