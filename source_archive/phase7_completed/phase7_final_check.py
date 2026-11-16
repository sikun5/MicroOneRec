#!/usr/bin/env python3
"""T8: current bytes/CPU structures and accepted evidence consistency.
No model forward, CUDA, recommendation generation, training or test metrics.
"""
from pathlib import Path
import argparse,collections,hashlib,json,os,sys,time
root=Path('/data/user/swu073/dsaa5020/MicroOneRec');os.chdir(root);sys.path[:0]=[str(root),str(root/'scripts')]
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);arg=ap.parse_args();out=Path(arg.out);start=time.time();cache={};verified={}
def j(p):return json.loads(Path(p).read_text())
def write(n,x):(out/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def progress(stage,**kw):write('progress.json',{'stage':stage,'wall':time.time(),**kw})
def digest(p):
 p=Path(p);st=p.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
 if key not in cache:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  after=p.stat();assert (after.st_ino,after.st_size,after.st_mtime_ns)==(st.st_ino,st.st_size,st.st_mtime_ns),('File changed',str(p));cache[key]=h.hexdigest()
 return cache[key]
def gate(rel,expected):
 p=Path(rel);p=p if p.is_absolute() else root/p;got=digest(p);assert got==expected,('Hash mismatch',str(p),got,expected)
 rel=str(p.relative_to(root)) if p.is_relative_to(root) else str(p)
 verified[rel]={'sha256':got,'size_bytes':p.stat().st_size};return got
progress('upstream_evidence')
ev=root/'outputs/v2/evidence';p7=ev/'phase7';a_dir=p7/'substitute_takeover_20261116T170236'
anchor={
 'outputs/v2/evidence/phase6/selected_checkpoints.json':'e902a6f401c4c15ccc3082a9ee01ed4481fe2f88b7bb88f14546d7be36bb9b5e',
 'outputs/v2/evidence/phase7/substitute_takeover_20261116T170236/A_artifact_inventory.json':'9da778e48349eb8bd2446efc9d9d017ccc3c5db9f3b4868f5883a51004e966d8',
 'outputs/v2/evidence/phase7/substitute_takeover_20261116T170236/A_final_handoff.json':'a826af9054ec542c1227af30e69057e62bfb7945a5ae44c6f2b25fcf8b6f1fc0',
 'outputs/v2/evidence/phase7/substitute_takeover_20261116T170236/A_resources.json':'d7ec9ec1cb78ac30a18bc254f8940ac348c36f53f2e3890105c999e9c918c5b1',
 'outputs/v2/evidence/phase7/substitute_takeover_20261116T170236/final_audit.json':'09d6f76a8745e8abcae37abf5f549b86362771b90aea749265338daf6a18b982',
 'outputs/v2/evidence/phase7/substitute_takeover_20261116T170236/load_check.json':'2c6675445590606a22f539de973ca1549f4ca2278945ef2ee205d9e7064d1ff9',
 'outputs/v2/evidence/phase7/h100_B_20261117/B_artifact_inventory.json':'a988373f99532d548dad2b787abd0c1e64d88c431632ff04668ad7c8b6fc608e',
 'outputs/v2/evidence/phase7/h100_B_20261117/B_handoff.json':'fe45206f00de9070c6a5bd5efb477df753caf42411fb58dedd460cd20bfe0401',
 'outputs/v2/evidence/phase7/h100_C_20261117/C_artifact_inventory.json':'b7cc270f3880ebf77c6a51f91db513abb4e883a4619c306dfb9aad843e1ff827',
 'outputs/v2/evidence/phase7/h100_C_20261117/C_handoff.json':'998735dbe98bab432dc5df764fce4882573aaeb31b38909233082f0aed188042',
 'outputs/v2/evidence/phase7/h100_C_20261117/C_S1_inventory.json':'79256790ec0e98652eb913b5c56181b975d655b2ca6cd83ea5390abaced9c19d',
 'scripts/v2_train_grpo_audit.py':'fda1ee5b525a9d49667effab51c5dc9d5493cb3c20f521c889da03c8f5797ab1',
 'outputs/v2/evidence/phase7/grpo_config_lock.json':'1f421c03b2d7b670ddc034ced8fef46700e20cb3c7bce6abbe15e053fb876b09'}
for rel,h in anchor.items():gate(rel,h)
p6=j(ev/'phase6/selected_checkpoints.json');p4=j(ev/'phase4/artifact_inventory.json');p5=j(ev/'phase5/artifact_inventory.json')
# Explicit known Phase4 refs; test CSV files are byte-hashed only, never parsed.
for rel,h in p4['scripts'].items():gate(rel,h)
for n,h in p4['evidence_core'].items():gate(ev/'phase4'/n,h)
for section,entries in p4['sections'].items():
 for rel,d in entries.items():gate(rel,d['sha256'])
for rel,d in p5['baselines'].items():gate(rel,d['sha256'])
for rel,d in p5['scripts'].items():gate(rel,d if isinstance(d,str) else d['sha256'])
for n,d in p5['evidence'].items():
 rel=n if n.startswith('outputs/') else str(ev/'phase5'/n);gate(rel,d if isinstance(d,str) else d['sha256'])
# Verify A/B/C accepted package files rather than accepting an inventory label alone.
for idir,name in [(a_dir,'A_artifact_inventory.json'),(p7/'h100_B_20261117','B_artifact_inventory.json'),(p7/'h100_C_20261117','C_artifact_inventory.json')]:
 inv=j(idir/name);entries=inv.get('files',{})
 if isinstance(entries,list):
  for d in entries:gate(d['path'],d['sha256'])
 else:
  for rel,d in entries.items():
   path=Path(rel)
   if str(path).startswith('/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec/'):path=root/str(path).split('/MicroOneRec/',1)[1]
   elif not path.is_absolute() and not str(path).startswith(('outputs/','scripts/','source_archive/')):path=idir/path
   gate(path,d['sha256'] if isinstance(d,dict) else d)
for n,d in j(p7/'h100_C_20261117/C_S1_inventory.json')['files'].items():gate(p7/'h100_C_20261117'/n,d['sha256'])
progress('model_payloads_and_bindings',upstream_files_verified=len(verified))
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from v2_sft_data import load_group_bindings,verify_group_bindings,verify_info_txt
torch.set_num_threads(4);atoms=j(root/'outputs/v2/tokenizer/token_id_map.json')['map'];assert len(atoms)==3308
paths={g+'-SFT':f'outputs/v2/checkpoints/{g}-SFT' for g in 'ABC'};paths.update({'A-RL':'outputs/v2/debug/phase7/audit_fresh_attempt002/final_checkpoint','B-RL':'outputs/v2/checkpoints/B-RL/final_checkpoint','C-RL':'outputs/v2/checkpoints/C-RL/final_checkpoint'})
expected={g+'-SFT':p6['per_group'][g]['selected_checkpoint']['top_level_model_safetensors_sha256'] for g in 'ABC'};expected.update({'A-RL':'bbd1aacfb343788296e3af6441018c199474184999f40985a8152c42457f66fe','B-RL':'9f38d1f230f6dddfa313c6f4667431d50440e75dc92a4aaa30444c8921bba973','C-RL':'33afe7d008f54c644d79afe2d08b0b333db9bf9e71b29130719a1468c6d37d1f'})
common_vocab=None;models={};structure=None;binding_reports={}
for model_id,rel in paths.items():
 g=model_id[0];expected_dtype=torch.bfloat16 if model_id.endswith('SFT') else torch.float32;ck=root/rel;gate(ck/'model.safetensors',expected[model_id]);tok=AutoTokenizer.from_pretrained(ck,local_files_only=True)
 assert len(tok)==154977 and all(tok.encode(s,add_special_tokens=False)==[int(i)] for s,i in atoms.items())
 if common_vocab is None:common_vocab=tok.get_vocab()
 assert tok.get_vocab()==common_vocab
 conf=j(ck/'config.json');assert conf['vocab_size']==154977
 shape={};elements=0
 with safe_open(str(ck/'model.safetensors'),framework='pt',device='cpu') as f:
  for n in f.keys():
   t=f.get_tensor(n);assert t.dtype==expected_dtype and bool(torch.isfinite(t).all()),(model_id,n);shape[n]=list(t.shape);elements+=t.numel()
 assert elements==599163904 and len(shape)==310
 if structure is None:structure=shape
 assert shape==structure
 if g not in binding_reports:
  b=load_group_bindings(g,root=str(root));bc=verify_group_bindings(b,tok,atoms);ic=verify_info_txt(b,root=str(root));assert bc['all_ok'] and ic['ok'];binding_reports[g]={'catalog':len(b['item_ids']),'binding':bc,'info':ic}
 metadata={}
 for file in ck.iterdir():
  if file.is_file():metadata[file.name]={'sha256':gate(file,digest(file)),'bytes':file.stat().st_size}
 if model_id.endswith('SFT'):
  for n,h in p6['per_group'][g]['tokenizer_files'].items():gate(ck/n,h)
 for n,h in p6['per_group'][g]['sid_faces'].items():gate(root/f'outputs/v2/data/{g}'/n,h)
 models[model_id]={'path':rel,'absolute_path':str(ck),'method':g,'model_sha256':expected[model_id],'tokenizer_sha256':digest(ck/'tokenizer.json'),'vocab154977':True,'atomic3308':True,'stored_tensors':310,'stored_elements':elements,'stored_dtype':str(expected_dtype),'cpu_payload_all_finite':True,'structure_matches_other_five':True,'files':metadata,'new_forward_or_generation':False,'prior_actual_load_evidence':('outputs/v2/evidence/phase6/selected_checkpoints.json: strict valid head100' if model_id.endswith('SFT') else ('outputs/v2/evidence/phase7/substitute_takeover_20261116T170236/load_check.json' if g=='A' else f'outputs/v2/evidence/phase7/h100_{g}_20261117/validation/load_check.json')),'SID_index':f'outputs/v2/data/{g}/Industrial_and_Scientific.index.json','SID_index_sha256':digest(root/f'outputs/v2/data/{g}/Industrial_and_Scientific.index.json'),'info':f'outputs/v2/data/{g}/info.txt','info_sha256':digest(root/f'outputs/v2/data/{g}/info.txt')}
 progress('model_payloads_and_bindings',completed_models=list(models))
progress('data_order_and_training_evidence')
streams={g:[json.loads(x) for x in open(root/f'outputs/v2/manifests/rl_stream_20000_{g}.jsonl')] for g in 'ABC'}
non_sid=['stream_index','copy_index','task_id','task_type','source_id','source_split','user_id','item_id','title_clean','history_titles']
for i in range(20000):
 ref=streams['A'][i]
 for g in 'ABC':
  r=streams[g][i];assert all(r[k]==ref[k] for k in non_sid)
  assert r['stream_index']==i and r['copy_index']==i//10000
  assert r['source_split'] not in ['valid','validation','test']
  if r['task_type']!='title2sid':assert r['source_split']=='train'
  if ref['task_type']!='seq2sid':assert r['prompt']==ref['prompt']
for g,rows in streams.items():
 counts=collections.Counter((r['task_id'],r['copy_index']) for r in rows);assert len(counts)==20000 and set(counts.values())=={1}
 assert len({r['task_id'] for r in rows})==10000
 assert collections.Counter(r['task_type'] for r in rows)=={'seq2sid':16000,'title2sid':2000,'texthist2sid':2000}
order_hash=hashlib.sha256(json.dumps([[r[k] for k in non_sid] for r in streams['A']],ensure_ascii=False).encode()).hexdigest()
auditA=j(a_dir/'final_audit.json');assert all(auditA['final_budget_checks'].values()) and auditA['rewards']['all_ok']
runmeta={g:j(root/('outputs/v2/debug/phase7/audit_fresh_attempt002/run_meta.json' if g=='A' else f'outputs/v2/checkpoints/{g}-RL/run_meta.json')) for g in 'ABC'}
assert runmeta['A']['config']==runmeta['B']['config']==runmeta['C']['config']
for g in 'ABC':assert runmeta[g]['final_global_step']==625 and runmeta[g]['n_stream_rows']==20000
runtimes={};reports={'A':auditA};checks=[]
for g in 'BC':
 p=p7/f'h100_{g}_20261117';d=j(p/'final_audit.json');assert d['groups']==20000 and d['raw_catalog_legal']==320000 and d['reward_rows']==640000 and d['cpu_ema_exact_match'];reports[g]=d
 assert j(p/'supervisor_exit.json')['exit_code']==0 and j(p/'post_exit.json')['exit_code']==0
 runtimes[g]=j(p/f'{g}_resources.json')
p4diag=j(ev/'phase4/diagnostic_checks.json');assert p4diag['coverage']['complete'] and p4diag['coverage']['clean_runs']==9 and p4diag['coverage']['perturb_combos']==27
controls=j(p7/'reward_target_checks.json');assert controls['summary']['all_ok'] and controls['summary']['checks']==16
limits=['A direct formal OS exit/peak allocated/reserved and some per-task KL/token diversity unavailable; historical accepted limits retained.','A native-before reference was not captured; initial reference reconstruction is explicitly different from native-before.','A real load checks are historical before migration; current T8 verifies exact bytes/CPU structure and binding, no new GPU inference.','A full training raw SID tensor trace was not stored like B/C; do not attribute B/C 320k raw measurements to A.','A two completed physical runs and extant failed restore evidence/cost uncertainty retained.','C S1 finite KL/gradient-norm spikes retained; log norm is pre-clip under ordinary source/config path, per-step post-clip norm and token-level log-ratios not captured.','Heterogeneous A800-MIG/H100 training: same configuration does not claim a controlled speed comparison or bitwise numerical trajectory.','PyTorch allocator peaks do not constitute a continuously measured whole-device GPU peak (native allocator/optimizer memory may be outside these counters).','One training seed only; no multiseed training robustness claim; no test metrics or Phase8 execution.','Old ROOT/environment/model compatibility links remain required.']
revision=j(root/'outputs/v2/environment/environment_lock.json')['deterministic']['models']['Qwen/Qwen3-0.6B-Base']['revision'];assert revision=='da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
result={'schema':'phase7-final-current-state-check-v1','root':str(root),'scope':'Current bytes/CPU payload/shape/vocab/binding/order/config plus accepted actual training audits; no new model forward or generation','models':models,'group_bindings':binding_reports,'cross_method_non_sid_fields_compared':non_sid,'cross_method_non_sid_rows_equal':20000,'cross_method_non_sid_order_sha256':order_hash,'per_method_budget':{'physical_passes':1,'logical_epochs':2,'updates':625,'groups':20000,'base_tasks':10000,'copies':[0,1],'reward_rows':640000},'same_actual_config':True,'actual_config':runmeta['A']['config'],'manual_reward_controls_prior_actual':controls['summary'],'manual_controls_scope_note':'T3 mechanism fixture is 9.99M, not full599M; actual T4/full formal evidence is separate.','phase4_diagnostic_coverage':p4diag['coverage'],'baseline_artifacts_current_hash_checked':p5['baselines'],'baseline_validation_only':True,'phase5_evaluation_protocol':'outputs/v2/evidence/phase5/evaluation_protocol.json','formal_inference_declared_not_executed':j(ev/'phase5/evaluation_protocol.json')['generation_contract']['gen_config_declared'],'base_revision':revision,'phase6_revision_typo':{'stored_text':p6['model']['revision'],'correct_from_original_environment_lock':revision,'old_file_unchanged':True},'limits':limits,'all_verified_files':verified,'test_scope':'Test CSVs may have been byte-hashed through accepted Phase4 manifest; not parsed for evaluation, no model inference or metric/model selection on test.','finished':time.time(),'wall_seconds':time.time()-start,'status':'READY_FOR_PACKAGE_WITH_DISCLOSED_LIMITS'}
write('final_check.json',result);progress('complete',models_checked=6,files_checked=len(verified));print(json.dumps({'status':result['status'],'models':6,'verified_files':len(verified),'wall_seconds':result['wall_seconds']}))
