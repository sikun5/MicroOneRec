import os,sys,json,time,hashlib,shutil,subprocess,shlex,difflib
from pathlib import Path
from datetime import datetime
ROOT=Path('/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec')
os.chdir(ROOT)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def cmd(a):
 r=subprocess.run(a,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 return {'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
start=datetime.now().astimezone();pid=1738654
out=ROOT/'outputs/v2/evidence/phase7'/('substitute_takeover_'+start.strftime('%Y%m%dT%H%M%S'))
out.mkdir(parents=True,exist_ok=False)
attempt=ROOT/'outputs/v2/debug/phase7/audit_fresh_attempt002'
report={'collected_at':start.isoformat(),'cwd':str(ROOT),'pid':pid,'attempt':str(attempt),'record_scope':'read-only existing process and artifacts; evidence copies only'}
pp=Path('/proc')/str(pid)
report['pid_exists']=pp.exists()
if pp.exists():
 argv=[a.decode() for a in (pp/'cmdline').read_bytes().split(b'\0') if a]
 report['argv']=argv;report['argv_shell']=shlex.join(argv);report['argv_sha256']=sha(pp/'cmdline')
 report['proc_cwd']=str((pp/'cwd').resolve());report['proc_identity']=cmd(['ps','-p',str(pid),'-o','pid,ppid,user,lstart,etime,stat,args'])
 ev=(pp/'environ').read_bytes().split(b'\0'); env={}
 for e in ev:
  if b'=' in e:
   k,v=e.split(b'=',1)
   if k.decode() in ['T5_RESUME_CKPT','V2_BY_VALUE_RESTORE','V2_REF_SYNC_EVENTS','V2_REF_DTYPE','PYTORCH_CUDA_ALLOC_CONF']:
    env[k.decode()]=v.decode()
 report['selected_process_environment']=env
 report['selected_environment_absent']=[k for k in ['T5_RESUME_CKPT','V2_BY_VALUE_RESTORE','V2_REF_SYNC_EVENTS','V2_REF_DTYPE','PYTORCH_CUDA_ALLOC_CONF'] if k not in env]
report['head']=cmd(['git','rev-parse','HEAD']);report['branch']=cmd(['git','branch','--show-current'])
report['run_meta']=json.loads((attempt/'run_meta.json').read_text());report['run_meta_sha256_at_collection']=sha(attempt/'run_meta.json')
metrics=[json.loads(x) for x in (attempt/'train_metrics.jsonl').read_text().splitlines() if x]
report['metrics_rows']=len(metrics);report['latest_metrics']=metrics[-1] if metrics else None
report['finite_numeric_metrics']=all(not isinstance(v,float) or __import__('math').isfinite(v) for r in metrics for v in r.values())
report['reward_rows']=sum(1 for _ in open(attempt/'reward_records.jsonl'))
report['sync_events']=[]
event=attempt/'ref_sync_events.jsonl'
if event.exists():
 report['sync_events']=[json.loads(x) for x in event.read_text().splitlines() if x]
 report['sync_file_sha256']=sha(event)
 shutil.copy2(event,out/'ref_sync_events.native.jsonl')
 report['sync_copy_sha256']=sha(out/'ref_sync_events.native.jsonl')
 report['sync_copy_matches']=report['sync_file_sha256']==report['sync_copy_sha256']
report['artifacts']={}
files=['scripts/v2_train_grpo_audit.py','scripts/v2_train_grpo.py','scripts/v2_grpo_stream.py','minionerec_trainer.py','outputs/v2/manifests/rl_stream_20000_A.jsonl','outputs/v2/evidence/phase7/grpo_config_lock.json','outputs/v2/data/A/info.txt','outputs/v2/data/A/token_id_map.json','outputs/v2/checkpoints/A-SFT/model.safetensors','outputs/v2/checkpoints/A-SFT/tokenizer.json']
for rel in files:
 p=ROOT/rel
 report['artifacts'][rel]={'exists':p.exists()}
 if p.is_file():report['artifacts'][rel].update(size_bytes=p.stat().st_size,sha256=sha(p))
for name in ['v2_train_grpo_audit.py','v2_train_grpo.py']:
 shutil.copy2(ROOT/'scripts'/name,out/name)
diff=''.join(difflib.unified_diff((ROOT/'scripts/v2_train_grpo.py').read_text().splitlines(True),(ROOT/'scripts/v2_train_grpo_audit.py').read_text().splitlines(True),fromfile='v2_train_grpo.py',tofile='v2_train_grpo_audit.py'))
(out/'audited_entry.diff').write_text(diff);report['actual_diff_sha256']=sha(out/'audited_entry.diff')
old=Path('/tmp/audit_diff.txt')
if old.exists():
 shutil.copy2(old,out/'original_audit_diff.txt');report['original_tmp_diff_sha256']=sha(old)
for rel in ['run_meta.json']:
 shutil.copy2(attempt/rel,out/('snapshot_'+rel))
report['disk_free_bytes']=shutil.disk_usage(ROOT).free
report['mem_available_kib']=[x.strip() for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')]
report['completed_at']=datetime.now().astimezone().isoformat()
(out/'takeover.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({'evidence_dir':str(out),'report_sha256':sha(out/'takeover.json'),'report':report},ensure_ascii=False,indent=2))
