from pathlib import Path
import json,os,subprocess,time
r=Path('/data/user/swu073/dsaa5020/MicroOneRec');o=r/'outputs/v2/debug/phase7/b_preflight/attempt_001'
commands=[('.venv-phase2','env_phase2'),('.venv-phase3','env_phase3'),('.venv-phase2','embedding'),('.venv-phase3','base'),('.venv-phase3','b_trainer')]
env=os.environ.copy();env.update({'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','WANDB_DISABLED':'true','WANDB_MODE':'disabled','TOKENIZERS_PARALLELISM':'false'})
for k in ['PYTHONPATH','PYTHONHOME','T5_RESUME_CKPT','V2_BY_VALUE_RESTORE','V2_REF_SYNC_EVENTS']:env.pop(k,None)
results=[];start=time.time()
try:
 for v,kind in commands:
  cmd=[str(r/v/'bin/python'),'-B',str(o/'probe.py'),kind];begin=time.time()
  with (o/(kind+'.log')).open('w') as f:
   p=subprocess.Popen(cmd,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
   (o/'status.json').write_text(json.dumps({'stage':kind,'child_pid':p.pid,'supervisor_pid':os.getpid(),'command':cmd,'start':begin,'previous':results},indent=2))
   code=p.wait(timeout=360)
  results.append({'kind':kind,'command':cmd,'exit_code':code,'start':begin,'end':time.time(),'log':str(o/(kind+'.log'))})
  (o/'run_records.json').write_text(json.dumps(results,indent=2))
 (o/'exit.json').write_text(json.dumps({'exit_code':0 if all(x['exit_code']==0 for x in results) else 1,'start':start,'end':time.time(),'results':results},indent=2))
except BaseException as e:
 (o/'exit.json').write_text(json.dumps({'exit_code':2,'error':repr(e),'start':start,'end':time.time(),'results':results},indent=2))
 raise
