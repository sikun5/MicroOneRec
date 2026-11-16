#!/usr/bin/env python3
"""Persistent supervisor, no credentials. One formal run per fresh evidence root."""
from pathlib import Path
import json,os,subprocess,sys,time,hashlib,traceback
cfg=json.loads(Path(sys.argv[1]).read_text());repo=Path(cfg['repo']);ev=Path(cfg['evidence']);out=repo/cfg['output'];os.chdir(repo)
def atomic(name,x):
 p=ev/name;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
assert not (ev/'supervisor_start.json').exists(),'No duplicate supervisor'
assert not out.exists(),'Do not overwrite any pre-existing formal output'
assert hashlib.sha256((repo/'scripts/v2_train_grpo_audit.py').read_bytes()).hexdigest()==cfg['entry_sha256']
meta={'pid':os.getpid(),'start':time.time(),'cwd':str(repo),'command':cfg['command'],'evidence':str(ev),'output':str(out)};atomic('supervisor_start.json',meta)
env=os.environ.copy();env.update(cfg['environment'])
rc=1
try:
 with (ev/'formal.log').open('w') as log:
  p=subprocess.Popen(cfg['command'],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
  meta['child_pid']=p.pid;atomic('supervisor_start.json',meta)
  rc=p.wait()
except BaseException as e:
 meta['error']=repr(e);meta['traceback']=traceback.format_exc()
finally:
 meta.update(end=time.time(),exit_code=rc);meta['wall_seconds']=meta['end']-meta['start'];atomic('supervisor_exit.json',meta)
sys.exit(0 if rc==0 else 1)
