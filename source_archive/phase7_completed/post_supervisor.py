from pathlib import Path
import json,os,subprocess,sys,time,traceback
cfg=json.loads(Path(sys.argv[1]).read_text());p=Path(cfg['evidence']);os.chdir(cfg['repo'])
assert json.loads((p/'supervisor_exit.json').read_text())['exit_code']==0
assert not (p/'post_start.json').exists()
r={'pid':os.getpid(),'start':time.time(),'stages':[]};(p/'post_start.json').write_text(json.dumps(r,indent=2))
rc=0
try:
 for name,cmd in cfg['stages']:
  s={'name':name,'command':cmd,'start':time.time()}
  with (p/(name+'.log')).open('w') as f:
   child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
   s['pid']=child.pid;r['stages'].append(s);(p/'post_progress.json').write_text(json.dumps(r,indent=2));rc=child.wait()
  s.update(end=time.time(),exit_code=rc);(p/'post_progress.json').write_text(json.dumps(r,indent=2))
  if rc:break
except BaseException as ex:r.update(error=repr(ex),traceback=traceback.format_exc());rc=1
finally:r.update(end=time.time(),exit_code=rc);(p/'post_exit.json').write_text(json.dumps(r,indent=2))
sys.exit(0 if rc==0 else 1)
