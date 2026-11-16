import os,json,subprocess,time,hashlib
from pathlib import Path
from datetime import datetime
root=Path('/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec');os.chdir(root);out=root/'outputs/v2/evidence/phase7/substitute_takeover_20261116T170236'
argv=[str(root/'.venv-phase3/bin/python'),str(out/'loadcheck.py'),'--method','A','--checkpoint','outputs/v2/debug/phase7/audit_fresh_attempt002/final_checkpoint','--out',str(out),'--training-pid','1738654']
r={'start_at':datetime.now().astimezone().isoformat(),'argv':argv,'cwd':str(root),'task':'validation-only load check, not formal training'}
(out/'load_check_launch.json').write_text(json.dumps(r,indent=2))
with open(out/'load_check.log','w') as log:
 p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ),start_new_session=True)
 r['pid']=p.pid;(out/'load_check_status.json').write_text(json.dumps(r,indent=2));rc=p.wait()
r['end_at']=datetime.now().astimezone().isoformat();r['exit_code']=rc
(out/'load_check_status.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
