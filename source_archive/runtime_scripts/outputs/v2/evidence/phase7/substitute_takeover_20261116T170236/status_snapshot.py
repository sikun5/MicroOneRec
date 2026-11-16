import os,json,time,hashlib,subprocess
from pathlib import Path
from datetime import datetime
ROOT=Path('/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec');os.chdir(ROOT);a=ROOT/'outputs/v2/debug/phase7/audit_fresh_attempt002';out=ROOT/'outputs/v2/evidence/phase7/substitute_takeover_20261116T170236';p=Path('/proc/1738654')
now=datetime.now().astimezone();metrics=[json.loads(x) for x in (a/'train_metrics.jsonl').read_text().splitlines() if x];train=[x for x in metrics if 'loss' in x]
r={'collected_at':now.isoformat(),'pid':1738654,'process_exists':p.exists(),'process_cmdline':((p/'cmdline').read_bytes().replace(b'\0',b' ').decode() if p.exists() else None),'last_train_metrics':train[-1] if train else None,'last_metrics':metrics[-1] if metrics else None,'run_meta':json.loads((a/'run_meta.json').read_text()),'metrics_rows':len(metrics),'reward_rows':sum(1 for _ in open(a/'reward_records.jsonl')),'sync_events':[json.loads(x) for x in (a/'ref_sync_events.jsonl').read_text().splitlines() if x],'final_checkpoint_exists':(a/'final_checkpoint/model.safetensors').exists()}
log=Path('/tmp/t5_fresh_audit.log')
if log.exists():
 b=log.read_bytes();r['log_size_bytes']=len(b);r['log_tail']=b[-5000:].decode(errors='replace').replace('\r','\n');r['done_marker_present']=b'V2_GRPO_DONE method=A steps=625' in b;r['traceback_present']=b'Traceback (most recent call last)' in b
s=out/('status_'+now.strftime('%Y%m%dT%H%M%S')+'.json');s.write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps({'path':str(s),'sha256':hashlib.sha256(s.read_bytes()).hexdigest(),'report':r},ensure_ascii=False,indent=2))
