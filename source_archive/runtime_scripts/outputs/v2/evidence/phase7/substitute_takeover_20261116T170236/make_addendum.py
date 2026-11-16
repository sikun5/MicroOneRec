import os,json,hashlib,shutil
from pathlib import Path
from datetime import datetime
ROOT=Path('/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec');os.chdir(ROOT)
out=ROOT/'outputs/v2/evidence/phase7/substitute_takeover_20261116T170236'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
d={'created_at':datetime.now().astimezone().isoformat(),'authorization':'working-record boss section: 代班接管证据接受、真实512事件及P1-2裁定','base_lock_path':'outputs/v2/evidence/phase7/grpo_config_lock.json','base_lock_sha256':sha(ROOT/'outputs/v2/evidence/phase7/grpo_config_lock.json'),'unchanged_base_lock':True,'checkpoint_retention_exception':{'planned_save_total_limit':4,'actual_A_save_total_limit':6,'A_attempt':'outputs/v2/debug/phase7/audit_fresh_attempt002','future_B_C_save_total_limit':6,'scope':'checkpoint disk retention only; no change to sampling/loss/optimizer/LR/sync625/final selection'},'A_peak_memory':'unknown; not collected; do not substitute T4','native512_event_sha256':sha(out/'ref_sync_events.native.jsonl'),'reconstructed_reference_audit_sha256':sha(out/'reference_baseline_and_expected.json')}
p=out/'config_addendum.json';p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
for src in ['/tmp/phase7_subworker_takeover_20261116.py','/tmp/phase7_subworker_refaudit_20261116.py']:
 shutil.copy2(src,out/Path(src).name)
pp=Path('/proc/1738647')
parent={'parent_pid':1738647,'exists':pp.exists()}
if pp.exists():
 parent['cmdline']=[x.decode() for x in (pp/'cmdline').read_bytes().split(b'\0') if x]
 parent['cwd']=str((pp/'cwd').resolve())
(out/'launcher_parent.json').write_text(json.dumps(parent,ensure_ascii=False,indent=2))
print(json.dumps({'addendum_path':str(p),'addendum_sha256':sha(p),'launcher_parent':parent,'token_id_map_sha256':sha(ROOT/'outputs/v2/tokenizer/token_id_map.json')},ensure_ascii=False,indent=2))
