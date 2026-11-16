from pathlib import Path
import hashlib,json,time,subprocess
root=Path.cwd();E=Path('outputs/v2/evidence/phase7');V=E/'final_handoff_20261117'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
start=time.time();inv=read(E/'artifact_inventory.json');fail=[];total=0
for name,x in inv['files'].items():
 p=Path(name);expect=x['sha256'];actual=sha(p);n=p.stat().st_size;size=x.get('bytes',x.get('size_bytes'))
 if actual!=expect or (size is not None and n!=size):fail.append({'path':name,'actual':actual,'expected':expect,'actual_size':n,'expected_size':size})
 total+=n
receipt=read(V/'package_receipt.json')
for name,x in receipt['root_summaries'].items():
 if sha(name)!=x['sha256']:fail.append({'root_summary_mismatch':name})
models=read(E/'final_checkpoints.json')['models'];modelnames=set(models)
assert modelnames=={'A-SFT','B-SFT','C-SFT','A-RL','B-RL','C-RL'}
for name,m in models.items():
 if m['model_sha256']!=sha(Path(m['path'])/'model.safetensors'):fail.append({'model_mismatch':name})
assert read(V/'p7_evidence_matrix.json').keys()=={'P7-1','P7-2','P7-3','P7-4','P7-5','P7-6'}
status=subprocess.check_output(['git','status','--porcelain=v1'],text=True)
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
if head!='1f5c01ccd6ee25f6989039733e15dff0072f9aad' or status:fail.append({'unexpected_git_state':{'head':head,'status':status}})
# Only inspect known task identities, never terminate other users' processes.
owned_pids=[328706,328707,352842,352843,353203,355541,355542,379578,380522,380523,380911,387678,387679,388850,388859]
states={}
for pid in owned_pids:
 p=Path('/proc')/str(pid)
 if p.exists():
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  states[str(pid)]={'exists':True,'current_cmdline':cmd,'note':'Potential PID reuse; no kill attempted'}
 else:states[str(pid)]={'exists':False}
ps=subprocess.check_output(['ps','-eo','pid,ppid,state,args'],text=True).splitlines()
needles=['/h100_telemetry.py','/h100_supervisor.py','/post_supervisor.py','/phase7_final_check.py','/sync_only_supervisor.py']
owned_matches=[line for line in ps if any(n in line for n in needles) and str(root) in line and 'python3 -B -' not in line]
if owned_matches:fail.append({'owned_training_or_audit_processes_still_active':owned_matches})
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True,capture_output=True)
gpu_proc=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True,capture_output=True)
if gpu.returncode or gpu_proc.returncode or gpu_proc.stdout.strip():fail.append({'gpu_process_check':gpu_proc.stdout,'gpu_errors':gpu.stderr+gpu_proc.stderr})
r={'scope':'Independent CPU byte/hash/index/package/owned-process/GPU-status check; no training or inference','start':start,'end':time.time(),'indexed_files':len(inv['files']),'logical_bytes_hashed':total,'failures':fail,'passed':not fail,'models':list(models),'P7_coverage':['P7-1','P7-2','P7-3','P7-4','P7-5','P7-6'],'head':head,'git_porcelain':status,'known_owned_pids':states,'active_owned_matches':owned_matches,'gpu_status':gpu.stdout,'gpu_compute_processes':gpu_proc.stdout,'connection_state':'Task SSH master still open only for final report transfer; will close and anchor local receipt afterwards.'}
p=V/'package_verification.json';assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));raise SystemExit(bool(fail))
