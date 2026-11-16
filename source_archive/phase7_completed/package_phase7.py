"""CPU-only packaging of already-accepted Phase7 evidence; no model generation."""
from pathlib import Path
import json,hashlib,time,subprocess,shlex
ROOT=Path.cwd(); E=Path('outputs/v2/evidence/phase7'); V=E/'final_handoff_20261117'; C=V/'check_attempt002'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def meta(p):p=Path(p);return {'sha256':sha(p),'bytes':p.stat().st_size}
def dump(p,x):
 p=Path(p);assert not p.exists(),str(p)
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def text(p,x):
 p=Path(p);assert not p.exists(),str(p);p.write_text(x)
def run(cmd):
 p=subprocess.run(cmd,text=True,capture_output=True);return {'command':cmd,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
final=read(C/'final_check.json');assert final['status']=='READY_FOR_PACKAGE_WITH_DISCLOSED_LIMITS'
files=dict(final['all_verified_files']);extra=read(V/'source_control_supplement.json');bound=read(V/'t4_counter_bindings.json')
t4=read('outputs/v2/debug/phase7/grpo_smoke/t4r2_inventory.json')
assert sha('outputs/v2/debug/phase7/grpo_smoke/t4r2_inventory.json')==bound['inventory_sha256']
full_smoke={}
for m,x in bound['counters'].items():
 p=x['path'];assert sha(p)==x['current_sha256']==t4['files'][p]['sha256'];d=read(p)
 flat=[i for row in d['micro_rows'] for i in row['stream_index']]
 assert flat==list(range(64)) and d['micro_forwards']==64 and d['optimizer_steps']==2 and d['loss_finite'] and d['params_changed']
 assert d['full_trainable'] and d['total_params']==599163904
 assert all(r['grad_info']['dtype']=='torch.float32' and r['grad_info']['numel']==599163904 and r['grad_info']['tensors']==310 for r in d['micro_rows'])
 full_smoke[m]={'source':p,'sha256':sha(p),'micro_forwards':64,'optimizer_steps':2,'candidate_rows_per_micro':16,'accumulation_window':32,'flattened_stream_indices_exact_0_to_63':True,'full_fp32_live_gradient_elements':599163904,'full_fp32_live_gradient_tensors':310,'params_changed':True,'loss_finite':True,'legacy_nested_scalar_check_value':d['consumed_stream_index_sorted_eq'],'legacy_flag_note':'The old scalar flag is false, but raw micro_rows store singleton lists. Independent flattening verifies exactly 0..63; do not rewrite old report.'}
 files[p]=meta(p)
for p in ['outputs/v2/debug/phase7/grpo_smoke/t4r2_inventory.json','outputs/v2/evidence/phase7/reward_target_checks.json','outputs/v2/evidence/phase7/rollout_gradient_checks.json','minionerec_trainer.py','LogitProcessor.py']:
 files[p]=meta(p)
for x in extra['manual_controls'].values():
 if isinstance(x,dict) and 'path' in x and Path(x['path']).is_file():files[x['path']]=meta(x['path'])
# Verify core source bytes against the actual HEAD object, not merely a textual source citation.
for p in ['minionerec_trainer.py','LogitProcessor.py','scripts/v2_train_grpo_audit.py']:
 h=hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+p])).hexdigest();assert sha(p)==h
assert sha('scripts/v2_train_grpo_audit.py')=='fda1ee5b525a9d49667effab51c5dc9d5493cb3c20f521c889da03c8f5797ab1'
A=E/'substitute_takeover_20261116T170236'; ad=read(A/'final_audit.json');ah=read(A/'A_final_handoff.json');ar=read(A/'A_resources.json')
bc={m:E/f'h100_{m}_20261117' for m in 'BC'}
reports={m:read(p/'final_audit.json') for m,p in bc.items()}
handoffs={m:read(p/f'{m}_handoff.json') for m,p in bc.items()}
resources={m:read(p/f'{m}_resources.json') for m,p in bc.items()}
selection=read('outputs/v2/evidence/phase6/selected_checkpoints.json')
models=final['models']
for name,d in models.items():
 assert d['model_sha256']==d['files']['model.safetensors']['sha256']
 assert d['cpu_payload_all_finite'] and d['stored_elements']==599163904 and d['stored_tensors']==310
 assert d['stored_dtype']==('torch.bfloat16' if name.endswith('SFT') else 'torch.float32')
 # The full checker hashes these bytes already. Rehash model once here to catch any intervening mutation.
 assert sha(Path(d['path'])/'model.safetensors')==d['model_sha256']
for m,d in reports.items():
 assert d['training_steps']==625 and d['groups']==20000 and d['reward_rows']==640000 and d['raw_catalog_legal']==320000
 assert d['cpu_ema_exact_match'] and d['native_before_captured'] and d['cpu_ema_different_elements']==0 and d['all_model_finite']
 assert d['max_reward_error']==d['max_summed_reward_fp32_error']==0
assert all(ad['final_budget_checks'].values())
checkpoints={'schema':'phase7-final-checkpoints-v1','created':time.time(),'status':'worker complete; awaiting boss Phase7 acceptance','remote_root':str(ROOT),'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'base_model':{'repo':'Qwen/Qwen3-0.6B-Base','revision':final['base_revision'],'legacy_phase6_revision_typo':final['phase6_revision_typo']},'canonical_A_path_policy':'Keep audited audit_fresh_attempt002 path; old outputs/v2/checkpoints/A-RL is unselected historical evidence, not a replacement.','models':models,'source_SFT_selection':{'path':'outputs/v2/evidence/phase6/selected_checkpoints.json','sha256':sha('outputs/v2/evidence/phase6/selected_checkpoints.json'),'selected_steps':{m:selection['per_group'][m]['best']['step'] for m in 'ABC'}},'shared_map':{'path':'outputs/v2/tokenizer/token_id_map.json',**meta('outputs/v2/tokenizer/token_id_map.json')},'SID_bindings':final['group_bindings'],'source_streams':{m:{'path':f'outputs/v2/manifests/rl_stream_20000_{m}.jsonl',**meta(f'outputs/v2/manifests/rl_stream_20000_{m}.jsonl')} for m in 'ABC'},'baselines':final['baseline_artifacts_current_hash_checked'],'phase4_diagnostics':final['phase4_diagnostic_coverage'],'phase8_inference_protocol':{'source':final['phase5_evaluation_protocol'],'declared_only_not_executed':True,'config':final['formal_inference_declared_not_executed']},'current_check_scope':final['scope'],'test_scope':final['test_scope'],'limits':final['limits'],'not_verified_exact_training_resume':True}
dump(E/'final_checkpoints.json',checkpoints)
# Historical resource records are preserved verbatim and explicitly distinguished from current H100 receipts.
res={'schema':'phase7-resource-summary-v1','A_legacy_MIG_verbatim':ar,'B_H100_verbatim':resources['B'],'C_H100_verbatim':resources['C'],'cost_scope_notes':['A two complete formal runs are both counted; total HF time 53877.4379s. Three extant OOM log records are not an exact physical launch/time ledger. Missing A process peaks/OS exit remain unknown.','B/C allocated/reserved are same-training-process PyTorch allocator peaks from before model load to final save; native paged allocations outside PyTorch and total-device/GPU-busy hours are not continuously measured.','A/MIG and B/C H100 times are descriptive heterogeneous costs, not a controlled speedup claim.','CPU audits, validation checks, smoke/debug and migration are separate categories; no unknown cost replaced by zero.','No new model forward/generation in T8; current six-model CPU finite/hash/structure checks plus prior actual loads are reused.'],'T8_CPU_checks':{'attempt001':read(V/'check_exit.json'),'attempt002':read(C/'check_exit.json'),'attempt001_failure':'Verifier mistakenly expected SFT FP32; source/init proves BF16 storage. No model modified. Failed evidence retained.','attempt002_script_wall_seconds':final['wall_seconds']},'C_S1_CPU_diagnostic':read(bc['C']/'C_S1_process.json') if (bc['C']/'C_S1_process.json').exists() else {'source':'h100_C_20261117/C_S1_inventory.json','note':'Read the indexed process receipt for actual 9.982590s exit0 diagnostic; no new GPU operation.'},'T4_full_size_smoke':{'index':'outputs/v2/debug/phase7/grpo_smoke/t4r2_inventory.json','sha256':bound['inventory_sha256'],'note':'Historical full-size smoke, not another formal GRPO run. v4 configuration/attempt history retained.'}}
dump(E/'resource_summary.json',res)
P7={
 'P7-1':{'status':'passed','evidence':['reward_target_checks.json','rollout_gradient_checks.json','grpo_config_lock.json','scripts/v2_train_grpo_audit.py'],'facts':'16/16 manual reward/target controls plus audited rule/rank AST equal original. Each formal run 640000 reward rows independently checked; B/C recomputed correctness/rank/sum max error0.','scope':'T3 tiny fixture is mechanism only; not full-size gradient evidence.'},
 'P7-2':{'status':'passed with disclosed measurement scope','evidence':['final_handoff_20261117/t4_counter_summary.json','substitute_takeover_20261116T170236/final_audit.json','h100_B_20261117/final_audit.json','h100_C_20261117/final_audit.json'],'facts':'Full-size T4 A/B/C 64 actual micro forwards, 2 optimizer updates, each16 candidates/group,32 accumulation,all599163904 FP32 live gradients. B/C formal320000 selected raw candidates each strictly four atomic codes+EOS/catalog valid; independent advantage discrepancy max7.152557e-7 due CPU/GPU float operations.','scope':'A historical full raw selected token stream not stored like B/C; A strict validation10/10 and accepted actual group/reward/native audits are the recorded scope. Internal discarded dead-prefix EOS fallback is known; no final selected B/C illegal output.'},
 'P7-3':{'status':'passed','evidence':['final_handoff_20261117/check_attempt002/final_check.json','manifests/rl_stream_20000_{A,B,C}.jsonl','grpo_config_lock.json'],'facts':'A/B/C share10000task IDs eachcopy0/1 exactly once:20000 groups,625 complete updates,640000dual-function reward rows. Current cross-method20000non-SID source/text/order fields exact; only SID encoding/prompt changes. G16/batch16/accum32 =512candidate effectivebatch. 1physical20k pass equals2logicalepochs. All actual GRPO configuration fields equal, approved save_total_limit6 overrides old v4 limit without rewriting v4.'},
 'P7-4':{'status':'passed with disclosed native-before scope','evidence':['final_handoff_20261117/t4_counter_summary.json','substitute_takeover_20261116T170236/reference_baseline_and_expected.json','h100_B_20261117/initialization.json','h100_C_20261117/initialization.json','h100_B_20261117/sync512/','h100_C_20261117/sync512/'],'facts':'All policy/master/grad FP32 full599M; BF16 computation; independent frozen complete BF16ref; real event512/.6 only. CPU full310tensor FP32EMA then BF16 matches actual after exactly all3.','scope':'A before is reconstructed initial frozen BF16ref and preserved native actualafter/policy512; B/C nativebefore/policy/after all captured. Ref is gradient-frozen but actual sync updates values.'},
 'P7-5':{'status':'passed','evidence':['final_checkpoints.json','final_handoff_20261117/check_attempt002/final_check.json','substitute_takeover_20261116T170236/load_check.json','h100_B_20261117/validation/load_check.json','h100_C_20261117/validation/load_check.json','phase6/selected_checkpoints.json'],'facts':'Three RL completed625/20k and preselected final checkpoints; three SFT retained. Current six fullpayloads all310tensors finite, same599163904shape and154977vocab/3308atomic/3686bindings. Prior actual loads succeed; RL fixed10 originalvalidation each strict10/hit0, Beam10return1.','scope':'T8 is current CPU/hash/structure, no fresh GPU inference. SFT storedBF16 vsRL storedFP32. Hits0/10 is diagnostic notHR@10 or test. Final checkpoints supplied for inference, not verified exact optimizer/RNG/ref resumption.'},
 'P7-6':{'status':'passed with accepted historical limitations and negative findings','evidence':['resource_summary.json','run_records.md','substitute_takeover_20261116T170236/final_audit.json','h100_B_20261117/group_loss.jsonl','h100_C_20261117/group_loss.jsonl','h100_C_20261117/C_S1_inventory.json'],'facts':'Per-task zero-advantage/reward stats,global KL/diversity/steps and A costs recorded. B/C passive fullgroup/raw telemetry and true OS exits/allocatorpeaks added. C step514 true maxKL10997.427993829/preclipgrad10472.15722656 retained.','scope':'A per-taskKL/token diversity,fullprocesspeaks,directformalOSexit unknown. Logged gradnorm is preclip; configured clipmax.3 proven source/config, no perstep postclipnorm captured. No causal spike attribution or speculative probabilities. No data/result-based rerun.'}}
dump(V/'t4_counter_summary.json',{'scope':'Existing full-size accepted T4 raw micro rows rechecked on CPU only','inventory_sha256':bound['inventory_sha256'],'methods':full_smoke})
dump(V/'p7_evidence_matrix.json',P7)
# Preserve actual accepted diagnostics as data without rounding selected examples into invented metric gains.
diag={'A':{'global_training_metrics':ad['metrics'],'reward_task_stats':ad['rewards']['task_stats'],'limits':ad['measurement_limits']},'B':{'actual_task_stats':reports['B']['type_stats'],'native_sync':reports['B']['native_sync_event'],'limits':handoffs['B']['known_limitations']},'C':{'actual_task_stats':reports['C']['type_stats'],'native_sync':reports['C']['native_sync_event'],'limits':handoffs['C']['known_limitations'],'S1_source_inventory':str(bc['C']/'C_S1_inventory.json')}}
dump(V/'diagnostic_summary.json',diag)
records=['# Phase7 final run records\n','This is an evidence handoff, not Phase8 evaluation. Remote cwd: `'+str(ROOT)+'`. HEAD `'+checkpoints['head']+'`. Core audit entry SHA `'+sha('scripts/v2_train_grpo_audit.py')+'`.\n','## Selected runs\n','A uses `outputs/v2/debug/phase7/audit_fresh_attempt002/final_checkpoint`; preserved original unselected A is counted separately in resource_summary. Actual historical command/source and launch evidence: `substitute_takeover_20261116T170236/A_run_records.md`, `launcher_parent.json`, `final_run_meta.json`. The A native-after event is real; initial-before is reconstructed and labeled.\n']
for m,p in bc.items():
 st=read(p/'supervisor_start.json');ex=read(p/'supervisor_exit.json');cfg=read(p/'launch_config.json')
 records+=['## '+m+'-RL\n','Actual process: supervisor '+str(st['pid'])+', train '+str(st['child_pid'])+'; UNIXstart '+str(st['start'])+', end '+str(ex['end'])+', OSexit '+str(ex['exit_code'])+'. `'+str(p/'launch_config.json')+'` holds actual environment and command.\n','```sh\ncd '+shlex.quote(str(ROOT))+'\n'+shlex.join(cfg['command'])+'\n```\n','The command above is the already-completed recorded invocation, not an instruction to rerun. All telemetry sources/configs/CPU controls and native tensors are in this run evidence inventory. Formal step625; full20k/640k reward audit and separate fresh-process validation accepted.\n']
records+=['## Current T8 CPU check\n','`'+str(C/'check_start.json')+'` records real interpreter/command. attempt001 exited1 due verifier assumingSFTFP32; attempt002 corrected only stage expectation toSFTBF16 and exited0. Models/old reports were not altered. `final_check.json`: six full finite payloads,333verified path hashes,20000equal nonSID source rows. Package builder rehashes sixmodelbytes and checks core HEAD bindings again.\n','## Diagnostic and resource boundaries\n','A per-task reward/zeroadvantage and globalKL/diversity available, but historical per-taskKL/token diversity, peaks/directOSexit remainunknown. B/C PyTorchallocated/reserved peaks are not continuous whole-device nativepaged memory. C S1 max spike is step514, not former tail623; actual loggednorm is preclip, no measured postclipnorm. Kernel4.18 warnings/oldpathsymlinks remain, no dependency/kernel upgrades.\n','One LLM seed42; no training-seed robustness or heterogeneoushardware bitwisereplay claim. Same preselected finals, no validation/test choosing RL attempts. All historical failed/debug artifacts retained. Current check only hashes testCSVbytes for identity via prior index; no test model inference, metrics, or modelselection. Beam50formal protocol is frozen for future Phase8, not executed.\n','## Status\n','Worker evidence complete except final self-owned connection cleanup receipt, which is anchored in local working-record after SSH close. Await boss Phase7 acceptance. No automatic Phase8, training restart, push, or background worker task after final closure.\n']
text(E/'run_records.md','\n'.join(records))
accept=['# Phase7 acceptance evidence\n','Worker requests boss acceptance; this file does not itself assert boss approval. Current six-modelCPU/hash check passed; prior actual training/load evidence is reused, no new GPU forward.\n','| Requirement | Worker evidence outcome | Evidence |\n|---|---|---|']
for k,d in P7.items():accept+=['| '+k+' | '+d['status']+' | '+', '.join('`'+x+'`' for x in d['evidence'])+' |']
for k,d in P7.items():accept+=['\n## '+k+'\n',d['facts']+'\n',d.get('scope','')+'\n']
accept+=['\n## Phase8 handoff / unchanged limits\n','Six models and two baseline refs in `final_checkpoints.json`, Phase4 nine clean/27perturb quantizer diagnostics complete. All model/tokenizer/SID/input hashes in final checker and final index. Base revision normalized from original40-character source; old Phase6 39-character typo is preserved and disclosed without changing old artifact. No new scientific effect claimed. C collapse/instability and validation0/10 are retained. UnresolvedP0: none identified by the recorded checks; historical unknown fields remain explicitly unknown.\n','Final process/GPU/SSH cleanup receipt is written after this package and anchored by local authoritative `working-record.md`; it is intentionally external to this self-contained artifact index to avoid a close-connection/selfhash cycle.\n']
text(E/'acceptance.md','\n'.join(accept))
# Index existing accepted artifacts + exact new summaries; do not recursively include all runtime/checkpoints.
for p in [E/'final_checkpoints.json',E/'resource_summary.json',E/'run_records.md',E/'acceptance.md']:
 files[str(p)]=meta(p)
for p in V.rglob('*'):
 if p.is_file() and p.name not in ['artifact_inventory.json','package_receipt.json','package_stdout.log']:
  files[str(p)]=meta(p)
# File index excludes itself and its receipt by explicit construction.
index={'schema':'phase7-final-artifact-inventory-v1','root':str(ROOT),'files':files,'n_files':len(files),'scope':'Selected six model files and bound data/tokenizer/evidence, Phase4/5 accepted indexes, actual A/B/C audit packages/S1, current CPU checker and canonical summaries. Not all retained runtime/intermediate checkpoint files.','excluded':['artifact_inventory.json (self digest anchored in working-record)','final_handoff_20261117/package_receipt.json (contains digest of this index)','local connection_cleanup.json written only after SSHclose','all unselected intermediate checkpoints/runtime not in these accepted manifests; retained without deletion'],'test_scope':final['test_scope']}
dump(E/'artifact_inventory.json',index)
receipt={'schema':'phase7-final-package-receipt-v1','created':time.time(),'root':str(ROOT),'root_summaries':{str(E/n):meta(E/n) for n in ['final_checkpoints.json','artifact_inventory.json','resource_summary.json','run_records.md','acceptance.md']},'p7_requirements':list(P7),'n_index_files':len(files),'six_models':{k:{'path':v['path'],'model_sha256':v['model_sha256'],'stored_dtype':v['stored_dtype']} for k,v in models.items()},'self_reference_policy':'Receipt/index hash anchored in working record, not recursively in own digest','T8_current_checker':meta(C/'final_check.json'),'open_P0':[],'new_gpu_forward':False,'phase8_executed':False}
dump(V/'package_receipt.json',receipt)
print(json.dumps(receipt))
