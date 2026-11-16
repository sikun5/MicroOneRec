#!/usr/bin/env python3
"""Passive hooks around the immutable audited entry. No extra forward, rollout,
loss, RNG draw, parameter update, optimizer/scheduler change or resume.
Safety assertions stop on selected illegal SID or nonfinite loss.
"""
import argparse,hashlib,importlib.util,json,os,sys,time,traceback
from pathlib import Path


def atomic(p,x):
 p=Path(p);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
def append(p,x):
 with open(p,'a') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n')
def raw_valid(ids,mask,catalog,eos,pad):
 raw=[int(x) for x in ids];m=[int(x) for x in mask]
 return len(raw)>=5 and tuple(raw[:4]) in catalog and raw[4]==eos and m[:5]==[1]*5 and all(x==0 for x in m[5:]) and all(x==pad for x in raw[5:])
def digest_parameters(model):
 h=hashlib.sha256()
 for p in model.parameters():h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def sample_parameters(model):
 h=hashlib.sha256()
 for _,p in list(model.named_parameters())[::19]:h.update(p.detach().reshape(-1)[:128].cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()

def self_test():
 import torch as T
 from types import SimpleNamespace
 rng=T.random.get_rng_state().clone();p=T.nn.Linear(3,2,bias=False);before=p.weight.detach().clone()
 # Only deterministic fixture construction follows; observe a known loss object unchanged.
 global torch;torch=T
 p.weight.data.copy_(T.arange(6,dtype=T.float32).reshape(2,3));rng=T.random.get_rng_state().clone()
 h=digest_parameters(p);s=sample_parameters(p)
 assert T.equal(rng,T.random.get_rng_state()) and p.weight.dtype==T.float32
 assert raw_valid([1,2,3,4,9],[1]*5,{(1,2,3,4)},9,0)
 assert not raw_valid([1,2,3,8,9],[1]*5,{(1,2,3,4)},9,0)
 assert not raw_valid([1,2,3,4,9,16],[1]*5+[0],{(1,2,3,4)},9,0)
 # Callback's CPU snapshot does not mutate values or consume RNG.
 clone={n:v.detach().cpu().contiguous().clone() for n,v in p.named_parameters()}
 assert digest_parameters(p)==h and sample_parameters(p)==s and T.equal(rng,T.random.get_rng_state())
 return {'pass':True,'catalog_guards':3,'snapshot_parameter_and_rng_unchanged':True,'scope':'CPU controls, not a training or GPU smoke'}

if __name__=='__main__' and '--self-test' in sys.argv:
 print(json.dumps(self_test()));sys.exit(0)

ap=argparse.ArgumentParser();ap.add_argument('--evidence',required=True);ap.add_argument('--entry',required=True);a,rest=ap.parse_known_args()
e=Path(a.evidence);e.mkdir(parents=True,exist_ok=True);started=time.time();status={'pid':os.getpid(),'start':started,'stage':'before_model_loading','global_step':0,'groups_observed':0,'candidates_observed':0,'peak_allocated':0,'peak_reserved':0,'reset_scope':'Before audited entry import/model loading through final save; all subsequent resets observed'}
import torch
assert torch.cuda.is_available()
original_reset=torch.cuda.reset_peak_memory_stats
original_reset()
def memory():
 status['peak_allocated']=max(status['peak_allocated'],torch.cuda.max_memory_allocated())
 status['peak_reserved']=max(status['peak_reserved'],torch.cuda.max_memory_reserved())
 status['current_allocated']=torch.cuda.memory_allocated();status['current_reserved']=torch.cuda.memory_reserved();status['updated']=time.time()
def observed_reset(*args,**kw):
 memory();append(e/'memory_resets.jsonl',{'wall':time.time(),'stage':status['stage'],'step':status['global_step'],'peak_allocated_before_reset':torch.cuda.max_memory_allocated(),'peak_reserved_before_reset':torch.cuda.max_memory_reserved()});return original_reset(*args,**kw)
torch.cuda.reset_peak_memory_stats=observed_reset

def save_status():memory();atomic(e/'progress.json',status)
entry=Path(a.entry).resolve();assert hashlib.sha256(entry.read_bytes()).hexdigest()=='fda1ee5b525a9d49667effab51c5dc9d5493cb3c20f521c889da03c8f5797ab1'
spec=importlib.util.spec_from_file_location('immutable_phase7_audit',str(entry));audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
from transformers import TrainerCallback
from safetensors.torch import save_file
original_sync=audit.V2SyncRefCallback.on_step_end

def snapshot(model,name):
 p=e/'sync512';p.mkdir(exist_ok=True)
 tensors={n:v.detach().cpu().contiguous() for n,v in model.named_parameters()}
 save_file(tensors,str(p/(name+'.safetensors')))
 return {'file':str(p/(name+'.safetensors')),'parameters_order_sha256':digest_parameters(model),'n_tensors':len(tensors),'elements':sum(v.numel() for v in tensors.values())}
def observed_sync(self,args,state,control,model=None,**kw):
 if self.ref_model is not None and state.global_step%self.steps==0:
  status['stage']='native_sync_snapshot';save_status()
  before=snapshot(self.ref_model,'ref_before');policy=snapshot(model,'policy')
  result=original_sync(self,args,state,control,model=model,**kw)
  after=snapshot(self.ref_model,'ref_after')
  atomic(e/'sync512/native_snapshot.json',{'global_step':state.global_step,'alpha':self.alpha,'before_native':before,'policy_native':policy,'after_native':after,'callback_event':self.events[-1],'wall':time.time()})
  status['stage']='training';save_status();return result
 return original_sync(self,args,state,control,model=model,**kw)
audit.V2SyncRefCallback.on_step_end=observed_sync

class Observer(TrainerCallback):
 def __init__(self,t):self.t=t;self.initial=sample_parameters(t.model);self.first_grad=False
 def on_train_begin(self,args,state,control,**kw):
  status.update(stage='training',max_steps=state.max_steps);save_status()
 def on_pre_optimizer_step(self,args,state,control,model=None,**kw):
  if not self.first_grad:
   grads=[p.grad for p in model.parameters() if p.grad is not None]
   report={'before_update':state.global_step+1,'gradient_tensors':len(grads),'gradient_elements':sum(g.numel() for g in grads),'dtypes':sorted({str(g.dtype) for g in grads}),'devices':sorted({str(g.device) for g in grads}),'finite':all(bool(torch.isfinite(g).all()) for g in grads)}
   atomic(e/'first_gradients.json',report);assert report['finite'];self.first_grad=True
 def on_step_end(self,args,state,control,model=None,**kw):
  status.update(global_step=state.global_step,stage='training',optimizer_created=self.t.optimizer is not None)
  if state.global_step<=2:append(e/'first_updates.jsonl',{'global_step':state.global_step,'sample_sha256':sample_parameters(model),'initial_sample_sha256':self.initial,'sample_changed':sample_parameters(model)!=self.initial,'optimizer':type(self.t.optimizer).__name__,'wall':time.time()})
  save_status()
 def on_log(self,args,state,control,logs=None,**kw):
  import math
  if logs:
   bad={k:v for k,v in logs.items() if isinstance(v,(int,float)) and not math.isfinite(v)}
   assert not bad,('Nonfinite trainer log',bad)
 def on_train_end(self,args,state,control,**kw):status.update(global_step=state.global_step,stage='train_ended_saving');save_status()

Base=audit.V2ReReTrainer
class ObservedTrainer(Base):
 def __init__(self,*args,**kw):
  super().__init__(*args,**kw)
  policy=list(self.model.parameters());ref=list(self.ref_model.parameters())
  init={'trainable':sum(p.numel() for p in policy if p.requires_grad),'total':sum(p.numel() for p in policy),'policy_dtype':sorted({str(p.dtype) for p in policy}),'ref_dtype':sorted({str(p.dtype) for p in ref}),'policy_devices':sorted({str(p.device) for p in policy}),'ref_devices':sorted({str(p.device) for p in ref}),'ref_grad':any(p.requires_grad for p in ref),'independent':all(p.data_ptr()!=q.data_ptr() for p,q in zip(policy,ref)),'world_size':self.accelerator.num_processes,'config':self.args.to_dict(),'wall':time.time()}
  assert init['trainable']==init['total']==599163904 and init['policy_dtype']==['torch.float32'] and init['ref_dtype']==['torch.bfloat16'] and not init['ref_grad'] and init['independent'] and init['world_size']==1
  atomic(e/'initialization.json',init)
  self.catalog={tuple(self.processing_class(line.split('\t')[0].strip(),add_special_tokens=False).input_ids) for line in open(self.info_file)}
  assert len(self.catalog)==3686 and all(len(x)==4 for x in self.catalog)
  self._observed_group=None;self.add_callback(Observer(self));status['stage']='initialized';save_status()
 def _prepare_inputs(self,inputs):
  assert isinstance(inputs,list) and len(inputs)==16
  identities={(x['stream_index'],x['task_id'],x['copy_index'],x['target_sid']) for x in inputs};assert len(identities)==1
  out=super()._prepare_inputs(inputs)
  ids=out['completion_ids'].detach().cpu().tolist();masks=out['completion_mask'].detach().cpu().tolist();adv=out['advantages'].detach().cpu().tolist();rews=out['sliced_rewards'].detach().cpu().tolist()
  valid=[raw_valid(r,m,self.catalog,self.processing_class.eos_token_id,self.processing_class.pad_token_id) for r,m in zip(ids,masks)]
  i=inputs[0];rec={'stream_index':i['stream_index'],'task_id':i['task_id'],'copy_index':i['copy_index'],'task_type':i['task_type'],'target_sid':i['target_sid'],'global_step_before':self.state.global_step,'completion_ids':ids,'completion_mask':masks,'advantages':adv,'summed_rewards':rews,'legal_count':sum(valid),'candidate_count':len(ids),'wall':time.time()}
  append(e/'raw_groups.jsonl',rec);assert len(ids)==16 and all(valid),('Illegal selected raw SID',rec)
  assert bool(torch.isfinite(out['ref_per_token_logps']).all()) and bool(torch.isfinite(out['advantages']).all())
  self._observed_group=rec;status['groups_observed']+=1;status['candidates_observed']+=len(ids)
  return out
 def compute_loss(self,model,inputs,*args,**kw):
  loss=super().compute_loss(model,inputs,*args,**kw)
  assert bool(torch.isfinite(loss.detach()).all()),'Nonfinite actual loss'
  g=self._observed_group
  append(e/'group_loss.jsonl',{'stream_index':g['stream_index'],'task_type':g['task_type'],'global_step_before':self.state.global_step,'loss':float(loss.detach()),'kl':float(self._metrics['kl'][-1]),'categorical_diversity':float(self._metrics['categorical_diversity'][-1]),'token_diversity':float(self._metrics['token_diversity'][-1]),'completion_length':float(self._metrics['completion_length'][-1])})
  return loss

audit.V2ReReTrainer=ObservedTrainer
sys.argv=[str(entry)]+rest
rc=1
try:
 audit.main();rc=0;status['stage']='complete'
except BaseException as ex:
 status.update(stage='failed',error=repr(ex),traceback=traceback.format_exc());raise
finally:
 status.update(exit_code=rc,end=time.time(),wall_seconds=time.time()-started);save_status();atomic(e/'in_process_exit.json',status)
