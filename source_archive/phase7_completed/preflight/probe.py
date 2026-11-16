"""Read-only B-RL preflight: imports, model loading and forward/generation only."""
from pathlib import Path
import gc,importlib,importlib.metadata as md,json,os,re,sys,time,traceback
ROOT=Path('/data/user/swu073/dsaa5020/MicroOneRec');OUT=ROOT/'outputs/v2/debug/phase7/b_preflight/attempt_001'
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
kind=sys.argv[1];r={'kind':kind,'pid':os.getpid(),'start':time.time(),'pass':False,'training':False,'optimizer_step':False}
def mem():
 import torch
 torch.cuda.synchronize()
 return {'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()}
try:
 import torch
 from transformers import AutoTokenizer,AutoModel,AutoModelForCausalLM
 torch.manual_seed(42)
 r['python']=sys.version;r['prefix']=sys.prefix;r['base_prefix']=sys.base_prefix
 r['versions']={n:md.version(n) for n in ['torch','transformers','trl','bitsandbytes','accelerate','datasets','numpy','tokenizers','safetensors','huggingface-hub']}
 assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
 r['cuda']={'runtime':torch.version.cuda,'device':torch.cuda.get_device_name(0),'device_count':torch.cuda.device_count(),'capability':torch.cuda.get_device_capability(0),'free_total':list(torch.cuda.mem_get_info())}
 assert r['cuda']['free_total'][0]>16*1024**3,'Insufficient free GPU for sequential load checks'
 if kind.startswith('env'):
  imports=['torch','transformers','tokenizers','safetensors','numpy','trl','bitsandbytes','accelerate','datasets']
  for name in imports:importlib.import_module(name)
  r['imports']=imports
  import bitsandbytes as bnb
  lib=bnb.cextension.lib
  r['bitsandbytes']={'native_type':type(lib).__name__,'compiled_with_cuda':getattr(lib,'compiled_with_cuda',None),'library':str(getattr(getattr(lib,'_lib',None),'_name',None))}
  x=torch.linspace(-1,1,4096,device='cuda',dtype=torch.float32)
  q,state=bnb.functional.quantize_blockwise(x);y=bnb.functional.dequantize_blockwise(q,state)
  assert torch.isfinite(y).all();r['native_cuda_quantize']={'finite':True,'max_abs_error':float((x-y).abs().max())}
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   z=torch.ones((16,16),device='cuda')@torch.ones((16,16),device='cuda')
  assert torch.isfinite(z).all() and float(z[0,0])==16
  r['bf16_matmul']={'dtype':str(z.dtype),'value':float(z[0,0])};r['memory']=mem()
 elif kind in ['base','embedding']:
  path=ROOT.parent/'models'/('Qwen3-0.6B-Base' if kind=='base' else 'Qwen3-Embedding-4B')
  tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
  cls=AutoModelForCausalLM if kind=='base' else AutoModel
  model=cls.from_pretrained(path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
  model.requires_grad_(False);model.eval()
  inputs=tok(['Industrial replacement component'],return_tensors='pt',padding=True,truncation=True,max_length=2048).to('cuda')
  with torch.inference_mode():
   output=model(**inputs)
   if kind=='embedding':
    h=output.last_hidden_state.float();mask=inputs.attention_mask.unsqueeze(-1).to(h.dtype)
    pooled=(h*mask).sum(1)/mask.sum(1);result=torch.nn.functional.normalize(pooled,dim=1,eps=1e-12)
    assert torch.isfinite(result).all();r['pooling']={'kind':'masked_mean_then_L2','shape':list(result.shape),'norm':float(result.norm()),'finite':True,'max_length':2048}
   else:
    logits=output.logits;assert torch.isfinite(logits).all();r['logits']={'shape':list(logits.shape),'finite':True}
  r.update({'model_path':str(path),'model_class':type(model).__name__,'parameters':sum(p.numel() for p in model.parameters()),'dtype':str(next(model.parameters()).dtype),'attention':model.config._attn_implementation,'memory':mem()})
 elif kind=='b_trainer':
  # Both imported as modules: no entry main()/Trainer.train() is called.
  normal=importlib.import_module('scripts.v2_train_grpo');audit=importlib.import_module('scripts.v2_train_grpo_audit')
  def forbidden(*a,**k):raise AssertionError('Training is forbidden in M3 preflight')
  for module in [normal,audit]:
   module.V2ReReTrainer.train=forbidden;module.V2ReReTrainer._inner_training_loop=forbidden
  r['imported_entries']={m.__name__:m.__file__ for m in [normal,audit]}
  for name in ['T5_RESUME_CKPT','V2_BY_VALUE_RESTORE','V2_REF_SYNC_EVENTS']:assert not os.environ.get(name),('Unexpected resume/telemetry env',name)
  os.environ['V2_REF_DTYPE']='bf16'
  torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
  ds,n=audit.build_dataset(str(ROOT/'outputs/v2/manifests/rl_stream_20000_B.jsonl'))
  cfg=audit.GRPOConfig(output_dir=str(OUT/'trainer_init_only'),per_device_train_batch_size=16,per_device_eval_batch_size=16,
   gradient_accumulation_steps=32,num_generations=16,num_train_epochs=1,learning_rate=1e-5,beta=1e-3,temperature=1.,
   max_completion_length=128,max_prompt_length=512,sync_ref_model=True,ref_model_sync_steps=512,ref_model_mixup_alpha=.6,
   bf16=True,optim='paged_adamw_32bit',lr_scheduler_type='cosine',warmup_ratio=.03,max_grad_norm=.3,
   logging_steps=1,save_strategy='steps',save_steps=.05,save_total_limit=4,eval_strategy='no',report_to=[],seed=42,
   dataloader_num_workers=0,gradient_checkpointing=True,log_completions=False)
  ckpt=str(ROOT/'outputs/v2/checkpoints/B-SFT');info=str(ROOT/'outputs/v2/data/B/info.txt')
  trainer=audit.V2ReReTrainer(model=ckpt,base_model=ckpt,reward_funcs=[audit.rule_reward,audit.make_ndcg_rule_reward(16)],args=cfg,
   beam_search=True,length_penalty=0.,add_gt=False,dynamic_sampling=False,test_during_training=False,test_beam=16,
   dapo=False,gspo=False,info_file=info,prompt2history=None,history2target=None,train_dataset=ds,eval_dataset=None,callbacks=[])
  assert trainer.state.global_step==0 and trainer.optimizer is None
  policy=trainer.model;ref=trainer.ref_model;tok=trainer.processing_class
  npol=sum(p.numel() for p in policy.parameters());nref=sum(p.numel() for p in ref.parameters())
  assert npol==nref==599163904 and sum(p.numel() for p in policy.parameters() if p.requires_grad)==npol
  assert not any(p.requires_grad for p in ref.parameters())
  assert {p.dtype for p in policy.parameters()}=={torch.float32} and {p.dtype for p in ref.parameters()}=={torch.bfloat16}
  assert {str(p.device) for p in policy.parameters()}=={'cuda:0'} and {str(p.device) for p in ref.parameters()}=={'cuda:0'}
  assert all(a.data_ptr()!=b.data_ptr() for a,b in zip(policy.parameters(),ref.parameters()))
  codes=json.loads((ROOT/'outputs/v2/tokenizer/token_id_map.json').read_text())['map']
  for token,tid in codes.items():assert tok.encode(token,add_special_tokens=False)==[tid]
  sids=[l.split('\t')[0].strip() for l in Path(info).read_text().splitlines()]
  legal={tuple(tok.encode(s,add_special_tokens=False)) for s in sids};assert len(sids)==len(legal)==3686 and all(len(s)==4 for s in legal)
  row=ds[0];prompt=row['prompt'];assert prompt.endswith('### Response:\n')
  enc=tok(prompt,return_tensors='pt',add_special_tokens=False).to(trainer.accelerator.device)
  enc={k:v[:,-512:] for k,v in enc.items()}
  policy.eval()
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   a=policy(**enc);b=ref(**enc);assert torch.isfinite(a.logits).all() and torch.isfinite(b.logits).all()
  r['forwards']={'policy_finite':True,'ref_finite':True,'logits_shape':list(a.logits.shape),'prompt_tokens':enc['input_ids'].shape[-1]}
  from LogitProcessor import ConstrainedLogitsProcessor
  from transformers import LogitsProcessorList,TemperatureLogitsWarper
  lp=LogitsProcessorList([TemperatureLogitsWarper(temperature=1.),ConstrainedLogitsProcessor(trainer.prefix_allowed_tokens_fn,16,ckpt,tok.eos_token_id)])
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   generated=policy.generate(**enc,generation_config=trainer.generation_config,logits_processor=lp)
  raw=generated[:,enc['input_ids'].shape[-1]:].cpu().tolist()
  for s in raw:assert len(s)==5 and tuple(s[:4]) in legal and s[4]==tok.eos_token_id,('Invalid raw SID',s)
  r['strict_generation']={'source':'one training-stream prompt, no hit/recommendation metrics','num_beams':16,'num_return_sequences':16,'legal_count':len(raw),'raw_suffixes':raw,'suffix_length':5,'eos':tok.eos_token_id}
  r['trainer']={'class':type(trainer).__name__,'policy_parameters':npol,'policy_trainable':npol,'ref_parameters':nref,'ref_trainable':0,
   'policy_dtype':'float32','ref_dtype':'bfloat16','device':str(trainer.accelerator.device),'world_size':trainer.accelerator.num_processes,
   'attention':policy.config._attn_implementation,'flash_sdp':torch.backends.cuda.flash_sdp_enabled(),'memory_efficient_sdp':torch.backends.cuda.mem_efficient_sdp_enabled(),
   'math_sdp':torch.backends.cuda.math_sdp_enabled(),'optimizer_created':trainer.optimizer is not None,'global_step':trainer.state.global_step,
   'stream_rows':n,'common_sid_atoms_checked':len(codes),'catalog_unique_sids':len(legal),'vocab':len(tok),'peft':hasattr(policy,'peft_config'),
   'config':{k:str(getattr(cfg,k)) for k in ['per_device_train_batch_size','gradient_accumulation_steps','num_generations','num_train_epochs','bf16','optim','beta','ref_model_sync_steps','ref_model_mixup_alpha','gradient_checkpointing']},'memory':mem()}
  assert not r['trainer']['peft'] and r['trainer']['world_size']==1
 else:raise ValueError(kind)
 r['pass']=True
except BaseException as e:
 r['error']=repr(e);r['traceback']=traceback.format_exc()
finally:
 r['end']=time.time();(OUT/(kind+'.json')).write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps({'kind':kind,'pass':r['pass'],'error':r.get('error'),'wall':r['end']-r['start']}),flush=True)
sys.exit(0 if r['pass'] else 1)
