#!/usr/bin/env python3
"""Independent fresh-process validation-only strict raw load check (10 fixed rows).
Never run while the current formal training process remains active.
"""
import argparse,ast,csv,hashlib,json,os,sys,time
from pathlib import Path
from datetime import datetime
ROOT=Path('/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec');os.chdir(ROOT);sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--method',required=True,choices=['A','B','C']);ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True);ap.add_argument('--training-pid',type=int,required=True);args=ap.parse_args()
 p=Path('/proc')/str(args.training_pid)
 if p.exists() and b'v2_train_grpo' in (p/'cmdline').read_bytes():raise RuntimeError('Formal training PID still active; do not load second GPU model')
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig,LogitsProcessor
 from v2_sft_data import SID_INSTRUCTION,Tokenizer,load_group_bindings,_generate_prompt,verify_group_bindings,verify_info_txt
 from v2_generation_adapter import validate_sid_sequence
 start=time.time();ck=Path(args.checkpoint);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 tok=AutoTokenizer.from_pretrained(ck,local_files_only=True);assert len(tok)==154977
 tok.pad_token=tok.eos_token;tok.pad_token_id=tok.eos_token_id;tok.padding_side='left';eos=tok.eos_token_id;assert eos==151643
 tm=json.loads((ROOT/'outputs/v2/tokenizer/token_id_map.json').read_text())['map'];atomic=set(tm.values());assert len(tm)==3308
 assert all(tok.encode(s,add_special_tokens=False)==[int(i)] for s,i in tm.items())
 bindings=load_group_bindings(args.method,root=str(ROOT));bc=verify_group_bindings(bindings,tok,tm);ic=verify_info_txt(bindings,root=str(ROOT));assert bc['all_ok'] and ic['ok']
 byitem={str(i):j for j,i in enumerate(bindings['item_ids'])};bytokens={tuple(ts):str(i) for i,ts in zip(bindings['item_ids'],bindings['token_ids'])};assert len(bytokens)==3686
 audit=json.loads((ROOT/'outputs/v2/manifests/data_audit.json').read_text());src=ROOT/audit['deterministic']['splits']['valid']['csv_path'];mapped=ROOT/f'outputs/v2/data/{args.method}/valid.csv'
 original=list(csv.DictReader(open(src,newline='')))[:10];remapped=list(csv.DictReader(open(mapped,newline='')))[:10];assert len(original)==len(remapped)==10
 trie={}
 for ts in bindings['token_ids']:
  node=trie
  for t in ts:node=node.setdefault(int(t),{})
  node[eos]={}
 class LP(LogitsProcessor):
  def __init__(self,n):self.n=n
  def __call__(self,ids,scores):
   masked=torch.full_like(scores,float('-inf'))
   for b in range(ids.shape[0]):
    node=trie
    for t in ids[b,self.n:].tolist():
     if t not in node:raise RuntimeError('Invalid generated prefix')
     node=node[t]
    allowed=list(node)
    if not allowed:raise RuntimeError('No legal continuation before EOS')
    masked[b,allowed]=scores[b,allowed]
   return masked
 model=AutoModelForCausalLM.from_pretrained(ck,torch_dtype=torch.bfloat16,local_files_only=True).cuda().eval();tk=Tokenizer(tok)
 cfg=GenerationConfig.from_model_config(model.config);cfg.do_sample=False;cfg.num_beams=10;cfg.num_return_sequences=1;cfg.max_new_tokens=6;cfg.temperature=1.;cfg.top_k=None;cfg.top_p=None;cfg.pad_token_id=eos;cfg.eos_token_id=eos;cfg.length_penalty=0.
 results=[]
 for i,(r,m) in enumerate(zip(original,remapped)):
  assert str(r['user_id'])==str(m['user_id']) and str(r['item_id'])==str(m['item_id'])
  history=[str(x) for x in ast.literal_eval(r['history_item_id'])];assert history==[str(x) for x in ast.literal_eval(m['history_item_id'])]
  history_sid=[bindings['sids'][byitem[x]] for x in history];assert history_sid==ast.literal_eval(m['history_item_sid'])
  ui=f"The user has interacted with items {', '.join(history_sid)} in chronological order. Can you predict the next possible item that the user may expect?"
  ids=tk.encode(SID_INSTRUCTION,bos=True,eos=False)+tk.encode(_generate_prompt(ui),bos=False,eos=False);ids=ids[-512:]
  inputs=torch.tensor([ids],device='cuda');mask=torch.ones_like(inputs)
  with torch.no_grad():g=model.generate(inputs,attention_mask=mask,generation_config=cfg,logits_processor=[LP(len(ids))])
  raw=g[0,len(ids):].tolist();legal,reason=validate_sid_sequence(raw,eos,atomic)
  member=len(raw)==5 and raw[-1]==eos and tuple(raw[:4]) in bytokens
  target=str(r['item_id']);target_tokens=bindings['token_ids'][byitem[target]];pred=bytokens.get(tuple(raw[:4])) if member else None
  results.append({'row_id':f'valid:{i}','target_item_id_from_original_csv':target,'target_token_ids_from_frozen_index':target_tokens,'raw':raw,'strict_legal':bool(legal) and member,'catalog_member':member,'validation_reason':str(reason),'pred_item_id':pred,'hit':pred==target,'beam_width':10,'returned_candidates':int(g.shape[0]),'prompt_tokens':len(ids)})
 report={'collected_at':datetime.now().astimezone().isoformat(),'method':args.method,'checkpoint':str(ck),'validation_only':True,'source_valid_csv':str(src),'source_valid_sha256':sha(src),'mapped_valid_sha256':sha(mapped),'model_sha256':sha(ck/'model.safetensors'),'tokenizer_sha256':sha(ck/'tokenizer.json'),'common_token_id_map_sha256':sha(ROOT/'outputs/v2/tokenizer/token_id_map.json'),'n_prompts':10,'strict_legal':sum(x['strict_legal'] for x in results),'hits_vs_independent_original_targets':sum(x['hit'] for x in results),'catalog_binding_check':bc,'info_check':ic,'atomic_tokens_checked':len(tm),'beam_width':10,'num_return_sequences':1,'generation':{'do_sample':False,'temperature':1.,'top_k':None,'top_p':None,'max_new_tokens':6,'length_penalty':0.},'model_parameters':sum(p.numel() for p in model.parameters()),'resources':{'wall_seconds':time.time()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'scope':'this independent load-check process only; not formal training'},'results':results}
 rp=out/'load_check.json';rp.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({'path':str(rp),'sha256':sha(rp),'strict_legal':report['strict_legal'],'hits':report['hits_vs_independent_original_targets'],'n':10,'resources':report['resources']},ensure_ascii=False))
 assert report['strict_legal']==10
if __name__=='__main__':main()
