#!/usr/bin/env python3
"""Phase7 T4-R2(1b): longest-real-prompt full forward/backward stress (G16).

Diagnostic-only knobs (disclosed): gradient_accumulation_steps=1 (single
micro -> one optimizer step; formal stays 32), max_steps=1. Everything else
real: real 599M fp32 policy + full bf16 ref + paged_adamw_32bit + real
constrained beam rollout on the LONGEST prompt in the A stream (by tokenizer
length), one full backward. Records prompt token length, rollout/fwb peak
allocated/reserved, wall, exit."""
import argparse, json, os, sys, time
ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
import torch
from transformers import TrainerCallback
from trl import GRPOConfig
from v2_train_grpo import V2ReReTrainer, RewardRecorder, rule_reward, make_ndcg_rule_reward

ap = argparse.ArgumentParser()
ap.add_argument("--method", default="A")
a = ap.parse_args()
BASE = "outputs/v2/debug/phase7/grpo_smoke/attempt_006/reports"
os.makedirs(BASE, exist_ok=True)

rows = [json.loads(l) for l in open(f"outputs/v2/manifests/rl_stream_20000_{a.method}.jsonl")]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(f"outputs/v2/checkpoints/{a.method}-SFT", local_files_only=True)
lens = []
for r in rows[:2048]:
    lens.append((len(tok(r["prompt"], add_special_tokens=False).input_ids), r["stream_index"]))
lens.sort(reverse=True)
tok_len, sidx = lens[0]
sel = [r for r in rows if r["stream_index"] == sidx][0]
print("LONGEST_PROMPT tokens=", tok_len, "chars=", len(sel["prompt"]), "stream_index=", sidx)

from datasets import Dataset as HFD
ds = HFD.from_dict({'prompt': [sel['prompt']]*16, 'task_id': [sel['task_id']]*16, 'task_type': [sel['task_type']]*16, 'copy_index': [sel['copy_index']]*16, 'stream_index': [sel['stream_index']]*16, 'target_sid': [sel['item_sid']]*16, 'target_token_ids': [sel['item_token_ids']]*16, 'item_id': [sel['item_id']]*16})
import random
import numpy as np
random.seed(42); np.random.seed(42); torch.manual_seed(42)
cfg = GRPOConfig(output_dir="/tmp/t4r2_stress_out", per_device_train_batch_size=16,
    gradient_accumulation_steps=1, num_generations=16, num_train_epochs=1, max_steps=1,
    learning_rate=1e-5, beta=1e-3, temperature=1.0, max_completion_length=128, max_prompt_length=512,
    sync_ref_model=True, ref_model_sync_steps=512, ref_model_mixup_alpha=0.6,
    bf16=True, optim="paged_adamw_32bit", lr_scheduler_type="cosine", warmup_ratio=0.03,
    max_grad_norm=0.3, logging_steps=1, save_strategy="no", eval_strategy="no", report_to=[],
    seed=42, dataloader_num_workers=0, gradient_checkpointing=True, log_completions=False,
    run_name="t4r2-longest-stress")
tr = V2ReReTrainer(model=f"outputs/v2/checkpoints/{a.method}-SFT",
    base_model=f"outputs/v2/checkpoints/{a.method}-SFT",
    reward_funcs=[rule_reward, make_ndcg_rule_reward(16)],
    args=cfg, beam_search=True, length_penalty=0.0, add_gt=False, dynamic_sampling=False,
    test_during_training=False, test_beam=16, dapo=False, gspo=False,
    info_file=f"outputs/v2/data/{a.method}/info.txt", prompt2history=None, history2target=None,
    train_dataset=ds, eval_dataset=None, callbacks=[])

peaks = {"pre_train_alloc": torch.cuda.memory_allocated(), "pre_train_reserved": torch.cuda.memory_reserved()}
torch.cuda.reset_peak_memory_stats()
t0 = time.time(); code = "ok"
try:
    tr.train()
except torch.cuda.OutOfMemoryError as e:
    code = f"OOM {str(e)[:200]}"
wall = time.time() - t0
doc = {"task": "Phase7 T4-R2 longest-real-prompt stress", "method": a.method,
       "longest_prompt_tokens": tok_len, "longest_prompt_chars": len(sel["prompt"]),
       "stream_index": sidx, "config_disclosure": {"accum": "1 (diagnostic; formal 32)",
         "steps": 1, "everything_else": "formal wiring"},
       "pre_peak": peaks, "post_peak_alloc": torch.cuda.max_memory_allocated(),
       "post_peak_reserved": torch.cuda.max_memory_reserved(), "wall_seconds": round(wall, 1),
       "exit": code}
json.dump(doc, open(f"{BASE}/t4r2_longest_stress.json", "w"), ensure_ascii=False, indent=1)
print("STRESS", json.dumps({k: doc[k] for k in ("longest_prompt_tokens", "post_peak_alloc", "post_peak_reserved", "wall_seconds", "exit")}))
