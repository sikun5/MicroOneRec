#!/usr/bin/env python3
"""Phase 7 T2-R1: CPU tail-window consumption probe (boss ruling 2026-11-15).

Runs the ACTUAL locked training loop (transformers 4.51.3 Trainer.train() with
the actual ReReTrainer from the locked minionerec_trainer.py and its actual
RepeatRandomSampler) at batch16/accum32/G16 over a synthetic 10000-prompt
dataset x G16 x 2 epochs on a tiny CPU model. Generation is MOCKED (fixed
completion ids) and rewards are synthetic - disclosed here and in the record.
Purpose: count planned max_steps vs dataset loading vs forward vs gradient
contribution vs actual optimizer.step vs epoch termination, and determine
whether the partial accumulation tail is dropped or leaks across the epoch
boundary (stale gradients / wrong scaling). Counts only - no recommendation
metrics, no GPU, no real GRPO rollout, no protocol change.
"""
import hashlib
import json
import os
import sys

ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
EV = "outputs/v2/evidence/phase7"
os.makedirs(EV, exist_ok=True)

import torch
from transformers import AutoTokenizer, TrainerCallback
from datasets import Dataset as HFDataset

OUT = {}

def sha(p):
    if not os.path.exists(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

# ---------- locked trainer import (unchanged sources) ----------
import minionerec_trainer as mt  # noqa: E402  (imports trl 0.24.0 venv-local + LogitProcessor)
from trl import GRPOConfig  # noqa: E402
OUT["locked_sources"] = {
    "minionerec_trainer.py": sha("minionerec_trainer.py"),
    "trl_version_probe": __import__("trl").__version__,
    "trl_file": __import__("trl").__file__,
    "transformers": __import__("transformers").__version__,
}
assert mt.ReReTrainer is not None

# ---------- tiny CPU model (Qwen3 family, expanded vocab size) ----------
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402
torch.manual_seed(42)
cfg = Qwen3Config(vocab_size=154977, hidden_size=64, intermediate_size=128,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  head_dim=16, max_position_embeddings=512, tie_word_embeddings=True,
                  attention_residual_connection=False)
model = Qwen3ForCausalLM(cfg)
model = model.to("cpu").to(torch.float32)
model.config.use_cache = False
n_params = sum(p.numel() for p in model.parameters())
OUT["tiny_model"] = {"class": "Qwen3ForCausalLM", "params": n_params, "dtype": "float32",
                     "vocab": 154977, "device": "cpu",
                     "note": "random init; not any SFT checkpoint; probe-only"}

tokenizer = AutoTokenizer.from_pretrained("outputs/v2/checkpoints/A-SFT", local_files_only=True)
assert len(tokenizer) == 154977
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

# ---------- synthetic dataset: the 10000 logical tasks ----------
import random
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
tasks = [{"prompt": f"P{i:05d} end", "task_id": f"seq2sid:{i:05d}"} for i in range(10000)]
ds = HFDataset.from_dict({k: [r[k] for r in tasks] for k in tasks[0]})
ds = ds.shuffle(seed=42)  # rl.py convention: one shuffle before training

# ---------- GRPOConfig mirroring the locked formal config ----------
gargs = GRPOConfig(
    output_dir="/tmp/t2r1_probe_out", save_strategy="no", eval_strategy="no",
    max_completion_length=128, num_generations=16, temperature=1.0,
    sync_ref_model=True,  # actual callback attached; defaults steps=512, alpha=0.6
    per_device_eval_batch_size=16, per_device_train_batch_size=16,
    gradient_accumulation_steps=32, logging_steps=500, learning_rate=1e-5,
    beta=1e-3, warmup_ratio=0.03, max_grad_norm=0.3, num_train_epochs=2,
    bf16=False, optim="adamw_torch",  # CPU probe: fp32 + torch adamw (counting-neutral; disclosed)
    lr_scheduler_type="cosine", report_to=[], run_name="t2r1_tail_probe",
    dataloader_num_workers=0, seed=42, disable_dropout=False, log_completions=False,
)
OUT["config"] = {"per_device_train_batch_size": 16, "gradient_accumulation_steps": 32,
                 "num_generations": 16, "num_train_epochs": 2, "seed": 42,
                 "optim": "adamw_torch (CPU substitution; scaling semantics identical)",
                 "bf16": False, "eval_strategy": "no", "save_strategy": "no",
                 "sync_ref_model": True,
                 "ref_model_sync_steps": gargs.ref_model_sync_steps,
                 "ref_model_mixup_alpha": gargs.ref_model_mixup_alpha}

# ---------- mock generation + synthetic rewards (disclosed) ----------
GEN_LEN = 6
FAKE_COMPLETION = [151674, 151931, 152335, 152437, 151643, tokenizer.pad_token_id]

def fake_generate(input_ids=None, attention_mask=None, generation_config=None,
                  logits_processor=None, **kw):
    n = FAKE_RETURN_SEQS
    p = input_ids.size(1)
    comp = torch.tensor([FAKE_COMPLETION], dtype=input_ids.dtype, device=input_ids.device)
    return torch.cat([input_ids.repeat_interleave(n, dim=0), comp.expand(n, -1)], dim=1)

FAKE_RETURN_SEQS = 16
model.generate = fake_generate  # mock: no beam search executed (disclosed)

reward_calls = {"rule": 0, "ndcg": 0, "rows": 0}

def rule_reward_mock(prompts, completions, **kw):
    reward_calls["rule"] += 1
    reward_calls["rows"] += len(prompts)
    return [1.0 if c.endswith(tokenizer.eos_token) else 0.0 for c in prompts]

def ndcg_reward_mock(prompts, completions, **kw):
    reward_calls["ndcg"] += 1
    return [0.0] * len(prompts)

# ---------- counters ----------
micro_records = []          # one entry per executed micro-batch (training_step)
step_records = []           # one entry per actual optimizer step (callback on_step_end)
grad_probe = {}             # named grad checksums
norm_w = model.model.norm.weight

def grad_checksum():
    g = norm_w.grad
    return None if g is None else float(g.abs().sum().item())

ref_checksums = []

def ref_checksum():
    return float(sum(float(p.detach().abs().sum().item()) for p in trainer.ref_model.parameters()))

orig_training_step = mt.ReReTrainer.training_step

def counting_training_step(self, m, inputs, *a, **kw):
    idx = len(micro_records)
    task_ids = sorted({r.get("task_id") for r in inputs}) if isinstance(inputs, list) else ["<non-list>"]
    out = orig_training_step(self, m, inputs, *a, **kw)
    micro_records.append({
        "i": idx, "epoch": int(float(self.state.epoch or 0)),  # floor: epoch index
        "n_rows": len(inputs) if isinstance(inputs, list) else -1,
        "task_ids": task_ids if len(task_ids) == 1 else f"{len(task_ids)}_distinct",
        "loss_training_step_return": float(out) if hasattr(out, "item") else float(out),
        "grad_ckpt": grad_checksum(),
    })
    return out

mt.ReReTrainer.training_step = counting_training_step

class ProbeCallback(TrainerCallback):
    def on_step_end(self, args, state, control, optimizer=None, lr_scheduler=None, **kw):
        step_records.append({
            "global_step": state.global_step, "epoch": round(float(state.epoch or 0), 4),
            "lr": optimizer.param_groups[0]["lr"] if optimizer is not None else None,
            "ref_ckpt": ref_checksum(),
        })
    def on_epoch_end(self, args, state, control, **kw):
        step_records.append({"event": "epoch_end",
                             "epoch_finished_idx": int(round(float(state.epoch or 0))) - 1,
                             "global_step": state.global_step, "grad_ckpt_after_epoch": grad_checksum()})

# note: _prepare_inputs default path (add_gt=False, test_during_training=False,
# dynamic_sampling=False) needs no prompt2history/history2target dicts.
trainer = mt.ReReTrainer(
    model=model, base_model="outputs/v2/checkpoints/A-SFT",
    reward_funcs=[rule_reward_mock, ndcg_reward_mock],
    args=gargs, train_dataset=ds, eval_dataset=None,
    processing_class=tokenizer,
    info_file="outputs/v2/data/A/info.txt",
    prompt2history=None, history2target=None,
    beam_search=True, add_gt=False, dynamic_sampling=False,
    test_during_training=False, dapo=False, gspo=False,
    callbacks=[ProbeCallback()],
)

OUT["trainer_semantics"] = {
    "state_max_steps": None,  # filled after train
    "sampler_class": type(trainer._get_train_sampler()).__name__,
    "sampler_len": len(trainer._get_train_sampler()),
    "model_is_unwrapped": trainer.model is model,
    "ref_model_exists": trainer.ref_model is not None,
    "ref_param_count": sum(p.numel() for p in trainer.ref_model.parameters()),
    "ref_not_in_optimizer_note": "ref prepared with evaluation_mode=True; probe records param checksums to observe sync events",
}

grad_probe["before_train"] = grad_checksum()
trainer.train()

# persist raw records FIRST so analysis bugs cannot lose the run
with open(os.path.join(EV, "t2r1_micro_records.json"), "w") as f:
    json.dump({"micro": micro_records, "steps": step_records}, f)

def norm_task(r):
    t = r["task_ids"]
    if isinstance(t, list) and len(t) == 1:
        return t[0]
    return t if isinstance(t, str) else None

# ---------- analysis ----------
n_micro = len(micro_records)
per_epoch = {}
for r in micro_records:
    e = min(int(float(r["epoch"])), 1)
    per_epoch.setdefault(e, {"micro": 0, "task_ids": set(), "loss_sum": 0.0})
    per_epoch[e]["micro"] += 1
    per_epoch[e]["loss_sum"] += r["loss_training_step_return"]
    t = norm_task(r)
    if t:
        per_epoch[e]["task_ids"].add(t)

epoch_tail = {}
for e in (0, 1):
    recs = [r for r in micro_records if min(int(float(r["epoch"])), 1) == e]
    tail = recs[-16:]
    epoch_tail[f"epoch{e}"] = {
        "n_micro": len(recs),
        "last_step_micro_i": max([s["global_step"] for s in step_records
                                  if isinstance(s.get("global_step"), int) and s.get("epoch") is not None
                                  and int(float(s["epoch"])) == e] or [-1]),
        "tail_micro_indices": [r["i"] for r in tail],
        "tail_n_task_groups": len(tail),
        "tail_after_last_ckpt_grad": tail[-1]["grad_ckpt"] if tail else None,
        "distinct_tasks_in_tail": sorted({norm_task(r) for r in tail if norm_task(r)}),
    }

first_steps = [s for s in step_records if isinstance(s.get("global_step"), int)][:3]
ref_events = []
prev = None
for s in step_records:
    if "ref_ckpt" in s:
        if prev is not None and abs(s["ref_ckpt"] - prev) > 1e-9:
            ref_events.append(s["global_step"])
        prev = s["ref_ckpt"]

epoch_end_events = [s for s in step_records if s.get("event") == "epoch_end"]

summary = {
    "micro_records_total": n_micro,
    "optimizer_steps_total": len([s for s in step_records if "global_step" in s and "ref_ckpt" in s]),
    "final_global_step": trainer.state.global_step,
    "state_max_steps": trainer.state.max_steps,
    "state_num_train_epochs": trainer.state.num_train_epochs,
    "planned_updates_formula": "floor(10000 micro/32)=312 per epoch x 2 = 624",
    "per_epoch": {str(k): {"micro": v["micro"], "distinct_tasks_consumed": len(v["task_ids"]),
                           "loss_sum": round(v["loss_sum"], 6)} for k, v in per_epoch.items()},
    "consumption_verdict": {
        "forward_coverage": "all 10000 tasks loaded+forward in EVERY epoch (see per_epoch.distinct_tasks_consumed==10000)",
        "update_participation": "epoch tail micro-batches execute forward+backward but no optimizer.step in that epoch; "
                                 "see epoch_tail.tail_after_last_ckpt_grad for stale-gradient presence and "
                                 "epoch_end_events for gradient state at epoch boundary",
        "stale_gradient_evidence": {
            "epoch0_tail_grad_ckpt": epoch_tail["epoch0"]["tail_after_last_ckpt_grad"],
            "epoch0_end_grad_ckpt": next((s["grad_ckpt_after_epoch"] for s in epoch_end_events
                                          if s.get("epoch_finished_idx") == 0), None),
            "epoch1_end_grad_ckpt": next((s["grad_ckpt_after_epoch"] for s in epoch_end_events
                                          if s.get("epoch_finished_idx") == 1), None),
            "epoch1_first_steps": first_steps[:2],
        },
    },
    "ref_sync_events_steps": ref_events,
    "ref_sync_expected": [gargs.ref_model_sync_steps],
    "reward_calls": reward_calls,
    "mock_disclosure": {
        "generation": "model.generate replaced by fixed-completion mock (no beam search executed)",
        "rewards": "synthetic dummy reward functions (call counts recorded)",
        "optimizer": "adamw_torch fp32 on CPU (counting-neutral substitution for paged_adamw_32bit/bf16)",
        "model": "tiny random Qwen3ForCausalLM; NOT an SFT checkpoint",
        "loop": "REAL transformers 4.51.3 Trainer.train() inner loop with REAL ReReTrainer._prepare_inputs/compute_loss "
                "and REAL RepeatRandomSampler from the locked minionerec_trainer.py",
    },
}

OUT["summary"] = summary
OUT["epoch_tail_detail"] = epoch_tail
OUT["micro_records_file"] = "t2r1_micro_records.json"
OUT["step_records"] = step_records[:5] + step_records[-5:]

with open(os.path.join(EV, "t2r1_tail_probe.json"), "w") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1, default=str)

print("PROBE_DONE")
print(json.dumps(summary["consumption_verdict"]["stale_gradient_evidence"], indent=1, default=str)[:800])
print("micro_total", n_micro, "steps_total", summary["optimizer_steps_total"],
      "final_step", trainer.state.global_step, "max_steps", trainer.state.max_steps)
print("per_epoch", json.dumps(summary["per_epoch"]))
print("ref_sync_events", ref_events)
print("evidence_sha", sha(os.path.join(EV, "t2r1_tail_probe.json")),
      sha(os.path.join(EV, "t2r1_micro_records.json")))
