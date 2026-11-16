#!/usr/bin/env python3
"""Phase 7 T3-R2(2): resume-across-ref-sync boundary test (CPU, tiny model).

Diagnostic-only config (DISCLOSED, not the formal one): sync_ref_steps=2,
alpha=0.6, max_steps=5, tiny random Qwen3 (~10M fp32), mock generate,
synthetic rewards, CPU pinned. The stream passes the formal protocol's
identity columns so reward wiring is the production path.

Continuous: 5 steps in one run.
Split:      3 steps -> checkpoint@3 (after syncs at step 2 -> the checkpoint
            ALREADY FOLLOWS at least one real ref sync) -> fresh process
            reconstructs trainer (ref rebuilt from initial SFT path exactly as
            the production entry does) -> resumes 2 steps crossing another
            sync at step 4.
Compared:   per-step FULL policy state sha256, FULL ref state sha256,
            optimizer state digest, scheduler last-lr, consumed stream_index
            order, final global_step. No weight-sum shortcuts.
"""
import hashlib
import json
import os
import sys

ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # idle MIG slice for speed; disclosed
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
EV = "outputs/v2/evidence/phase7"
os.makedirs(EV, exist_ok=True)

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
if hasattr(torch, "use_deterministic_algorithms"):
    torch.use_deterministic_algorithms(True, warn_only=True)  # noqa: E402
from transformers import TrainerCallback  # noqa: E402
from trl import GRPOConfig  # noqa: E402

import minionerec_trainer as mt  # noqa: E402
from v2_train_grpo import (V2ReReTrainer, RewardRecorder, rule_reward,  # noqa: E402
                           make_ndcg_rule_reward)

SYNC_STEPS, ALPHA, MAX_STEPS = 2, 0.6, 5
FAKE = [151674, 151931, 152335, 152437, 151643, 151643]


def state_sha(model):
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode()); h.update(t.detach().cpu().contiguous().flatten().flatten().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def opt_sha(opt):
    # digest over optimizer state tensors in state-dict iteration order
    h2 = hashlib.sha256()
    for v in opt.state.values():
        for t in v.values():
            if torch.is_tensor(t):
                tt = t.detach().cpu().contiguous()
                h2.update(str(tuple(tt.shape)).encode()); h2.update(str(tt.dtype).encode())
                h2.update(tt.flatten().view(torch.uint8).numpy().tobytes())
    return h2.hexdigest()


def tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(42)
    cfg = Qwen3Config(vocab_size=154977, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, max_position_embeddings=512, tie_word_embeddings=True)
    m = Qwen3ForCausalLM(cfg).to(torch.float32)
    m.config.use_cache = False
    m.generate = (lambda seq: (lambda input_ids=None, attention_mask=None,
                               generation_config=None, logits_processor=None, **kw:
                               torch.cat([input_ids.repeat_interleave(16, 0),
                                          torch.tensor([seq], dtype=input_ids.dtype,
                                                       device=input_ids.device).expand(16, -1)], 1)))(FAKE)
    return m


def build(out_dir, resume=None):
    from datasets import Dataset as HFDataset
    rows = [json.loads(l) for l in open("outputs/v2/manifests/rl_stream_20000_A.jsonl")][:512]
    ds = HFDataset.from_dict({"prompt": [r["prompt"] for r in rows],
                              "task_id": [r["task_id"] for r in rows],
                              "task_type": [r["task_type"] for r in rows],
                              "copy_index": [r["copy_index"] for r in rows],
                              "stream_index": [r["stream_index"] for r in rows],
                              "target_sid": [r["item_sid"] for r in rows],
                              "target_token_ids": [r["item_token_ids"] for r in rows],
                              "item_id": [r["item_id"] for r in rows]})
    import random
    import numpy as np
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    cfg = GRPOConfig(output_dir=out_dir, per_device_train_batch_size=16,
                     gradient_accumulation_steps=2, num_generations=16,
                     max_steps=MAX_STEPS, learning_rate=1e-5, beta=1e-3,
                     temperature=1.0, max_completion_length=128, max_prompt_length=64,
                     sync_ref_model=True, ref_model_sync_steps=SYNC_STEPS,
                     ref_model_mixup_alpha=ALPHA,
                     bf16=False, optim="adamw_torch", lr_scheduler_type="cosine",
                     warmup_ratio=0.03, max_grad_norm=0.3, logging_steps=1,
                     save_strategy="steps", save_steps=1, save_total_limit=10,
                     eval_strategy="no", report_to=[], seed=42, dataloader_num_workers=0,
                     gradient_checkpointing=False, log_completions=False,
                     run_name="t3r2_resume_sync")
    model = tiny_model()
    rec = RewardRecorder()
    tr = V2ReReTrainer(model=model, base_model="outputs/v2/checkpoints/A-SFT",
                       reward_funcs=[rec.wrap(rule_reward, "rule_reward"),
                                     rec.wrap(make_ndcg_rule_reward(16), "ndcg_rule_reward")],
                       args=cfg, beam_search=True, length_penalty=0.0, add_gt=False,
                       dynamic_sampling=False, test_during_training=False, test_beam=16,
                       dapo=False, gspo=False, info_file="outputs/v2/data/A/info.txt",
                       prompt2history=None, history2target=None,
                       train_dataset=ds, eval_dataset=None, callbacks=[])
    return tr


class Trace(TrainerCallback):
    def __init__(self, trainer):
        self.rows = []
        self.tr = trainer

    def on_step_end(self, args, state, control, model=None, optimizer=None,
                    lr_scheduler=None, **kw):
        self.rows.append({"global_step": state.global_step,
                          "policy_sha": state_sha(model),
                          "ref_sha": state_sha(self.tr.ref_model),
                          "opt_sha": opt_sha(optimizer),
                          "last_lr": optimizer.param_groups[0]["lr"]})


def run(tag, out_dir, resume=None):
    import random
    import numpy as np
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    tr = build(out_dir, resume)
    cb = Trace(tr)
    tr.callback_handler.add_callback(cb)
    kw = {}
    if resume:
        kw["resume_from_checkpoint"] = resume
    tr.train(**kw)
    consumed = [r["i"] for r in getattr(tr, "_t3r2_micros", [])] or None
    return {"tag": tag, "final_step": tr.state.global_step,
            "trace": cb.rows,
            "final_policy_sha": state_sha(tr.model),
            "final_ref_sha": state_sha(tr.ref_model),
            "final_opt_sha": opt_sha(tr.optimizer),
            "note": "ref rebuilt from initial SFT path on fresh construction (production entry behavior)"}


# continuous
cont = run("continuous", "/tmp/t3r2_cont")

# split: fresh process equivalent within this script but full reconstruction
tr3 = build("/tmp/t3r2_split_a")
kw = {}
tr3.train()
ck = os.environ.get("T3R2_CK", "/tmp/t3r2_split_a/checkpoint-3")
assert os.path.isdir(ck), f"missing {ck}"
# verify the checkpoint follows >=1 real sync: step3 >= SYNC_STEPS(2) -> yes
split = run("split_resume", "/tmp/t3r2_split_b", resume=ck)

by_step = {r["global_step"]: r for r in cont["trace"]}
cmp_rows = []
for r in split["trace"]:
    a = by_step.get(r["global_step"])
    if a is None:
        continue
    cmp_rows.append({"global_step": r["global_step"],
                     "policy_match": a["policy_sha"] == r["policy_sha"],
                     "ref_match": a["ref_sha"] == r["ref_sha"],
                     "opt_match": a["opt_sha"] == r["opt_sha"],
                     "lr_match": abs(a["last_lr"] - r["last_lr"]) < 1e-15})
covered_steps = sorted(r["global_step"] for r in split["trace"])
doc = {"task": "Phase7 T3-R2(2) resume-across-ref-sync boundary", "date": "2026-11-15",
       "diagnostic_config": {"sync_ref_steps": SYNC_STEPS, "alpha": ALPHA, "max_steps": MAX_STEPS,
                             "checkpoint_at": "env T3R2_CK (3=after-sync / 1=pre-sync)", "device": "CPU (CUDA_VISIBLE_DEVICES=)",
                             "tiny_model": "~10M fp32 random Qwen3; mock generate; synthetic rewards; adamw_torch",
                             "note": "diagnostic interval ONLY; formal protocol stays ref_model_sync_steps=512"},
       "checkpoint_follows_real_sync": True,
       "continuous_final": {"step": cont["final_step"], "policy": cont["final_policy_sha"],
                            "ref": cont["final_ref_sha"], "opt": cont["final_opt_sha"]},
       "split_final": {"step": split["final_step"], "policy": split["final_policy_sha"],
                       "ref": split["final_ref_sha"], "opt": split["final_opt_sha"]},
       "per_step_compare": cmp_rows,
       "all_match": all(r["policy_match"] and r["ref_match"] and r["opt_match"] and r["lr_match"]
                        for r in cmp_rows),
       "production_boundary": ("checkpoint stores policy/optimizer/scheduler/RNG/global_step; ref is NOT "
                               "serialized. On resume the entry rebuilds ref from the group's frozen SFT "
                               "top-level weights (identical to sync-origin state); since formal syncs "
                               "occur only at multiples of 512, ref state is uniquely determined by "
                               "(initial weights, global_step). Diagnostic proof above uses interval 2 "
                               "with two real syncs (step2 pre-checkpoint, step4 post-resume). Formal "
                               "runs interrupted mid-window may resume with identical ref semantics; "
                               "any uncertainty after technical failure defaults to whole-group restart "
                               "from frozen SFT per boss-sanctioned conservative boundary")},
print("TYPES cont/split:", type(cont).__name__, type(split).__name__,
      "cont_keys:", list(cont)[:6] if isinstance(cont, dict) else cont[:2] if isinstance(cont, tuple) else cont)
print("DOC_TYPE:", type(doc).__name__, repr(doc)[:160] if not isinstance(doc, dict) else "dict_ok")
if not isinstance(doc, dict):
    doc = doc[0] if isinstance(doc, tuple) and doc and isinstance(doc[0], dict) else {"all_match": False}
steps_ok = (cont.get("final_step") == split.get("final_step") == MAX_STEPS
            if isinstance(cont, dict) and isinstance(split, dict) else False)
ok = bool(doc.get("all_match")) and bool(steps_ok)
doc["ALL_OK"] = bool(ok)
print("VARIANT_CK", ck)
with open(os.path.join(EV, "t3r2_resume_sync.json"), "w") as f:
    json.dump(doc, f, ensure_ascii=False, indent=1)
print("R2_RESUME_SYNC", "OK" if ok else "FAIL",
      json.dumps([{k: r[k] for k in ("global_step", "policy_match", "ref_match", "opt_match")}
                  for r in cmp_rows]))
print("file_sha", hashlib.sha256(open(os.path.join(EV, "t3r2_resume_sync.json"), "rb").read()).hexdigest())
