#!/usr/bin/env python3
"""Phase 7 v2 GRPO entry (boss-released T3).

Adapts the ORIGINAL rl.py entry around the LOCKED minionerec_trainer.ReReTrainer.
Adaptation list (exhaustive; loss math, reward math, generation strategy,
optimizer are UNCHANGED):

A1 Data: consumes ONE global 10k manifest via the materialized two-logical-epoch
   stream (scripts/v2_grpo_stream.py; T2-R1 ruling: exact 625 full windows)
   instead of the original three dataset classes (ConcatDataset + sample=10000
   + one shuffle). Dataset order = stream file order; num_train_epochs=1
   (the stream itself spans 2 logical epochs; rows carry copy_index).
A2 Sampler: RepeatOrderedSampler (deterministic stream order x G) replaces
   RepeatRandomSampler -> no train-time RNG in sampling, 320000 rows = 20000
   micro-batches = 625 exact optimizer windows. Verified in T3-R1 (2).
A3 Rewards: SAME two functions (exact-match rule + ndcg-style group ranking,
   identical formulas incl. a_j=1/log2(j+2) normalization) but targets are
   recovered from per-row identity columns (target_sid carried through the
   batch) instead of the original prompt2history/history2target dicts, which
   silently overwrite the 296+18 same-history-different-target groups found in
   T2. task_id is NOT shown to the model (never enters prompt text).
A4 Removed the original rl.py device-probe LLM copy (rl.py loaded a second
   full LLM with device_map="auto" only to read .device); device comes from
   the trainer policy.
A5 test_during_training=False and eval_strategy="no": closes the original's
   extra beam-20 top-k generation during training (locked protocol: no extra
   Top-K generation) and validation-as-selection; validation remains dev-only
   diagnostics outside the formal run.
A6 use_vllm locked False (trl 0.24 lacks the per-device vllm config fields;
   original path unused).
A7 add_gt/dynamic_sampling/dapo/gspo locked False (original optional features,
   off in the frozen protocol).
A8 report_to=[] with WANDB disabled: original used offline wandb; metrics are
   persisted here as JSONL diagnostics instead (same numbers, different sink).
A9 Signature columns widened to carry identity/target columns through
   remove_unused_columns (wiring only).
Everything else (constrained beam search over the full-catalog trie from
group info.txt, EOS/pad handling, completion masking, KL k3 estimator, ref
model construction + SyncRefModelCallback, paged_adamw_32bit, cosine schedule,
warmup ratio 0.03, max_grad_norm 0.3, bf16, seed) is the locked original.
"""
import argparse
import json
import hashlib
import math
import os
import sys
import time

ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
sys.path.insert(0, ROOT)

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from datasets import Dataset as HFDataset  # noqa: E402
from transformers import TrainerCallback  # noqa: E402
from trl import GRPOConfig  # noqa: E402

from minionerec_trainer import ReReTrainer  # noqa: E402  (locked trainer)


class RepeatOrderedSampler(torch.utils.data.Sampler):
    """Yields dataset positions in fixed file order, each repeated G times
    consecutively (positions 0,0,...,0 (G), 1,1,...,1 (G), ...). Deterministic;
    consumes no RNG. Grouping invariant: per_device_train_batch_size=G rows of
    one micro-batch are exactly one prompt group."""

    def __init__(self, data_source, num_generations):
        self.data_source = data_source
        self.G = num_generations

    def __iter__(self):
        n = len(self.data_source)
        for i in range(n):
            for _ in range(self.G):
                yield i

    def __len__(self):
        return len(self.data_source) * self.G


def rule_reward(prompts, completions, target_sid=None, **kw):
    """Original rl.py rule_reward math verbatim; target from identity column."""
    targets = [t.strip("\n\" ") for t in target_sid]
    return [1.0 if c.strip("\n\" ") == t else 0.0 for c, t in zip(completions, targets)]


def make_ndcg_rule_reward(num_generations):
    ndcg_rewards = [-1.0 / math.log2(i + 2) for i in range(num_generations)]
    ndcg_rewards = [-e / sum(ndcg_rewards) for e in ndcg_rewards]

    def ndcg_rule_reward(prompts, completions, target_sid=None, **kw):
        """Original rl.py ndcg_rule_reward math verbatim (group = consecutive G
        completions in arrival order = one prompt group under A2)."""
        targets = [t.strip("\n\"") for t in target_sid]
        rewards, flag, lis = [], False, []
        for i, completion in enumerate(completions):
            if completion.strip("\n\"") == targets[i]:
                flag = True
                lis.append(0.0)
            else:
                lis.append(ndcg_rewards[i % num_generations])
            if (i + 1) % num_generations == 0:
                rewards.extend(lis if flag else [0.0] * num_generations)
                flag, lis = False, []
        return rewards
    return ndcg_rule_reward


class V2ReReTrainer(ReReTrainer):
    """Only wiring overrides; no training-loop changes.

    strict_catalog_trie (default True): rebuilds the decoding trie after init
    WITHOUT the inherited trailing-newline transition, enforcing the frozen
    Phase5 raw contract [4 SID atoms, EOS]. Header-key alignment (depth-3 keys
    = response-header tokens, matching ConstrainedLogitsProcessor.count==0)
    mirrors the locked trainer's build exactly, minus the "\n" hop.
    """

    strict_catalog_trie = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.strict_catalog_trie:
            self._rebuild_strict_catalog_trie()
        resume_ckpt = kwargs.pop("resume_from_checkpoint", os.environ.get("T5_RESUME_CKPT", ""))
        by_value_restore = bool(resume_ckpt) and os.environ.get("V2_BY_VALUE_RESTORE", "") == "1"
        self._v2_resume_ckpt = resume_ckpt
        self._v2_by_value_restore = by_value_restore
        ref_dtype = kwargs.pop("ref_dtype", os.environ.get("V2_REF_DTYPE", "bf16"))
        self._v2_ref_dtype = ref_dtype
        if ref_dtype == "bf16" and self.ref_model is not None and getattr(
                self.args, "sync_ref_model", False):
            self.ref_model = self.ref_model.to(torch.bfloat16)
            self._install_bf16_sync_callback()

    def _rebuild_strict_catalog_trie(self):
        tok = self.processing_class
        header = tok("### Response:\n", add_special_tokens=False).input_ids
        assert len(header) == 3, f"response header tokens != 3: {header}"
        eos = tok.eos_token_id
        self.hash_dict = {}
        for line in open(self.info_file):
            sid = line.split("\t")[0].strip()
            ids = tok(sid, add_special_tokens=False).input_ids
            if len(ids) != 4:
                continue  # malformed info line (none in v2 info files)
            seq = list(header) + ids + [eos]
            for i in range(3, len(seq)):
                key = "-".join(map(str, seq[:3] if i == 3 else seq[3:i]))
                self.hash_dict.setdefault(key, set()).add(seq[i])
        self.hash_dict = {k: sorted(v) for k, v in self.hash_dict.items()}

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "task_id", "task_type", "copy_index",
                                       "stream_index", "target_sid", "target_token_ids",
                                       "item_id"]

    def _install_bf16_sync_callback(self):
        from trl import SyncRefModelCallback as _BuiltIn
        kept = [c for c in self.callback_handler.callbacks
                if not isinstance(c, _BuiltIn)]
        self.callback_handler.callbacks = kept
        self.callback_handler.add_callback(V2SyncRefCallback(
            self.ref_model, self.accelerator,
            steps=getattr(self.args, "ref_model_sync_steps", 512),
            alpha=getattr(self.args, "ref_model_mixup_alpha", 0.6)))

    def _by_value_state_preseed(self, resume_ckpt):
        import torch as T
        for p in self.model.parameters():
            p.grad = T.zeros_like(p)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        print("V2_BY_VALUE: paged buffers pre-seeded", flush=True)

    def _override_opt_load(self, resume_ckpt):
        import torch as T
        opt = self.optimizer
        cpu_state = T.load(os.path.join(resume_ckpt, "optimizer.pt"), map_location="cpu")

        def by_value(state_dict):
            saved = state_dict.get("state", {})
            saved_ids = list(saved.keys())
            live_ids = list(opt.state.keys())
            n = 0
            for i, sid in enumerate(saved_ids):
                lid = live_ids[i] if i < len(live_ids) else None
                if lid is None:
                    continue
                st = opt.state.setdefault(lid, {})
                for k, v in saved[sid].items():
                    if T.is_tensor(v) and k in st and T.is_tensor(st[k]):
                        st[k].copy_(v.to(st[k].dtype) if st[k].dtype != v.dtype else v)
                    else:
                        st[k] = v
                n += 1
            print(f"V2_BY_VALUE: optimizer state restored for {n} params (by value, paged buffers preserved)", flush=True)

        opt.load_state_dict = by_value
        print("V2_BY_VALUE: optimizer.load_state_dict overridden (cpu->paged by value)", flush=True)

    def _get_train_sampler(self, train_dataset=None):
        ds = self.train_dataset if train_dataset is None else train_dataset
        return RepeatOrderedSampler(ds, self.num_generations)


class V2SyncRefCallback(TrainerCallback):
    """Approved T4-R1 adaptation: full reference stored in BF16, EMA computed in
    FP32 intermediates (master-average semantics preserved; rounding only at
    bf16 storage). Replaces trl's fp32 in-place callback which cannot mix
    fp32 policy grads with a bf16 ref. Events are logged for audit."""

    def __init__(self, ref_model, accelerator, steps=512, alpha=0.6):
        self.ref_model = ref_model
        self.accelerator = accelerator
        self.steps = steps
        self.alpha = alpha
        self.events = []
        self.events_path = os.environ.get("V2_REF_SYNC_EVENTS", "")

    def _persist(self, ev):
        if self.events_path:
            with open(self.events_path, "a") as f:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    def _ema(self, policy):
        a = self.alpha
        with torch.no_grad():
            for p_ref, p_pol in zip(self.ref_model.parameters(), policy.parameters()):
                t = p_ref.detach().float()
                t.mul_(1.0 - a).add_(p_pol.detach().float(), alpha=a)
                p_ref.data.copy_(t.to(p_ref.dtype))

    def on_step_end(self, args, state, control, model=None, **kw):
        if self.ref_model is None or state.global_step % self.steps != 0:
            return
        self._ema(model)
        h = hashlib.sha256()
        for p in self.ref_model.parameters():
            h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        ev = {"event": "ref_ema_sync", "global_step": state.global_step,
              "alpha": self.alpha, "steps_interval": self.steps,
              "intermediate": "fp32", "store_dtype": "bf16",
              "ref_bf16_sha256": h.hexdigest(),
              "n_ref_params": sum(1 for _ in self.ref_model.parameters())}
        self.events.append(ev)
        self._persist(ev)


class DiagnosticsCallback(TrainerCallback):
    """Persist per-step trainer metrics + per-group reward records to JSONL
    files outside checkpoint rotation (locked protocol: diagnostics must not
    be lost to save_total_limit rotation)."""

    def __init__(self, metrics_path, reward_records_path, recorder):
        self.metrics_path = metrics_path
        self.reward_records_path = reward_records_path
        self.recorder = recorder

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs:
            return
        rec = {"global_step": state.global_step, "epoch": round(float(state.epoch or 0), 6),
               "wall": time.time(), **{k: v for k, v in logs.items()
                                        if isinstance(v, (int, float, str, bool))}}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.recorder.flush(self.reward_records_path)


class RewardRecorder:
    """Wraps the two reward functions to record per-candidate rewards with
    stable identity (stream_index/task_id/task_type/copy_index)."""

    def __init__(self):
        self.buf = []
        self.n_calls = {"rule_reward": 0, "ndcg_rule_reward": 0}

    def wrap(self, fn, name):
        def wrapped(prompts, completions, task_id=None, task_type=None,
                    stream_index=None, copy_index=None, **kw):
            self.n_calls[name] += 1
            out = fn(prompts, completions,
                     target_sid=kw.get("target_sid"), **{})
            for k, v in zip(zip(stream_index or [], task_id or [],
                                task_type or [], copy_index or []), out):
                self.buf.append({"stream_index": k[0], "task_id": k[1],
                                 "task_type": k[2], "copy_index": k[3],
                                 "reward_fn": name, "reward": float(v)})
            return out
        wrapped.__name__ = name
        return wrapped

    def flush(self, path):
        if not self.buf:
            return
        with open(path, "a") as f:
            for r in self.buf:
                f.write(json.dumps(r) + "\n")
        self.buf = []


def build_dataset(stream_path):
    rows = [json.loads(l) for l in open(stream_path)]
    data = {
        "prompt": [r["prompt"] for r in rows],
        "task_id": [r["task_id"] for r in rows],
        "task_type": [r["task_type"] for r in rows],
        "copy_index": [r["copy_index"] for r in rows],
        "stream_index": [r["stream_index"] for r in rows],
        "target_sid": [r["item_sid"] for r in rows],
        "target_token_ids": [r["item_token_ids"] for r in rows],
        "item_id": [r["item_id"] for r in rows],
    }
    return HFDataset.from_dict(data), len(rows)


def main():
    ap = argparse.ArgumentParser(description="Phase7 v2 GRPO (see module docstring for adaptation list)")
    ap.add_argument("--model-path", required=True, help="group SFT top-level dir (policy AND ref init)")
    ap.add_argument("--base-model", default=None, help="tokenizer/info source dir (default: model-path)")
    ap.add_argument("--method", required=True, choices=["A", "B", "C"])
    ap.add_argument("--stream-manifest", default=None,
                    help="rl_stream_20000_{G}.jsonl (default: outputs/v2/manifests/ by method)")
    ap.add_argument("--info-file", default=None, help="group info.txt (default: outputs/v2/data/{G}/info.txt)")
    ap.add_argument("--train-batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=32)
    ap.add_argument("--num-generations", type=int, default=16)
    ap.add_argument("--num-train-epochs", type=int, default=1,
                    help="stream spans 2 logical epochs; 1 pass over the stream = the formal budget")
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=1e-3)
    ap.add_argument("--max-completion-length", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--sync-ref-model", default="True")
    ap.add_argument("--ref-sync-steps", type=int, default=512)
    ap.add_argument("--ref-mixup-alpha", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--beams", type=int, default=None, help="defaults to num-generations (constrained beam G)")
    ap.add_argument("--bf16", default="True")
    ap.add_argument("--optim", default="paged_adamw_32bit")
    ap.add_argument("--lr-scheduler-type", default="cosine")
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-grad-norm", type=float, default=0.3)
    ap.add_argument("--logging-steps", type=int, default=1)
    ap.add_argument("--save-steps", type=float, default=0.1)
    ap.add_argument("--save-total-limit", type=int, default=20)
    ap.add_argument("--gradient-checkpointing", default="False")
    ap.add_argument("--resume-from-checkpoint", default=None)
    ap.add_argument("--ref-dtype", default="bf16", choices=["fp32", "bf16"],
                    help="T4-R1 approved: bf16 storage + fp32-intermediate EMA (fp32 keeps original semantics)")
    args = ap.parse_args()

    G = args.num_generations
    base_model = args.base_model or args.model_path
    stream = args.stream_manifest or f"outputs/v2/manifests/rl_stream_20000_{args.method}.jsonl"
    info_file = args.info_file or f"outputs/v2/data/{args.method}/info.txt"

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    train_dataset, n_rows = build_dataset(stream)

    sync_ref = str(args.sync_ref_model).lower() in ("1", "true", "yes")
    cfg = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=G,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        temperature=args.temperature,
        max_completion_length=args.max_completion_length,
        max_prompt_length=512,
        sync_ref_model=sync_ref,
        ref_model_sync_steps=args.ref_sync_steps,
        ref_model_mixup_alpha=args.ref_mixup_alpha,
        bf16=str(args.bf16).lower() in ("1", "true", "yes"),
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="no",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=0,
        gradient_checkpointing=str(args.gradient_checkpointing).lower() in ("1", "true", "yes"),
        log_completions=False,
        run_name=f"{args.method}-RL",
    )

    recorder = RewardRecorder()
    rule_w = recorder.wrap(rule_reward, "rule_reward")
    ndcg_w = recorder.wrap(make_ndcg_rule_reward(G), "ndcg_rule_reward")

    os.environ["V2_REF_DTYPE"] = args.ref_dtype
    trainer = V2ReReTrainer(
        model=args.model_path,
        base_model=base_model,
        reward_funcs=[rule_w, ndcg_w],
        args=cfg,
        beam_search=True,
        length_penalty=0.0,
        add_gt=False,
        dynamic_sampling=False,
        test_during_training=False,
        test_beam=args.beams or G,
        dapo=False,
        gspo=False,
        info_file=info_file,
        prompt2history=None,
        history2target=None,
        train_dataset=train_dataset,
        eval_dataset=None,
        callbacks=[DiagnosticsCallback(
            os.path.join(args.output_dir, "train_metrics.jsonl"),
            os.path.join(args.output_dir, "reward_records.jsonl"), recorder)],
    )

    os.makedirs(args.output_dir, exist_ok=True)
    run_meta = {
        "stage": "phase7-v2-grpo", "method": args.method, "stream_manifest": stream,
        "n_stream_rows": n_rows, "info_file": info_file,
        "entry_sha256": __import__("hashlib").sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
        "trainer_source": "minionerec_trainer.ReReTrainer (locked)",
        "adaptations": "see module docstring A1-A9",
        "config": {k: str(getattr(cfg, k)) for k in
                   ("per_device_train_batch_size", "gradient_accumulation_steps", "num_generations",
                    "num_train_epochs", "learning_rate", "beta", "temperature", "max_completion_length",
                    "max_prompt_length", "sync_ref_model", "ref_model_sync_steps", "ref_model_mixup_alpha",
                    "bf16", "optim", "lr_scheduler_type", "warmup_ratio", "max_grad_norm",
                    "save_steps", "save_total_limit", "seed", "gradient_checkpointing")},
        "start_wall": time.time(),
    }
    with open(os.path.join(args.output_dir, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=1)

    kwargs = {}
    if args.resume_from_checkpoint:
        kwargs["resume_from_checkpoint"] = args.resume_from_checkpoint
    trainer.train(**kwargs)

    trainer.save_model(args.output_dir)
    final_dir = os.path.join(args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(final_dir)
    trainer.processing_class.save_pretrained(final_dir)

    run_meta["end_wall"] = time.time()
    run_meta["final_global_step"] = trainer.state.global_step
    run_meta["reward_calls"] = recorder.n_calls
    with open(os.path.join(args.output_dir, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=1)
    print(f"V2_GRPO_DONE method={args.method} steps={trainer.state.global_step}")


if __name__ == "__main__":
    main()
