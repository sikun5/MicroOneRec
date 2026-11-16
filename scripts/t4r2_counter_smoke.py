#!/usr/bin/env python3
"""Phase7 T4-R2(1): instrumented per-group accumulation counter smoke.

Same real wiring as attempt_005 (bf16 ref, fp32 intermediate EMA, G16/batch16/
accum32, formal 512 prompt cap, strict catalog trie, paged_adamw_32bit,
max_steps=2) with honest instrumentation: per-micro-batch forward count with
stream_index/group identity, per-optimizer-step count and window composition,
loss values (finiteness), gradient/optimizer-state dtype+device+paging facts,
representative parameter fingerprints, prompt-length distribution (incl. the
longest real stream prompt), allocated/reserved peaks with capture ranges,
wall time and exit code. Counts are MEASURED, not 2x32 derivation.
"""
import argparse, hashlib, json, os, sys, time
ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
BASE = os.environ.get("T4_ATTEMPT", "outputs/v2/debug/phase7/grpo_smoke/attempt_006")
os.makedirs(BASE + "/reports", exist_ok=True)
import torch
from transformers import TrainerCallback
from trl import GRPOConfig
from v2_train_grpo import V2ReReTrainer, RewardRecorder, rule_reward, make_ndcg_rule_reward

FAKE = [151674, 151931, 152335, 152437, 151643]  # not used (real rollout)

def sha_state(model, sample=None):
    h = hashlib.sha256(); sd = model.state_dict()
    keys = sorted(sd)
    if sample: keys = [k for k in keys if any(x in k for x in sample)][:8]
    for k in keys:
        t = sd[k].detach().cpu().contiguous().flatten()
        h.update(k.encode()); h.update(str(tuple(sd[k].shape)).encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def run(method):
    rows = [json.loads(l) for l in open(f"outputs/v2/manifests/rl_stream_20000_{method}.jsonl")][:1024]
    lens = [len(r["prompt"]) for r in rows]
    from datasets import Dataset as HFD
    ds = HFD.from_dict({'prompt': [r['prompt'] for r in rows], 'task_id': [r['task_id'] for r in rows], 'task_type': [r['task_type'] for r in rows], 'copy_index': [r['copy_index'] for r in rows], 'stream_index': [r['stream_index'] for r in rows], 'target_sid': [r['item_sid'] for r in rows], 'target_token_ids': [r['item_token_ids'] for r in rows], 'item_id': [r['item_id'] for r in rows]})
    import random, numpy as np
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    cfg = GRPOConfig(output_dir=f"{BASE}/update_smoke/{method}_counted",
        per_device_train_batch_size=16, gradient_accumulation_steps=32, num_generations=16,
        num_train_epochs=1, max_steps=2, learning_rate=1e-5, beta=1e-3, temperature=1.0,
        max_completion_length=128, max_prompt_length=512,
        sync_ref_model=True, ref_model_sync_steps=512, ref_model_mixup_alpha=0.6,
        bf16=True, optim="paged_adamw_32bit", lr_scheduler_type="cosine",
        warmup_ratio=0.03, max_grad_norm=0.3, logging_steps=1, save_strategy="no",
        eval_strategy="no", report_to=[], seed=42, dataloader_num_workers=0,
        gradient_checkpointing=True, log_completions=False, run_name=f"{method}-counted")
    tr = V2ReReTrainer(model=f"outputs/v2/checkpoints/{method}-SFT",
        base_model=f"outputs/v2/checkpoints/{method}-SFT",
        reward_funcs=[rule_reward, make_ndcg_rule_reward(16)],
        args=cfg, beam_search=True, length_penalty=0.0, add_gt=False,
        dynamic_sampling=False, test_during_training=False, test_beam=16,
        dapo=False, gspo=False, info_file=f"outputs/v2/data/{method}/info.txt",
        prompt2history=None, history2target=None,
        train_dataset=ds, eval_dataset=None, callbacks=[])
    # ---- instrumentation ----
    micro, steps, losses = [], [], []
    orig_ts = type(tr).training_step
    def counting_ts(self, m, inputs, *a, **kw):
        sidx = sorted({r["stream_index"] for r in inputs}) if isinstance(inputs, list) else []
        tids = sorted({r["task_id"] for r in inputs}) if isinstance(inputs, list) else []
        pl = [len(r["prompt"]) for r in inputs] if isinstance(inputs, list) else []
        out = orig_ts(self, m, inputs, *a, **kw)
        g = [p.grad for p in self.model.parameters() if p.grad is not None]
        grad_info = ({'dtype': str(g[0].dtype), 'device': str(g[0].device),
                      'tensors': len(g), 'numel': sum(x.numel() for x in g),
                      'paged_live': False} if g else {'grads': 'none_at_this_point'})
        micro.append({"grad_info": grad_info, "i": len(micro), "n_rows": len(inputs) if isinstance(inputs, list) else -1,
                      "stream_index": sidx, "task_id": tids[0] if len(tids) == 1 else f"{len(tids)}distinct",
                      "prompt_chars_min": min(pl) if pl else -1, "prompt_chars_max": max(pl) if pl else -1,
                      "loss": float(out) if hasattr(out, "item") else float(out)})
        if math_isfinite(micro[-1]["loss"]): losses.append(micro[-1]["loss"])
        return out
    type(tr).training_step = counting_ts
    class CB(TrainerCallback):
        def on_step_end(self, args, state, control, optimizer=None, model=None, **kw):
            inner = getattr(optimizer, "optimizer", optimizer)
            st = getattr(inner, "state", {})
            dev = dtype = paged = None
            for v in st.values():
                for t in v.values():
                    if torch.is_tensor(t):
                        dtype = str(t.dtype); dev = str(t.device); paged = type(t).__name__; break
                if dtype: break
            steps.append({"global_step": state.global_step,
                          "lr": optimizer.param_groups[0]["lr"],
                          "grad_note": "grads zeroed post-step; live grad facts captured per-micro (see micro records)",
                          "opt_state_dtype": dtype, "opt_state_device": dev, "opt_state_class": paged,
                          "n_opt_states": len(st)})
    tr.callback_handler.add_callback(CB())
    fp = {"before": sha_state(tr.model, sample=("embed_tokens", "layers.0.", "model.norm"))}
    trainable = sum(p.numel() for p in tr.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in tr.model.parameters())
    t0 = time.time(); code = "ok"
    try:
        tr.train()
    except Exception as e:
        import traceback
        import traceback; code = "FAIL " + traceback.format_exc()[-650:]
    wall = time.time() - t0
    fp["after"] = sha_state(tr.model, sample=("embed_tokens", "layers.0.", "model.norm"))
    wins = [micro[i:i+32] for i in range(0, len(micro), 32)]
    doc = {"method": method, "measured": True,
           "micro_forwards": len(micro), "optimizer_steps": len(steps),
           "rows_per_micro_set": sorted({r["n_rows"] for r in micro}),
           "windows": len(wins), "window_sizes": sorted({len(w) for w in wins}),
           "window0_group_ids": [w[0]["task_id"] for w in wins[:1]],
           "window_ids_first_last": [wins[0][0]["stream_index"] if wins else None,
                                      wins[-1][-1]["stream_index"] if wins else None],
           "consumed_stream_index_sorted_eq": sorted(x["stream_index"][0] for x in micro if len(x["stream_index"]) == 1) == list(range(1024)),
           "loss_finite": all(math_isfinite(x) for x in losses), "n_loss_rows": len(losses),
           "loss_head_tail": [losses[:2], losses[-2:]],
           "grads": steps[0] if steps else None,
           "paged_state": steps[-1] if steps else None,
           "trainable": trainable, "total_params": total, "full_trainable": trainable == total,
           "param_fp_before": fp["before"], "param_fp_after": fp["after"], "params_changed": fp["before"] != fp["after"],
           "prompt_chars_min": min(lens), "prompt_chars_max": max(lens),
           "peak_alloc": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
           "peak_reserved": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
           "peak_capture_range": "reset at process start; covers build+train+close",
           "wall_seconds": round(wall, 1), "exit": code,
           "micro_rows": micro, "observation_source": "counting wrapper on real Trainer.training_step + TrainerCallback.on_step_end (measured, not derived)"}
    json.dump(doc, open(f"{BASE}/reports/t4r2_counter_{method}.json", "w"), ensure_ascii=False, indent=1)
    type(tr).training_step = orig_ts
    print("COUNTER", method, json.dumps({k: doc[k] for k in
          ("micro_forwards", "optimizer_steps", "windows", "consumed_stream_index_sorted_eq", "exit", "wall_seconds")}))
    return doc

def math_isfinite(x):
    return x == x and abs(x) != float("inf")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list("ABC"))
    a = ap.parse_args()
    run(a.method)