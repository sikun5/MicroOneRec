#!/usr/bin/env python3
"""Phase 7 T4: real-wiring G16 memory probe & two-update smoke per group (boss-released).

Real 599M policy+ref from the group's frozen SFT top-level (fp32 load, original
rl.py behavior), paged_adamw_32bit, bf16 autocast, G16 / batch16 / accum32,
formal max_prompt_length=512, real constrained beam rollout over the full
catalog trie, real two-reward wiring on identity columns, gradient
checkpointing ON (allowed mitigation), max_steps=2 (=2 real updates, 64
logical groups). NO mock generation, NO synthetic rewards, NO reduced G/batch.

Artifacts -> outputs/v2/debug/phase7/grpo_smoke/attempt_001/ (debug-only; the
formal checkpoint directories are never touched).
  memory_probe/{G}/part_{x}.json  per-phase allocated/reserved peaks
  update_smoke/{G}/group_snapshot.json  one full real 16-candidate group
  update_smoke/{G}/smoke_ckpt/          2-update smoke weights (debug)
  update_smoke/{G}/load_check.json      fresh-process load + strict raw gen
Reports aggregate -> .../reports/ and evidence/phase7/.
"""
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
BASE = os.environ.get("T4_ATTEMPT", "outputs/v2/debug/phase7/grpo_smoke/attempt_001")
os.makedirs(BASE, exist_ok=True)
for d in ("configs", "control_checks", "reports"):
    os.makedirs(os.path.join(BASE, d), exist_ok=True)
for g in "ABC":
    os.makedirs(os.path.join(BASE, "memory_probe", g), exist_ok=True)
    os.makedirs(os.path.join(BASE, "update_smoke", g), exist_ok=True)

import torch  # noqa: E402
from transformers import TrainerCallback  # noqa: E402
from trl import GRPOConfig  # noqa: E402

from v2_train_grpo import V2ReReTrainer, RewardRecorder, rule_reward, make_ndcg_rule_reward  # noqa: E402
from v2_generation_adapter import validate_sid_sequence  # noqa: E402 (Phase5 frozen)

CK = "outputs/v2/checkpoints"


def sha_state(model, sample=None):
    h = hashlib.sha256()
    sd = model.state_dict()
    keys = sorted(sd)
    if sample:
        keys = [k for k in keys if any(s in k for s in sample)][:6]
    for k in keys:
        t = sd[k].detach().cpu().contiguous().flatten()
        h.update(k.encode())
        h.update(str(tuple(sd[k].shape)).encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class MemProbe(TrainerCallback):
    """Per-phase peak memory: generation (per micro) and optimizer step."""

    def __init__(self, path):
        self.path = path
        self.rows = []

    def on_step_begin(self, args, state, control, **kw):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # allowed mitigation: release stale blocks between updates
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kw):
        row = {"global_step": state.global_step,
               "allocated_now": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
               "reserved_now": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
               "peak_alloc": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
               "peak_reserved": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0}
        self.rows.append(row)
        with open(self.path, "w") as f:
            json.dump(self.rows, f, indent=1)


class GroupCapture(TrainerCallback):
    """Capture the designated first micro-batch of update 1 in full."""

    def __init__(self, path, capture_step=1):
        self.path = path
        self.capture_step = capture_step
        self.done = False

    def on_step_end(self, args, state, control, model=None, **kw):
        cap = getattr(model, "_v2_group_capture", None)
        if cap and not self.done and state.global_step >= self.capture_step:
            self.done = True
            with open(self.path, "w") as f:
                json.dump(cap, f, ensure_ascii=False)
            model._v2_group_capture = None


class V2SmokeTrainer(V2ReReTrainer):
    """Adds: per-micro generation peak sampling + designated group capture +
    representative param fingerprints. Training math untouched."""

    capture_micro = None

    def _prepare_inputs(self, inputs):
        was = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            was = torch.cuda.memory_allocated()
        out = super()._prepare_inputs(inputs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # release generation KV/fragmentation before backward
        if torch.cuda.is_available() and not getattr(self, "_cap_done", False):
            cap = {"stream_index": [r.get("stream_index") for r in inputs][:16],
                   "task_id": [r.get("task_id") for r in inputs][:16],
                   "task_type": [r.get("task_type") for r in inputs][:16],
                   "target_sid": [r.get("target_sid") for r in inputs][:16],
                   "prompt_len": int(out["prompt_ids"].size(1)) if hasattr(out, "get") else None,
                   "completion_len": int(out["completion_ids"].size(1)),
                   "completion_ids": out["completion_ids"].tolist(),
                   "completion_mask": out["completion_mask"].tolist(),
                   "advantages": out["advantages"].tolist(),
                   "sliced_rewards": out["sliced_rewards"].tolist(),
                   "ref_per_token_logps_head": out["ref_per_token_logps"][:, :3].tolist(),
                   "gen_peak_alloc": torch.cuda.max_memory_allocated(),
                   "gen_peak_reserved": torch.cuda.max_memory_reserved()}
            caps = getattr(self, "_captures", [])
            caps.append(cap)
            if len(caps) == 1:
                self._cap_done = True
                with open(self._capture_path, "w") as f:
                    json.dump(caps[0], f, ensure_ascii=False)
        return out


def load_stream(method, n=1024):
    rows = [json.loads(l) for l in open(f"outputs/v2/manifests/rl_stream_20000_{method}.jsonl")][:n]
    from datasets import Dataset as HFDataset
    return HFDataset.from_dict({
        "prompt": [r["prompt"] for r in rows], "task_id": [r["task_id"] for r in rows],
        "task_type": [r["task_type"] for r in rows], "copy_index": [r["copy_index"] for r in rows],
        "stream_index": [r["stream_index"] for r in rows],
        "target_sid": [r["item_sid"] for r in rows],
        "target_token_ids": [r["item_token_ids"] for r in rows],
        "item_id": [r["item_id"] for r in rows]}), len(rows)


def run_group(method, steps=2, sync_steps=512, tag="main"):
    out_dir = f"{BASE}/update_smoke/{method}" + ("" if tag == "main" else f"_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    ds, n = load_stream(method, steps * 32 * 16)
    mem = MemProbe(f"{BASE}/memory_probe/{method}_{tag}.json")
    rec = RewardRecorder()
    import random
    import numpy as np
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    cfg = GRPOConfig(
        output_dir=out_dir, per_device_train_batch_size=16, gradient_accumulation_steps=32,
        num_generations=16, num_train_epochs=1, max_steps=steps,
        learning_rate=1e-5, beta=1e-3, temperature=1.0,
        max_completion_length=128, max_prompt_length=512,
        sync_ref_model=True, ref_model_sync_steps=sync_steps, ref_model_mixup_alpha=0.6,
        bf16=True, optim="paged_adamw_32bit", lr_scheduler_type="cosine",
        warmup_ratio=0.03, max_grad_norm=0.3, logging_steps=1,
        save_strategy="no", eval_strategy="no", report_to=[],
        seed=42, dataloader_num_workers=0, gradient_checkpointing=True,
        log_completions=False, run_name=f"{method}-RL-smoke-{tag}")
    tr = V2SmokeTrainer(
        model=f"{CK}/{method}-SFT", base_model=f"{CK}/{method}-SFT",
        reward_funcs=[rec.wrap(rule_reward, "rule_reward"),
                      rec.wrap(make_ndcg_rule_reward(16), "ndcg_rule_reward")],
        args=cfg, beam_search=True, length_penalty=0.0, add_gt=False,
        dynamic_sampling=False, test_during_training=False, test_beam=16,
        dapo=False, gspo=False, info_file=f"outputs/v2/data/{method}/info.txt",
        prompt2history=None, history2target=None,
        train_dataset=ds, eval_dataset=None, callbacks=[mem])
    tr._capture_path = f"{BASE}/update_smoke/{method}/group_snapshot{'_' + tag if tag != 'main' else ''}.json"
    tr._captures = []
    fp = {"before": sha_state(tr.model, sample=("embed_tokens", "layers.0.self_attn", "model.norm"))}
    n_trainable = sum(p.numel() for p in tr.model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in tr.model.parameters())
    t0 = time.time()
    rc = 0
    try:
        tr.train()
    except torch.cuda.OutOfMemoryError as e:
        rc = 1
        with open(f"{BASE}/memory_probe/{method}_{tag}_OOM.json", "w") as f:
            json.dump({"OOM": str(e)[:500], "config": cfg.to_dict() if hasattr(cfg, "to_dict") else str(cfg)}, f)
        raise
    wall = time.time() - t0
    fp["after"] = sha_state(tr.model, sample=("embed_tokens", "layers.0.self_attn", "model.norm"))
    result = {"method": method, "tag": tag, "steps": steps, "sync_steps": sync_steps,
              "wall_seconds": round(wall, 1), "exit": "ok" if rc == 0 else "oom",
              "trainable": n_trainable, "all_params": n_all,
              "full_trainable": n_trainable == n_all,
              "param_fingerprint_before": fp["before"], "param_fingerprint_after": fp["after"],
              "params_changed": fp["before"] != fp["after"],
              "final_metrics_last": None,
              "peak_after_train_alloc": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
              "peak_after_train_reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0}
    if tr.state.global_step >= steps:
        tr.save_model(f"{out_dir}/smoke_ckpt")
        result["saved"] = f"{out_dir}/smoke_ckpt"
    with open(f"{BASE}/update_smoke/{method}/smoke_result{'_' + tag if tag != 'main' else ''}.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("SMOKE_RESULT", json.dumps({k: result[k] for k in
          ("method", "tag", "steps", "wall_seconds", "params_changed", "full_trainable")}))
    return result


def load_check(method):
    """Fresh-process load + strict constrained generation on 10 valid prompts."""
    ck = f"{BASE}/update_smoke/{method}/smoke_ckpt"
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, ROOT)  # v2_generation_adapter lives at root
    from v2_sft_data import SID_INSTRUCTION, Tokenizer, load_group_bindings, _generate_prompt
    tokz = AutoTokenizer.from_pretrained(ck, local_files_only=True)
    assert len(tokz) == 154977
    tokz.pad_token = tokz.eos_token
    tokz.pad_token_id = tokz.eos_token_id
    tokz.padding_side = "left"
    tkw = Tokenizer(tokz)
    model = AutoModelForCausalLM.from_pretrained(ck, torch_dtype=torch.bfloat16, local_files_only=True)
    model.config.use_cache = False
    model = model.cuda().eval()
    bindings = load_group_bindings(method, root=ROOT)
    tok_map = json.load(open("outputs/v2/tokenizer/token_id_map.json"))["map"]
    atomic = set(tok_map.values())
    EOS = tokz.eos_token_id
    trie = {}
    for toks in bindings["token_ids"]:
        node = trie
        for t in toks:
            node = node.setdefault(t, {})
        node[EOS] = {}
    valid = pd.read_csv(f"outputs/v2/data/{method}/valid.csv").head(10)

    class StrictLP:
        def __init__(self):
            self.pl = 0

        def __call__(self, ids, scores):
            out = []
            b = scores.size(0)
            for row in range(b):
                gen = ids[row, self.pl:].tolist()
                node = trie
                ok = True
                for t in gen:
                    if t in node:
                        node = node[t]
                    else:
                        ok = False
                        break
                allowed = sorted(node.keys()) if ok else []
                mask = torch.full_like(scores[row], float("-inf"))
                if allowed:
                    mask[torch.tensor(allowed, device=scores.device, dtype=torch.long)] = 0.0
                out.append(mask)
            return torch.stack(out)

    res = []
    legal = 0
    for _, r in valid.iterrows():
        hist = eval(r["history_item_sid"]) if isinstance(r["history_item_sid"], str) else []
        ui = (f"The user has interacted with items {', '.join(hist)} in chronological order. "
              f"Can you predict the next possible item that the user may expect?")
        ids = tkw.encode(SID_INSTRUCTION, bos=True, eos=False) + tkw.encode(_generate_prompt(ui), bos=False, eos=False)
        lp = StrictLP(); lp.pl = len(ids)
        with torch.no_grad():
            gen = model.generate(torch.tensor([ids]).cuda(), attention_mask=torch.ones((1, len(ids)), dtype=torch.long).cuda(),
                                 max_new_tokens=6, num_beams=10, do_sample=False, num_return_sequences=1,
                                 logits_processor=[lp], pad_token_id=EOS, length_penalty=0.0)
        out = gen[0, len(ids):].tolist()
        ok, msg = validate_sid_sequence(out, EOS, atomic)
        legal += int(bool(ok))
        res.append({"raw_ids": out, "strict_legal": bool(ok)})
    doc = {"method": method, "checkpoint": ck, "n_prompts": 10, "strict_legal": legal,
           "raw_first": res[0]["raw_ids"], "results": res}
    with open(f"{BASE}/update_smoke/{method}/load_check.json", "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print("LOAD_CHECK", method, f"{legal}/10 strict_legal")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list("ABC"))
    ap.add_argument("--phase", default="main", choices=["main", "sync_probe", "load_check"])
    ap.add_argument("--steps", type=int, default=2)
    args = ap.parse_args()
    if args.phase == "main":
        run_group(args.method, steps=args.steps, sync_steps=512, tag="main")
    elif args.phase == "sync_probe":
        run_group(args.method, steps=5, sync_steps=4, tag="syncdiag")
    else:
        load_check(args.method)
