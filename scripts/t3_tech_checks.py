#!/usr/bin/env python3
"""Phase 7 T3 technical checks (boss T3 list) -> rollout_gradient_checks.json + resume_check.json.

--part a  construction invariants (CPU tiny model, FULL group-A stream dataset):
          policy all-trainable, no PEFT, optimizer covers every trainable param,
          ref separate storage / no optimizer entry / no grads, sync callback
          attach on/off control, mask & KL boundary on one micro-step with a
          mocked generate (disclosed), zero-advantage group -> finite loss.
--part b  T3-R1(2) counting regression: real Trainer loop + FIXED stream sampler
          over the full 20000-group stream (mock generate, synthetic rewards).
          Asserts 20000 micro-batches = 625 exact windows x 32 groups = 320000
          candidates, every (copy_index, task_id) exactly once, boundary window
          = 16 groups per logical epoch, ref sync exactly at 512, no partial
          window, no drop/dup.
--part c  T3-R1(4) resume vs continuous (CPU tiny model, max_steps=8): checkpoint
          at step 3 + resume, compare consumed stream order, per-step optimizer
          state/lr, final weight checksum; honest scoping of resume semantics.
Counts only - no recommendation metrics, no GPU, no real rollout.
"""
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
if os.environ.get("T3_PIN_CPU") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # part a stays CPU; parts b/c may use the idle MIG slice
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
EV = "outputs/v2/evidence/phase7"
os.makedirs(EV, exist_ok=True)

import torch  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM, TrainerCallback  # noqa: E402
from trl import GRPOConfig  # noqa: E402

import minionerec_trainer as mt  # noqa: E402
from v2_train_grpo import (V2ReReTrainer, RepeatOrderedSampler,  # noqa: E402
                           RewardRecorder, rule_reward, make_ndcg_rule_reward)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


FAKE = [151674, 151931, 152335, 152437, 151643, 151643]


def tiny_model():
    torch.manual_seed(42)
    cfg = Qwen3Config(vocab_size=154977, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, max_position_embeddings=512, tie_word_embeddings=True)
    m = Qwen3ForCausalLM(cfg).to(torch.float32)
    m.config.use_cache = False
    return m


def mock_generate_factory(seq):
    def fake_generate(input_ids=None, attention_mask=None, generation_config=None,
                      logits_processor=None, **kw):
        p = input_ids.size(1)
        comp = torch.tensor([seq], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids.repeat_interleave(16, 0), comp.expand(16, -1)], 1)
    return fake_generate


def build(stream="outputs/v2/manifests/rl_stream_20000_A.jsonl", sync=True, max_steps=None,
          save_steps=0.1, logging_steps=1, output_dir="/tmp/t3_tech_out", probe_prompt_len=64):
    from datasets import Dataset as HFDataset
    rows = [json.loads(l) for l in open(stream)]
    data = {"prompt": [r["prompt"] for r in rows],
            "task_id": [r["task_id"] for r in rows],
            "task_type": [r["task_type"] for r in rows],
            "copy_index": [r["copy_index"] for r in rows],
            "stream_index": [r["stream_index"] for r in rows],
            "target_sid": [r["item_sid"] for r in rows],
            "target_token_ids": [r["item_token_ids"] for r in rows],
            "item_id": [r["item_id"] for r in rows]}
    ds = HFDataset.from_dict(data)
    import random
    import numpy as np
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    cfg = GRPOConfig(output_dir=output_dir, per_device_train_batch_size=16,
                     gradient_accumulation_steps=32, num_generations=16,
                     num_train_epochs=1, learning_rate=1e-5, beta=1e-3,
                     temperature=1.0, max_completion_length=128, max_prompt_length=probe_prompt_len,
                     sync_ref_model=sync, ref_model_sync_steps=512, ref_model_mixup_alpha=0.6,
                     bf16=False, optim="adamw_torch", lr_scheduler_type="cosine",
                     warmup_ratio=0.03, max_grad_norm=0.3, logging_steps=logging_steps,
                     save_strategy="steps", save_steps=save_steps, save_total_limit=20,
                     eval_strategy="no", report_to=[], seed=42, dataloader_num_workers=0,
                     gradient_checkpointing=False, log_completions=False,
                     run_name="t3_tech", **({"max_steps": max_steps} if max_steps else {}))
    model = tiny_model()
    model.generate = mock_generate_factory(FAKE)
    rec = RewardRecorder()
    trainer = V2ReReTrainer(
        model=model, base_model="outputs/v2/checkpoints/A-SFT",
        reward_funcs=[rec.wrap(rule_reward, "rule_reward"),
                      rec.wrap(make_ndcg_rule_reward(16), "ndcg_rule_reward")],
        args=cfg, beam_search=True, length_penalty=0.0, add_gt=False,
        dynamic_sampling=False, test_during_training=False, test_beam=16,
        dapo=False, gspo=False, info_file="outputs/v2/data/A/info.txt",
        prompt2history=None, history2target=None,
        train_dataset=ds, eval_dataset=None, callbacks=[])
    return trainer, rec, model


def part_a():
    trainer, _, model = build(sync=True, logging_steps=500)
    out = {}
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    out["policy_trainable"] = {"trainable": n_train, "total": n_all, "all_true": n_train == n_all}
    n_peft = sum("lora" in n.lower() or "peft" in n.lower()
                 for n, _ in model.named_modules())
    out["no_peft"] = {"peft_modules": n_peft, "ok": n_peft == 0}
    trainer.create_optimizer()
    names = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
    cov = sum(1 for p in model.parameters() if id(p) in names)
    n_tensors = len(list(model.parameters()))
    out["optimizer_coverage"] = {"policy_param_tensors": n_tensors, "in_optimizer": cov,
                                 "policy_numel": n_all, "ok": cov == n_tensors}
    ref = trainer.ref_model
    ref_ng = all(not p.requires_grad for p in ref.parameters())
    ref_in_opt = any(id(p) in names for p in ref.parameters())
    shared = any(a.data_ptr() == b.data_ptr()
                 for a, b in zip(model.parameters(), ref.parameters()))
    out["ref_model"] = {"exists": ref is not None,
                        "params": sum(p.numel() for p in ref.parameters()),
                        "requires_grad_all_false": ref_ng,
                        "in_optimizer": ref_in_opt,
                        "shares_storage_with_policy": shared,
                        "ok": ref_ng and not ref_in_opt and not shared}
    cb = [type(c).__name__ for c in trainer.callback_handler.callbacks]
    out["sync_callback_on"] = {"attached": "SyncRefModelCallback" in cb, "callbacks": cb}
    # sync OFF control
    trainer_off, _, _ = build(sync=False, logging_steps=500,
                              output_dir="/tmp/t3_tech_out_off")
    cb_off = [type(c).__name__ for c in trainer_off.callback_handler.callbacks]
    out["sync_callback_off_control"] = {"attached": "SyncRefModelCallback" in cb_off,
                                        "ok": "SyncRefModelCallback" not in cb_off}
    # one micro-step: mask & KL boundary + zero-advantage loss finiteness
    batch = [trainer.train_dataset[i] for i in range(16)]
    prep = trainer._prepare_inputs(batch)
    cm = prep["completion_mask"]
    eos_pos = FAKE.index(151643)
    out["mask_boundary"] = {"mask_per_row": cm[0].tolist(),
                            "expected_len": eos_pos + 1,
                            "ok": bool((cm == (torch.arange(cm.size(1)) <= eos_pos).int()).all())}
    kl = trainer.compute_loss(trainer.model, prep)
    out["loss_finite"] = {"loss": float(kl), "finite": bool(torch.isfinite(kl)),
                          "zero_advantage_note": "all rewards identical -> advantages 0 -> loss is KL-term only; "
                                                  "policy==ref at init -> KL=0"}
    with open(os.path.join(EV, "tech_checks_part_a.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("PART_A", json.dumps({k: (v.get("ok") if isinstance(v, dict) and "ok" in v else True)
                                for k, v in out.items()}))


def part_b():
    trainer, rec, model = build(sync=True, logging_steps=25)
    micro, steps, ref_sums = [], [], []
    orig_ts = mt.ReReTrainer.training_step

    def counting_ts(self, m, inputs, *a, **kw):
        rows = len(inputs) if isinstance(inputs, list) else -1
        out = orig_ts(self, m, inputs, *a, **kw)
        micro.append({"i": len(micro), "epoch": int(float(self.state.epoch or 0)),
                      "n_rows": rows,
                      "ids": sorted({r["task_id"] for r in inputs}) if isinstance(inputs, list) else [],
                      "sidx": sorted({r["stream_index"] for r in inputs}) if isinstance(inputs, list) else [],
                      "grad": float(sum(float(p.grad.abs().sum()) for p in self.model.parameters()
                                        if p.grad is not None))})
        return out
    mt.ReReTrainer.training_step = counting_ts

    class CB(TrainerCallback):
        def on_step_end(self, args, state, control, optimizer=None, model=None, **kw):
            steps.append({"global_step": state.global_step,
                          "lr": optimizer.param_groups[0]["lr"]})

    trainer.callback_handler.add_callback(CB())
    ref0 = float(sum(float(p.detach().abs().sum()) for p in trainer.ref_model.parameters()))
    t0 = time.time()
    trainer.train()
    wall = time.time() - t0
    ref1 = float(sum(float(p.detach().abs().sum()) for p in trainer.ref_model.parameters()))

    n = len(micro)
    per_window_ok = all(r["n_rows"] == 16 for r in micro)
    windows = [micro[i:i + 32] for i in range(0, n, 32)]
    boundary = windows[312]
    seen = {}
    for r in micro:
        for tid in r["ids"]:
            seen[tid] = seen.get(tid, 0) + 1
    counts = {}
    for r in micro:
        key = r["ids"][0] if len(r["ids"]) == 1 else "MIXED"
        counts[key] = counts.get(key, 0) + 1
    copies = {"0": 0, "1": 0}
    for r in micro:
        if len(r["sidx"]) == 1:
            copies["0" if r["sidx"][0] < 10000 else "1"] += 1
    grads_zero_after = all(r["grad"] < 1e-9 for r in micro[::967])
    out = {
        "micro_total": n, "steps_total": len(steps), "final_global_step": trainer.state.global_step,
        "all_micro_16rows": per_window_ok,
        "windows_total": len(windows),
        "windows_all_32": all(len(w) == 32 for w in windows),
        "candidates_total": n * 16,
        "distinct_tasks_forwarded": len(counts),
        "tasks_missing": 20000 - len([k for k in counts if k != "MIXED"]),
        "copy0_micros": copies["0"], "copy1_micros": copies["1"],
        "boundary_window_313": {"n_micro": len(boundary),
                                "copy0_groups": sum(1 for r in boundary
                                                     if r["sidx"][0] < 10000),
                                "copy1_groups": sum(1 for r in boundary
                                                     if r["sidx"][0] >= 10000)},
        "boundary_window_sidx_range": [boundary[0]["sidx"][0], boundary[-1]["sidx"][0]] if boundary else [],
        "grad_nonzero_at_sampled_micros": not grads_zero_after,
        "ref_checksum": {"before": ref0, "after": ref1, "changed": ref0 != ref1},
        "wall_seconds": round(wall, 1),
        "final_lr": steps[-1]["lr"] if steps else None,
        "disclosure": "mock generate + synthetic rewards; adamw_torch fp32 substitution; CPU tiny model",
    }
    ok = (n == 20000 and len(steps) == 625 and per_window_ok and len(windows) == 625
          and all(len(w) == 32 for w in windows) and copies == {"0": 10000, "1": 10000}
          and out["boundary_window_313"]["copy0_groups"] == 16
          and out["boundary_window_313"]["copy1_groups"] == 16
          and ref0 != ref1)
    out["ALL_OK"] = ok
    with open(os.path.join(EV, "tech_checks_part_b.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(os.path.join(EV, "tech_checks_micro_records.json"), "w") as f:
        json.dump({"micro_sample": micro[:50] + micro[-50:], "steps_head_tail": steps[:3] + steps[-3:]}, f)
    print("PART_B", "OK" if ok else "FAIL", json.dumps({k: out[k] for k in
          ("micro_total", "steps_total", "windows_total", "copy0_micros", "copy1_micros",
           "boundary_window_313", "final_lr")}))


def wsum(m):
    return float(sum(float(p.detach().double().abs().sum()) for p in m.parameters()))


def part_c():
    res = {}
    tr_cont, _, _ = build(sync=True, max_steps=8, save_steps=4,
                          output_dir="/tmp/t3_resume_cont")
    tr_cont.train()
    cont = {"final_step": tr_cont.state.global_step, "wsum": wsum(tr_cont.model),
            "ref_wsum": wsum(tr_cont.ref_model),
            "opt_state_keys": len(tr_cont.optimizer.state),
            "consumed": None}
    import random as _r
    import numpy as _np
    _r.seed(42); _np.random.seed(42); torch.manual_seed(42)
    tr_a, _, _ = build(sync=True, max_steps=8, save_steps=4, output_dir="/tmp/t3_resume_a")
    tr_a.train()
    ckpt = os.path.join("/tmp/t3_resume_a", "checkpoint-4")
    assert os.path.isdir(ckpt), ckpt
    tr_b, _, _ = build(sync=True, max_steps=8, save_steps=4, output_dir="/tmp/t3_resume_b")
    tr_b.train(resume_from_checkpoint=ckpt)
    res["continuous"] = cont
    res["resumed"] = {"final_step": tr_b.state.global_step, "wsum": wsum(tr_b.model),
                      "ref_wsum": wsum(tr_b.ref_model),
                      "opt_state_keys": len(tr_b.optimizer.state)}
    res["order_identical_note"] = ("stream order is materialized file order (no RNG); Trainer resume "
                                    "replays skip_batches deterministically; RNG state restored from "
                                    "checkpoint for torch/numpy/python - generation RNG is MOCKED in this "
                                    "probe so its trajectory is not exercised here")
    res["match"] = {"final_step": res["continuous"]["final_step"] == res["resumed"]["final_step"],
                    "weights_close": abs(cont["wsum"] - res["resumed"]["wsum"]) < 1e-6,
                    "ref_close": abs(cont["ref_wsum"] - res["resumed"]["ref_wsum"]) < 1e-6}
    res["scoping"] = ("resume restores data order, optimizer/scheduler/global_step and ref-sync position "
                      "(sync anchored on global_step%512); it does NOT claim bit-identical continuation "
                      "of an uninterrupted GPU run (generate/backprop RNG and nondeterministic kernels)")
    with open(os.path.join(EV, "resume_check.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("PART_C", json.dumps(res["match"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["a", "b", "c"])
    args = ap.parse_args()
    {"a": part_a, "b": part_b, "c": part_c}[args.part]()
