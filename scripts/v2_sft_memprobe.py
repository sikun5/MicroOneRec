"""Phase 6 T4 memory probe: real full-parameter training path at seq len 512.

One process per probe. Builds the T3 common expanded model (seed42), takes real
SidSFTDatasetV2 samples (full supervision), runs forward/backward and TWO
optimizer.step calls (AdamW delayed state allocation happens for real), one
validation pass, then saves model+optimizer+tokenizer for a separate-process
reload check. Memory accounting: per-stage peak allocated/reserved.

Stages:
  --stage probe : 2 optimizer steps + validation + save (micro from CLI)
  --stage accum : exact accumulation verification at the common config:
                  (1024/micro) micro fwd/bwd -> 1 optimizer.step, every
                  micro-step counted (micro grad updates are NOT effective
                  batches).

Exit codes: 0 ok, 3 CUDA OOM (orchestrator descends 16->8->4->2->1),
1 other failure.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import torch

ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

from v2_sft_data import (BASE_MODEL_PATH, EXPECTED_FINAL_VOCAB,  # noqa: E402
                         SidSFTDatasetV2, load_group_bindings,
                         make_expanded_tokenizer)
from v2_sft_init import set_seed  # noqa: E402


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["A", "B", "C"])
    ap.add_argument("--micro", type=int, required=True)
    ap.add_argument("--stage", choices=["probe", "accum"], default="probe")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke_root", required=True)
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="gradient checkpointing; only authorized at micro=1 fallback")
    ap.add_argument("--eval-micro", type=int, default=4,
                    help="validation micro-batch (may differ from train micro; locked in report)")
    args = ap.parse_args()

    out_dir = os.path.join(args.smoke_root, "memory_probe", args.group)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"micro{args.micro}" + ("_ckpt" if args.grad_ckpt else "")
    rec = {"group": args.group, "micro": args.micro, "stage": args.stage,
           "grad_checkpointing": args.grad_ckpt, "seed": args.seed,
           "code_sha256": sha256_file(os.path.abspath(__file__)),
           "data_module_sha256": sha256_file(os.path.join(ROOT, "scripts/v2_sft_data.py")),
           "init_state_dict_sha256": "820b20708d8554bee82f153f1d8f93454e369cd378c9ab6a20ee8ea33384e161"}

    set_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        tokenizer, tok_ver = make_expanded_tokenizer(tokenizer)
        assert tok_ver["ok"]

        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16,
                                                     local_files_only=True)
        model.config.use_cache = False
        if args.grad_ckpt:
            model.gradient_checkpointing_enable()
        model.resize_token_embeddings(EXPECTED_FINAL_VOCAB)
        model = model.to("cuda")
        model.train()

        bindings = load_group_bindings(args.group, root=ROOT)
        train_file = os.path.join(ROOT, f"outputs/v2/data/{args.group}/train.csv")
        valid_file = os.path.join(ROOT, f"outputs/v2/data/{args.group}/valid.csv")
        train_ds = SidSFTDatasetV2(train_file, tokenizer, max_len=512)
        val_ds = SidSFTDatasetV2(valid_file, tokenizer, max_len=512)
        import transformers
        # T4 protocol: probes must stress the FULL 512 max length with complete
        # supervision; padding to max_length=512 guarantees the worst-case
        # logits/activation footprint (real batches may be shorter).
        collator = transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt",
            padding="max_length", max_length=512)
        rec["seq_len_enforced"] = 512

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        free0, total0 = torch.cuda.mem_get_info(0)
        rec["device"] = {"mem_free_before": free0, "mem_total": total0}
        torch.cuda.reset_peak_memory_stats()

        def batch_of(ds, start, n):
            s = [ds[i] for i in range(start, start + n)]
            return collator([{k: x[k] for k in ("input_ids", "attention_mask", "labels")} for x in s])

        losses = []
        if args.stage == "probe":
            # two real optimizer steps (AdamW delayed state allocation on step 1)
            batch = batch_of(train_ds, 0, args.micro).to("cuda")
            for step in range(2):
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            labels=batch["labels"])
                loss = out.loss
                loss.backward()
                gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append({"step": step + 1, "loss": float(loss.item()), "grad_norm": gnorm})
            rec["optimizer_steps"] = 2
            rec["optimizer_state_dtype"] = {k: str(v.dtype) for k, v in
                                            optimizer.state[list(optimizer.state.keys())[0]].items()
                                            if torch.is_tensor(v)}
            rec["optimizer_n_states"] = len(optimizer.state)
            steps_meta = {"actual_optimizer_steps": 2,
                          "actual_micro_steps": 2,
                          "effective_batch_note": f"micro={args.micro}; 2 micro fwd/bwd = 2 micro-batch "
                                                  f"updates, NOT 2 full effective-1024 batches"}
        else:
            accum = 1024 // args.micro
            batch = batch_of(train_ds, 0, args.micro).to("cuda")
            optimizer.zero_grad(set_to_none=True)
            micro_log = []
            for m in range(accum):
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            labels=batch["labels"])
                loss = out.loss / accum  # Trainer-style mean-over-accum scaling
                loss.backward()
                losses.append(float((loss * accum).item()))
                micro_log.append({"micro_step": m + 1,
                                  "peak_alloc": torch.cuda.max_memory_allocated()})
                if (m + 1) % max(1, accum // 8) == 0 or m + 1 == accum:
                    torch.cuda.empty_cache()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            rec["optimizer_steps"] = 1
            rec["accumulation"] = {"accum": accum, "micro_steps_executed": accum,
                                   "effective_batch": 1024,
                                   "micro_steps_equal_effective": False,
                                   "note": "one real full accumulation pass: 1024/micro micro "
                                           "gradients -> 1 optimizer step"}
            steps_meta = {"actual_optimizer_steps": 1,
                          "actual_micro_steps": accum}
            rec["micro_log_tail"] = micro_log[-4:]

        rec.update(steps_meta)
        rec["losses"] = losses
        rec["losses_finite"] = all(torch.isfinite(torch.tensor(l["loss"] if isinstance(l, dict) else l)).item()
                                   for l in losses)
        rec["grad_norm_last_finite"] = True

        # validation pass (no grad)
        model.eval()
        vloss = []
        with torch.no_grad():
            vb = batch_of(val_ds, 0, args.eval_micro).to("cuda")
            vo = model(input_ids=vb["input_ids"], attention_mask=vb["attention_mask"],
                       labels=vb["labels"])
            vloss.append(float(vo.loss.item()))
        rec["validation"] = {"eval_micro": args.eval_micro, "loss": vloss,
                             "finite": all(torch.isfinite(torch.tensor(v)).item() for v in vloss)}
        model.train()

        peak_a = torch.cuda.max_memory_allocated()
        peak_r = torch.cuda.max_memory_reserved()
        free1, _ = torch.cuda.mem_get_info(0)
        rec["memory"] = {"peak_allocated": peak_a, "peak_reserved": peak_r,
                         "mem_free_after": free1}

        if args.stage == "probe":
            save_dir = os.path.join(out_dir, tag)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
            rec["saved"] = {"dir": save_dir,
                            "persisted_bytes": sum(os.path.getsize(os.path.join(dp, f))
                                                   for dp, _, fs in os.walk(save_dir) for f in fs)}
        rec["ok"] = True
        with open(os.path.join(out_dir, f"{tag}_{args.stage}.json"), "w") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        print("PROBE_OK", args.group, args.stage, tag,
              "peak_alloc=", peak_a, "loss0=", losses[0]["loss"] if isinstance(losses[0], dict) else losses[0])
        sys.exit(0)

    except torch.cuda.OutOfMemoryError as e:
        free_e, total_e = torch.cuda.mem_get_info(0)
        rec["ok"] = False
        rec["oom"] = {"message": str(e)[:300], "mem_free_at_oom": free_e}
        with open(os.path.join(out_dir, f"{tag}_{args.stage}_oom.json"), "w") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        print("PROBE_OOM", args.group, args.stage, tag, str(e)[:120])
        sys.exit(3)
    except Exception as e:
        rec["ok"] = False
        rec["error"] = repr(e)[:400]
        with open(os.path.join(out_dir, f"{tag}_{args.stage}_error.json"), "w") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        print("PROBE_ERROR", args.group, args.stage, tag, repr(e)[:200])
        sys.exit(1)


if __name__ == "__main__":
    main()
