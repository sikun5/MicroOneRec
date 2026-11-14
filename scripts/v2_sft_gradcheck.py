"""Phase 6 T3 gradient check: real single forward+backward per group (debug).

Debug evidence lives under outputs/v2/debug/phase6/sft_smoke/attempt_001/.
For each group (fresh process): build the common expanded model (seed42),
take real samples from each of the three task datasets, pad with the same
collator used by training, run forward+backward ONCE, and verify:
- every parameter requires_grad (full-parameter training)
- loss is finite
- representative gradients are finite and non-zero: new-token embedding rows,
  a base embedding row, sampled attention (q/k/v/o) and MLP (up/down) weights
No optimizer.step is run (that belongs to T4 stress probes).
"""
import argparse
import hashlib
import json
import os
import sys

import torch

ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

from v2_sft_data import (BASE_MODEL_PATH, EXPECTED_FINAL_VOCAB,  # noqa: E402
                         FusionSeqRecDatasetV2, SidItemFeatDatasetV2,
                         SidSFTDatasetV2, load_group_bindings,
                         make_expanded_tokenizer)
from v2_sft_init import set_seed  # noqa: E402


def grad_norm(t):
    if t.grad is None:
        return None
    g = t.grad.detach().float()
    n = g.norm().item()
    finite = bool(torch.isfinite(g).all().item())
    return {"norm": n, "finite": finite, "nonzero": n > 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["A", "B", "C"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke_root", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = os.path.join(args.smoke_root, "gradient_checks")
    os.makedirs(out_dir, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer, tok_ver = make_expanded_tokenizer(tokenizer)
    assert tok_ver["ok"]

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16,
                                                 local_files_only=True)
    model.config.use_cache = False
    model.resize_token_embeddings(EXPECTED_FINAL_VOCAB)
    model = model.to("cuda")  # single-device probe (MIG); forward+backward on GPU
    model.train()

    bindings = load_group_bindings(args.group, root=ROOT)
    train_file = os.path.join(ROOT, f"outputs/v2/data/{args.group}/train.csv")

    n_per_task = 2
    item_full = SidItemFeatDatasetV2(bindings, tokenizer, max_len=512)
    n_sid2title = 2 * 3686 // 2  # 3686 sid2title samples precede title2sid block
    tasks = {
        "sid_sft": SidSFTDatasetV2(train_file, tokenizer, max_len=512)[:n_per_task],
        "item_feat": [item_full[0], item_full[1],
                      item_full[3686], item_full[3687]],  # sid2title x2 + canonical title2sid x2
        "fusion_seqrec": FusionSeqRecDatasetV2(train_file, bindings, tokenizer, max_len=512)[:n_per_task],
    }
    del item_full
    collator = __import__("transformers").DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

    results = {"group": args.group, "seed": args.seed, "tasks": {}}
    all_req = all(p.requires_grad for p in model.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results["params"] = {"total": n_total, "trainable": n_trainable,
                         "all_requires_grad": all_req}

    losses = []
    for task_name, samples in tasks.items():
        batch = collator([{k: s[k] for k in ("input_ids", "attention_mask", "labels")} for s in samples])
        model.zero_grad(set_to_none=True)
        out = model(input_ids=batch["input_ids"].cuda(),
                    attention_mask=batch["attention_mask"].cuda(),
                    labels=batch["labels"].cuda())
        loss = out.loss
        loss.backward()
        embed = model.get_input_embeddings().weight
        # row-level gradients must be read from the accumulated FULL-weight grad
        # (row slices share storage and carry no .grad of their own); probe rows
        # actually consumed by this batch (boss: unused rows may legitimately be zero)
        ids_in_batch = sorted({t for row in batch["input_ids"].tolist() for t in row})
        used_new = [t for t in ids_in_batch if t >= 151669][:2]
        used_base = [t for t in ids_in_batch if t < 151669][:2]
        eg = embed.grad
        def row_grad(idx):
            if eg is None or idx is None:
                return None
            g = eg[idx].detach().float()
            n = g.norm().item()
            return {"row": idx, "norm": n, "finite": bool(torch.isfinite(g).all().item()),
                    "nonzero": n > 0.0}
        probe = {"loss": float(loss.item()),
                 "loss_finite": bool(torch.isfinite(loss).item()),
                 "grad_new_token_rows": [row_grad(t) for t in used_new],
                 "grad_base_token_rows": [row_grad(t) for t in used_base]}
        # representative attention/MLP rows across a few layers
        layers = model.model.layers
        for li in (0, 14, 27):
            blk = layers[li]
            attn = blk.self_attn
            mlp = blk.mlp
            probe[f"L{li}_q"] = grad_norm(attn.q_proj.weight)
            probe[f"L{li}_k"] = grad_norm(attn.k_proj.weight)
            probe[f"L{li}_v"] = grad_norm(attn.v_proj.weight)
            probe[f"L{li}_o"] = grad_norm(attn.o_proj.weight)
            probe[f"L{li}_mlp_up"] = grad_norm(mlp.up_proj.weight)
            probe[f"L{li}_mlp_down"] = grad_norm(mlp.down_proj.weight)
        # embedding row-level: which new-token rows got nonzero grad (samples use few)
        rows_nonzero = int((embed.grad.abs().float().sum(dim=1) > 0).sum().item()) if embed.grad is not None else None
        probe["embedding_rows_with_nonzero_grad"] = rows_nonzero
        probe["all_representative_ok"] = (
            probe["loss_finite"]
            and len(probe["grad_new_token_rows"]) >= 1
            and all(v and v["finite"] and v["nonzero"] for v in probe["grad_new_token_rows"])
            and all(v and v["finite"] and v["nonzero"] for v in probe["grad_base_token_rows"]))
        losses.append(loss.item())
        results["tasks"][task_name] = probe
        model.zero_grad(set_to_none=True)

    results["losses"] = losses
    results["losses_finite"] = all(bool(torch.isfinite(torch.tensor(l)).item()) for l in losses)
    results["all_ok"] = (results["losses_finite"] and all_req
                         and all(t["all_representative_ok"] for t in results["tasks"].values()))

    with open(os.path.join(out_dir, f"{args.group}_gradcheck.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("GRADCHECK_OK", args.group, "all_ok=", results["all_ok"], "losses=", losses)


if __name__ == "__main__":
    main()
