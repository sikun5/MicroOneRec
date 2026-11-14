"""Phase 6 controlled SFT training entry (v2).

Controlled replacement for the original `sft.py` train() (left untouched).
Differences vs the original entry (recorded):
1. Common-vocab initialization: tokenizer expands with shared_vocab.json in
   frozen order, verified against token_id_map.json entry-by-entry; the
   original TokenExtender (per-group occurrence-sorted registration) is NOT
   used - forbidden by protocol.
2. resize_token_embeddings(154977) after expansion; full-parameter training
   (freeze_LLM is fixed False; no PEFT/LoRA/QLoRA code paths).
3. Datasets come from scripts/v2_sft_data.py V2 classes (per-item title2sid,
   explicit source IDs, canonical order) instead of the legacy classes.
4. Runtime meta (model revision, hashes of code/data/index, token verification,
   task counts, eval/save semantics) is written to output_dir/run_meta.json
   BEFORE training starts.
Everything else follows the original training semantics: Trainer with
bf16, adamw_torch, warmup_steps=20 + default linear schedule, eval/save every
5% of planned total steps, save_total_limit=1, load_best_model_at_end=True
(default metric eval_loss, lower=better), EarlyStoppingCallback(patience=3),
DataCollatorForSeq2Seq(-100 label padding), train = ConcatDataset(3 tasks)
shuffled once with seed42, validation = main SID recommendation task only.
"""
import hashlib
import json
import os
import random
import sys

import argparse
import numpy as np
import torch
import transformers
from datasets import Dataset as HFDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback

ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from v2_sft_data import (BASE_MODEL_PATH, FusionSeqRecDatasetV2,  # noqa: E402
                         SidItemFeatDatasetV2, SidSFTDatasetV2,
                         load_group_bindings, make_expanded_tokenizer)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train(
    base_model: str = BASE_MODEL_PATH,
    train_file: str = "",
    eval_file: str = "",
    output_dir: str = "",
    seed: int = 42,
    batch_size: int = 1024,
    micro_batch_size: int = 16,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    category: str = "Industrial_and_Scientific",
    output_root: str = ROOT,
    eval_step: float = 0.05,  # fraction of planned total steps (original semantic)
    save_total_limit: int = 1,
    warmup_steps: int = 20,
    early_stopping_patience: int = 3,
    eval_micro: int = 4,  # validation batch locked separately (T4: 4; may differ from train micro)
):
    assert output_dir, "--output_dir required"
    assert not os.path.abspath(output_dir).startswith(
        os.path.join(ROOT, "outputs/v2/debug")), "formal training must not write into debug root"
    set_seed(seed)

    group = os.path.basename(os.path.dirname(train_file.rstrip("/")))
    assert group in ("A", "B", "C"), f"cannot infer group from train_file: {train_file}"
    bindings = load_group_bindings(group, root=output_root)

    # ---- tokenizer + common vocab (controlled path; TokenExtender NOT used)
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True,
                                              trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer, tok_ver = make_expanded_tokenizer(tokenizer)
    assert tok_ver["ok"], f"common vocab expansion failed: {tok_ver}"

    # ---- model: full-parameter bf16
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16,
                                                 local_files_only=True)
    model.config.use_cache = False
    model.resize_token_embeddings(len(tokenizer))
    assert all(p.requires_grad for p in model.parameters()), "non-trainable parameter found"
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # runtime in-process init state hash (T6+ persistence improvement over A,
    # where only a same-seed same-path reconstruction was available)
    _h = hashlib.sha256()
    _sd = model.state_dict()
    for _k in sorted(_sd.keys()):
        _t = _sd[_k].detach().cpu().contiguous()
        _h.update(_k.encode())
        _h.update(str(tuple(_t.shape)).encode())
        _h.update(str(_t.dtype).encode())
        _h.update(_t.view(torch.uint8).numpy().tobytes())
    init_state_sha256 = _h.hexdigest()
    del _h, _sd

    # ---- runtime meta BEFORE training
    os.makedirs(output_dir, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    meta = {
        "group": group,
        "seed": seed,
        "base_model": base_model,
        "tokenizer": tok_ver,
        "param_total": n_total,
        "param_trainable": n_trainable,
        "train_file": train_file,
        "train_file_sha256": sha256_file(train_file),
        "eval_file": eval_file,
        "eval_file_sha256": sha256_file(eval_file),
        "index_sha256": sha256_file(bindings["paths"]["index"]),
        "item_sha256": sha256_file(bindings["paths"]["item"]),
        "reverse_sha256": sha256_file(bindings["paths"]["reverse"]),
        "shared_vocab_sha256": sha256_file(os.path.join(ROOT, "outputs/v2/tokenizer/shared_vocab.json")),
        "token_id_map_sha256": sha256_file(os.path.join(ROOT, "outputs/v2/tokenizer/token_id_map.json")),
        "entry_sha256": sha256_file(os.path.abspath(__file__)),
        "data_module_sha256": sha256_file(os.path.join(here, "v2_sft_data.py")),
        "init_state_sha256_runtime": init_state_sha256,
        "training_semantics": {
            "effective_batch": batch_size,
            "micro_batch": micro_batch_size,
            "grad_accum": batch_size // micro_batch_size,
            "lr": learning_rate,
            "optimizer": "adamw_torch",
            "scheduler": "transformer default linear with warmup_steps=20 (original custom cosine defined but never passed)",
            "warmup_steps": warmup_steps,
            "epochs_max": num_epochs,
            "cutoff": cutoff_len,
            "eval_fraction": eval_step,
            "save_strategy": "steps@same 5% fraction; save_total_limit=1",
            "best_model": "load_best_model_at_end=True, metric=eval_loss, greater_is_better=False",
            "early_stopping_patience": early_stopping_patience,
            "train_shuffle": "HF Dataset.shuffle(seed=42) once over ConcatDataset(sid_sft,item_feat,fusion)",
            "validation_task": "main SID recommendation only (SidSFTDatasetV2 on eval_file)",
            "eval_micro": eval_micro,
            "config_note": "micro/accum = T4-frozen common feasible config (micro2, accum=512 for effective 1024)",
            "collator": "DataCollatorForSeq2Seq(pad -100, pad_to_multiple_of=8)",
            "wandb": "report_to=None",
        },
    }
    with open(os.path.join(output_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    # ---- datasets (three tasks concatenated; validation = main task only)
    train_sets = [
        SidSFTDatasetV2(train_file, tokenizer, max_len=cutoff_len),
        SidItemFeatDatasetV2(bindings, tokenizer, max_len=cutoff_len),
        FusionSeqRecDatasetV2(train_file, bindings, tokenizer, max_len=cutoff_len),
    ]
    counts = [len(s) for s in train_sets]
    meta["task_counts"] = {"sid_sft": counts[0], "item_feat": counts[1], "fusion_seqrec": counts[2]}
    with open(os.path.join(output_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    from torch.utils.data import ConcatDataset
    train_concat = ConcatDataset(train_sets)
    keys = list(train_concat[0].keys())
    hf_train = HFDataset.from_dict({k: [s[k] for s in train_concat] for k in keys})
    hf_train = hf_train.shuffle(seed=42)
    val = SidSFTDatasetV2(eval_file, tokenizer, max_len=cutoff_len)
    hf_val = HFDataset.from_dict({
        "input_ids": [s["input_ids"] for s in val.samples],
        "attention_mask": [s["attention_mask"] for s in val.samples],
        "labels": [s["labels"] for s in val.samples],
    })

    gradient_accumulation_steps = batch_size // micro_batch_size
    trainer = transformers.Trainer(
        model=model,
        train_dataset=hf_train,
        eval_dataset=hf_val,
        args=transformers.TrainingArguments(
            run_name=f"{group}-SFT",
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=eval_micro,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="steps",
            eval_steps=eval_step,
            save_strategy="steps",
            save_steps=eval_step,
            output_dir=output_dir,
            save_total_limit=save_total_limit,
            load_best_model_at_end=True,
            group_by_length=False,
            report_to=None,
            seed=seed,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    state = trainer.state
    vram_peak = torch.cuda.max_memory_allocated()
    with open(os.path.join(output_dir, "training_log_history.json"), "w") as f:
        json.dump({"group": group, "log_history": state.log_history,
                   "global_step": state.global_step,
                   "max_steps": state.max_steps,
                   "best_metric": state.best_metric,
                   "best_model_checkpoint": state.best_model_checkpoint,
                   "note": "full Trainer log_history persisted at process exit (T6+ improvement: complete validation/loss trajectory survives checkpoint rotation and stdout buffering)"},
                  f, ensure_ascii=False, indent=1)
    selection = {
        "group": group,
        "total_steps_planned_raw": state.max_steps,
        "actual_global_steps": state.global_step,
        "best_metric": state.best_metric,
        "best_model_checkpoint": state.best_model_checkpoint,
        "load_best_model_at_end": True,
        "task_counts": meta["task_counts"],
        "init_state_sha256_runtime": meta["init_state_sha256_runtime"],
        "vram_peak_allocated_runtime": vram_peak,
    }
    with open(os.path.join(output_dir, "selection_manifest.json"), "w") as f:
        json.dump(selection, f, ensure_ascii=False, indent=1)
    print("SFT_DONE", group, "best=", state.best_metric, "ckpt=", state.best_model_checkpoint)


def _cli():
    ap = argparse.ArgumentParser(description="Phase 6 controlled SFT entry (v2)")
    ap.add_argument("--base_model", default=BASE_MODEL_PATH)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--eval_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--micro_batch_size", type=int, default=16)
    ap.add_argument("--num_epochs", type=int, default=10)
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--cutoff_len", type=int, default=512)
    ap.add_argument("--category", default="Industrial_and_Scientific")
    ap.add_argument("--output_root", default=ROOT)
    ap.add_argument("--eval_step", type=float, default=0.05)
    ap.add_argument("--save_total_limit", type=int, default=1)
    ap.add_argument("--warmup_steps", type=int, default=20)
    ap.add_argument("--early_stopping_patience", type=int, default=3)
    ap.add_argument("--eval_micro", type=int, default=4)
    a = ap.parse_args()
    train(**vars(a))


if __name__ == "__main__":
    _cli()
