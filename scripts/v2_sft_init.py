"""Phase 6 controlled initialization builder (one process per group).

Builds the common expanded model for one group in a fresh process:
seed42 -> tokenizer add_tokens(shared_vocab order) -> base model (bf16) ->
resize_token_embeddings(154977). Dumps a verification JSON including the full
state_dict SHA-256, parameter counts, tie status, and the 267-legacy-pad-row
strategy evidence. Three groups must produce identical hashes.
"""
import argparse
import hashlib
import json
import os
import sys

import torch

ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from v2_sft_data import (BASE_MODEL_PATH, EXPECTED_FINAL_VOCAB,  # noqa: E402
                         make_expanded_tokenizer)


def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hash_state_dict(model):
    h = hashlib.sha256()
    sd = model.state_dict()
    keys = sorted(sd.keys())
    for k in keys:
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest(), len(keys)


def hash_rows(model, start, end):
    """SHA-256 of embedding rows [start:end) (uint8 reinterpreted bytes)."""
    w = model.get_input_embeddings().weight.detach().cpu().contiguous()
    seg = w[start:end]
    return hashlib.sha256(seg.view(torch.uint8).numpy().tobytes()).hexdigest()


def base_embed_rows_hash(start, end):
    """Rows of the ORIGINAL base safetensors embedding (no model load)."""
    from safetensors import safe_open
    p = os.path.join(BASE_MODEL_PATH, "model.safetensors")
    with safe_open(p, framework="pt") as f:
        for k in f.keys():
            if "embed_tokens" in k:
                w = f.get_tensor(k)[start:end].contiguous()
                return hashlib.sha256(w.view(torch.uint8).numpy().tobytes()).hexdigest(), k
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["A", "B", "C"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True,
                                              trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer, tok_ver = make_expanded_tokenizer(tokenizer)
    assert tok_ver["ok"], f"tokenizer expansion failed: {tok_ver}"

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16,
                                                 local_files_only=True)
    model.config.use_cache = False
    model.resize_token_embeddings(EXPECTED_FINAL_VOCAB)

    # full-parameter: everything requires grad by default; assert explicitly
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    tied_ptrs = (lm_head is not None and lm_head.weight.data_ptr() == embed.weight.data_ptr())

    sd_sha, n_keys = hash_state_dict(model)
    # 267-row strategy evidence: rows [151669,151936) are the base model's own
    # legacy padding rows (kept untouched); rows [151936,154977) are the 3041
    # newly initialized rows (3308 tokens - 267 already-existing rows).
    base_legacy_sha, base_key = base_embed_rows_hash(151669, 151936)
    rec = {
        "group": args.group,
        "seed": args.seed,
        "base_model": BASE_MODEL_PATH,
        "tokenizer": tok_ver,
        "embed_shape": list(embed.weight.shape),
        "embed_dtype": str(embed.weight.dtype),
        "param_total": n_total,
        "param_trainable": n_trainable,
        "all_requires_grad": all(p.requires_grad for p in model.parameters()),
        "tie_word_embeddings_config": bool(model.config.tie_word_embeddings),
        "lm_head_tied_by_ptr": bool(tied_ptrs),
        "state_dict_sha256": sd_sha,
        "state_dict_n_keys": n_keys,
        "row_strategy": {
            "model_vocab_original": 151936,
            "base_tokenizer_len": 151669,
            "legacy_pad_rows": [151669, 151936],
            "legacy_pad_rows_hash_post_resize": hash_rows(model, 151669, 151936),
            "legacy_pad_rows_hash_base_safetensors": base_legacy_sha,
            "legacy_pad_rows_untouched": None,
            "new_rows": [151936, 154977],
            "new_rows_n": 154977 - 151936,
            "new_rows_hash": hash_rows(model, 151936, 154977),
            "policy": "keep original 151936 rows (incl. 267 model pad rows) unchanged; "
                      "append 3041 new rows initialized under seed42; tokenizer rows "
                      "0..151668 unchanged, 151669..154976 = 3308 common tokens",
        },
    }
    rec["row_strategy"]["legacy_pad_rows_untouched"] = (
        rec["row_strategy"]["legacy_pad_rows_hash_post_resize"] == base_legacy_sha)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print("INIT_OK", args.group, rec["state_dict_sha256"][:16],
          "untouched=", rec["row_strategy"]["legacy_pad_rows_untouched"])


if __name__ == "__main__":
    main()
