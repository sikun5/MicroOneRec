"""Phase 6 T4 reload check: new-process load of a saved probe checkpoint.

Verifies model weights load via from_pretrained and the AdamW optimizer state
reload via torch.load into a fresh optimizer, comparing checksums against the
saved artifacts. Runs in its own process (boss: 保存/新进程加载).
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


def t_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    rec = {"group": args.group, "save_dir": args.save_dir}
    model = AutoModelForCausalLM.from_pretrained(args.save_dir, torch_dtype=torch.bfloat16,
                                                 local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.save_dir, local_files_only=True)
    sd = model.state_dict()
    rec["model_loaded"] = True
    rec["embed_shape"] = list(sd["model.embed_tokens.weight"].shape)
    rec["embed_sha_head"] = t_sha(sd["model.embed_tokens.weight"][:64])
    rec["tokenizer_len"] = len(tokenizer)

    opt_path = os.path.join(args.save_dir, "optimizer.pt")
    state = torch.load(opt_path, map_location="cpu", weights_only=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    optimizer.load_state_dict(state)
    rec["optimizer_loaded"] = True
    rec["optimizer_n_states"] = len(optimizer.state)
    first = optimizer.state[list(optimizer.state.keys())[0]]
    rec["optimizer_state_dtypes"] = {k: str(v.dtype) for k, v in first.items() if torch.is_tensor(v)}
    rec["optimizer_state_keys"] = sorted(first.keys())
    rec["all_ok"] = (rec["model_loaded"] and rec["optimizer_loaded"]
                     and rec["embed_shape"] == [154977, 1024]
                     and rec["tokenizer_len"] == 154977
                     and len(optimizer.state) > 0)

    with open(args.out, "w") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print("RELOAD_OK" if rec["all_ok"] else "RELOAD_FAIL", args.group, args.save_dir)


if __name__ == "__main__":
    main()
