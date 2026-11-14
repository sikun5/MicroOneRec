"""Phase 6 T4 smoke reload: new-process load of the smoke checkpoint and
same-caliber eval loss comparison against the recorded final loss."""
import argparse
import json
import os
import sys

import torch

ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

from v2_sft_data import (BASE_MODEL_PATH, EXPECTED_FINAL_VOCAB,  # noqa: E402
                         SidSFTDatasetV2, make_expanded_tokenizer)
from v2_sft_init import set_seed  # noqa: E402
from v2_sft_smoke100 import eval_loss  # noqa: E402  (same caliber code path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--smoke_root", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = os.path.join(args.smoke_root, "overfit_100", args.group)
    ckpt = os.path.join(out_dir, "smoke_ckpt")
    set_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    assert len(tokenizer) == EXPECTED_FINAL_VOCAB
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16,
                                                 local_files_only=True)
    model.config.use_cache = False
    model = model.to("cuda")
    mini = os.path.join(out_dir, f"train100_{args.group}.csv")
    ds = SidSFTDatasetV2(mini, tokenizer, max_len=512)
    samples = [ds[i] for i in range(len(ds))]
    import transformers
    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    reload_loss = eval_loss(model, collator, samples)
    recorded = json.load(open(os.path.join(out_dir, "smoke_result.json")))
    final = recorded["final_eval_loss"]
    rec = {"group": args.group, "reload_eval_loss": reload_loss,
           "recorded_final_eval_loss": final,
           "abs_diff": abs(reload_loss - final),
           "match_tol_1e_4": abs(reload_loss - final) < 1e-4,
           "all_ok": abs(reload_loss - final) < 1e-4}
    with open(os.path.join(out_dir, "reload_eval.json"), "w") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print("SMOKE_RELOAD_OK" if rec["all_ok"] else "SMOKE_RELOAD_DIFF",
          args.group, "reload=", round(reload_loss, 6), "recorded=", round(final, 6))


if __name__ == "__main__":
    main()
