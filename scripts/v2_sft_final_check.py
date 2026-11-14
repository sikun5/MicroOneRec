#!/usr/bin/env python3
"""Phase 6 T5 final checks for one group.

1. Selection consistency: the saved top-level model.safetensors (weights after
   load_best_model_at_end) must hash-equal the best checkpoint's weights
   (checkpoint-429/model.safetensors).
2. Fresh-process reload of the selected checkpoint + strict-trie constrained
   generation (4 SID atoms then EOS only, T4-accepted form) over a fixed
   validation list (first 100 rows of valid.csv, deterministic). No test data.
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

from v2_generation_adapter import validate_sid_sequence  # noqa: E402
from v2_sft_data import (SID_INSTRUCTION, Tokenizer,  # noqa: E402
                         load_group_bindings, make_expanded_tokenizer,
                         _generate_prompt)
from v2_sft_init import set_seed  # noqa: E402

EOS, NL = 151643, 198
N_VAL_CHECK = 100


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_prompt_ids(tokw, row):
    hist = eval(row["history_item_sid"]) if isinstance(row["history_item_sid"], str) else []
    history = ", ".join(hist)
    user_input = (f"The user has interacted with items {history} in chronological order. "
                  f"Can you predict the next possible item that the user may expect?")
    pre = tokw.encode(SID_INSTRUCTION, bos=True, eos=False)
    return pre + tokw.encode(_generate_prompt(user_input), bos=False, eos=False)


class StrictTrieProcessor:
    def __init__(self, tokw, bindings):
        self.eos = tokw.eos_id
        trie = {}
        for toks in bindings["token_ids"]:
            node = trie
            for t in toks:
                node = node.setdefault(t, {})
            node[self.eos] = {}
        self.trie = trie
        self.prompt_len = None

    def __call__(self, input_ids, scores):
        pl = self.prompt_len
        for row in range(input_ids.shape[0]):
            prefix = input_ids[row, pl:].tolist()
            node = self.trie
            ok = True
            for t in prefix:
                if t in node:
                    node = node[t]
                else:
                    ok = False
                    break
            allowed = set(node.keys()) if ok else set()
            mask = torch.full_like(scores[row], float("-inf"))
            if allowed:
                idx = torch.tensor(sorted(allowed), device=scores.device, dtype=torch.long)
                mask[idx] = 0.0
            scores[row] = scores[row] + mask
        return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["A", "B", "C"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.out_dir, "selection_manifest.json")))
    best_dir = manifest.get("best_model_checkpoint", "")
    rec = {"group": args.group, "selection_manifest": manifest,
           "code_sha256": sha256_file(os.path.abspath(__file__))}

    # 1. selection consistency
    top_weights = os.path.join(args.out_dir, "model.safetensors")
    best_weights = os.path.join(best_dir, "model.safetensors")
    rec["top_model_sha256"] = sha256_file(top_weights)
    rec["best_ckpt_model_sha256"] = sha256_file(best_weights)
    rec["selection_consistent"] = rec["top_model_sha256"] == rec["best_ckpt_model_sha256"]
    rec["persisted_bytes_top"] = os.path.getsize(top_weights)
    du = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(args.out_dir) for f in fs)
    rec["persisted_bytes_total_dir"] = du

    # 2. fresh reload + strict generation on fixed validation list
    set_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pandas as pd

    tokenizer = AutoTokenizer.from_pretrained(args.out_dir, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    assert len(tokenizer) == 154977
    tokw = Tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.out_dir, torch_dtype=torch.bfloat16,
                                                 local_files_only=True)
    model.config.use_cache = False
    model = model.to("cuda")
    model.eval()

    bindings = load_group_bindings(args.group, root=ROOT)
    valid = pd.read_csv(os.path.join(ROOT, f"outputs/v2/data/{args.group}/valid.csv"))
    rows = valid.head(N_VAL_CHECK)
    tok_map = json.load(open(os.path.join(ROOT, "outputs/v2/tokenizer/token_id_map.json")))["map"]
    atomic_ids = set(tok_map.values())
    proc = StrictTrieProcessor(tokw, bindings)

    results, n_strict, hits, nl_tried = [], 0, 0, 0
    for _, row in rows.iterrows():
        prompt_ids = build_prompt_ids(tokw, {"history_item_sid": row["history_item_sid"]})
        item_id = str(row["item_id"])
        target_toks = bindings["token_ids"][bindings["item_ids"].index(item_id)]
        proc.prompt_len = len(prompt_ids)
        gen = model.generate(
            input_ids=torch.tensor([prompt_ids], device="cuda"),
            attention_mask=torch.ones((1, len(prompt_ids)), device="cuda", dtype=torch.long),
            max_new_tokens=6, num_beams=10, do_sample=False, num_return_sequences=1,
            logits_processor=[proc], pad_token_id=EOS, length_penalty=0.0)
        out_ids = gen[0, len(prompt_ids):].tolist()
        ok, msg = validate_sid_sequence(out_ids, EOS, atomic_ids)
        n_strict += int(bool(ok))
        nl_tried += int(NL in out_ids)
        match = out_ids[:4] == list(target_toks)
        hits += int(match)
        results.append({"row_id": str(row["row_id"]), "raw_token_ids": out_ids,
                        "strict_legal": bool(ok), "match": match,
                        "target_token_ids": list(target_toks)})
    n = len(results)
    rec["validation_list"] = {"source": f"outputs/v2/data/{args.group}/valid.csv head({N_VAL_CHECK})",
                               "n": n, "deterministic": "fixed head rows, beam10 deterministic"}
    rec["strict_legal_4sid_eos"] = n_strict
    rec["strict_legal_rate"] = n_strict / n
    rec["newline_in_outputs"] = nl_tried
    rec["hits_hit_at_1"] = hits
    rec["phase5_validator"] = "v2_generation_adapter.validate_sid_sequence (SHA 67be20d4...)"
    rec["examples"] = {"first_hit": next((r for r in results if r["match"]), None),
                        "first_non_hit": next((r for r in results if not r["match"]), None)}
    rec["all_generation_raw"] = results
    rec["all_ok"] = (rec["selection_consistent"] and n_strict == n)
    with open(os.path.join(args.out_dir, "final_check.json"), "w") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print("FINAL_CHECK_OK" if rec["all_ok"] else "FINAL_CHECK_ISSUES",
          args.group, "sel_consistent=", rec["selection_consistent"],
          f"strict={n_strict}/{n} hits={hits}/{n} nl={nl_tried}")


if __name__ == "__main__":
    main()
