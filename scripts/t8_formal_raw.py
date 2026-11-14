#!/usr/bin/env python3
"""Phase 6 T8-R2: persist per-sample raw completions of the FORMAL checkpoints.

The T5-T7 final_check runs did not persist per-sample raw token ids (only
summary numbers + 2 examples). Per the boss's T8-R2 authorization, this
script REGENERATES the fixed validation list (valid.csv head(100)) with the
selected (top-level = best) checkpoint using the identical deterministic
pipeline (seed42, strict trie, beam10), persists ALL 100 raw completions per
group, and re-validates them against an INDEPENDENT reference (valid.csv
item_id -> group index token_ids, not final_check's embedded targets).
Cross-checks against the recorded final_check summary numbers. No retraining.
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
                         load_group_bindings, _generate_prompt)
from v2_sft_init import set_seed  # noqa: E402

EOS, NL = 151643, 198


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

    set_seed(args.seed)
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer

    bindings = load_group_bindings(args.group, root=ROOT)
    tokenizer = AutoTokenizer.from_pretrained(args.out_dir, local_files_only=True)
    assert len(tokenizer) == 154977
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokw = Tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.out_dir, torch_dtype=torch.bfloat16,
                                                 local_files_only=True)
    model.config.use_cache = False
    model = model.to("cuda")
    model.eval()

    valid = pd.read_csv(os.path.join(ROOT, f"outputs/v2/data/{args.group}/valid.csv")).head(100)
    tok_map = json.load(open(os.path.join(ROOT, "outputs/v2/tokenizer/token_id_map.json")))["map"]
    atomic_ids = set(tok_map.values())
    proc = StrictTrieProcessor(tokw, bindings)

    results, n_strict, hits = [], 0, 0
    for i, (_, row) in enumerate(valid.iterrows()):
        prompt_ids = build_prompt_ids(tokw, {"history_item_sid": row["history_item_sid"]})
        item_id = str(row["item_id"])
        target_toks = bindings["token_ids"][bindings["item_ids"].index(item_id)]  # independent reference
        proc.prompt_len = len(prompt_ids)
        gen = model.generate(
            input_ids=torch.tensor([prompt_ids], device="cuda"),
            attention_mask=torch.ones((1, len(prompt_ids)), device="cuda", dtype=torch.long),
            max_new_tokens=6, num_beams=10, do_sample=False, num_return_sequences=1,
            logits_processor=[proc], pad_token_id=EOS, length_penalty=0.0)
        out_ids = gen[0, len(prompt_ids):].tolist()
        ok, msg = validate_sid_sequence(out_ids, EOS, atomic_ids)
        n_strict += int(bool(ok))
        match = out_ids[:4] == target_toks
        hits += int(match)
        results.append({"row_id": str(row["row_id"]), "item_id": item_id,
                         "raw_token_ids": out_ids, "phase5_validate": msg,
                         "strict_legal": bool(ok), "match": match,
                         "target_token_ids_independent": target_toks})

    fc = json.load(open(os.path.join(args.out_dir, "final_check.json")))
    rec = {"stage": "t8-formal-valid-raw-persist", "group": args.group, "seed": args.seed,
           "checkpoint": args.out_dir,
           "code_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
           "note": ("T5-T7 final_check did not persist per-sample raw ids; this regeneration uses the "
                     "identical deterministic pipeline (seed42, strict trie, beam10, same fixed valid "
                     "head(100)) authorized by boss T8-R2; no retraining. Cross-checked against the "
                     "recorded final_check summary."),
           "n_samples": len(results),
           "strict_legal_4sid_eos": n_strict,
           "strict_legal_rate": n_strict / len(results),
           "hits_hit_at_1": hits,
           "cross_check_vs_final_check": {
               "final_check_strict_legal": fc["strict_legal_4sid_eos"],
               "final_check_hits": fc["hits_hit_at_1"],
               "strict_match": n_strict == fc["strict_legal_4sid_eos"],
               "hits_match": hits == fc["hits_hit_at_1"]},
           "reference": "independent: valid.csv row item_id -> group index token_ids",
           "all_raw": results}
    rec["all_ok"] = (n_strict == len(results)
                      and rec["cross_check_vs_final_check"]["strict_match"]
                      and rec["cross_check_vs_final_check"]["hits_match"])
    with open(os.path.join(args.out_dir, "formal_valid_raw_outputs.json"), "w") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print("FORMAL_RAW_OK", args.group, f"strict={n_strict}/100 hits={hits}/100 "
          f"cross={rec['all_ok']}")


if __name__ == "__main__":
    main()
