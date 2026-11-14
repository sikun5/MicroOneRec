#!/usr/bin/env python3
"""Phase 6 T4 strict-decode regeneration (boss supplement: real 4-SID+EOS).

Regenerates the 100 smoke prompts from the ALREADY-SAVED smoke checkpoints
(no retraining) with the trie constraint modified to the strict Phase-5
contract shape: after the 4 SID atomic tokens the ONLY continuation allowed
is EOS (newline excluded). Reports the real model output distribution:
how many completions are exactly [4 atoms, EOS], hits, and Phase-5 validator
results on the raw generations (no post-hoc stripping).
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
    """Trie over the group SID token sequences; after 4 atoms ONLY EOS allowed
    (newline forbidden) - the strict Phase-5 candidate shape."""

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
    ap.add_argument("--smoke_root", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = os.path.join(args.smoke_root, "overfit_100", args.group)
    set_seed(args.seed)
    import pandas as pd
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    bindings = load_group_bindings(args.group, root=ROOT)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(out_dir, "smoke_ckpt"),
                                              local_files_only=True)
    assert len(tokenizer) == 154977
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokw = Tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(os.path.join(out_dir, "smoke_ckpt"),
                                                 torch_dtype=torch.bfloat16,
                                                 local_files_only=True)
    model.config.use_cache = False
    model = model.to("cuda")
    model.eval()

    rows = pd.read_csv(os.path.join(out_dir, f"train100_{args.group}.csv"))
    manifest = [json.loads(l) for l in open(os.path.join(args.smoke_root, "samples_manifest.jsonl"))]
    tok_map = json.load(open(os.path.join(ROOT, "outputs/v2/tokenizer/token_id_map.json")))["map"]
    atomic_ids = set(tok_map.values())
    proc = StrictTrieProcessor(tokw, bindings)

    results = []
    n_strict, hits, tried_nl = 0, 0, 0
    for idx, rec_m in enumerate(manifest):
        row = rows.iloc[idx]
        prompt_ids = build_prompt_ids(tokw, {"history_item_sid": row["history_item_sid"]})
        target_toks = bindings["token_ids"][bindings["item_ids"].index(str(rec_m["item_id"]))]
        proc.prompt_len = len(prompt_ids)
        gen = model.generate(
            input_ids=torch.tensor([prompt_ids], device="cuda"),
            attention_mask=torch.ones((1, len(prompt_ids)), device="cuda", dtype=torch.long),
            max_new_tokens=6, num_beams=10, do_sample=False, num_return_sequences=1,
            logits_processor=[proc], pad_token_id=EOS, length_penalty=0.0)
        out_ids = gen[0, len(prompt_ids):].tolist()
        ok, msg = validate_sid_sequence(out_ids, EOS, atomic_ids)
        n_strict += int(bool(ok))
        if NL in out_ids:
            tried_nl += 1
        match = out_ids[:4] == list(target_toks) and len(out_ids) >= 4
        hits += int(bool(ok) and match)
        results.append({"row_id": rec_m["row_id"], "raw_token_ids": out_ids,
                        "phase5_validate": msg, "strict_legal": bool(ok),
                        "match": bool(ok) and match,
                        "target_token_ids": list(target_toks)})
    n = len(results)
    strict_examples = {"first_hit": next((r for r in results if r["match"]), None),
                       "first_non_hit": next((r for r in results if not r["match"]), None)}
    rec = {"stage": "t4-strict-decode-regen", "group": args.group, "seed": args.seed,
           "checkpoint": os.path.join(out_dir, "smoke_ckpt"),
           "code_sha256": sha256_file(os.path.abspath(__file__)),
           "constraint": "strict trie: 4 SID atoms then EOS only (newline forbidden); beam10 deterministic; NO post-hoc stripping",
           "n_samples": n, "strict_legal_4sid_eos": n_strict,
           "strict_legal_rate": n_strict / n,
           "hits_hit_at_1": hits,
           "attempts_containing_newline": tried_nl,
           "annotated_examples": {k: (dict(v, annotated=None) if v is None else v) for k, v in strict_examples.items()},
           "all_raw": results}
    with open(os.path.join(out_dir, "strict_regen_outputs.json"), "w") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print("REGEN_OK", args.group, f"strict_legal={n_strict}/{n} hits={hits}/{n} nl_tried={tried_nl}")


if __name__ == "__main__":
    main()
