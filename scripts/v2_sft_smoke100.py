"""Phase 6 T4 100-sample smoke: overfit check + constrained Beam10 generation.

Stages:
  --stage manifest : one-shot builder. seed42 selection of 100 conflict-free
      complete-SID recommendation rows from group A train.csv (conflict =
      target sid equals last history sid; legality = every sid maps through
      the group index). Saves samples_manifest.jsonl with raw source fields
      and per-group SID values; verifies the same row_ids exist in B/C.
  --stage train --group G : T3 common init -> same-口径 eval loss on the fixed
      batch (before), full-parameter debug training within the frozen budget
      (micro=common feasible config, accum=64/micro, <=30 epochs, stop at
      eval-loss drop >=80% vs init), same eval loss (after), constrained
      deterministic Beam10 generation (group SID trie: 4 atomic tokens + EOS,
      Phase-5 contract), raw token outputs, exact-match hits, illegal rate,
      save checkpoint.

Debug-only: checkpoints stay in the smoke root and are NEVER used to
initialize formal SFT runs.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

import torch

ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

from v2_sft_data import (BASE_MODEL_PATH, EXPECTED_FINAL_VOCAB,  # noqa: E402
                         SID_INSTRUCTION, SidSFTDatasetV2, Tokenizer,
                         load_group_bindings, make_expanded_tokenizer,
                         _generate_prompt)
from v2_sft_init import set_seed  # noqa: E402

SMOKE_MANIFEST = os.path.join(ROOT, "outputs/v2/debug/phase6/sft_smoke/attempt_001/samples_manifest.jsonl")
N_SMOKE = 100
MAX_EPOCHS = 30
MEMO_TARGET = 0.5  # stop only at memorization-level eval loss (v6; drop-ratio
# stop fired too early at 1 epoch and produced hits=0 twice, see history)
EVAL_MICRO = 4  # reduced from 16 after OOM at fp32 logits upcast (16x512x154977x4B); eval batch may be lowered independently per protocol, locked here


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_manifest():
    import pandas as pd
    set_seed(42)
    bA = load_group_bindings("A")
    legal_A = set(bA["sids"])
    df = pd.read_csv(os.path.join(ROOT, "outputs/v2/data/A/train.csv"))
    dfA = {str(r["row_id"]): r for _, r in df.iterrows()}
    bB = load_group_bindings("B")
    bC = load_group_bindings("C")
    maps = {"A": dict(zip(bA["item_ids"], bA["sids"])),
            "B": dict(zip(bB["item_ids"], bB["sids"])),
            "C": dict(zip(bC["item_ids"], bC["sids"]))}
    csvs = {g: pd.read_csv(os.path.join(ROOT, f"outputs/v2/data/{g}/train.csv")) for g in ("B", "C")}
    idx = {g: {str(r["row_id"]): r for _, r in csvs[g].iterrows()} for g in ("B", "C")}

    def conflict_free(g, r):
        hist = eval(r["history_item_sid"]) if isinstance(r["history_item_sid"], str) else []
        target = str(r["item_sid"])
        if not hist or hist[-1] == target:
            return False
        legal = set(maps[g].values())
        return target in legal and all(h in legal for h in hist)

    cand = [rid for rid, r in dfA.items() if conflict_free("A", r)]
    rng = random.Random(42)
    chosen = sorted(rng.sample(cand, N_SMOKE), key=lambda rid: int(rid.split(":")[1]))
    out = []
    for rid in chosen:
        rA = dfA[rid]
        rec = {"row_id": rid, "user_id": str(rA["user_id"]), "item_id": str(rA["item_id"]),
               "item_title": str(rA["item_title"]),
               "history_len": len(eval(rA["history_item_sid"])),
               "sid_A": str(rA["item_sid"]), "sid_B": str(idx["B"][rid]["item_sid"]),
               "sid_C": str(idx["C"][rid]["item_sid"]),
               "selection_rule": "seed42 random.Random(42).sample over conflict-free complete-SID train rows; sorted by row index"}
        # cross-group integrity: same item_id everywhere, sids match group maps
        assert str(idx["B"][rid]["item_id"]) == rec["item_id"] == str(idx["C"][rid]["item_id"])
        assert rec["sid_B"] == maps["B"][rec["item_id"]] and rec["sid_C"] == maps["C"][rec["item_id"]]
        assert conflict_free("B", idx["B"][rid]) and conflict_free("C", idx["C"][rid])
        out.append(rec)
    with open(SMOKE_MANIFEST, "w") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("MANIFEST_OK", len(out), "conflict_free_candidates=", len(cand))


def build_prompt_ids(tokw, row):
    """Prompt ids via the data.Tokenizer wrapper (byte-identical to the dataset path)."""
    hist = eval(row["history_item_sid"]) if isinstance(row["history_item_sid"], str) else []
    history = ", ".join(hist)
    user_input = (f"The user has interacted with items {history} in chronological order. "
                  f"Can you predict the next possible item that the user may expect?")
    pre = tokw.encode(SID_INSTRUCTION, bos=True, eos=False)
    ids = pre + tokw.encode(_generate_prompt(user_input), bos=False, eos=False)
    return ids


def eval_loss(model, collator, samples, device="cuda"):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(samples), EVAL_MICRO):
            s = samples[i:i + EVAL_MICRO]
            batch = collator([{k: x[k] for k in ("input_ids", "attention_mask", "labels")} for x in s]).to(device)
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        labels=batch["labels"])
            tot += float(out.loss.item()) * len(s)
            n += len(s)
    model.train()
    torch.cuda.empty_cache()
    return tot / n


class LegalSIDProcessor:
    """Constrain beam decoding to the group's legal SID token sequences.

    Trie over the group's 3686 SID token-id sequences (4 atomic tokens), then
    the newline token, then EOS (Phase-5 four-token+EOS contract). Raw scores
    of disallowed tokens are masked to -inf.
    """

    def __init__(self, tokw, bindings):
        eos = tokw.eos_id
        nl = tokw.encode("\n", bos=False, eos=False)
        assert len(nl) == 1, f"newline must encode to one token, got {nl}"
        self.nl, self.eos = nl[0], eos
        trie = {}
        for toks in bindings["token_ids"]:
            node = trie
            for t in toks:
                node = node.setdefault(t, {})
            node[self.nl] = {self.eos: {}}
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
            if len(prefix) == 0 and not allowed:
                allowed = set(self.trie.keys())
            mask = torch.full_like(scores[row], float("-inf"))
            if allowed:
                idx = torch.tensor(sorted(allowed), device=scores.device, dtype=torch.long)
                mask[idx] = 0.0
            scores[row] = scores[row] + mask
        return scores


def stage_train(group, smoke_root, micro, seed, accum=None, max_epochs=MAX_EPOCHS):
    out_dir = os.path.join(smoke_root, "overfit_100", group)
    os.makedirs(out_dir, exist_ok=True)
    if accum is None:
        accum = 64 // micro
    cfg = {"stage": "t4-smoke100-frozen", "group": group, "seed": seed, "micro": micro,
           "accum": accum, "effective_batch": micro * accum, "max_epochs": max_epochs,
           "stop_rule": f"eval_loss <= {MEMO_TARGET} (memorization level) or max_epochs; v6 amendment after v4/v5 drop-ratio stops yielded hits=0",
           "lr": 3e-4, "beam": {"num_beams": 10, "do_sample": False, "deterministic": True,
                                 "constraint": "group SID trie (4 atomic tokens + EOS), LegalSIDProcessor"},
           "manifest": SMOKE_MANIFEST,
           "code_sha256": sha256_file(os.path.abspath(__file__)),
           "init_state_dict_sha256": "820b20708d8554bee82f153f1d8f93454e369cd378c9ab6a20ee8ea33384e161",
           "debug_note": "smoke checkpoint never initializes formal SFT; budget independent of formal protocol"}
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)

    import pandas as pd
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(seed)
    manifest = [json.loads(l) for l in open(SMOKE_MANIFEST)]
    df = pd.read_csv(os.path.join(ROOT, f"outputs/v2/data/{group}/train.csv"))
    want = {r["row_id"] for r in manifest}
    rows = df[df["row_id"].astype(str).isin(want)].copy()
    assert len(rows) == N_SMOKE, f"group {group}: expected 100 rows, got {len(rows)}"
    mini = os.path.join(out_dir, f"train100_{group}.csv")
    rows.to_csv(mini, index=False)

    bindings = load_group_bindings(group, root=ROOT)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer, tok_ver = make_expanded_tokenizer(tokenizer)
    assert tok_ver["ok"]

    ds = SidSFTDatasetV2(mini, tokenizer, max_len=512)
    samples = [ds[i] for i in range(len(ds))]
    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16,
                                                 local_files_only=True)
    model.config.use_cache = False
    model.resize_token_embeddings(EXPECTED_FINAL_VOCAB)
    model = model.to("cuda")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    t0 = time.time()
    init_loss = eval_loss(model, collator, samples)
    history = [{"epoch": 0, "eval_loss": init_loss}]
    final_train_loss = None
    stop_reason = "max_epochs"
    for epoch in range(1, max_epochs + 1):
        ep_losses = []
        optimizer.zero_grad(set_to_none=True)
        for i in range(0, len(samples), micro):
            s = samples[i:i + micro]
            batch = collator([{k: x[k] for k in ("input_ids", "attention_mask", "labels")} for x in s]).to("cuda")
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        labels=batch["labels"])
            loss = out.loss / accum
            loss.backward()
            ep_losses.append(float(out.loss.item()))
            if (i // micro + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        final_train_loss = sum(ep_losses) / len(ep_losses)
        el = eval_loss(model, collator, samples)
        history.append({"epoch": epoch, "train_loss": final_train_loss, "eval_loss": el})
        print(f"[smoke {group}] epoch {epoch} train={final_train_loss:.4f} eval={el:.4f}", flush=True)
        if not torch.isfinite(torch.tensor(el)):
            stop_reason = "non_finite"
            break
        if el <= MEMO_TARGET:
            stop_reason = "memorized(eval<=0.5)"
            break
    wall = time.time() - t0
    final_loss = history[-1]["eval_loss"]

    # persist the smoke checkpoint BEFORE generation (a generation bug must not
    # lose the training evidence)
    model.save_pretrained(os.path.join(out_dir, "smoke_ckpt"))
    tokenizer.save_pretrained(os.path.join(out_dir, "smoke_ckpt"))
    persisted = sum(os.path.getsize(os.path.join(dp, f))
                    for dp, _, fs in os.walk(os.path.join(out_dir, "smoke_ckpt")) for f in fs)
    del optimizer
    torch.cuda.empty_cache()

    # constrained deterministic Beam10 generation over all 100 samples
    tokw = Tokenizer(tokenizer)
    proc = LegalSIDProcessor(tokw, bindings)
    model.eval()
    hits, illegal, raw_out = 0, 0, []
    for idx, rec_m in enumerate(manifest):
        row = rows.iloc[idx]
        prompt_ids = build_prompt_ids(tokw, {"history_item_sid": row["history_item_sid"]})
        target_sid = rec_m[f"sid_{group}"]
        target_toks = bindings["token_ids"][bindings["item_ids"].index(str(rec_m["item_id"]))]
        proc.prompt_len = len(prompt_ids)
        gen = model.generate(
            input_ids=torch.tensor([prompt_ids], device="cuda"),
            attention_mask=torch.ones((1, len(prompt_ids)), device="cuda", dtype=torch.long),
            max_new_tokens=7, num_beams=10, do_sample=False, num_return_sequences=1,
            logits_processor=[proc], pad_token_id=tokenizer.eos_token_id,
            length_penalty=0.0)
        out_ids = gen[0, len(prompt_ids):].tolist()
        legal = out_ids[:4] in [list(t) for t in bindings["token_ids"]] and \
            len(out_ids) >= 5 and out_ids[4] == proc.nl and (len(out_ids) == 5 or out_ids[5] == proc.eos)
        if not legal:
            illegal += 1
        if out_ids[:4] == list(target_toks):
            hits += 1
        raw_out.append({"row_id": rec_m["row_id"], "raw_token_ids": out_ids,
                        "target_token_ids": list(target_toks),
                        "match": out_ids[:4] == list(target_toks), "legal": legal})
    illegal_rate = illegal / N_SMOKE

    rec = {"group": group, "init_eval_loss": init_loss, "final_eval_loss": final_loss,
           "drop_ratio": final_loss / init_loss if init_loss else None,
           "losses_finite": all(torch.isfinite(torch.tensor(h["eval_loss"])).item() for h in history),
           "stop_reason": stop_reason, "epochs_run": len(history) - 1,
           "wall_seconds": round(wall, 1), "micro": micro, "accum": accum,
           "hits": hits, "illegal_rate": illegal_rate,
           "loss_history": history,
           "generation_samples_head": raw_out[:5],
           "raw_outputs_file": "generation_outputs.json"}
    with open(os.path.join(out_dir, "generation_outputs.json"), "w") as f:
        json.dump(raw_out, f, ensure_ascii=False)
    with open(os.path.join(out_dir, "smoke_result.json"), "w") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print("SMOKE_OK", group, "init=", round(init_loss, 4), "final=", round(final_loss, 4),
          "hits=", hits, "illegal=", illegal_rate, "stop=", stop_reason, "wall=", round(wall, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["manifest", "train"])
    ap.add_argument("--group", choices=["A", "B", "C"])
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--accum", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke_root", default=os.path.join(
        ROOT, "outputs/v2/debug/phase6/sft_smoke/attempt_001"))
    args = ap.parse_args()
    if args.stage == "manifest":
        stage_manifest()
    else:
        stage_train(args.group, args.smoke_root, args.micro, args.seed,
                    accum=args.accum, max_epochs=args.epochs)


if __name__ == "__main__":
    main()
