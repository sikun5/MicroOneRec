#!/usr/bin/env python3
"""Phase7 T5 completion: fresh-process load of A-RL final_checkpoint + strict raw 10 valid prompts."""
import json, os, sys
ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
import torch, pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from v2_sft_data import SID_INSTRUCTION, Tokenizer, load_group_bindings, _generate_prompt
from v2_generation_adapter import validate_sid_sequence
ck = "outputs/v2/checkpoints/A-RL/final_checkpoint"
tok = AutoTokenizer.from_pretrained(ck, local_files_only=True)
assert len(tok) == 154977
tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id; tok.padding_side = "left"
tk = Tokenizer(tok)
model = AutoModelForCausalLM.from_pretrained(ck, torch_dtype=torch.bfloat16, local_files_only=True).cuda().eval()
bindings = load_group_bindings("A", root=ROOT)
tmap = json.load(open("outputs/v2/tokenizer/token_id_map.json"))["map"]
atomic = set(tmap.values()); EOS = tok.eos_token_id
trie = {}
for toks in bindings["token_ids"]:
    node = trie
    for t in toks: node = node.setdefault(t, {})
    node[EOS] = {}
class LP:
    def __init__(self, pl): self.pl = pl
    def __call__(self, ids, scores):
        out = []
        for row in range(ids.size(0)):
            gen = ids[row, self.pl:].tolist(); node = trie; ok = True
            for t in gen:
                if t in node: node = node[t]
                else: ok = False; break
            allowed = sorted(node.keys()) if ok else []
            m = torch.full_like(scores[row], float("-inf"))
            if allowed: m[torch.tensor(allowed, device=scores.device, dtype=torch.long)] = 0.0
            out.append(m)
        return torch.stack(out)
valid = pd.read_csv("outputs/v2/data/A/valid.csv").head(10)
res, legal, hits = [], 0, 0
for _, r in valid.iterrows():
    hist = eval(r["history_item_sid"]) if isinstance(r["history_item_sid"], str) else []
    ui = f"The user has interacted with items {', '.join(hist)} in chronological order. Can you predict the next possible item that the user may expect?"
    ids = tk.encode(SID_INSTRUCTION, bos=True, eos=False) + tk.encode(_generate_prompt(ui), bos=False, eos=False)
    lp = LP(len(ids))
    with torch.no_grad():
        g = model.generate(torch.tensor([ids]).cuda(), attention_mask=torch.ones((1, len(ids)), dtype=torch.long).cuda(),
            max_new_tokens=6, num_beams=10, do_sample=False, num_return_sequences=1, logits_processor=[lp], pad_token_id=EOS, length_penalty=0.0)
    out = g[0, len(ids):].tolist()
    ok, _ = validate_sid_sequence(out, EOS, atomic)
    legal += int(bool(ok))
    tgt = bindings["token_ids"][bindings["item_ids"].index(str(r["item_id"]))]
    h = int(out[:4] == tgt)
    hits += h
    res.append({"raw": out, "legal": bool(ok), "hit": bool(h)})
doc = {"task": "Phase7 T5 completion check", "method": "A", "checkpoint": ck,
       "n_prompts": 10, "strict_legal": legal, "hits_vs_valid_targets": hits, "results": res}
os.makedirs("outputs/v2/evidence/phase7", exist_ok=True)
json.dump(doc, open("outputs/v2/evidence/phase7/t5_a_rl_load_check.json", "w"), ensure_ascii=False, indent=1)
import hashlib
print("FINAL_LOAD_CHECK legal=%d/10 hits=%d/10" % (legal, hits))
print("sha", hashlib.sha256(open("outputs/v2/evidence/phase7/t5_a_rl_load_check.json","rb").read()).hexdigest())
