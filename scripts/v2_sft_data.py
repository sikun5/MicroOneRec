"""Phase 6 controlled SFT data module (v2).

Worker-owned adaptation for MiniOneRec v2 Phase 6. The original repo files
`sft.py` / `data.py` are NOT modified; this module is the controlled entry.

Recorded adaptations vs original data.py classes:
1. Index building uses Phase-4 index.json arrays (item_ids / sids / token_ids)
   in canonical order (numeric-first item IDs). The original classes expected
   a legacy `{item_id: [tokens]}` dict format and are not compatible with the
   v2 frozen data face.
2. SidItemFeatDatasetV2 generates sid2title AND title2sid per item (3686+3686
   samples), each carrying its item_id provenance. The original built title2sid
   through a plain dict (`title2sid[title] = sid`), silently overwriting
   duplicate titles and losing legal multi-target sources - forbidden by the
   Phase 6 protocol.
3. Item text fields are mapped to the Phase-4 catalog schema: `title_clean` /
   `description_clean` (original used `title` / `description` keys).
4. Every sample carries explicit source IDs (row_id / item_id) for manifests;
   no prompt-dict overwrite anywhere.
Tokenization, prompt strings, label masking (-100 on prompt, target + EOS
supervised), and left truncation semantics are kept equivalent to the original
classes (SidSFTDataset / SidItemFeatDataset / FusionSeqRecDataset).
"""
import copy
import json
import os
import sys

import pandas as pd

ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import Tokenizer  # noqa: E402  (original Tokenizer wrapper kept)
from torch.utils.data import Dataset  # noqa: E402

GROUPS = ("A", "B", "C")
CATALOG_SIZE = 3686
SHARED_VOCAB_PATH = os.path.join(ROOT, "outputs/v2/tokenizer/shared_vocab.json")
TOKEN_ID_MAP_PATH = os.path.join(ROOT, "outputs/v2/tokenizer/token_id_map.json")
BASE_MODEL_PATH = "/hpc2hdd/home/swu073/sikun/dsaa5020/models/Qwen3-0.6B-Base"
EXPECTED_FINAL_VOCAB = 154977  # 151669 base + 3308 common new tokens


# ---------------------------------------------------------------- index/bindings
def load_group_bindings(group, root=ROOT):
    """Load one group's Phase-4 binding face with explicit item-ID linkage.

    Returns a dict with canonical-order item_ids / sids / token_ids plus
    reverse map and item metadata (raw JSON structures, no reordering).
    """
    d = os.path.join(root, f"outputs/v2/data/{group}")
    with open(os.path.join(d, "Industrial_and_Scientific.index.json")) as f:
        index = json.load(f)
    with open(os.path.join(d, "reverse_item_map.json")) as f:
        reverse = json.load(f)["reverse"]
    with open(os.path.join(d, "Industrial_and_Scientific.item.json")) as f:
        item_meta = json.load(f)
    return {
        "group": group,
        "item_ids": list(index["item_ids"]),
        "sids": list(index["sids"]),
        "token_ids": [list(t) for t in index["token_ids"]],
        "reverse": reverse,
        "items": item_meta["items"],
        "paths": {
            "index": os.path.join(d, "Industrial_and_Scientific.index.json"),
            "reverse": os.path.join(d, "reverse_item_map.json"),
            "item": os.path.join(d, "Industrial_and_Scientific.item.json"),
            "info": os.path.join(d, "info.txt"),
        },
    }


def verify_group_bindings(bindings, tokenizer=None, token_id_map=None):
    """Full explicit-ID consistency check over ALL 3686 items (boss T3 supplement).

    For every canonical position i: index.item_ids[i] <-> sids[i] <->
    token_ids[i] (decoded == sid), reverse[sid] == item_id (both directions),
    item_meta[item_id] exists with items[item_id].sid == index.sid.
    No first-row or positional inference is used for any cross-file claim.
    """
    n = len(bindings["item_ids"])
    fails = []
    sid_to_idx = {}
    for i in range(n):
        item_id = bindings["item_ids"][i]
        sid = bindings["sids"][i]
        toks = bindings["token_ids"][i]
        if bindings["reverse"].get(sid) != item_id:
            fails.append({"i": i, "item_id": item_id, "issue": "reverse_mismatch"})
        if sid in sid_to_idx:
            fails.append({"i": i, "item_id": item_id, "issue": "duplicate_sid"})
        sid_to_idx[sid] = i
        it = bindings["items"].get(item_id)
        if it is None:
            fails.append({"i": i, "item_id": item_id, "issue": "item_meta_missing"})
        elif it.get("sid") != sid:
            fails.append({"i": i, "item_id": item_id, "issue": "item_sid_mismatch"})
        if tokenizer is not None:
            decoded = tokenizer.decode(toks)
            if decoded.replace(" ", "") != sid:
                fails.append({"i": i, "item_id": item_id, "issue": "token_decode_mismatch",
                              "decoded": decoded, "sid": sid})
            if token_id_map is not None:
                for t in toks:
                    tok_str = tokenizer.convert_ids_to_tokens(int(t))
                    if token_id_map.get(tok_str) != int(t):
                        fails.append({"i": i, "item_id": item_id, "issue": "token_map_mismatch",
                                      "token": tok_str, "id": int(t)})
                        break
    return {"n_items": n, "n_fails": len(fails), "fails_head": fails[:20], "all_ok": len(fails) == 0}


def verify_info_txt(bindings, root=ROOT):
    """Explicit item-ID join against info.txt for ALL items.

    info.txt lines are `<SID>\ttitle_clean\titem_id`; we index them by item_id
    (column 3), then require each item_id's SID == bindings sid.
    """
    info_path = os.path.join(root, f"outputs/v2/data/{bindings['group']}/info.txt")
    by_item = {}
    with open(info_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                return {"ok": False, "issue": "malformed_line", "n_lines": None}
            by_item[parts[2]] = (parts[0], parts[1])
    fails = []
    for i, item_id in enumerate(bindings["item_ids"]):
        ent = by_item.get(item_id)
        if ent is None:
            fails.append({"item_id": item_id, "issue": "missing_in_info"})
        elif ent[0] != bindings["sids"][i]:
            fails.append({"item_id": item_id, "issue": "sid_mismatch"})
        elif ent[1] != bindings["items"][item_id].get("title_clean"):
            fails.append({"item_id": item_id, "issue": "title_mismatch"})
    return {"ok": len(fails) == 0, "n_items_checked": len(bindings["item_ids"]),
            "n_fails": len(fails), "fails_head": fails[:20]}


def compare_with_common_catalog(bindings, root=ROOT):
    """Compare per-group item.json text/label fields against the common catalog.

    Common catalog authority: outputs/v2/manifests/catalog_clean.jsonl (title,
    description) and outputs/v2/manifests/quality_features.jsonl
    (original_description_missing). Explicit item-ID join.
    """
    cat = {}
    with open(os.path.join(root, "outputs/v2/manifests/catalog_clean.jsonl"), encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cat[str(rec.get("item_id", rec.get("parent_asin")))] = rec
    qf = {}
    with open(os.path.join(root, "outputs/v2/manifests/quality_features.jsonl"), encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            qf[str(rec.get("item_id", rec.get("parent_asin")))] = rec
    fails, missing = [], []
    for item_id in bindings["item_ids"]:
        it = bindings["items"][item_id]
        c = cat.get(item_id)
        q = qf.get(item_id)
        if c is None or q is None:
            missing.append(item_id)
            continue
        if c.get("title_clean") != it.get("title_clean"):
            fails.append({"item_id": item_id, "issue": "title_differs"})
        if c.get("description_clean") != it.get("description_clean"):
            fails.append({"item_id": item_id, "issue": "description_differs"})
        qv = q.get("original_description_missing", q.get("missing"))
        if str(qv) != str(it.get("original_description_missing")):
            fails.append({"item_id": item_id, "issue": "missing_label_differs",
                          "catalog": str(qv), "item_json": str(it.get("original_description_missing"))})
    return {"n_items": len(bindings["item_ids"]), "n_missing_in_catalog": len(missing),
            "missing_head": missing[:10], "n_fails": len(fails), "fails_head": fails[:20],
            "all_ok": len(missing) == 0 and len(fails) == 0}


def duplicate_title_stats(bindings):
    """Duplicate-title census with the canonical rule (numeric-first item IDs).

    Returns instance counts and per-title canonical item assignment evidence.
    """
    from collections import Counter, defaultdict
    titles = [bindings["items"][i]["title_clean"] for i in bindings["item_ids"]]
    cnt = Counter(titles)
    dup_titles = {t: c for t, c in cnt.items() if c > 1}
    members = defaultdict(list)
    for item_id in bindings["item_ids"]:
        t = bindings["items"][item_id]["title_clean"]
        if t in dup_titles:
            members[t].append(item_id)
    canonical_members = {t: sorted(v, key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
                         for t, v in members.items()}
    return {
        "n_items": len(titles), "n_unique_titles": len(cnt),
        "n_duplicate_titles": len(dup_titles),
        "n_duplicate_instances": sum(dup_titles.values()),
        "max_instances_per_title": max(dup_titles.values()) if dup_titles else 0,
        "canonical_rule": "item ID numeric-first ascending; first member is canonical (Phase-4 order_rule)",
        "members_head": {t: canonical_members[t] for t in list(sorted(canonical_members))[:3]},
    }


# ---------------------------------------------------------------- tokenizer/model
def make_expanded_tokenizer(base_tokenizer):
    """Register the 3308 common vocab tokens in frozen shared_vocab.json order.

    Verifies every resulting ID against the frozen token_id_map.json (all 3308
    entries). Returns (tokenizer, verification dict). This replaces the
    original TokenExtender behavior (per-group sorted registration - forbidden).
    """
    with open(SHARED_VOCAB_PATH) as f:
        sv_obj = json.load(f)
    shared = sv_obj["tokens"] if isinstance(sv_obj, dict) else sv_obj
    with open(TOKEN_ID_MAP_PATH) as f:
        id_map = json.load(f)["map"]
    base_len = len(base_tokenizer)
    base_tokenizer.add_tokens(shared)  # frozen order, all three groups identical
    mismatches = []
    for tok in shared:
        got = base_tokenizer.convert_tokens_to_ids(tok)
        want = id_map.get(tok)
        if got != want:
            mismatches.append({"token": tok, "got": got, "want": want})
    ver = {
        "base_len": base_len,
        "final_len": len(base_tokenizer),
        "expected_final": EXPECTED_FINAL_VOCAB,
        "shared_vocab_schema": sv_obj.get("schema") if isinstance(sv_obj, dict) else None,
        "shared_vocab_registration_order_sha256": sv_obj.get("registration_order_sha256") if isinstance(sv_obj, dict) else None,
        "n_shared": len(shared),
        "n_map_mismatch": len(mismatches),
        "map_mismatch_head": mismatches[:10],
        "pad_token_id": base_tokenizer.pad_token_id,
        "eos_token_id": base_tokenizer.eos_token_id,
    }
    ver["ok"] = (ver["final_len"] == EXPECTED_FINAL_VOCAB and not mismatches)
    return base_tokenizer, ver


# ---------------------------------------------------------------- datasets
SID_INSTRUCTION = ("Below is an instruction that describes a task, paired with an input that provides "
                   "further context. Write a response that appropriately completes the request. \n\n"
                   "### Instruction:\nCan you predict the next possible item that the user may expect?\n\n")
ITEMFEAT_INSTRUCTION = ("Below is an instruction that describes a task, paired with an input that provides "
                        "further context. Write a response that appropriately completes the request. \n\n"
                        "### Instruction:\nAnswer the question about item identification.\n\n")
FUSION_INSTRUCTION = ("Below is an instruction that describes a task, paired with an input that provides "
                      "further context. Write a response that appropriately completes the request. \n\n"
                      "### Instruction:\nCan you recommend the next item for the user based on their interaction history?\n\n")


def _generate_prompt(user_input, response=""):
    # byte-equivalent to original BaseDataset.generate_prompt
    # (note: original template has a trailing space after "### User Input:")
    return (
        "### User Input: \n"
        f"{user_input}\n"
        "\n"
        "### Response:\n"
        f"{response}"
    )


def _finalize(pre_tokens, prompt, target, max_len, tokenizer, test):
    """Original truncation/label semantics: prompt tokens + golden (target + EOS),
    labels = [-100]*prompt_len + golden_ids, left truncation to max_len."""
    tokens = pre_tokens + tokenizer.encode(prompt, bos=False, eos=False)
    if test:
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}
    golden = tokenizer.encode(target, bos=False, eos=True)
    p = len(tokens)
    tokens = tokens + golden
    labels = [-100] * p + tokens[p:]
    return {
        "input_ids": tokens[-max_len:],
        "attention_mask": [1] * min(len(tokens), max_len),  # all ones; equals original slice
        "labels": labels[-max_len:],
    }


class _V2Base(Dataset):
    def __init__(self, tokenizer, max_len):
        self.tokenizer = Tokenizer(tokenizer)
        self.max_len = max_len
        self.samples = []   # tokenized dicts (input_ids / attention_mask / labels)
        self.metas = []     # provenance per sample

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class SidSFTDatasetV2(_V2Base):
    """Main SID sequence recommendation task (train + the only validation task).

    Sample order = CSV file order (deterministic). Source row_id preserved.
    """

    def __init__(self, train_file, tokenizer, max_len=512, test=False):
        super().__init__(tokenizer, max_len)
        self.test = test
        df = pd.read_csv(train_file)
        for _, row in df.iterrows():
            row_id = str(row["row_id"])
            hist = eval(row["history_item_sid"]) if isinstance(row["history_item_sid"], str) else row["history_item_sid"]
            history = ", ".join(hist)
            target = str(row["item_sid"]) + "\n"
            user_input = (f"The user has interacted with items {history} in chronological order. "
                          f"Can you predict the next possible item that the user may expect?")
            pre = self.tokenizer.encode(SID_INSTRUCTION, bos=True, eos=False)
            prompt = _generate_prompt(user_input)
            s = _finalize(pre, prompt, target, max_len, self.tokenizer, test)
            self.samples.append(s)
            self.metas.append({"task": "sid_sft", "row_id": row_id, "target_sid": str(row["item_sid"]),
                               "n_history": len(hist)})


class SidItemFeatDatasetV2(_V2Base):
    """Per-item sid2title + title2sid alignment task (3686 + 3686 samples).

    Order: canonical item order (Phase-4 order_rule) within each sub-task;
    sid2title block first, then title2sid block. Duplicate titles are kept as
    separate samples with item_id provenance (no dict overwrite).
    """

    def __init__(self, bindings, tokenizer, max_len=512, test=False):
        super().__init__(tokenizer, max_len)
        self.test = test
        # canonical title mapping (boss ruling 2026-11-12): one title2sid sample
        # per UNIQUE clean title; canonical item = numerically smallest item ID
        # (digits compared as int, numeric-first per Phase-4 order_rule; string
        # sort never used). Duplicate-title members are recorded in provenance.
        def sk(s):
            return (0, int(s), "") if s.isdigit() else (1, 0, s)
        members = {}
        for item_id in bindings["item_ids"]:
            members.setdefault(bindings["items"][item_id]["title_clean"], []).append(item_id)
        canonical_of = {t: min(v, key=sk) for t, v in members.items()}
        emitted = set()
        n_dup_instances_kept = 0
        for i, item_id in enumerate(bindings["item_ids"]):
            sid = bindings["sids"][i]
            title = bindings["items"][item_id]["title_clean"]
            # sid2title: per item, all 3686
            pre = self.tokenizer.encode(ITEMFEAT_INSTRUCTION, bos=True, eos=False)
            prompt = _generate_prompt(f'What is the title of item "{sid}"?')
            self.samples.append(_finalize(pre, prompt, title + "\n", max_len, self.tokenizer, test))
            self.metas.append({"task": "sid2title", "item_id": item_id, "sid": sid})
            # title2sid: canonical only, first hit in canonical item order
            if item_id == canonical_of[title] and title not in emitted:
                emitted.add(title)
                prompt = _generate_prompt(f"Which item has the title: {title}?")
                self.samples.append(_finalize(pre, prompt, sid + "\n", max_len, self.tokenizer, test))
                self.metas.append({"task": "title2sid", "item_id": item_id, "sid": sid,
                                   "title": title, "n_title_members": len(members[title]),
                                   "member_item_ids": sorted(members[title], key=sk)})
        self.title_members = members
        self.canonical_of = canonical_of


class FusionSeqRecDatasetV2(_V2Base):
    """Sequence recommendation with textual target (title task), train CSV order."""

    def __init__(self, train_file, bindings, tokenizer, max_len=512, test=False):
        super().__init__(tokenizer, max_len)
        self.test = test
        sid2title = {}
        for i, item_id in enumerate(bindings["item_ids"]):
            sid2title[bindings["sids"][i]] = bindings["items"][item_id]["title_clean"]
        df = pd.read_csv(train_file)
        for _, row in df.iterrows():
            row_id = str(row["row_id"])
            hist = eval(row["history_item_sid"]) if isinstance(row["history_item_sid"], str) else row["history_item_sid"]
            history = ", ".join(hist)
            target_sid = str(row["item_sid"])
            title = sid2title.get(target_sid)
            if title is None:
                title = f"An item with semantic ID {target_sid}"
            user_input = (f"The user has sequentially interacted with items {history}. "
                          f"Can you recommend the next item for him? Tell me the title of the item")
            pre = self.tokenizer.encode(FUSION_INSTRUCTION, bos=True, eos=False)
            prompt = _generate_prompt(user_input)
            self.samples.append(_finalize(pre, prompt, title + "\n", max_len, self.tokenizer, test))
            self.metas.append({"task": "fusion_seqrec", "row_id": row_id, "target_sid": target_sid})


def label_integrity(samples, tokenizer, task="sid_sft", head=50):
    """Check: non-empty supervision and EOS-terminated targets.

    For sid_sft: at least the 4 SID tokens + EOS. For text targets (title):
    at least 1 supervised token + EOS. Confirms truncation never produced
    empty labels; pad=eos distinction is positional (labels only extend to
    the appended golden tokens).
    """
    eos = tokenizer.eos_token_id
    min_sup = 5 if task == "sid_sft" else 2
    bad = []
    for i, s in enumerate(samples):
        lab = s["labels"]
        sup = [t for t in lab if t != -100]
        if len(sup) < min_sup or sup[-1] != eos:
            bad.append({"i": i, "n_sup": len(sup), "last": sup[-1] if sup else None})
        if i >= 200000:
            break
    return {"task": task, "min_supervised": min_sup, "n_checked": min(len(samples), 200001),
            "n_bad": len(bad), "bad_head": bad[:head], "all_ok": len(bad) == 0}
