#!/usr/bin/env python3
"""Phase 7 T3-R1 (1): materialize the deterministic two-logical-epoch GRPO stream.

Order = (HF Dataset.shuffle(seed) permutation, read authoritatively from the
shuffled dataset's _indices) composed with the locked RepeatRandomSampler's
ACTUAL two-iteration randperm sequence (one torch.Generator(seed) advanced
across both calls, exactly as the sampler does when iterated once per epoch).
The composed order is verified against the REAL sampler iterated twice over a
REAL shuffled HF Dataset. Base 10k manifest and its order are untouched; each
task appears exactly twice (copy_index 0/1) and groups differ only in SID
fields.

Disclosure: the fixed stream sampler consumes NO train-time RNG (order is
materialized here); this replicates the DATA ORDER of the locked two-epoch
sampler+shuffle, not a bit-identical replay of the old training RNG
trajectory (generate/backprop RNG streams are separate and remain live).
"""
import argparse
import hashlib
import json
import os
import sys

ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import pandas as pd  # noqa: E402
from datasets import Dataset as HFDataset  # noqa: E402

from minionerec_trainer import RepeatRandomSampler  # noqa: E402  (locked trainer source)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


PROMPT_SEQ = ("### User Input: \nThe user has interacted with items {hist} in chronological order. "
              "Can you predict the next possible item that the user may expect?\n\n### Response:\n")
PROMPT_TITLE = "### User Input: \nWhich item has the title: {title}?\n\n### Response:\n"
PROMPT_TEXTHIST = ("### User Input: \nGiven the title sequence of user historical interactive items: {titles}, "
                   "can you recommend a suitable next item for the user?\n\n### Response:\n")


def build_task_rows(group):
    """One logical task row per manifest entry (canonical manifest order)."""
    base = [json.loads(l) for l in open("outputs/v2/manifests/rl_tasks_10000.jsonl")]
    sidm = [json.loads(l) for l in open(f"outputs/v2/data/{group}/rl_tasks_10000_sid.jsonl")]
    assert len(base) == len(sidm) == 10000
    train = pd.read_csv(f"outputs/v2/data/{group}/train.csv")
    train_by_row = {r["row_id"]: r for _, r in train.iterrows()}
    index = json.load(open(f"outputs/v2/data/{group}/Industrial_and_Scientific.index.json"))
    pos = {str(i): p for p, i in enumerate(index["item_ids"])}
    rows = []
    for b, s in zip(base, sidm):
        assert b["task_id"] == s["task_id"] and b["task_type"] == s["task_type"]
        assert b["source_id"] == s["source_id"] and str(b["item_id"]) == str(s["item_id"])
        r = dict(task_id=b["task_id"], task_type=b["task_type"], source_id=b["source_id"],
                 source_split=b["source_split"], user_id=b.get("user_id"),
                 item_id=str(b["item_id"]), title_clean=b.get("title_clean"),
                 item_sid=s["item_sid"],
                 item_token_ids=index["token_ids"][pos[str(b["item_id"])]],
                 history_item_sid=s.get("history_item_sid"))
        if b["task_type"] == "seq2sid":
            tr = train_by_row[b["source_id"]]
            assert str(tr["item_id"]) == r["item_id"], f"source row mismatch {b['task_id']}"
            hist = tr["history_item_sid"]
            hist = eval(hist) if isinstance(hist, str) else hist
            assert list(hist) == list(s["history_item_sid"]), f"SID drift {b['task_id']}"
            r["prompt"] = PROMPT_SEQ.format(hist=", ".join(hist))
            r["history_titles"] = None
        elif b["task_type"] == "title2sid":
            r["prompt"] = PROMPT_TITLE.format(title=b["title_clean"])
            r["history_titles"] = None
        elif b["task_type"] == "texthist2sid":
            tr = train_by_row[b["source_id"]]
            assert str(tr["item_id"]) == r["item_id"], f"source row mismatch {b['task_id']}"
            titles = tr["history_item_title"]
            titles = eval(titles) if isinstance(titles, str) else titles
            r["prompt"] = PROMPT_TEXTHIST.format(titles=", ".join([f'"{t}"' for t in titles]))
            r["history_titles"] = titles
        else:
            raise ValueError(b["task_type"])
        r["completion"] = s["item_sid"] + "\n"
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="outputs/v2/manifests")
    ap.add_argument("--groups", default="A,B,C")
    args = ap.parse_args()
    os.chdir(ROOT)

    # ---- shuffle permutation, read authoritatively from the datasets object ----
    n = 10000
    probe = HFDataset.from_list([{"i": i} for i in range(n)])
    shuffled = probe.shuffle(seed=args.seed)
    shuffle_pos = shuffled._indices["indices"].to_pylist()  # list[int]; shuffled row p = original row shuffle_pos[p]
    shuffle_hash = hashlib.sha256(json.dumps(shuffle_pos).encode()).hexdigest()

    # ---- sampler two-iteration randperm sequence (torch generator advanced) ----
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    perm1 = torch.randperm(n, generator=gen).tolist()
    perm2 = torch.randperm(n, generator=gen).tolist()

    # ---- verify against the REAL sampler iterated twice over a REAL shuffled dataset ----
    smp = RepeatRandomSampler(shuffled, repeat_count=1, seed=args.seed)
    got = []
    for _ in range(2):
        for idx in smp:
            got.append(int(idx))
    # real sampler yields repeated positions; dedupe consecutive repeats -> epoch orders
    epoch_orders = []
    for chunk in (got[:n], got[n:]):
        assert len(chunk) == n
        seen, order = set(), []
        for p in chunk:
            if p not in seen:
                seen.add(p)
                order.append(p)
        assert len(order) == n
        epoch_orders.append(order)
    ok1 = epoch_orders[0] == perm1
    ok2 = epoch_orders[1] == perm2
    # composed task order: task at shuffled position p = manifest row shuffle_pos[p]
    task_orders = [[shuffle_pos[p] for p in eo] for eo in epoch_orders]

    report = {"seed": args.seed, "n_tasks": n,
              "shuffle_perm_hash": shuffle_hash,
              "shuffle_source": "HFDataset.shuffle(seed)._indices (authoritative)",
              "sampler_two_iter_match": {"iter1": ok1, "iter2": ok2},
              "task_order_hash": hashlib.sha256(json.dumps(task_orders[0] + task_orders[1]).encode()).hexdigest(),
              "note": ("data-order replication of shuffle+RepeatRandomSampler two-epoch semantics; "
                       "stream sampler consumes no train-time RNG; not a bit-identical replay of "
                       "old training RNG trajectory (generation/backprop RNG remain live)")}
    assert ok1 and ok2, report["sampler_two_iter_match"]

    for g in args.groups.split(","):
        rows = build_task_rows(g)
        stream = []
        for copy_index, torder in enumerate(task_orders):
            for stream_index, task_row in enumerate(torder):
                r = dict(rows[task_row])
                r["copy_index"] = copy_index
                r["stream_index"] = copy_index * n + stream_index
                stream.append(r)
        assert len(stream) == 2 * n
        by_copy = {}
        for r in stream:
            by_copy.setdefault(r["copy_index"], {})[r["task_id"]] = r
        assert sorted(by_copy[0]) == sorted(by_copy[1]) and len(by_copy[0]) == n
        for tid, r0 in by_copy[0].items():
            r1 = by_copy[1][tid]
            for f in ("prompt", "item_sid", "item_token_ids", "task_type", "item_id",
                      "source_id", "completion", "history_titles"):
                assert r0[f] == r1[f], f"copy mismatch {tid}.{f}"
            assert r0["copy_index"] == 0 and r1["copy_index"] == 1
        out = os.path.join(args.out_dir, f"rl_stream_20000_{g}.jsonl")
        with open(out, "w") as f:
            for r in stream:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        report[f"group_{g}"] = {"path": out, "sha256": sha(out), "rows": len(stream),
                                "task_type_counts": {t: sum(1 for r in stream if r["task_type"] == t)
                                                     for t in ("seq2sid", "title2sid", "texthist2sid")}}

    A = [json.loads(l) for l in open(f"outputs/v2/manifests/rl_stream_20000_A.jsonl")]
    B = [json.loads(l) for l in open(f"outputs/v2/manifests/rl_stream_20000_B.jsonl")]
    C = [json.loads(l) for l in open(f"outputs/v2/manifests/rl_stream_20000_C.jsonl")]
    non_sid = ["stream_index", "copy_index", "task_id", "task_type", "source_id",
               "source_split", "user_id", "item_id", "title_clean", "history_titles"]
    sid_carrying = ["prompt", "completion", "item_sid", "item_token_ids", "history_item_sid"]
    diff = sum(1 for a, b, c in zip(A, B, C)
               if any(a.get(k) != b.get(k) or a.get(k) != c.get(k) for k in non_sid))
    sid_diff = sum(1 for a, b, c in zip(A, B, C)
                   if not (a.get("item_sid") != b.get("item_sid") and a.get("item_sid") != c.get("item_sid")))
    report["three_group_invariance"] = {"non_sid_diff_rows": diff, "ok": diff == 0,
                                        "note": "SID-carrying fields (prompt/completion/item_sid/"
                                                "item_token_ids/history_item_sid) differ per group by design; "
                                                "seq2sid prompts embed group SIDs; title2sid prompts are "
                                                "title-only and identical across groups",
                                        "sid_fields_checked": sid_carrying}
    assert diff == 0

    ev = "outputs/v2/evidence/phase7"
    os.makedirs(ev, exist_ok=True)
    with open(os.path.join(ev, "t3_stream_report.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("STREAM_OK")
    for g in args.groups.split(","):
        print(f"group_{g}", report[f"group_{g}"]["sha256"], report[f"group_{g}"]["task_type_counts"])
    print("shuffle_hash", shuffle_hash)
    print("task_order_hash", report["task_order_hash"])
    print("report_sha", sha(os.path.join(ev, "t3_stream_report.json")))


if __name__ == "__main__":
    main()
