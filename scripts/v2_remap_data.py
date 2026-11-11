#!/usr/bin/env python
"""Phase 4 T5: row / catalog / RL-task remapping onto each group's T4 SID index.
SCHEMA: phase4-v2_remap_data-1.0

Explicit remapping step of the T4 export flow (guide phase4_sid_diagnostics_export.md
2.3): consumes data/{A,B,C}/Industrial_and_Scientific.index.json, rebuilds the SID
columns of the original CSVs in original row order, emits per-group item.json /
info.txt / rl_tasks_10000_sid.jsonl, and audits every non-SID field row by row.
Any unmapped target/history id ABORTS (no row skipping). No user dedup, no
longest-history conversion, no RL training — interface only.
"""
import os, sys, json, csv, hashlib, ast, argparse, time
os.environ.setdefault("OMP_NUM_THREADS", "8")
from pathlib import Path
from collections import Counter, defaultdict

REPO = Path(__file__).resolve().parent.parent
METHODS = ("A", "B", "C")
SPLITS = ("train", "valid", "test")
CSV_PATH = "data/Amazon/{split}/Industrial_and_Scientific_5_2016-10-2018-11.csv"
HEADER = ["user_id", "history_item_title", "item_title", "history_item_id",
          "item_id", "history_item_sid", "item_sid"]

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def parse_ids(raw):
    v = ast.literal_eval(raw)
    return [str(x) for x in v]

def run(args):
    data_root = REPO / "outputs/v2/data"
    ev = REPO / args.evidence_dir
    ev.mkdir(parents=True, exist_ok=True)
    cat = {}
    for line in open(REPO / "outputs/v2/manifests/catalog_clean.jsonl"):
        r = json.loads(line)
        cat[str(r["item_id"])] = r

    # original manifest tasks (verbatim order)
    tasks_orig = [json.loads(l) for l in open(REPO / "outputs/v2/manifests/rl_tasks_10000.jsonl")]
    orig_task_ids = [t["task_id"] for t in tasks_orig]
    orig_source_ids = [t["source_id"] for t in tasks_orig]
    type_counts = Counter(t["task_type"] for t in tasks_orig)

    alignment = {"schema": "phase4-row-alignment-1.0", "date": "2026-11-11", "groups": {}}
    rl_checks = {"schema": "phase4-rl-mapping-checks-1.0", "date": "2026-11-11",
                 "n_tasks": len(tasks_orig), "type_counts": dict(type_counts),
                 "groups": {}}
    all_ok = True

    for m in METHODS:
        t0 = time.time()
        idx = json.load(open(data_root / m / "Industrial_and_Scientific.index.json"))
        ids = [str(i) for i in idx["item_ids"]]
        id2sid = dict(zip(ids, idx["sids"]))
        assert len(id2sid) == 3686

        # ---- catalog item.json (same text across groups, per-group SID) ----
        item = {}
        for iid in ids:
            c = cat[iid]
            item[iid] = {"title_clean": c["title_clean"], "description_clean": c["description_clean"],
                         "sid": id2sid[iid], "original_description_missing": bool(c["original_description_missing"])}
        (data_root / m / "Industrial_and_Scientific.item.json").write_text(
            json.dumps({"schema": "phase4-item-catalog-1.0", "method": m, "n_items": len(item),
                        "items": item}, ensure_ascii=False, indent=1))

        # ---- CSV remap: rebuild SID columns, keep every other field byte-equal ----
        split_stats = {}
        total_rows = 0
        for split in SPLITS:
            src = REPO / CSV_PATH.format(split=split)
            dst = data_root / m / f"{split}.csv"
            with open(src, newline="") as f:
                rows = list(csv.reader(f))
            header, body = rows[0], rows[1:]
            assert header == HEADER, f"unexpected header in {src}: {header}"
            align = {"n_rows": len(body), "non_sid_fields_equal": True,
                     "sid_rebuilt": True, "order_preserved": True}
            miss = None
            with open(dst, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["row_id"] + header)
                for k, row in enumerate(body):
                    uid, hist_title, item_title, hist_ids_raw, item_id_raw = row[:5]
                    item_id = row[4].strip()
                    hist_ids = parse_ids(row[3])
                    if item_id not in id2sid:
                        miss = f"{split} row {k}: unmapped target item {item_id}"
                        break
                    new_hist = []
                    for hid in hist_ids:
                        if hid not in id2sid:
                            miss = f"{split} row {k}: unmapped history item {hid}"
                            break
                        new_hist.append(id2sid[hid])
                    if miss:
                        break
                    # row_id = split:zero-based original data-row index (Phase 1 rows_manifest contract)
                    new_row = [f"{split}:{k}", uid, hist_title, item_title, hist_ids_raw, item_id_raw,
                               str(new_hist), id2sid[item_id]]
                    # non-SID fields byte-equal vs original
                    if new_row[1:6] != row[:5]:
                        align["non_sid_fields_equal"] = False
                    w.writerow(new_row)
                else:
                    pass
            if miss:
                raise RuntimeError(f"UNMAPPED_ID_ABORT {m}: {miss}")
            split_stats[split] = {"n_rows": len(body), **{k: v for k, v in align.items() if k != "n_rows"}}
            total_rows += len(body)

        L = 0
        for split in SPLITS:
            with open(REPO / CSV_PATH.format(split=split), newline="") as f:
                for row in list(csv.reader(f))[1:]:
                    L = max(L, len(parse_ids(row[3])))
        # info.txt = the consumable legal-SID catalog for this group, in the LOCKED
        # upstream consumer contract (convert_dataset_gpr.create_item_info_file):
        # "<sid>\t<title>\t<item_id>" per line, 3686 lines, index order
        title_by_id = {iid: cat[iid]["title_clean"] for iid in ids}
        (data_root / m / "info.txt").write_text(
            "\n".join(f"{sid}\t{title_by_id[iid]}\t{iid}"
                      for iid, sid in zip(ids, idx["sids"])) + "\n")
        # history-length measurement per split (explicit semantics; separate from L-layers=3 / SID-len=4)
        hist_stats = {}
        for split in SPLITS:
            with open(REPO / CSV_PATH.format(split=split), newline="") as f:
                lens = [len(parse_ids(r[3])) for r in list(csv.reader(f))[1:]]
            from collections import Counter as C2
            hist_stats[split] = {"max": max(lens), "dist": dict(sorted(C2(lens).items()))}
        (data_root / m / "dataset_stats.json").write_text(json.dumps({
            "schema": "phase4-dataset-stats-1.0", "method": m,
            "semantics": {"rq_layers": 3, "sid_token_len": 4,
                           "history_len_max_per_split": {s: hist_stats[s]["max"] for s in SPLITS}},
            "history_len_dist": {s: hist_stats[s]["dist"] for s in SPLITS},
            "n_items": 3686, "splits_rows": {"train": 36259, "valid": 4532, "test": 4533},
            "note": "info.txt is the legal-SID catalog (one SID per line); this file is the stats summary",
            "source": f"outputs/v2/data/{m}/Industrial_and_Scientific.index.json (Phase 4 T4)",
        }, ensure_ascii=False, indent=1))

        # ---- audits on this group's outputs + originals ----
        orig_rows = {}
        for split in SPLITS:
            with open(REPO / CSV_PATH.format(split=split), newline="") as f:
                orig_rows[split] = list(csv.reader(f))[1:]
        rem_rows = {}
        for split in SPLITS:
            with open(data_root / m / f"{split}.csv", newline="") as f:
                rem_rows[split] = list(csv.reader(f))[1:]
        n_equal = 0
        for split in SPLITS:
            o, r = orig_rows[split], rem_rows[split]
            assert len(o) == len(r)
            for k, (ro, rr) in enumerate(zip(o, r)):
                exp_hist = str([id2sid[h] for h in parse_ids(ro[3])])
                if rr[0] == f"{split}:{k}" and ro[:5] == rr[1:6] and rr[6] == exp_hist and rr[7] == id2sid[ro[4].strip()]:
                    n_equal += 1
        # row_id contract: unique across all 45324 rows; matches Phase 1 rows_manifest on join
        all_row_ids = [rr[0] for sp in SPLITS for rr in rem_rows[sp]]
        row_ids_unique = len(all_row_ids) == 45324 and len(set(all_row_ids)) == 45324
        manifest_by_id = {}
        with open(REPO / "outputs/v2/manifests/rows_manifest.jsonl") as f:
            for line in f:
                mr = json.loads(line)
                manifest_by_id[mr["row_id"]] = (mr["user_id"], mr["item_id"])
        join_ok = 0
        for sp in SPLITS:
            for rr in rem_rows[sp]:
                if manifest_by_id.get(rr[0]) == (rr[1], rr[5].strip()):
                    join_ok += 1
        test_users = len({r[1] for r in rem_rows["test"]})
        test_missing_targets = sum(1 for r in rem_rows["test"]
                                   if cat[r[5].strip()]["original_description_missing"])
        train_lines = [tuple(r) for r in orig_rows["train"]]
        dup_rows = len(train_lines) - len(set(train_lines))

        # ---- RL interface: per-task list, original fields verbatim + SID fields ----
        rl_out = REPO / "outputs/v2/data" / m / "rl_tasks_10000_sid.jsonl"
        mapped = 0
        conflict = defaultdict(set)
        with open(rl_out, "w") as f:
            for t in tasks_orig:
                t2 = dict(t)
                if t["item_id"] not in id2sid:
                    raise RuntimeError(f"UNMAPPED_ID_ABORT {m} rl task {t['task_id']}: target {t['item_id']}")
                t2["item_sid"] = id2sid[t["item_id"]]
                if "history_item_id_raw" in t:
                    hids = parse_ids(t["history_item_id_raw"])
                    hs = []
                    for hid in hids:
                        if hid not in id2sid:
                            raise RuntimeError(f"UNMAPPED_ID_ABORT {m} rl task {t['task_id']}: history {hid}")
                        hs.append(id2sid[hid])
                    t2["history_item_sid"] = hs
                f.write(json.dumps(t2, ensure_ascii=False) + "\n")
                mapped += 1
                        # CONTENT-LEVEL prompt key (Phase 1 canonical): only fields that enter the
                # actual model input. user_id/task_id/source_id are metadata and are NOT in
                # the key; the target item title/id/SID never enters the key for
                # seq2sid/texthist2sid (title2sid is the title->SID exception: title IS input).
                if t["task_type"] == "seq2sid":
                    pk = tuple(parse_ids(t["history_item_id_raw"]))
                elif t["task_type"] == "texthist2sid":
                    pk = tuple(parse_ids(t["history_item_id_raw"]))
                else:
                    pk = t.get("title_clean", "")
                conflict[(t["task_type"], pk)].add(t["item_id"])

        task_ids_out, source_ids_out = [], []
        for line in open(rl_out):
            r = json.loads(line)
            task_ids_out.append(r["task_id"]); source_ids_out.append(r["source_id"])
        order_ok = task_ids_out == orig_task_ids and source_ids_out == orig_source_ids
        conflict_groups = {k: sorted(v) for k, v in conflict.items() if len(v) > 1}
        # per-type breakdown (Phase 1 canonical content keys)
        per_type = {}
        for tt in ("seq2sid", "texthist2sid", "title2sid"):
            groups_tt = {k[1]: v for k, v in conflict.items() if k[0] == tt}
            multi_tt = {k: v for k, v in groups_tt.items() if len(v) > 1}
            inv_tt = sum(1 for t in tasks_orig if t["task_type"] == tt
                         and (k[1] in multi_tt for k in [next((kk for kk in conflict if kk[0] == tt and kk[1] == self_key(t)), None)])
                         ) if False else None
            per_type[tt] = {"distinct_prompts": len(groups_tt), "multi_target_groups": len(multi_tt)}
        # involved rows per type (tasks whose key group is multi-target)
        def self_key(t):
            if t["task_type"] == "title2sid":
                return (t["task_type"], t.get("title_clean", ""))
            return (t["task_type"], tuple(parse_ids(t["history_item_id_raw"])))
        per_type_involved = {tt: sum(1 for t in tasks_orig if t["task_type"] == tt
                                     and len(conflict.get(self_key(t), set())) > 1)
                             for tt in ("seq2sid", "texthist2sid", "title2sid")}
        inv = sum(per_type_involved.values())
        inv_srcs = {t["source_id"] for t in tasks_orig if len(conflict.get(self_key(t), set())) > 1}
        # SID-mapped content keys per group: 1:1 id->sid must reproduce identical grouping
        sid_conflict = defaultdict(set)
        for t in tasks_orig:
            if t["task_type"] == "title2sid":
                pk = (t["task_type"], t.get("title_clean", ""))
            else:
                pk = (t["task_type"], tuple(id2sid[h] for h in parse_ids(t["history_item_id_raw"])))
            sid_conflict[pk].add(id2sid[t["item_id"]])
        sid_multi = sum(1 for v in sid_conflict.values() if len(v) > 1)
        seq_multi = len([k for k in conflict_groups if k[0] == "seq2sid"])
        th_multi = len([k for k in conflict_groups if k[0] == "texthist2sid"])
        ti_multi = len([k for k in conflict_groups if k[0] == "title2sid"])
        seq_example = next(((k, v) for k, v in conflict_groups.items() if k[0] == "seq2sid"), None)
        seq_srcs = [t["source_id"] for t in tasks_orig
                    if seq_example and self_key(t) == seq_example[0]][:4]

        # catalog title conflicts + duplicate source ids (dataset-level, same for all groups)
        title_map = defaultdict(set)
        for iid, c in cat.items():
            title_map[c["title_clean"]].add(iid)
        title_conflicts = sum(1 for v in title_map.values() if len(v) > 1)
        dup_source = sum(1 for v in Counter(orig_source_ids).values() if v > 1)

        alignment["groups"][m] = {
            "split_rows": {s: split_stats[s]["n_rows"] for s in SPLITS},
            "total_rows": total_rows,
            "expected_total": 45324,
            "rows_field_equal": n_equal,
            "test_unique_users": test_users, "expected_users": 1987,
            "test_missing_desc_targets": test_missing_targets, "expected_missing": 447,
            "train_duplicate_rows_kept": dup_rows, "expected_duplicates": 3,
            "all_rows_rebuilt_from_index": True,
            "row_ids_unique_45324": row_ids_unique,
            "rows_manifest_join_matched": join_ok,
            "row_id_format": "split:zero-based original data-row index (identical across A/B/C)",
        }
        # info.txt consumable read check (no training), under the LOCKED consumer contract:
        # (a) trainer-style: split('\t')[0] -> 3686 unique four-token SIDs
        # (b) full TSV: sid/title/item_id columns consistent with index + item catalog
        # (c) all in the ACCEPTED T4 prefix trie (token-ID sequences from index)
        def _trie_accepts(trie, seq):
            node = trie
            for t in seq:
                if t not in node:
                    return False
                node = node[t]
            return "__leaf__" in node
        trie = json.load(open(data_root / m / "prefix_trie.json"))["trie"]
        info_lines = [l for l in (data_root / m / "info.txt").read_text().split("\n") if l]
        import re as _re
        pat_sid = _re.compile(r"^<a_\d+><b_\d+><c_\d+><d_\d+>$")
        idx_tok = idx["token_ids"]  # info.txt lines are in index order
        tsv = [l.split("\t") for l in info_lines]
        trainer_first = [r[0].strip() for r in tsv]
        info_ok = (len(info_lines) == 3686 and all(len(r) == 3 for r in tsv)
                   and len(set(trainer_first)) == 3686 and all(pat_sid.match(x) for x in trainer_first)
                   and trainer_first == idx["sids"]
                   and [r[2].strip() for r in tsv] == [str(i) for i in idx["item_ids"]]
                   and all(title_by_id[r[2].strip()] == r[1] for r in tsv)
                   and all(_trie_accepts(trie, [str(t) for t in tq]) for tq in idx_tok))
        rl_checks["groups"][m] = {
            "mapped_tasks": mapped, "task_source_order_preserved": order_ok,
            "prompt_key_definition": {
                "seq2sid": "tuple(history item ids) — content only; user_id/source_id/task_id excluded as metadata",
                "texthist2sid": "tuple(history item ids) — task carries no title field; same content rule",
                "title2sid": "title_clean (title->SID exception: title is the input)",
            },
            "per_type_conflicts": {
                "seq2sid": {**per_type["seq2sid"], "involved_task_rows": per_type_involved["seq2sid"]},
                "texthist2sid": {**per_type["texthist2sid"], "involved_task_rows": per_type_involved["texthist2sid"]},
                "title2sid": {**per_type["title2sid"], "involved_task_rows": per_type_involved["title2sid"]},
            },
            "phase1_reconciliation": {
                "seq2sid_expected": {"multi_groups": 296, "distinct_prompts": 7215},
                "texthist2sid_expected": {"multi_groups": 18, "distinct_prompts": 968},
                "measured_matches_phase1": (per_type["seq2sid"]["multi_target_groups"] == 296
                                            and per_type["seq2sid"]["distinct_prompts"] == 7215
                                            and per_type["texthist2sid"]["multi_target_groups"] == 18
                                            and per_type["texthist2sid"]["distinct_prompts"] == 968),
            },
            "same_prompt_multi_target_groups": len(conflict_groups),
            "involved_task_rows": inv,
            "involved_distinct_source_ids": len(inv_srcs),
            "sid_mapped_content_multi_target_groups": sid_multi,
            "example_conflict": {"task_type": seq_example[0][0],
                                 "history_key": json.dumps(seq_example[0][1], ensure_ascii=False)[:200],
                                 "targets": sorted(seq_example[1])[:6], "n_targets": len(seq_example[1]),
                                 "source_ids_sample": seq_srcs} if seq_example else None,
            "max_targets_in_one_group": max((len(v) for v in conflict_groups.values()), default=0),
            "interface": "per-task list (task_id keyed); no prompt-key dict collapsing",
            "title_conflict_item_groups": title_conflicts,
            "duplicate_source_ids": dup_source,
            "info_read_check": {
                "consumer_contract": "convert_dataset_gpr.create_item_info_file writes '<sid>\t<title>\t<item_id>' per line; minionerec_trainer.py:529 parses names via line.split('\t')[0].strip(); rl_gpr.py:79 same line-read",
                "consumer_sources_sha256": {
                    "convert_dataset_gpr.py": "a7a6d70db79b25901a2fc494f096662794044a1047e8dbf92c6428a5fc644b9d",
                    "minionerec_trainer.py": "d925718de95c54365da037f24c4884d75381efdaabbc697ad228021450067fe3",
                    "rl_gpr.py": "2dad52fd60885e779ea726c3478373a7f3ea24544b5da9615eb68d29daff456b"},
                "n_lines": len(info_lines),
                "trainer_style_parse_unique_sids": len(set(trainer_first)),
                "all_four_token": bool(all(pat_sid.match(x) for x in trainer_first)),
                "sids_match_index_order": trainer_first == idx["sids"],
                "item_ids_match_index": [r[2].strip() for r in tsv] == [str(i) for i in idx["item_ids"]],
                "titles_match_catalog": bool(all(title_by_id[r[2].strip()] == r[1] for r in tsv)),
                "all_in_trie_non_cartesian": bool(all(
                    _trie_accepts(trie, [str(t) for t in tq]) for tq in idx_tok)), "ok": bool(info_ok)},
        }
        gok = (total_rows == 45324 and n_equal == 45324 and test_users == 1987
               and test_missing_targets == 447 and dup_rows == 3 and mapped == 10000 and order_ok
               and row_ids_unique and join_ok == 45324)
        alignment["groups"][m]["passed"] = bool(gok)
        # texthist TEXT-level prompt keys: real model-input text history taken verbatim
        # from the source CSV row's history_item_title field (list of titles, in order)
        rows_by_split = {sp: orig_rows[sp] for sp in SPLITS}
        def _texthist_text_key(t):
            sp, ri = t["source_id"].split(":")
            titles = ast.literal_eval(rows_by_split[sp][int(ri)][1])
            return tuple(str(x) for x in titles)
        tg_text = defaultdict(set)
        for t in tasks_orig:
            if t["task_type"] == "texthist2sid":
                tg_text[_texthist_text_key(t)].add(t["item_id"])
        th_text_multi = {k: v for k, v in tg_text.items() if len(v) > 1}
        th_text_inv = sum(1 for t in tasks_orig if t["task_type"] == "texthist2sid"
                          and _texthist_text_key(t) in th_text_multi)
        th_text_srcs = sorted({t["source_id"] for t in tasks_orig if t["task_type"] == "texthist2sid"
                               and _texthist_text_key(t) in th_text_multi})
        th_text_tids = sorted({t["task_id"] for t in tasks_orig if t["task_type"] == "texthist2sid"
                               and _texthist_text_key(t) in th_text_multi})
        th_text_example = next(iter(th_text_multi.items())) if th_text_multi else None
        tg_sid_text = defaultdict(set)
        for t in tasks_orig:
            if t["task_type"] == "texthist2sid":
                tg_sid_text[_texthist_text_key(t)].add(id2sid[t["item_id"]])
        th_sid_text_multi = sum(1 for v in tg_sid_text.values() if len(v) > 1)
        rl_checks["groups"][m]["texthist_text_level_keys"] = {
            "text_field": "source row history_item_title (verbatim titles list, in order; the exact input strings)",
            "distinct_prompts": len(tg_text), "multi_target_groups": len(th_text_multi),
            "involved_task_rows": th_text_inv,
            "involved_distinct_source_ids": len(th_text_srcs),
            "involved_distinct_task_ids": len(th_text_tids),
            "id_level_kept_for_reference": {"distinct_prompts": 968, "multi_target_groups": 18, "involved_task_rows": 39},
            "sid_mapped_multi_target_groups_same_structure": th_sid_text_multi,
            "example": {"titles": json.dumps(list(th_text_example[0]), ensure_ascii=False)[:220],
                        "targets": sorted(th_text_example[1])[:6],
                        "source_ids_sample": th_text_srcs[:4]} if th_text_example else None,
        }
        rl_checks["groups"][m]["passed"] = bool(gok and mapped == 10000 and info_ok)
        all_ok &= gok
        print(f"{m}: rows={total_rows} equal={n_equal} users={test_users} miss={test_missing_targets} "
              f"dup={dup_rows} rl={mapped} order={order_ok} rowid_unique={row_ids_unique} "
              f"join={join_ok} wall={time.time()-t0:.1f}s", flush=True)

    alignment["all_passed"] = bool(all_ok)
    rl_checks["all_passed"] = bool(all_ok)
    (ev / "row_alignment.json").write_text(json.dumps(alignment, ensure_ascii=False, indent=1))
    (ev / "rl_mapping_checks.json").write_text(json.dumps(rl_checks, ensure_ascii=False, indent=1))

    # files inventory
    files = sorted(p for r in [REPO / "outputs/v2/data"] for p in r.rglob("*") if p.is_file())
    res = {"schema": "phase4-t5-files-1.0", "date": "2026-11-11", "n_files": len(files),
           "files": {str(p): {"sha256": sha(p), "bytes": p.stat().st_size} for p in files}}
    (ev / "t5_files_sha256.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("ROW_ALIGNMENT_ALL:", alignment["all_passed"], flush=True)
    print("RL_CHECKS_ALL:", rl_checks["all_passed"], flush=True)
    print("T5FILES_SHA256:", sha(ev / "t5_files_sha256.json"), flush=True)
    print("T5_REMAP_DONE", flush=True)

def main():
    ap = argparse.ArgumentParser(description="Phase 4 T5: remap rows/catalog/RL tasks onto T4 SID index")
    ap.add_argument("--evidence-dir", default="outputs/v2/evidence/phase4")
    ap.add_argument("--data-root", default="outputs/v2/data")
    args = ap.parse_args()
    run(args)

if __name__ == "__main__":
    main()
