#!/usr/bin/env python3
"""Phase 6 / T3 orchestrator: initialization, data integrity, manifests, gradients.

Worker-owned checker. Produces:
  outputs/v2/evidence/phase6/initialization_checks.json
  outputs/v2/evidence/phase6/data_gradient_checks.json   (aggregated)
  outputs/v2/evidence/phase6/task_manifests/{G}_{task}.json
Debug run artifacts -> outputs/v2/debug/phase6/sft_smoke/attempt_001/
No training beyond single forward+backward probes; no recommendation metrics;
test split touched for identity/integrity only.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
OUT = os.path.join(ROOT, "outputs/v2/evidence/phase6")
SMOKE = os.path.join(ROOT, "outputs/v2/debug/phase6/sft_smoke/attempt_001")
SCRIPTS = os.path.join(ROOT, "scripts")
GROUPS = ["A", "B", "C"]
PY = os.path.join(ROOT, ".venv-phase3/bin/python")

CHECKS = []


def check(name, passed, detail):
    CHECKS.append({"name": name, "passed": bool(passed), "detail": str(detail)[:400]})


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, timeout=1800):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-init", action="store_true",
                    help="reuse existing initialization_checks.json (boss ruling: identical-source init not redone)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(SMOKE, "configs"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "task_manifests"), exist_ok=True)
    t0 = time.time()
    run_log = {"commands": [], "skip_init": args.skip_init}

    # archive rejected pervitem design artifacts as history (boss: keep history)
    hist = os.path.join(SMOKE, "history_t3_title2sid_pervitem_rejected")
    os.makedirs(hist, exist_ok=True)
    for fn in os.listdir(os.path.join(OUT, "task_manifests")):
        srcp = os.path.join(OUT, "task_manifests", fn)
        dstp = os.path.join(hist, fn)
        if not os.path.exists(dstp):
            with open(srcp, "rb") as a, open(dstp, "wb") as b:
                b.write(a.read())
    for extra in ("task_manifest_summary.json", "data_gradient_checks.json"):
        p = os.path.join(OUT, extra)
        if os.path.exists(p):
            dstp = os.path.join(hist, extra)
            if not os.path.exists(dstp):
                with open(p, "rb") as a, open(dstp, "wb") as b:
                    b.write(a.read())

    # ---------- 0. source hashes ----------
    src = {}
    for fn in ("v2_sft_data.py", "v2_sft_init.py", "v2_sft.py", "v2_sft_gradcheck.py", "t3_checks.py"):
        p = os.path.join(SCRIPTS, fn) if fn != "t3_checks.py" else os.path.join(OUT, fn)
        src[fn] = sha256_file(p)
        if fn != "t3_checks.py":
            with open(p, "rb") as a, open(os.path.join(SMOKE, "configs", fn), "wb") as b:
                b.write(a.read())

    # ---------- 1. initialization: fresh processes, or reuse per boss ruling ----------
    init_dir = os.path.join(SMOKE, "init_checks")
    os.makedirs(init_dir, exist_ok=True)
    init = {}
    existing = os.path.join(OUT, "initialization_checks.json")
    if args.skip_init and os.path.exists(existing):
        init = json.load(open(existing))["per_group"]
        init["A_repeat"] = {"reused": True}
        init_checks = json.load(open(existing))
        init_checks["reused_this_run"] = True
        print("[init] REUSED existing initialization_checks.json (no init-logic change)", flush=True)
        run_log["commands"].append({"cmd": "SKIP init (reuse)", "rc": 0})
    else:
        for g in GROUPS + ["A_repeat"]:
            grp = "A" if g == "A_repeat" else g
            out_json = os.path.join(init_dir, f"{g}_init.json")
            rc, so, se = run([PY, os.path.join(SCRIPTS, "v2_sft_init.py"),
                              "--group", grp, "--out", out_json], timeout=1800)
            ok = rc == 0 and os.path.exists(out_json)
            init[g] = json.load(open(out_json)) if ok else {"error": (so + se)[-400:], "rc": rc}
            print(f"[init {g}] rc={rc} ok={ok}", flush=True)
            run_log["commands"].append({"cmd": f"v2_sft_init.py --group {grp}", "rc": rc})
        init_checks = {
            "per_group": {g: init[g] for g in GROUPS},
            "state_dict_sha256": {g: init[g].get("state_dict_sha256") for g in GROUPS},
            "three_groups_identical": len({init[g].get("state_dict_sha256") for g in GROUPS}) == 1,
            "a_repeat_identical": init["A_repeat"].get("state_dict_sha256") == init["A"].get("state_dict_sha256"),
            "row_strategy_common": init["A"].get("row_strategy"),
        }
        with open(os.path.join(OUT, "initialization_checks.json"), "w") as f:
            json.dump(init_checks, f, ensure_ascii=False, indent=1)
    hashes = init_checks["state_dict_sha256"]
    all_same = init_checks["three_groups_identical"]
    repeat_same = init_checks.get("a_repeat_identical", False)
    check("init_three_groups_identical", all_same,
          json.dumps({g: (h or "")[:16] for g, h in hashes.items()}))
    check("init_a_repeat_identical", repeat_same, str(repeat_same))
    check("init_all_requires_grad", all(init[g].get("all_requires_grad") for g in GROUPS),
          f"trainable={init['A'].get('param_trainable')} total={init['A'].get('param_total')}")
    check("init_lm_head_tied", all(init[g].get("lm_head_tied_by_ptr") for g in GROUPS),
          str(init["A"].get("lm_head_tied_by_ptr")))
    check("init_legacy_rows_untouched",
          all(init[g].get("row_strategy", {}).get("legacy_pad_rows_untouched") for g in GROUPS),
          str(init["A"].get("row_strategy", {}).get("legacy_pad_rows_untouched")))
    check("init_token_map_ok", all(init[g].get("tokenizer", {}).get("ok") for g in GROUPS),
          str(init["A"].get("tokenizer", {}).get("n_map_mismatch")))

    # ---------- 2. data integrity (CPU, expanded tokenizer for decode check) ----------
    sys.path.insert(0, SCRIPTS)
    from transformers import AutoTokenizer
    from v2_sft_data import (FusionSeqRecDatasetV2, SidItemFeatDatasetV2,  # noqa: E402
                             SidSFTDatasetV2, compare_with_common_catalog,
                             duplicate_title_stats, load_group_bindings,
                             make_expanded_tokenizer, verify_group_bindings,
                             verify_info_txt, label_integrity)

    tokenizer = AutoTokenizer.from_pretrained(
        "/hpc2hdd/home/swu073/sikun/dsaa5020/models/Qwen3-0.6B-Base", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer, tok_ver = make_expanded_tokenizer(tokenizer)
    check("data_token_map_ok", tok_ver["ok"], f"final={tok_ver['final_len']} mismatches={tok_ver['n_map_mismatch']}")

    import csv as _csv
    data = {"groups": {}, "tokenizer_expansion": tok_ver}
    for g in GROUPS:
        b = load_group_bindings(g)
        vb = verify_group_bindings(b, tokenizer=tokenizer,
                                   token_id_map=json.load(open(os.path.join(
                                       ROOT, "outputs/v2/tokenizer/token_id_map.json")))["map"])
        vi = verify_info_txt(b)
        vc = compare_with_common_catalog(b)
        dup = duplicate_title_stats(b)
        # CSV row integrity (all three splits; identity/integrity only)
        splits = {}
        sid_of = {b["item_ids"][i]: b["sids"][i] for i in range(len(b["item_ids"]))}
        for split in ("train.csv", "valid.csv", "test.csv"):
            path = os.path.join(ROOT, f"outputs/v2/data/{g}/{split}")
            rows = 0
            bad = []
            seen_rid = set()
            with open(path, newline="", encoding="utf-8") as f:
                rd = _csv.DictReader(f)
                for row in rd:
                    rows += 1
                    rid = row["row_id"]
                    if rid in seen_rid:
                        bad.append({"row": rows, "issue": "dup_row_id"})
                    seen_rid.add(rid)
                    try:
                        hid = [str(x) for x in eval(row["history_item_id"])]
                        hsid = [str(x) for x in eval(row["history_item_sid"])]
                    except Exception:
                        bad.append({"row": rows, "issue": "history_parse"})
                        continue
                    if len(hid) != len(hsid):
                        bad.append({"row": rows, "issue": "history_len_mismatch"})
                    else:
                        for iid, isid in zip(hid, hsid):
                            if sid_of.get(iid) != isid:
                                bad.append({"row": rows, "issue": "history_sid_mismatch", "item": iid})
                                break
                    if sid_of.get(str(row["item_id"])) != str(row["item_sid"]):
                        bad.append({"row": rows, "issue": "target_sid_mismatch"})
            splits[split] = {"n_rows": rows, "n_bad": len(bad), "bad_head": bad[:10],
                             "n_unique_row_ids": len(seen_rid)}
        data["groups"][g] = {"bindings": vb, "info_txt": vi, "common_catalog": vc,
                             "duplicate_titles": dup, "csv_splits": splits}
        print(f"[data {g}] bindings_ok={vb['all_ok']} info_ok={vi['ok']} catalog_ok={vc['all_ok']} "
              f"csv_bad={[splits[s]['n_bad'] for s in splits]}", flush=True)
        check(f"bindings_{g}_3686_full", vb["all_ok"], f"n={vb['n_items']} fails={vb['n_fails']}")
        check(f"info_txt_{g}_full", vi["ok"], f"n={vi.get('n_items_checked')} fails={vi.get('n_fails')}")
        check(f"common_catalog_{g}_full", vc["all_ok"],
              f"missing={vc['n_missing_in_catalog']} fails={vc['n_fails']}")
        check(f"csv_integrity_{g}", all(splits[s]["n_bad"] == 0 for s in splits),
              json.dumps({s: splits[s]["n_bad"] for s in splits}))

    # ---------- 3. task manifests + label integrity (dataset builds) ----------
    manifests = {}
    for g in GROUPS:
        b = load_group_bindings(g)
        train_file = os.path.join(ROOT, f"outputs/v2/data/{g}/train.csv")
        valid_file = os.path.join(ROOT, f"outputs/v2/data/{g}/valid.csv")
        t_sid = SidSFTDatasetV2(train_file, tokenizer, max_len=512)
        t_item = SidItemFeatDatasetV2(b, tokenizer, max_len=512)
        t_fus = FusionSeqRecDatasetV2(train_file, b, tokenizer, max_len=512)
        v_sid = SidSFTDatasetV2(valid_file, tokenizer, max_len=512)
        print(f"[build {g}] sid={len(t_sid)} item={len(t_item)} fusion={len(t_fus)} val={len(v_sid)}", flush=True)

        def _ids_hash(seq):
            h = hashlib.sha256()
            for s in seq:
                h.update((s + "\x1f").encode())
            return h.hexdigest()

        m_dir = os.path.join(OUT, "task_manifests")
        tasks_meta = {
            "sid_sft": (t_sid, [m["row_id"] for m in t_sid.metas],
                        "CSV file order; label = [-100]*prompt + target_sid_tokens + EOS; pad=-100"),
            "item_feat": (t_item, [m["item_id"] for m in t_item.metas],
                          "canonical item order; sid2title block (3686) then title2sid per-item (3686); duplicate titles kept with item_id provenance (no dict overwrite)"),
            "fusion_seqrec": (t_fus, [m["row_id"] for m in t_fus.metas],
                              "CSV file order; label = [-100]*prompt + title_tokens + EOS"),
        }
        for task, (ds, ids, rule) in tasks_meta.items():
            rec = {
                "task": task, "group": g, "n_samples": len(ds),
                "source_id_field": "item_id" if task == "item_feat" else "row_id",
                "source_ids_sha256": _ids_hash(ids),
                "source_ids_head": ids[:10],
                "order_rule": rule,
                "train_shuffle": "HF Dataset.shuffle(seed=42) applied ONCE over the 3-task concat before Trainer (frozen)",
                "validation_scope": "validation = SidSFTDatasetV2 on valid.csv only (main task)",
            }
            if task == "item_feat":
                rec["duplicate_title_census"] = data["groups"][g]["duplicate_titles"]
                rec["canonical_rule"] = ("title2sid: one sample per UNIQUE clean title; canonical item = "
                                         "numerically smallest item ID (digits as int; string sort never used); "
                                         "boss ruling 2026-11-12")
                # explicit canonical verification: every dup title picked numeric-min
                canon_bad = []
                for tid, itmeta in enumerate(t_item.metas):
                    if itmeta["task"] == "title2sid":
                        c = t_item.canonical_of[itmeta["title"]]
                        if c != itmeta["item_id"]:
                            canon_bad.append({"title": itmeta["title"][:40], "picked": itmeta["item_id"], "expected": c})
                rec["canonical_verification"] = {"n_title2sid": sum(1 for m in t_item.metas if m["task"] == "title2sid"),
                                                  "n_bad": len(canon_bad), "bad_head": canon_bad[:5],
                                                  "all_numeric_min": len(canon_bad) == 0}
                rec["order_cross_group_note"] = "canonical member IDs derive from identical title sets across groups; order = canonical item positions"
            if task == "sid_sft":
                rec["valid_rows"] = len(v_sid)
            with open(os.path.join(m_dir, f"{g}_{task}.json"), "w") as f:
                json.dump(rec, f, ensure_ascii=False, indent=1)
            manifests[f"{g}_{task}"] = rec
        li = [label_integrity(t_sid.samples, tokenizer, task="sid_sft"),
              label_integrity(v_sid.samples, tokenizer, task="sid_sft"),
              label_integrity(t_item.samples, tokenizer, task="text"),
              label_integrity(t_fus.samples, tokenizer, task="text")]
        data["groups"][g]["label_integrity"] = li
        check(f"label_integrity_{g}", all(x["all_ok"] for x in li),
              json.dumps([{ "t": x["task"], "bad": x["n_bad"]} for x in li]))
        canon_all_ok = all(m["item_id"] == min(m["member_item_ids"], key=lambda s: (0, int(s), "") if s.isdigit() else (1, 0, s))
                           for m in t_item.metas if m["task"] == "title2sid")
        check(f"manifest_counts_{g}", len(t_sid) == 36259 and len(t_item) == 7332 and len(t_fus) == 36259,
              f"sid={len(t_sid)} item={len(t_item)}(sid2title=3686+title2sid=3646) fusion={len(t_fus)}")
        check(f"title2sid_canonical_numeric_min_{g}", canon_all_ok and len(t_item.canonical_of) == 3646,
              f"unique_titles={len(t_item.canonical_of)} n_title2sid={sum(1 for m in t_item.metas if m['task']=='title2sid')}")
        del t_sid, t_item, t_fus, v_sid

    with open(os.path.join(OUT, "task_manifest_summary.json"), "w") as f:
        json.dump({"manifests": {k: {kk: vv for kk, vv in v.items() if kk != "duplicate_title_census"}
                                 for k, v in manifests.items()}, "data": data},
                  f, ensure_ascii=False, indent=1)

    # ---------- 4. gradient checks (GPU, fresh processes) ----------
    grad_dir = os.path.join(SMOKE, "gradient_checks")
    os.makedirs(grad_dir, exist_ok=True)
    grad = {}
    for g in GROUPS:
        out_json = os.path.join(grad_dir, f"{g}_gradcheck.json")
        rc, so, se = run([PY, os.path.join(SCRIPTS, "v2_sft_gradcheck.py"),
                          "--group", g, "--smoke_root", SMOKE], timeout=1800)
        ok = rc == 0 and os.path.exists(out_json)
        grad[g] = json.load(open(out_json)) if ok else {"error": (so + se)[-400:], "rc": rc}
        print(f"[grad {g}] rc={rc} ok={ok}", flush=True)
    clarification = {
        "measured_field": "embedding_rows_with_nonzero_grad = count of rows with |grad|>0 over the FULL embedding matrix (from the whole tied weight's .grad), per task per group",
        "actual_values": {g: {t: grad[g].get("tasks", {}).get(t, {}).get("embedding_rows_with_nonzero_grad")
                               for t in ("sid_sft", "item_feat", "fusion_seqrec")} for g in GROUPS if isinstance(grad.get(g), dict)},
        "interpretation": ("embedding_rows_with_nonzero_grad = 154977/154977 for every task and group: "
                            "with tie_word_embeddings=true the lm_head weight IS the input embedding "
                            "(same tensor, verified by data_ptr equality in initialization_checks), and "
                            "loss = Qwen3ForCausalLM.forward(labels=...) -> transformers ForCausalLMLoss, "
                            "FULL-vocab cross-entropy over 154977 logits with no restricted output; the "
                            "output-side gradient is dense over the whole vocabulary, so every embedding "
                            "row receives nonzero gradient. The earlier claim 'nonzero rows == tokens used "
                            "in the batch (unused rows zero)' was WRONG and unverified; input-side-only "
                            "sparsity was NOT isolated (unmeasured)."),
        "representative_rows": ("independent representative-row checks remain valid: consumed new-token rows "
                                 "(>=151669) and base rows, L0/14/27 attention q/k/v/o + MLP up/down, all "
                                 "finite and nonzero; requires_grad=all params does NOT imply every row was "
                                 "updated - parameter updates happen only after optimizer.step (T4 scope)"),
    }
    grad_agg = {
        "per_group": grad,
        "all_ok": all(grad[g].get("all_ok") for g in GROUPS),
        "clarification": clarification,
        "note": "single forward+backward probes only; no optimizer.step (T4 scope); "
                "run artifacts under outputs/v2/debug/phase6/sft_smoke/attempt_001/",
    }
    with open(os.path.join(OUT, "data_gradient_checks.json"), "w") as f:
        json.dump(grad_agg, f, ensure_ascii=False, indent=1)
    check("gradcheck_three_groups", grad_agg["all_ok"],
          json.dumps({g: grad[g].get("losses") for g in GROUPS})[:300])
    check("gradcheck_full_params", all(grad[g].get("params", {}).get("all_requires_grad") for g in GROUPS),
          str(grad["A"].get("params")))

    # ---------- 5. finalize ----------
    summary = {
        "stage": "phase6-t3-adaptation",
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "cwd": ROOT,
        "source_hashes": src,
        "smoke_root": SMOKE,
        "commands_and_rc": run_log["commands"],
        "skip_init": args.skip_init,
        "checks": CHECKS,
        "all_pass": all(c["passed"] for c in CHECKS),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT, "t3_checks.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("=== T3 SUMMARY ===")
    print("all_pass=", summary["all_pass"])
    for c in CHECKS:
        print(("PASS" if c["passed"] else "FAIL"), c["name"], "|", c["detail"][:140])


if __name__ == "__main__":
    main()
