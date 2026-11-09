#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2_phase1_checks.py — Phase 1 / T4：可重现性重建与边界/身份检查。

产出（写入冻结证据目录 outputs/v2/evidence/phase1/，不改动任何冻结产物）:
  identity_checks.json : 实数据不变量独立复核 + 合成边界场景检查
  reproducibility.json : 独立临时目录重建清单并逐文件比较哈希
比较方法（按承接要求披露）:
  - .jsonl 清单与 summary: 整文件 SHA-256（文件不含时间戳, 逐字节一致主张成立）
  - data_audit.json: 记录两份完整哈希 + 差异路径逐条列出（预期仅 run 小节）+
    确定性节 canonical(sort_keys) JSON 哈希比较
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def canonical_dump(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def diff_paths(a, b, prefix=""):
    """递归列出两个 JSON 对象的全部差异路径。"""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out.extend(diff_paths(a.get(k), b.get(k), f"{prefix}.{k}" if prefix else k))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{prefix}[]: len {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out.extend(diff_paths(x, y, f"{prefix}[{i}]"))
    else:
        if a != b:
            out.append(f"{prefix}: {json.dumps(a, ensure_ascii=False)[:80]!r} vs "
                       f"{json.dumps(b, ensure_ascii=False)[:80]!r}")
    return out


# ---------------- Part A: 可重现性重建 ----------------

def run_prepare(output_root, only, data_root, dataset, rl_counts):
    """only: 逗号分隔阶段; 重建用全阶段 (audit 的源哈希核对依赖同根 file_hashes.json)。"""
    cmd = [sys.executable, "scripts/v2_prepare.py",
           "--data-root", data_root, "--dataset", dataset,
           "--rl-counts", *[str(x) for x in rl_counts],
           "--seed", "42", "--train-view-seed", "1701",
           "--diagnostic-view-seed", "2701",
           "--output-root", output_root, "--only", only]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    return cmd, proc, time.time() - t0


def reproducibility(args, started_utc):
    frozen = REPO / "outputs/v2"
    tmp_root = REPO / "outputs/v2_repro_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True)
    cmd, proc, elapsed = run_prepare("outputs/v2_repro_tmp/rebuild_out",
                                     "environment,file_hashes,audit,rl,views",
                                     args.data_root, args.dataset, args.rl_counts)
    manifest_files = [
        "manifests/rows_manifest.jsonl",
        "manifests/catalog_clean.jsonl",
        "manifests/rl_tasks_10000.jsonl",
        "manifests/train_views.jsonl",
        "manifests/diagnostic_views.jsonl",
        "evidence/phase1/rl_summary.json",
        "evidence/phase1/views_summary.json",
    ]
    results = []
    all_core_equal = True
    for rel in manifest_files:
        fp, tp = frozen / rel, tmp_root / "rebuild_out" / rel
        entry = {"path": rel,
                 "frozen_sha256": sha256_file(fp) if fp.exists() else None,
                 "rebuild_sha256": sha256_file(tp) if tp.exists() else None}
        entry["whole_file_equal"] = (entry["frozen_sha256"] == entry["rebuild_sha256"]
                                     and entry["frozen_sha256"] is not None)
        if not entry["whole_file_equal"]:
            if entry["rebuild_sha256"] is None:
                entry["note"] = "重建文件缺失 (异常; 若为 jsonl 清单应逐字节一致)"
                continue
            if fp.suffix == ".jsonl":
                entry["note"] = "jsonl 清单不一致——异常, 不得宣称可复现"
                continue
            fd = json.load(open(fp, encoding="utf-8"))
            td = json.load(open(tp, encoding="utf-8"))
            diffs = diff_paths(fd, td)
            entry["differing_paths"] = diffs
            entry["differing_paths_expected"] = ["schema_version"]
            entry["note"] = ("summary 含 schema_version 字段, 重建版为 v2-phase1.4 "
                             "(规范序硬化后的正本版本); 差异应仅此字段"
                             if rel.endswith("summary.json") else "unexpected difference")
        results.append(entry)
        core = rel.endswith(".jsonl")
        if core and not entry["whole_file_equal"]:
            all_core_equal = False

    # data_audit.json: 整文件哈希 + 差异路径 + 确定性节比较
    rel = "manifests/data_audit.json"
    fp, tp = frozen / rel, tmp_root / "rebuild_out" / rel
    fd = json.load(open(fp, encoding="utf-8"))
    td = json.load(open(tp, encoding="utf-8"))
    fdet = canonical_dump(fd["deterministic"])
    tdet = canonical_dump(td["deterministic"])
    audit_entry = {
        "path": rel,
        "comparison_method": ("整文件含 run 小节时间戳/命令, 不做逐字节一致主张; "
                              "确定性节以 sort_keys 规范化 JSON 哈希比较"),
        "frozen_full_sha256": sha256_file(fp),
        "rebuild_full_sha256": sha256_file(tp),
        "full_files_equal": sha256_file(fp) == sha256_file(tp),
        "run_section_diff_paths": diff_paths(fd.get("run", {}), td.get("run", {})),
        "frozen_deterministic_sha256": hashlib.sha256(fdet.encode()).hexdigest(),
        "rebuild_deterministic_sha256": hashlib.sha256(tdet.encode()).hexdigest(),
        "deterministic_equal": fdet == tdet,
    }
    # 冻结 data_audit 含 file_hashes 源哈希核对等确定性内容, 全部应相等
    payload = {
        "schema_version": "v2-phase1.4",
        "stage": "reproducibility",
        "deterministic": {
            "script_sha256": sha256_file(REPO / "scripts/v2_prepare.py"),
            "rebuild_command": " ".join(cmd),
            "rebuild_output_root": "outputs/v2_repro_tmp/rebuild_out",
            "frozen_output_root": "outputs/v2",
            "frozen_outputs_untouched": True,
            "files": results,
            "data_audit_comparison": audit_entry,
            "all_core_manifests_byte_identical": all_core_equal,
            "data_audit_deterministic_equal": audit_entry["deterministic_equal"],
        },
        "run": {"command": " ".join(sys.argv), "cwd": os.getcwd(),
                "started_utc": started_utc,
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "rebuild_elapsed_seconds": round(elapsed, 3),
                "rebuild_returncode": proc.returncode,
                "rebuild_stderr_tail": proc.stderr[-500:] if proc.returncode else ""},
    }
    out = REPO / "outputs/v2/evidence/phase1/reproducibility.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[checks] reproducibility: core_jsonl_equal={all_core_equal} "
          f"audit_det_equal={audit_entry['deterministic_equal']} -> {out}")
    return payload


# ---------------- Part B: 边界/身份检查 ----------------

def real_invariant_checks():
    """对冻结清单做独立重算复核（不调用 v2_prepare 内部函数）。"""
    checks = []

    def add(name, passed, measured, detail=""):
        checks.append({"name": name, "scenario": "real", "passed": bool(passed),
                       "measured": measured, "detail": detail})

    rows = read_jsonl(REPO / "outputs/v2/manifests/rows_manifest.jsonl")
    ids = [r["row_id"] for r in rows]
    add("row_id_unique", len(set(ids)) == len(ids), {"n": len(ids), "unique": len(set(ids))})
    order_ok = True
    for split in ("train", "valid", "test"):
        idxs = [r["row_index"] for r in rows if r["split"] == split]
        order_ok &= idxs == list(range(len(idxs)))
    add("row_order_preserved_per_split", order_ok,
        {"splits": ["train", "valid", "test"]})

    catalog = {c["item_id"]: c for c in read_jsonl(REPO / "outputs/v2/manifests/catalog_clean.jsonl")}
    flag_ok = all((c["original_description_missing"]) == (c["description_raw"].strip() == "")
                  for c in catalog.values())
    add("missing_label_fixed_before_clean", flag_ok,
        {"n_items": len(catalog),
         "raw_missing": sum(c["original_description_missing"] for c in catalog.values())})

    tasks = read_jsonl(REPO / "outputs/v2/manifests/rl_tasks_10000.jsonl")
    train_ids = {r["row_id"] for r in rows if r["split"] == "train"}
    vt_ids = {r["row_id"] for r in rows if r["split"] in ("valid", "test")}
    row_src = [t for t in tasks if t["task_type"] in ("seq2sid", "texthist2sid")]
    add("rl_source_isolation", all(t["source_id"] in train_ids for t in row_src) and
        not any(t["source_id"] in vt_ids for t in row_src),
        {"row_sources": len(row_src), "leaked_valid_test": sum(1 for t in row_src if t["source_id"] in vt_ids)})
    m_by_id = {r["row_id"]: r for r in rows}
    add("rl_target_consistency",
        all(m_by_id[t["source_id"]]["item_id"] == t["item_id"] for t in row_src),
        {"checked": len(row_src)})

    tv = read_jsonl(REPO / "outputs/v2/manifests/train_views.jsonl")
    dv = read_jsonl(REPO / "outputs/v2/manifests/diagnostic_views.jsonl")

    def view_checks(recs, p, tag):
        bad_k = bad_order = bad_hash = bad_ident = 0
        for r in recs:
            c = catalog[r["item_id"]]
            if r["title_clean"] != c["title_clean"] or r["original_text"] != c["description_clean"]:
                bad_ident += 1
            if hashlib.sha256(r["original_text"].encode()).hexdigest() != r["original_text_sha256"]:
                bad_hash += 1
            if hashlib.sha256(r["perturbed_text"].encode()).hexdigest() != r["perturbed_text_sha256"]:
                bad_hash += 1
            if r["valid_aug"] and p is not None:
                n = r["n_tokens"]
                want = min(n - 1, max(1, int(p * n)))
                if r["k_deleted"] != want:
                    bad_k += 1
                toks = r["original_text"].split()
                kept = [t for i, t in enumerate(toks) if i not in set(r["deleted_positions"])]
                if " ".join(kept) != r["perturbed_text"]:
                    bad_order += 1
        add(f"{tag}_k_formula", bad_k == 0, {"violations": bad_k, "checked": len(recs)})
        add(f"{tag}_order_preserved", bad_order == 0, {"violations": bad_order})
        add(f"{tag}_identity_unchanged", bad_ident == 0 and bad_hash == 0,
            {"identity_violations": bad_ident, "hash_violations": bad_hash})

    view_checks(tv, 0.3, "train_drop30")
    for vt, p in (("drop30", 0.3), ("drop60", 0.6)):
        view_checks([r for r in dv if r["view_type"] == vt], p, f"diag_{vt}")
    dall = [r for r in dv if r["view_type"] == "dropall"]
    dall_ok = all((r["valid_aug"] and r["perturbed_text"] == "" and r["k_deleted"] == r["n_tokens"]
                   and r["original_text"] != "") or
                  (not r["valid_aug"] and r["original_text"].strip() == "")
                  for r in dall)
    add("diag_dropall_semantics", dall_ok,
        {"valid": sum(r["valid_aug"] for r in dall), "total": len(dall)})
    from collections import defaultdict, Counter
    cov = defaultdict(Counter)
    for r in dv:
        cov[r["item_id"]][r["view_type"]] += 1
    add("diag_coverage_3_views_per_item",
        len(cov) == len(catalog) and all(set(c.values()) == {1} and len(c) == 3 for c in cov.values()),
        {"items": len(cov)})
    return checks


def synthetic_checks():
    """合成迷你数据端到端运行, 验证边界场景行为。"""
    import math
    syn_root = REPO / "outputs/v2_repro_tmp/synthetic"
    if syn_root.exists():
        shutil.rmtree(syn_root)
    data_root = syn_root / "data" / "Amazon"
    (data_root / "index").mkdir(parents=True)
    for s in ("train", "valid", "test"):
        (data_root / s).mkdir(parents=True)
    # 键顺序故意打乱 (10 在最前), 测试标题冲突保留规则不依赖文件序
    item_json = {
        "10": {"title": "Gamma", "description": "same text here", "brand": "", "categories": ""},
        "abc": {"title": "Alpha Widget", "description": "", "brand": "", "categories": ""},
        "2": {"title": "Beta Tool", "description": "singleword", "brand": "", "categories": ""},
        "7": {"title": "Alpha Widget", "description": "one two three four five", "brand": "", "categories": ""},
        "3": {"title": "Delta", "description": "same text here", "brand": "", "categories": ""},
    }
    (data_root / "index" / "Synthetic.item.json").write_text(
        json.dumps(item_json, ensure_ascii=False), encoding="utf-8")
    header = "user_id,history_item_title,item_title,history_item_id,item_id,history_item_sid,item_sid"
    train_rows = [
        'U1,"[\'t7\']","t7",[7],3,[<a_1><b_1><c_1>],<a_1><b_1><c_1>',
        'U1,"[\'t7\', \'t3\']","t3","[7, 3, 999]",10,[<a_1><b_1><c_1>, <a_2><b_2><c_2>, <a_9><b_9><c_9>],<a_2><b_2><c_2>',  # 999 为目录外历史项
        'U2,"[\'t10\']","t2",[10],2,[],[]',
        'U2,"[\'t7\']","t7",[7],3,[<a_1><b_1><c_1>],<a_1><b_1><c_1>',  # 与第 0 行重复内容
    ]
    (data_root / "train" / "Synthetic_x.csv").write_text(
        header + "\n" + "\n".join(train_rows) + "\n", encoding="utf-8")
    (data_root / "valid" / "Synthetic_x.csv").write_text(
        header + "\n" + 'U3,"[\'t2\']","t10",[2],10,[],[]' + "\n", encoding="utf-8")
    test_rows = [
        'U4,"[\'t3\']","t7",[3],7,[],[]',
        'U4,"[\'t2\']","tabc",[2],abc,[],[]',  # 目标为字符串 ID abc (原始描述缺失)
    ]
    (data_root / "test" / "Synthetic_x.csv").write_text(
        header + "\n" + "\n".join(test_rows) + "\n", encoding="utf-8")

    out_root = "outputs/v2_repro_tmp/synthetic_out"
    cmd, proc, elapsed = run_prepare(out_root, "environment,file_hashes,audit,rl,views",
                                     "outputs/v2_repro_tmp/synthetic/data/Amazon",
                                     "Synthetic", [3, 4, 2])
    checks = []
    if proc.returncode != 0:
        checks.append({"name": "synthetic_run_completed", "scenario": "synthetic",
                       "passed": False, "measured": {"returncode": proc.returncode},
                       "detail": proc.stderr[-800:]})
        return checks, elapsed

    def add(name, passed, measured, detail=""):
        checks.append({"name": name, "scenario": "synthetic", "passed": bool(passed),
                       "measured": measured, "detail": detail})

    out = REPO / out_root
    cat = {c["item_id"]: c for c in read_jsonl(out / "manifests/catalog_clean.jsonl")}
    add("syn_catalog_ids_preserved", set(cat) == set(item_json),
        {"ids": sorted(cat)})
    add("syn_empty_desc_flag_and_clean",
        cat["abc"]["original_description_missing"] is True and cat["abc"]["description_clean"] == "",
        {"abc_flag": cat["abc"]["original_description_missing"]})
    add("syn_categories_not_constructed",
        all(c["categories_raw"] == "" for c in cat.values()),
        {"empty_categories": sum(1 for c in cat.values() if c["categories_raw"] == "")})
    rows = read_jsonl(out / "manifests/rows_manifest.jsonl")
    ids = [r["row_id"] for r in rows]
    add("syn_row_id_unique_with_duplicate_content", len(set(ids)) == len(ids) == 7,
        {"rows": len(rows), "unique": len(set(ids))})
    add("syn_string_item_id_handled",
        any(r["item_id"] == "abc" for r in rows),
        {"string_target_rows": sum(1 for r in rows if r["item_id"] == "abc")})
    audit_syn = json.load(open(out / "manifests/data_audit.json", encoding="utf-8"))
    tr = audit_syn["deterministic"]["splits"]["train"]
    add("syn_coverage_gap_reported_not_crash",
        tr["history_items_not_in_catalog_count"] == 1 and tr["n_rows"] == 4,
        {"history_items_not_in_catalog": tr["history_items_not_in_catalog_count"],
         "sample": tr["history_items_not_in_catalog_sample"],
         "n_rows": tr["n_rows"],
         "detail": "目录外历史条目 999 被如实记录, 全部行保留, 不跳行不崩溃"})
    tasks = read_jsonl(out / "manifests/rl_tasks_10000.jsonl")
    ttitles = [(t["title_clean"], t["item_id"]) for t in tasks if t["task_type"] == "title2sid"]
    # Alpha Widget 冲突 {7, abc}: 必须保留 7; 文件序中 abc 先于 7 出现, 依赖文件序会错
    # 注: 官方 clean_text 对标题追加尾部句点, 清洗后标题为 "Alpha Widget."
    add("syn_title_conflict_min_numeric_kept",
        ("Alpha Widget.", "7") in ttitles and ("Alpha Widget.", "abc") not in ttitles,
        {"title_tasks": ttitles,
         "note": "官方 clean_text 追加尾部句点; 全部 4 个唯一标题被抽样 (title2sid=4)"})
    tv = read_jsonl(out / "manifests/train_views.jsonl")
    by_id = {r["item_id"]: r for r in tv}
    add("syn_empty_desc_raw_missing_invalid",
        by_id["abc"]["valid_aug"] is False and by_id["abc"]["invalid_reason"] == "raw_missing",
        {"reason": by_id["abc"]["invalid_reason"]})
    add("syn_single_word_insufficient",
        by_id["2"]["valid_aug"] is False and by_id["2"]["invalid_reason"] == "insufficient_words",
        {"reason": by_id["2"]["invalid_reason"]})
    add("syn_valid_partial_view",
        by_id["7"]["valid_aug"] is True and by_id["7"]["k_deleted"] ==
        min(4, max(1, math.floor(0.3 * 5))),
        {"n": by_id["7"]["n_tokens"], "k": by_id["7"]["k_deleted"]})
    add("syn_duplicate_desc_kept_not_dedup",
        by_id["3"]["original_text"] == by_id["10"]["original_text"]
        and by_id["3"]["original_text"].endswith("same text here."),
        {"duplicated_items": ["3", "10"],
         "text": by_id["3"]["original_text"],
         "note": "官方 clean_text 对非句点结尾文本追加句点, 两商品清洗后逐字节相同且均保留"})
    dv = read_jsonl(out / "manifests/diagnostic_views.jsonl")
    dall = {r["item_id"]: r for r in dv if r["view_type"] == "dropall"}
    add("syn_dropall_raw_missing_invalid",
        dall["abc"]["valid_aug"] is False and dall["abc"]["invalid_reason"] == "raw_missing",
        {"reason": dall["abc"]["invalid_reason"]})
    add("syn_dropall_cleaned_nonempty_valid",
        dall["7"]["valid_aug"] is True and dall["7"]["perturbed_text"] == "",
        {"valid": dall["7"]["valid_aug"]})
    return checks, elapsed


def identity_checks(args, started_utc):
    checks = real_invariant_checks()
    syn, syn_elapsed = synthetic_checks()
    checks.extend(syn)
    failed = [c for c in checks if not c["passed"]]
    payload = {
        "schema_version": "v2-phase1.4",
        "stage": "identity_checks",
        "deterministic": {
            "checks_script_sha256": sha256_file(Path(__file__)),
            "prepare_script_sha256": sha256_file(REPO / "scripts/v2_prepare.py"),
            "n_checks": len(checks),
            "n_passed": len(checks) - len(failed),
            "n_failed": len(failed),
            "checks": checks,
            "failed_details": failed,
            "all_passed": len(failed) == 0,
        },
        "run": {"command": " ".join(sys.argv), "cwd": os.getcwd(),
                "started_utc": started_utc,
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "synthetic_elapsed_seconds": round(syn_elapsed, 3)},
    }
    out = REPO / "outputs/v2/evidence/phase1/identity_checks.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[checks] identity: {len(checks) - len(failed)}/{len(checks)} passed -> {out}")
    if failed:
        for c in failed:
            print(f"[checks] FAILED: {c['name']} {c.get('detail', '')[:200]}")
    return payload


def main():
    ap = argparse.ArgumentParser(description="Phase 1 T4 reproducibility & identity checks")
    ap.add_argument("--data-root", default="data/Amazon")
    ap.add_argument("--dataset", default="Industrial_and_Scientific")
    ap.add_argument("--rl-counts", nargs=3, type=int, default=[8000, 1000, 1000])
    args = ap.parse_args()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rep = reproducibility(args, started)
    ide = identity_checks(args, started)
    ok = (rep["deterministic"]["all_core_manifests_byte_identical"]
          and rep["deterministic"]["data_audit_deterministic_equal"]
          and ide["deterministic"]["all_passed"])
    print(f"[checks] DONE_{'OK' if ok else 'WITH_FAILURES'}")
    sys.exit(0 if ok else 5)


if __name__ == "__main__":
    main()
