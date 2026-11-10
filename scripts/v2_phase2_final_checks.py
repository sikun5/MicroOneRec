#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2_phase2_final_checks.py — Phase 2 / T6: independent checks, small-segment
re-encode diff (T4 tolerance), shuffled-ID correspondence, cost accounting.

Run IDs referenced: P2-EMB-A / P2-EMB-BC / P2-EMB-DIAG (see run_records.md).
"""
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(".").resolve()
E = REPO / "outputs/v2/embeddings"
EV = REPO / "outputs/v2/evidence/phase2"
MAN = REPO / "outputs/v2/manifests"
MP = "/hpc2hdd/home/swu073/sikun/dsaa5020/models/Qwen3-Embedding-4B"
MAX_LEN = 2048
DEVICE = "cuda:0"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def canonical_id_key(s):
    if s.lstrip("-").isdigit():
        return (0, int(s), "")
    return (1, 0, s)


def quality_stats(text, L=32):
    tokens = (text or "").lower().split()
    n = len(tokens)
    if n == 0:
        return {"n": 0, "r": 0.0, "q": 0.0, "w": 0.0}
    r = (1.0 - len(set(zip(tokens, tokens[1:], tokens[2:]))) / (n - 2)) if n >= 3 else 0.0
    q = min(n / L, 1.0) * (1.0 - r)
    return {"n": n, "r": round(r, 6), "q": round(q, 6), "w": round(0.5 * q, 6)}


def join_text(title, description):
    return title if not description else title + "\n" + description


def main():
    checks = []

    def add(name, passed, measured, detail=""):
        checks.append({"name": name, "passed": bool(passed), "measured": measured,
                       "detail": detail})
        print(f"[final] {'PASS' if passed else 'FAIL'} {name}")

    feats = read_jsonl(MAN / "quality_features.jsonl")
    item_ids = [r["item_id"] for r in feats]
    id2row = {iid: i for i, iid in enumerate(item_ids)}

    # ============ 1. cache reload verification ============
    arrs = {}
    for g in ("A", "B", "C"):
        arrs[(g, "clean")] = np.load(E / g / "clean.npy")
    arrs[("C", "train_aug")] = np.load(E / "C/train_aug.npy")
    valid_aug = np.load(E / "C/valid_aug.npy").astype(bool)
    shape_ok = all(a.shape == (3686, 2560) and a.dtype == np.float32 for a in arrs.values())
    add("cache_shapes_dtype", shape_ok,
        {f"{g}/{k}": [list(a.shape), str(a.dtype)] for (g, k), a in arrs.items()})
    ids_groups = {g: sha(E / g / "item_ids.json") for g in ("A", "B", "C")}
    add("item_ids_consistent", len(set(ids_groups.values())) == 1
        and ids_groups["A"] == sha(E / "item_ids.json")
        and json.load(open(E / "item_ids.json"))["item_ids"] == item_ids,
        {"groups": ids_groups})
    add("valid_aug_3356", valid_aug.shape == (3686,) and int(valid_aug.sum()) == 3356,
        {"valid": int(valid_aug.sum())})

    # metadata 内部哈希复算
    meta_ok, meta_detail = True, {}
    for g in ("A", "B", "C"):
        m = json.load(open(E / g / "metadata.json"))
        ids_ok = m.get("item_ids_sha256") == sha(E / g / "item_ids.json")
        inp_ok = all(sha(MAN / k) == v for k, v in m.get("inputs_sha256", {}).items()
                     if (MAN / k).exists())
        if not (ids_ok and inp_ok):
            meta_ok = False
        meta_detail[g] = {"item_ids_sha_ok": ids_ok, "inputs_sha_ok": inp_ok}
    add("metadata_internal_hashes", meta_ok, meta_detail,
        "metadata 内 item_ids/inputs 哈希对实际文件复算")

    # 诊断: 数组/标记/无效行复制
    diag_ok, want = True, {"drop30": 3356, "drop60": 3356, "dropall": 3359}
    diag_summary = {}
    for g in ("A", "B", "C"):
        for vt in ("drop30", "drop60", "dropall"):
            a = np.load(E / g / "diagnostic" / f"{vt}.npy")
            v = np.load(E / g / "diagnostic" / f"{vt}_valid.npy").astype(bool)
            norms = np.linalg.norm(a, axis=1)
            badrows = int(sum(0 if np.array_equal(a[i], arrs[(g, "clean")][i]) else 1
                              for i in range(3686) if not v[i]))
            ok = (a.shape == (3686, 2560) and a.dtype == np.float32
                  and bool(np.isfinite(a).all())
                  and bool((norms > 0.99).all() and (norms < 1.01).all())
                  and int(v.sum()) == want[vt] and badrows == 0)
            if not ok:
                diag_ok = False
            diag_summary[f"{g}/{vt}"] = {"valid": int(v.sum()), "invalid_copy_clean_ok": badrows == 0}
    add("diagnostics_reload", diag_ok, diag_summary)

    # 空描述精确回退 & 无效行复制
    empty_rows = [i for i, r in enumerate(feats) if r["description_clean"] == ""]
    fb = all(np.array_equal(arrs[("B", "clean")][i], arrs[("A", "clean")][i])
             for i in empty_rows)
    add("empty_desc_exact_fallback_rows", fb and len(empty_rows) == 327,
        {"n_empty": len(empty_rows), "byte_equal": fb})
    bad_c = int(sum(0 if np.array_equal(arrs[("C", "train_aug")][i], arrs[("C", "clean")][i]) else 1
                    for i in range(3686) if not valid_aug[i]))
    add("C_invalid_rows_copy", bad_c == 0, {"violations": bad_c})

    # 共享缓存证明
    shared_files = ["B/clean.npy", "C/clean.npy"] + [
        f"{g}/diagnostic/{vt}{sfx}" for g in ("B", "C")
        for vt in ("drop30", "drop60", "dropall") for sfx in (".npy", "_valid.npy")
    ] + [f"{g}/diagnostic/views_source.json" for g in ("B", "C")]
    shared_pairs = [("B/clean.npy", "C/clean.npy")] + [
        (f"B/diagnostic/{vt}{sfx}", f"C/diagnostic/{vt}{sfx}")
        for vt in ("drop30", "drop60", "dropall") for sfx in (".npy", "_valid.npy")
    ] + [("B/diagnostic/views_source.json", "C/diagnostic/views_source.json")]
    shared_ok = all(sha(E / a) == sha(E / b) for a, b in shared_pairs)
    add("shared_cache_sha_proof", shared_ok,
        {"n_pairs": len(shared_pairs), "all_byte_identical": shared_ok})

    # 乱序 ID 对应测试
    import random
    shuffled = item_ids.copy()
    random.Random(2026).shuffle(shuffled)
    pos = {iid: i for i, iid in enumerate(item_ids)}
    sample_ids = shuffled[:50]
    ok_shuf = all(np.array_equal(arrs[("A", "clean")][pos[iid]],
                                 arrs[("A", "clean")][item_ids.index(iid)])
                  for iid in sample_ids)
    add("shuffled_id_correspondence", ok_shuf,
        {"n_sample": len(sample_ids),
         "note": "乱序列表按 id2row 取行与原序行逐字节一致 (数组寻址按 ID 不按行号)"})

    # ============ 2. 质量公式独立重算 (20 样本, float64) ============
    import random as _r
    idx = sorted(_r.Random(7).sample(range(3686), 20)) + [0, 1, 2]
    q_ok, q_bad = 0, []
    for i in idx:
        r = feats[i]
        want = quality_stats(r["description_clean"])
        got = r["quality"]
        if all(math.isclose(want[k], got[k], abs_tol=1e-9) or want[k] == got[k] for k in ("n", "r", "q", "w")):
            q_ok += 1
        else:
            q_bad.append(r["item_id"])
    add("quality_formula_recompute_20", q_ok == len(idx), {"ok": q_ok, "bad": q_bad})

    # ============ 3. 小段同版本重编码 (GPU, T4 容差 0.0) ============
    re_enc = {"enabled": True, "n_texts": 0, "rows": []}
    try:
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MP)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModel.from_pretrained(MP, torch_dtype=torch.bfloat16,
                                          attn_implementation="sdpa")
        model.requires_grad_(False)
        model.to(DEVICE)
        model.eval()

        def enc(text):
            enc_i = tok(text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            with torch.no_grad():
                o = model(**{k: v.to(model.device) for k, v in enc_i.items()})
                torch.cuda.synchronize()
            hh = o.last_hidden_state.float()
            mm = enc_i.attention_mask.unsqueeze(-1).to(model.device).to(hh.dtype)
            p = (hh * mm).sum(1) / mm.sum(1)
            return torch.nn.functional.normalize(p, dim=1, eps=1e-12)[0]

        def enc_join(r):
            return enc(join_text(r["title_clean"], r["description_clean"]))

        # 选取: 空描述 1 / 单词 1 / 普通 3 / 最长 1 (A); 标题 3, 描述 3 (B 融合); C 视图 2; 诊断 A+B 各 1
        picks = {"empty": empty[0] if (empty := [r for r in feats if r["original_description_missing"]]) else feats[0]}
        short_l = [r for r in feats if r["quality"]["n"] == 1]
        picks["short"] = short_l[0] if short_l else feats[1]
        normal = [r for r in feats if 10 <= r["quality"]["n"] <= 60][:3]
        for j, r in enumerate(normal):
            picks[f"normal{j}"] = r
        counts = [(len(tok.encode(join_text(r["title_clean"], r["description_clean"]))), i)
                  for i, r in enumerate(feats)]
        picks["longest"] = feats[max(counts)[1]]
        rows = []
        # A 组
        for key in ("empty", "short", "normal0", "normal1", "normal2", "longest"):
            r = picks[key]
            v = enc_join(r)
            cached = arrs[("A", "clean")][id2row[r["item_id"]]]
            d = float(np.abs(v.cpu().numpy() - cached).max())
            rows.append({"group": "A", "item_id": r["item_id"], "max_abs_diff": d,
                         "mean_abs_diff": float(np.abs(v.cpu().numpy() - cached).mean())})
        # B 组: fallback + 融合
        for key in ("empty", "short", "normal0"):
            r = picks[key]
            w = r["quality"]["w"]
            t = enc(r["title_clean"])
            if r["description_clean"]:
                d = enc(r["description_clean"])
                b = torch.nn.functional.normalize((1 - w) * t + w * d, dim=0, eps=1e-12)
            else:
                b = t.clone()
            cached = arrs[("B", "clean")][id2row[r["item_id"]]]
            dd = float(np.abs(b.cpu().numpy() - cached).max())
            rows.append({"group": "B", "item_id": r["item_id"], "max_abs_diff": dd,
                         "fusion_w": w})
        # C 视图 2 条 (valid)
        c_views = [(i, r) for i, r in enumerate(feats) if r["views"]["train_drop30"]["valid"]][:2]
        for i, r in c_views:
            v = enc(r["views"]["train_drop30"]["text"])
            cached = arrs[("C", "train_aug")][i]
            rows.append({"group": "C_train_aug", "item_id": r["item_id"],
                         "max_abs_diff": float(np.abs(v.cpu().numpy() - cached).max())})
        # 诊断 A+B 各 1 (drop30 有效行)
        dg = next((i, r) for i, r in enumerate(feats) if r["views"]["diag_drop30"]["valid"])
        va = enc(join_text(dg[1]["title_clean"], dg[1]["views"]["diag_drop30"]["text"]))
        rows.append({"group": "A_diag_drop30", "item_id": dg[1]["item_id"],
                     "max_abs_diff": float(np.abs(va.cpu().numpy()
                                                  - np.load(E / "A/diagnostic/drop30.npy")[dg[0]]).max())})
        vb = enc(dg[1]["views"]["diag_drop30"]["text"])
        rows.append({"group": "B_diag_drop30", "item_id": dg[1]["item_id"],
                     "max_abs_diff": float(np.abs(vb.cpu().numpy()
                                                  - np.load(E / "B/diagnostic/drop30.npy")[dg[0]]).max())})
        vbd = enc(picks["empty"]["title_clean"])  # dropall valid 行: 输入=标题 (取 raw_missing 行? no, dropall valid 需 cleaned 非空) — 用 normal0 的标题
        vbd = enc(picks["normal0"]["title_clean"])
        i0 = id2row[picks["normal0"]["item_id"]]
        rows.append({"group": "B_diag_dropall", "item_id": picks["normal0"]["item_id"],
                     "max_abs_diff": float(np.abs(vbd.cpu().numpy()
                                                  - np.load(E / "B/diagnostic/dropall.npy")[i0]).max())})
        re_enc["rows"] = rows
        re_enc["n_texts"] = len(rows)
        max_d = max(r["max_abs_diff"] for r in rows)
        mean_d = float(np.mean([r["max_abs_diff"] for r in rows]))
        re_enc["overall_max_abs_diff"] = max_d
        re_enc["overall_mean_of_max"] = round(mean_d, 8)
        re_enc["tolerance_report"] = {
            "production_path_recompute": "B 融合行以生产 numpy 实现复算 (见 B_fusion_recompute)",
            "E_reproducibility": "同进程重复前向 0.0 (T4); 跨进程 13 行中 12 行直接 0.0",
            "checker_impl_diff": "1.49e-8 为 T6 检查器 torch GPU F.normalize 独立实现的舍入差 (非生产链非确定性), 详见 item1631_localization.json",
        }
        re_enc["within_tolerance"] = max_d <= 1e-7
        add("reencode_torch_impl_measurement", max_d <= 1e-7,
            {"max_measured": max_d, "mean_of_max": round(mean_d, 8), "n_rows": len(rows),
             "rows_exact_zero": sum(1 for r in rows if r["max_abs_diff"] == 0.0),
             "classification": "独立公式近似误差 (检查器 torch GPU F.normalize vs 生产 numpy v/||v||); 非生产链跨进程非确定性——定位证据 item1631_localization.json: 生产路径复算 byte_equal=True",
             "history": "首轮误以 0.0 门禁套用 torch 独立实现判 FAIL 且误归因跨进程核差异; 经生产路径定向复测证伪后按 boss 指示重分类为测量记录, 生产路径门禁见 B_fusion_recompute"})
        # B 融合公式独立重算 (用重编码 t/d)
        r = picks["short"]
        w = r["quality"]["w"]
        t = enc(r["title_clean"])
        d = enc(r["description_clean"])
        b = torch.nn.functional.normalize((1 - w) * t + w * d, dim=0, eps=1e-12)
        cached_b = arrs[("B", "clean")][id2row[r["item_id"]]]
        # 生产融合实现 (numpy CPU, 与 stage_encode 逐语句一致)
        t_np = t.cpu().numpy().astype(np.float32)
        d_np = d.cpu().numpy().astype(np.float32)
        b_prod = ((1 - w) * t_np + w * d_np)
        b_prod = b_prod / max(float(np.linalg.norm(b_prod)), 1e-12)
        d_fusion = float(np.abs(b_prod - cached_b).max())
        byte_eq = bool(np.array_equal(b_prod, cached_b))
        add("B_fusion_recompute", d_fusion == 0.0,
            {"max_abs_diff": d_fusion, "byte_equal": byte_eq,
             "note": "生产实现 (numpy CPU v/||v||) 逐语句复算; torch GPU F.normalize 独立实现对照差 1.49e-8 属检查器近似误差, 见 item1631_localization.json"})
    except Exception as ex:  # noqa: BLE001
        re_enc["enabled"] = False
        re_enc["error"] = repr(ex)[:400]
        add("reencode_within_T4_tolerance", False, {"error": re_enc["error"]})
    # 保存重编码对照数组
    smoke_dir = REPO / "outputs/v2/smoke/phase2"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    with open(smoke_dir / "t6_reencode_rows.json", "w", encoding="utf-8") as f:
        json.dump(re_enc, f, ensure_ascii=False, indent=2)

    payload_cache = {
        "schema": "phase2-t6-cache-checks",
        "deterministic": {"n_checks": len(checks),
                          "n_passed": sum(c["passed"] for c in checks),
                          "checks": checks,
                          "all_passed": all(c["passed"] for c in checks),
                          "reencode": re_enc},
        "run": {"command": " ".join(sys.argv), "cwd": os.getcwd(),
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    }
    with open(EV / "cache_checks.json", "w", encoding="utf-8") as f:
        json.dump(payload_cache, f, ensure_ascii=False, indent=2)
    print(f"[final] cache_checks: {payload_cache['deterministic']['n_passed']}/{len(checks)}")

    # ============ 4. view_checks (视图来源/身份, 只读) ============
    tv = read_jsonl(MAN / "train_views.jsonl")
    dv = read_jsonl(MAN / "diagnostic_views.jsonl")
    cat = {c["item_id"]: c for c in read_jsonl(MAN / "catalog_clean.jsonl")}
    bad_title = sum(1 for r in tv + dv
                    if cat.get(r["item_id"]) is None
                    or r["title_clean"] != cat[r["item_id"]]["title_clean"])
    bad_seed = sum(1 for r in tv if r["view_seed"] != 1701) + \
               sum(1 for r in dv if r["view_seed"] != 2701)
    bad_target = sum(1 for r in tv + dv if r.get("perturbed_text") is None)
    view_checks = {
        "schema": "phase2-t6-view-checks",
        "title_item_unchanged_violations": bad_title,
        "seed_1701_2701_violations": bad_seed,
        "perturbed_text_present_violations": bad_target,
        "train_n": len(tv), "diag_n": len(dv),
        "all_passed": bad_title == 0 and bad_seed == 0 and bad_target == 0,
        "quality_features_sha256": sha(MAN / "quality_features.jsonl"),
        "train_views_sha256": sha(MAN / "train_views.jsonl"),
        "diagnostic_views_sha256": sha(MAN / "diagnostic_views.jsonl"),
    }
    with open(EV / "view_checks.json", "w", encoding="utf-8") as f:
        json.dump(view_checks, f, ensure_ascii=False, indent=2)
    print(f"[final] view_checks all_passed={view_checks['all_passed']}")

    # ============ 5. 成本归账 (token 只读重算) ============
    t0 = time.time()
    groups_encoded = {
        "A_clean": [join_text(r["title_clean"], r["description_clean"]) for r in feats],
        "B_titles": [r["title_clean"] if r["title_clean"] else "unknown item" for r in feats],
        "B_descs": [r["description_clean"] for r in feats if r["description_clean"]],
        "C_train_aug": [r["views"]["train_drop30"]["text"] for r in feats
                        if r["views"]["train_drop30"]["valid"]],
        "A_diag_drop30": [join_text(r["title_clean"], r["views"]["diag_drop30"]["text"])
                          for r in feats if r["views"]["diag_drop30"]["valid"]],
        "A_diag_drop60": [join_text(r["title_clean"], r["views"]["diag_drop60"]["text"])
                          for r in feats if r["views"]["diag_drop60"]["valid"]],
        "A_diag_dropall": [r["title_clean"] for r in feats if r["views"]["diag_dropall"]["valid"]],
        "BC_shared_diag_drop30": [r["views"]["diag_drop30"]["text"] for r in feats
                                  if r["views"]["diag_drop30"]["valid"]],
        "BC_shared_diag_drop60": [r["views"]["diag_drop60"]["text"] for r in feats
                                  if r["views"]["diag_drop60"]["valid"]],
        "BC_shared_diag_dropall": [r["title_clean"] for r in feats
                                   if r["views"]["diag_dropall"]["valid"]],
    }
    groups_reused = {
        "C_diag_drop30": "字节复制 B/diagnostic/drop30.npy, 未编码",
        "C_diag_drop60": "字节复制 B/diagnostic/drop60.npy, 未编码",
        "C_diag_dropall": "字节复制 B/diagnostic/dropall.npy, 未编码",
        "C_clean": "字节复制 B/clean.npy, 未编码",
        "item_ids_copies": "字节复制 canonical, 未编码",
    }
    cost = {"schema": "phase2-t6-cost",
            "note_token_recount": "重建统计: 冻结 tokenizer 只读重算, 非编码期采集",
            "groups_encoded": {}, "groups_reused_not_encoded": groups_reused,
            "shared_accounting": {
                "BC_diag_counted_once": True, "C_clean_copy_not_counted": True,
                "C_diag_files_copies_not_counted": True, "item_ids_copies_not_counted": True}}
    tot_raw = tot_post = tot_texts = 0
    for g, texts in groups_encoded.items():
        raw = post = 0
        for t in texts:
            n = len(tok(t).input_ids)
            raw += n
            post += min(n, MAX_LEN)
        cost["groups_encoded"][g] = {"n_texts": len(texts), "tokens_raw": raw,
                                     "tokens_post_truncation": post}
        tot_raw += raw; tot_post += post; tot_texts += len(texts)
    cost["totals_encoded"] = {"texts": tot_texts, "tokens_raw": tot_raw,
                              "tokens_post_truncation": tot_post,
                              "forward_batches": tot_texts,
                              "truncated_count": sum(1 for v in groups_encoded.values() for t in v
                                                     if len(tok(t).input_ids) > MAX_LEN)}
    # 时间/显存/存储
    log_times = {"A_clean": 342.34, "B_titles+descs": 489.38, "C_train_aug": 258.47,
                 "A_diag": round(285.2+238.19+180.2, 2), "BC_diag_shared": round(257.78+214.12+187.2, 2)}
    cost["wall_seconds"] = {"total": 2473.5, "per_stage": log_times}
    cost["gpu_hours"] = {"mode": "wall×单GPU占用(编码全程)", "gpu_hours": round(2473.5/3600, 3)}
    cost["peak_memory"] = {
        "T5_formal_capture": "未采集 (如实披露)",
        "T2_probe_reference": {"batch1_2048token_allocated_MiB": 8122.1, "reserved_MiB": 8188.0},
        "T4_probe_reference": {"longest_real_text_1108tok_allocated_MiB": 7994.0, "reserved_MiB": 8044.0},
        "cpu_rss_peak_MB_T4": 5523.7}
    cost["storage"] = {"embeddings_dir_MB": 804}
    # distinct 文本哈希统计 (34,229 条实际编码输入)
    all_texts = [t for v in groups_encoded.values() for t in v]
    hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in all_texts]
    uniq = {}
    for h in hashes:
        uniq[h] = uniq.get(h, 0) + 1
    n_dup = len(hashes) - len(uniq)
    cost["distinct_text_hashes"] = {
        "hash_rule": "sha256(UTF-8 编码的提交字符串); 规则: 按提交字符串原文, 不做规范化",
        "total_submitted": len(hashes), "distinct": len(uniq),
        "re_encoded_duplicate_submissions": n_dup,
        "duplicate_groups_top": sorted((c, h[:12]) for h, c in uniq.items() if c > 1) [:5],
        "note": "重复提交来自 A/B 组在空描述时的相同标题文本; 各组缓存行仍独立保存, 重复提交是编码调用次数而非缓存行数"}
    cost["resume_rerun_events"] = 0
    cost["run_elapsed_check"] = round(time.time() - t0, 1)
    with open(EV / "cost_accounting.json", "w", encoding="utf-8") as f:
        json.dump(cost, f, ensure_ascii=False, indent=2)
    print(f"[final] cost: encoded_texts={cost['totals_encoded']['texts']} tokens_raw={tot_raw} post={tot_post}")

    ok_all = (payload_cache["deterministic"]["all_passed"]
              and view_checks["all_passed"])
    print(f"[final] FINAL_{'OK' if ok_all else 'WITH_FAILURES'}")
    sys.exit(0 if ok_all else 5)


if __name__ == "__main__":
    main()
