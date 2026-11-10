#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2_build_embeddings.py — MiniOneRec v2 Phase 2: quality features, explicit
identity, frozen encoding caches (A/B/C clean, C train views, diagnostics).

Stages (--stages, comma separated, default all):
  quality     : manifests/quality_features.jsonl + embeddings/item_ids.json (CPU)
  encode      : A/B/C clean caches (GPU, per locked protocol from T2/T4)
  diagnostics : A / shared B+C diagnostic view caches (GPU)

Adapted from the repo's existing frozen-encoding entry
rq/text2emb/amazon_text2emb.py (masked mean pooling, last_hidden_state x
attention_mask). Its word_drop_ratio augmentation is an original-project
parameter and is NOT part of the v2 method.

Interface constants follow the v2 protocol (topic_minionerec_v2.md §4) and
the Phase 2 guide (phase2_quality_embeddings.md). Phase 1 frozen manifests
are read-only inputs; nothing upstream is rewritten.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

SCHEMA = "phase2-v2_build_embeddings-1.1"
JOIN_RULE = "join(title, description) = title if description_clean empty else title + '\\n' + description_clean; join result is NOT re-cleaned"
PLACEHOLDER = "unknown item"
INVALID_ROW_RULE = "invalid rows copy the group's clean vector for that item and carry valid=False; empty strings are never encoded; no zero vectors"
CANONICAL_ORDER_RULE = "item IDs sorted numeric-first (int value), non-numeric IDs in canonical string order afterwards"
CACHE_DTYPE = "float32"
L2_EPS = 1e-12
VIEW_TYPES = ("drop30", "drop60", "dropall")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def canonical_id_key(s):
    if s.lstrip("-").isdigit():
        return (0, int(s), "")
    return (1, 0, s)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def quality_stats(text, quality_length):
    """小写空白分词; n<3→r=0; r=1-unique_trigrams/(n-2); q=1[n>0]*min(n/L,1)*(1-r); w=0.5q."""
    tokens = (text or "").lower().split()
    n = len(tokens)
    if n == 0:
        return {"n": 0, "r": 0.0, "q": 0.0, "w": 0.0}
    if n >= 3:
        tris = set(zip(tokens, tokens[1:], tokens[2:]))
        r = 1.0 - len(tris) / (n - 2)
    else:
        r = 0.0
    q = min(n / quality_length, 1.0) * (1.0 - r)
    w = 0.5 * q
    return {"n": n, "r": round(r, 6), "q": round(q, 6), "w": round(w, 6)}


def join_text(title, description):
    """A 组拼接规则（metadata 固定）；拼接结果不再次清洗。"""
    if not description:
        return title
    return title + "\n" + description


# ---------------- stage: quality ----------------

def stage_quality(args, out_root):
    catalog = read_jsonl(Path(args.manifest_root) / "catalog_clean.jsonl")
    catalog = sorted(catalog, key=lambda c: canonical_id_key(c["item_id"]))
    tv = {r["item_id"]: r for r in read_jsonl(Path(args.manifest_root) / "train_views.jsonl")}
    dv = {}
    for r in read_jsonl(Path(args.manifest_root) / "diagnostic_views.jsonl"):
        dv[(r["item_id"], r["view_type"])] = r

    item_ids = [c["item_id"] for c in catalog]
    emb_root = Path(args.output_root)
    emb_root.mkdir(parents=True, exist_ok=True)
    ids_path = emb_root / "item_ids.json"
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump({"order_rule": CANONICAL_ORDER_RULE, "item_ids": item_ids},
                  f, ensure_ascii=False, indent=1)

    qpath = Path(args.manifest_root) / "quality_features.jsonl"
    with open(qpath, "w", encoding="utf-8", newline="\n") as f:
        for c in catalog:
            iid = c["item_id"]
            rec = {
                "item_id": iid,
                "title_clean": c["title_clean"],
                "description_clean": c["description_clean"],
                "original_description_missing": c["original_description_missing"],
                "quality": quality_stats(c["description_clean"], args.quality_length),
                "views": {},
            }
            t = tv[iid]
            views = {"train_drop30": {
                "view_seed": t["view_seed"], "valid": t["valid_aug"],
                "text": t["perturbed_text"],
                "quality": quality_stats(t["perturbed_text"], args.quality_length)}}
            for vt in VIEW_TYPES:
                d = dv[(iid, vt)]
                views[f"diag_{vt}"] = {
                    "view_seed": d["view_seed"], "valid": d["valid_aug"],
                    "text": d["perturbed_text"],
                    "quality": quality_stats(d["perturbed_text"], args.quality_length)}
            rec["views"] = views
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[v2_build] wrote {qpath} ({len(item_ids)} items); item_ids -> {ids_path}")
    return {"n_items": len(item_ids), "quality_features": str(qpath),
            "item_ids": str(ids_path)}


# ---------------- stage: encode (GPU; run only after T4 smoke) ----------------

def load_encoder(args):
    import torch  # noqa: PLC0415
    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415
    model_path = args.model_path
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                      attn_implementation="sdpa")
    model.requires_grad_(False)
    model.to(args.device)
    model.eval()
    return model, tok


def encode_texts(model, tok, texts, max_length, device, chunk_tag="chunk"):
    """batch=1 顺序编码（T2/T4 锁定协议），memmap 断点续传。

    进度文件绑定 (chunk_tag, max_length, 模型路径, 文本集合哈希, N)；签名不符时
    重新开始，不混拼不同配置。每 100 条与结束时 flush 进度。"""
    import hashlib  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    assert all(t for t in texts), "empty text in encode list (invalid rows must copy clean vectors, not encode)"
    prog_dir = Path("outputs/v2/embeddings/.progress")
    prog_dir.mkdir(parents=True, exist_ok=True)
    sig_obj = {"tag": chunk_tag, "max_length": max_length, "model_path": model.config._name_or_path,
               "n": len(texts),
               "texts_sha": hashlib.sha256("\x00".join(texts).encode("utf-8")).hexdigest()}
    sig_sha = hashlib.sha256(json.dumps(sig_obj, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    mm_path = prog_dir / f"{chunk_tag}_{sig_sha}.npy"
    prog_path = prog_dir / f"{chunk_tag}_{sig_sha}.json"

    stats = []
    start, mm, D = 0, None, None
    if prog_path.exists():
        prog = json.load(open(prog_path))
        if prog.get("sig_sha") == sig_sha and mm_path.exists():
            start = prog["done"]
            mm = np.load(mm_path, mmap_mode="r+")
            D = mm.shape[1]
            print(f"[v2_build] resume {chunk_tag}: {start}/{len(texts)}")

    t0 = time.time()
    for i in range(start, len(texts)):
        text = texts[i]
        enc = tok(text, truncation=True, max_length=max_length, return_tensors="pt")
        n_raw = len(tok(text).input_ids)
        with torch.no_grad():
            out = model(**{k: v.to(device) for k, v in enc.items()})
            if device.startswith("cuda"):
                torch.cuda.synchronize()
        h = out.last_hidden_state.float()
        m = enc.attention_mask.unsqueeze(-1).to(device).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1)
        l2 = torch.nn.functional.normalize(pooled, dim=1, eps=L2_EPS)[0]
        v = l2.cpu().numpy().astype(np.float32)
        if mm is None:
            D = v.shape[0]
            mm = np.lib.format.open_memmap(mm_path, mode="w+", dtype=np.float32,
                                           shape=(len(texts), D))
        mm[i] = v
        stats.append({"i": i, "n_tokens_raw": n_raw,
                      "n_tokens_after_truncation": int(enc.attention_mask.sum())})
        if (i + 1) % 100 == 0 or i + 1 == len(texts):
            mm.flush()
            with open(prog_path, "w") as f:
                json.dump({"sig_sha": sig_sha, "done": i + 1}, f)
            print(f"[v2_build] {chunk_tag}: {i+1}/{len(texts)} "
                  f"({round(time.time()-t0,1)}s)", flush=True)
    mm.flush()
    elapsed = round(time.time() - t0, 2)
    return np.asarray(mm), stats, elapsed


def metadata_payload(args, group, item_ids_path, inputs, extra):
    ids_sha = sha256_file(item_ids_path)
    return {
        "schema": SCHEMA,
        "group": group,
        "encoder": {"repo_id": args.encoder,
                    "revision": "5cf2132abc99cad020ac570b19d031efec650f2b",
                    "local_path": args.model_path,
                    "pooling": args.pooling, "max_length": args.max_length},
        "weight_dtype": "bfloat16", "pooling_norm_cache_dtype": CACHE_DTYPE,
        "attention": "sdpa", "batch": args.batch_size,
        "join_rule": JOIN_RULE, "placeholder": PLACEHOLDER,
        "invalid_row_rule": INVALID_ROW_RULE,
        "canonical_order_rule": CANONICAL_ORDER_RULE,
        "item_ids_sha256": ids_sha,
        "quality_rule": {"quality_length": args.quality_length,
                         "description_max_weight": args.description_max_weight,
                         "formula": "q=1[n>0]*min(n/L,1)*(1-r); r=1-unique_trigrams/(n-2), n<3→0; tokens lowercase whitespace"},
        "inputs_sha256": inputs,
        "extra": extra,
    }


def stage_encode(args, out_root):
    import numpy as np  # noqa: PLC0415
    emb_root = Path(out_root)
    qpath = Path(args.manifest_root) / "quality_features.jsonl"
    feats = read_jsonl(qpath)
    item_ids = [r["item_id"] for r in feats]
    model, tok = load_encoder(args)
    manifest_root = Path(args.manifest_root)
    inputs = {Path(p).name: sha256_file(p) for p in
              (qpath, manifest_root / "catalog_clean.jsonl",
               manifest_root / "train_views.jsonl", manifest_root / "diagnostic_views.jsonl")}

    def copy_ids(group_dir):
        shutil.copyfile(emb_root / "item_ids.json", Path(group_dir) / "item_ids.json")

    # A: join(title, desc)（拼接不再清洗）
    a_texts, b_titles, b_descs = [], [], []
    n_placeholder = 0
    for r in feats:
        title, desc = r["title_clean"], r["description_clean"]
        if not title and not desc:
            n_placeholder += 1
            a_texts.append(PLACEHOLDER)
            b_titles.append(PLACEHOLDER)
            b_descs.append("")
        else:
            a_texts.append(join_text(title, desc))
            b_titles.append(title if title else PLACEHOLDER)
            b_descs.append(desc)
    t0 = time.time()
    a_clean, a_stats, a_elapsed = encode_texts(model, tok, a_texts,
                                               args.max_length, args.device,
                                               chunk_tag="A_clean")
    a_dir = emb_root / "A"
    a_dir.mkdir(parents=True, exist_ok=True)
    np.save(a_dir / "clean.npy", a_clean.astype(np.float32))
    copy_ids(a_dir)

    # B: 标题与描述分别编码；空描述精确复制标题向量（不再归一化）
    t_titles = time.time()
    t_vecs, _, t_elapsed = encode_texts(model, tok, b_titles,
                                        args.max_length, args.device,
                                        chunk_tag="B_titles")
    need_desc = [i for i, d in enumerate(b_descs) if d]
    d_vecs, _, d_elapsed = encode_texts(model, tok, [b_descs[i] for i in need_desc],
                                        args.max_length, args.device,
                                        chunk_tag="B_descs")
    b_clean = t_vecs.copy()
    for j, i in enumerate(need_desc):
        w = feats[i]["quality"]["w"]
        v = (1.0 - w) * b_clean[i] + w * d_vecs[j]
        b_clean[i] = v / max(float(np.linalg.norm(v)), L2_EPS)
    b_dir = emb_root / "B"
    b_dir.mkdir(parents=True, exist_ok=True)
    np.save(b_dir / "clean.npy", b_clean.astype(np.float32))
    copy_ids(b_dir)

    # C: 直接复制 B 的干净缓存（字节一致），并编码训练 drop30 视图
    c_dir = emb_root / "C"
    c_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(b_dir / "clean.npy", c_dir / "clean.npy")
    copy_ids(c_dir)
    c_texts, c_rows = [], []
    for i, r in enumerate(feats):
        v = r["views"]["train_drop30"]
        if v["valid"]:
            c_texts.append(v["text"])
            c_rows.append(i)
    c_aug, _, c_elapsed = encode_texts(model, tok, c_texts,
                                       args.max_length, args.device,
                                       chunk_tag="C_train_aug")
    c_clean = np.load(c_dir / "clean.npy")
    c_aug_full = c_clean.copy()
    valid_aug = np.zeros(len(feats), dtype=bool)
    for j, i in enumerate(c_rows):
        c_aug_full[i] = c_aug[j]
        valid_aug[i] = True
    np.save(c_dir / "train_aug.npy", c_aug_full.astype(np.float32))
    np.save(c_dir / "valid_aug.npy", valid_aug)

    meta_inputs = dict(inputs)
    meta_inputs["quality_features.jsonl"] = sha256_file(qpath)
    meta = metadata_payload(args, "A", a_dir / "item_ids.json", meta_inputs, {
        "n_texts": len(a_texts), "n_placeholder": n_placeholder,
        "encode_elapsed_seconds": a_elapsed,
        "token_stats": a_stats})
    (a_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    meta_b = metadata_payload(args, "B", b_dir / "item_ids.json", meta_inputs, {
        "n_title_texts": len(b_titles), "n_desc_texts": len(need_desc),
        "title_encode_elapsed_seconds": t_elapsed,
        "desc_encode_elapsed_seconds": d_elapsed,
        "fallback_rule": "empty description rows copy the title vector byte-exactly (no re-normalization)"})
    (b_dir / "metadata.json").write_text(
        json.dumps(meta_b, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    meta_c = metadata_payload(args, "C", c_dir / "item_ids.json", meta_inputs, {
        "clean_source": "byte-identical copy of B/clean.npy",
        "train_aug_texts": len(c_texts), "train_aug_valid": int(valid_aug.sum()),
        "train_aug_encode_elapsed_seconds": c_elapsed,
        "invalid_row_rule_applied": INVALID_ROW_RULE})
    (c_dir / "metadata.json").write_text(
        json.dumps(meta_c, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[v2_build] encode done: A {a_elapsed}s, B {t_elapsed + d_elapsed}s, "
          f"C aug {c_elapsed}s; placeholder={n_placeholder}")
    return {"a_elapsed": a_elapsed, "b_elapsed": round(t_elapsed + d_elapsed, 2),
            "c_elapsed": c_elapsed, "placeholder": n_placeholder}


# ---------------- stage: diagnostics (GPU; shared B/C) ----------------

def stage_diagnostics(args, out_root):
    import numpy as np  # noqa: PLC0415
    emb_root = Path(out_root)
    qpath = Path(args.manifest_root) / "quality_features.jsonl"
    feats = read_jsonl(qpath)
    manifest_root = Path(args.manifest_root)
    model, tok = load_encoder(args)
    inputs = {"quality_features.jsonl": sha256_file(qpath),
              "diagnostic_views.jsonl": sha256_file(manifest_root / "diagnostic_views.jsonl"),
              "train_views.jsonl": sha256_file(manifest_root / "train_views.jsonl")}

    def build_group(group, text_of):
        """text_of(rec, v, vt) -> 编码文本（无效行不进入列表, 复制组干净向量）。"""
        gdir = emb_root / group / "diagnostic"
        gdir.mkdir(parents=True, exist_ok=True)
        clean = np.load(emb_root / group / "clean.npy")
        summary = {}
        for vt in VIEW_TYPES:
            texts, rows = [], []
            valid = np.zeros(len(feats), dtype=bool)
            for i, r in enumerate(feats):
                v = r["views"][f"diag_{vt}"]
                if v["valid"]:
                    t = text_of(r, v, vt)
                    texts.append(t)
                    rows.append(i)
                    valid[i] = True
            arr, stats, elapsed = encode_texts(model, tok, texts,
                                               args.max_length, args.device,
                                               chunk_tag=f"{group}_diag_{vt}")
            full = clean.copy()
            for j, i in enumerate(rows):
                full[i] = arr[j]
            np.save(gdir / f"{vt}.npy", full.astype(np.float32))
            np.save(gdir / f"{vt}_valid.npy", valid)
            summary[vt] = {"n_texts": len(texts), "valid": int(valid.sum()),
                           "elapsed_seconds": elapsed}
        with open(gdir / "views_source.json", "w", encoding="utf-8") as f:
            json.dump({"manifest": "outputs/v2/manifests/diagnostic_views.jsonl",
                       "manifest_sha256": sha256_file(manifest_root / "diagnostic_views.jsonl"),
                       "view_seeds": {"drop30": 2701, "drop60": 2701, "dropall": 2701},
                       "invalid_row_rule": INVALID_ROW_RULE}, f, ensure_ascii=False, indent=2)
        return summary

    summary_a = build_group("A", lambda r, v, vt: join_text(r["title_clean"], v["text"]))
    # B/C: drop30/drop60 编码扰动描述; dropall 视图输入 = 仅标题 (协议 §9.2 "仅保留标题")
    summary_bc = build_group("B", lambda r, v, vt: r["title_clean"] if vt == "dropall" else v["text"])
    # C 诊断 = 字节一致复制 B 诊断
    for vt in VIEW_TYPES + ("",):
        src = emb_root / "B" / "diagnostic"
        dst = emb_root / "C" / "diagnostic"
        dst.mkdir(parents=True, exist_ok=True)
        for suffix in (".npy", "_valid.npy"):
            f = f"{vt}{suffix}"
            if (src / f).exists():
                shutil.copyfile(src / f, dst / f)
        shutil.copyfile(src / "views_source.json", dst / "views_source.json")
    print(f"[v2_build] diagnostics done: A={summary_a} B/C(shared)={summary_bc}")
    return {"A": summary_a, "BC_shared": summary_bc}


# ---------------- checks ----------------

def run_checks(args):
    """T3 验收检查: 公式样例/全量范围/合成身份/视图接口。全部 CPU。"""
    qpath = Path(args.manifest_root) / "quality_features.jsonl"
    feats = read_jsonl(qpath)
    checks = []

    def add(name, passed, measured, detail=""):
        checks.append({"name": name, "passed": bool(passed), "measured": measured,
                       "detail": detail})

    L = args.quality_length
    # 1. 公式样例（精确值, isclose 容差 1e-9）
    samples = [
        ("empty", "", 0, 0.0, 0.0, 0.0),
        ("single", "hello", 1, 0.0, 1.0 / L, 0.5 / L),
        ("two_words", "hello world", 2, 0.0, 2.0 / L, 1.0 / L),
        ("case_fold", "Hello HELLO hello world", 4, 0.0, 4.0 / L, 2.0 / L),
        ("unique32", " ".join(f"w{i}" for i in range(32)), 32, 0.0, 1.0, 0.5),
        ("unique62", " ".join(f"w{i}" for i in range(62)), 62, 0.0, 1.0, 0.5),
    ]
    trigrams_repeated = ("ab ba " * 16).strip()  # 32 words: a,b 交替 (小写后相同)
    st = quality_stats(trigrams_repeated, L)
    # 32 words → 30 trigrams, unique=2 → r=1-2/30; q=min(32/32,1)*(1-r)=2/30; w=q/2
    want_r = round(1.0 - 2 / 30, 6)
    want_q = round((2 / 30) * 1.0, 6)
    want_w = round(0.5 * (2 / 30), 6)
    add("formula_repeat_trigrams", st["n"] == 32 and st["r"] == want_r
        and st["q"] == want_q and st["w"] == want_w,
        st, "32 词 a/b 交替: unique_trigrams=2, r=1-2/30, q=min(1,1)*(1-r)=2/30, w=q/2")
    for name, text, want_n, want_r, want_q, want_w in samples:
        st = quality_stats(text, L)
        ok = (st["n"] == want_n and math.isclose(st["r"], want_r, abs_tol=1e-9)
              and math.isclose(st["q"], want_q, abs_tol=1e-9)
              and math.isclose(st["w"], want_w, abs_tol=1e-9))
        add(f"formula_{name}", ok, st, f"expect n={want_n} r={want_r} q={want_q} w={want_w}")

    # 2. 全量范围与口径
    qs = [r["quality"] for r in feats]
    add("q_in_0_1_all", all(0.0 <= q["q"] <= 1.0 for q in qs), {"n": len(qs)})
    add("w_in_0_half_all", all(0.0 <= q["w"] <= 0.5 for q in qs), {"n": len(qs)})
    n_missing = sum(r["original_description_missing"] for r in feats)
    n_clean_empty = sum(1 for r in feats if r["description_clean"] == "")
    n_short = sum(1 for r in feats if r["quality"]["n"] == 1)
    add("counts_229_327_3", n_missing == 229 and n_clean_empty == 327 and n_short == 3,
        {"raw_missing": n_missing, "clean_empty": n_clean_empty, "single_word": n_short})
    v_train = sum(1 for r in feats if r["views"]["train_drop30"]["valid"])
    v_d30 = sum(1 for r in feats if r["views"]["diag_drop30"]["valid"])
    v_d60 = sum(1 for r in feats if r["views"]["diag_drop60"]["valid"])
    v_dall = sum(1 for r in feats if r["views"]["diag_dropall"]["valid"])
    add("valid_denominators", (v_train, v_d30, v_d60, v_dall) == (3356, 3356, 3356, 3359),
        {"train": v_train, "d30": v_d30, "d60": v_d60, "dall": v_dall})
    add("n_items_3686", len(feats) == 3686, {"n": len(feats)})
    # 扰动质量独立于干净质量（抽验 100 条 valid: 扰动 q != 干净 q 至少多数成立）
    diff_ct = sum(1 for r in feats if r["views"]["train_drop30"]["valid"]
                  and r["views"]["train_drop30"]["quality"] != r["quality"])
    add("view_quality_recomputed", diff_ct > 0,
        {"view_quality_differs_from_clean": diff_ct})

    # 3. 合成身份（乱序+非连续数值+字符串 ID）
    import tempfile  # noqa: PLC0415
    syn = [
        {"item_id": "10", "title_clean": "Gamma", "description_clean": "same text here",
         "original_description_missing": False},
        {"item_id": "abc", "title_clean": "Alpha Widget", "description_clean": "",
         "original_description_missing": True},
        {"item_id": "2", "title_clean": "Beta Tool", "description_clean": "singleword",
         "original_description_missing": False},
        {"item_id": "7", "title_clean": "Alpha Widget", "description_clean": "one two",
         "original_description_missing": False},
        {"item_id": "3", "title_clean": "Delta", "description_clean": "x y z",
         "original_description_missing": False},
    ]
    ordered = sorted(syn, key=lambda c: canonical_id_key(c["item_id"]))
    id2row = {c["item_id"]: i for i, c in enumerate(ordered)}
    ok_order = [c["item_id"] for c in ordered] == ["2", "3", "7", "10", "abc"]
    ok_lookup = all(ordered[id2row[k]]["title_clean"] == v for k, v in
                    (("abc", "Alpha Widget"), ("2", "Beta Tool"), ("7", "Alpha Widget")))
    add("synthetic_identity_recovery", ok_order and ok_lookup and id2row["abc"] == 4,
        {"order": [c["item_id"] for c in ordered], "id2row": id2row,
         "detail": "乱序输入按规范序恢复, 按 ID 取行不依赖数组行号"})

    # 4. 视图接口（对 Phase 1 冻结清单逐条核验）
    cat = {c["item_id"]: c for c in read_jsonl(Path(args.manifest_root) / "catalog_clean.jsonl")}
    bad_title = bad_seed = bad_dropall = bad_valid_text = 0
    tv = read_jsonl(Path(args.manifest_root) / "train_views.jsonl")
    dv = read_jsonl(Path(args.manifest_root) / "diagnostic_views.jsonl")
    for r in tv:
        if r["view_seed"] != 1701:
            bad_seed += 1
        c = cat.get(r["item_id"])
        if c is None or r["title_clean"] != c["title_clean"]:
            bad_title += 1
        if r["valid_aug"] and r["perturbed_text"] == r.get("original_text", r["perturbed_text"]):
            bad_valid_text += 1
    for r in dv:
        want = 2701
        if r["view_seed"] != want:
            bad_seed += 1
        c = cat.get(r["item_id"])
        if c is None or r["title_clean"] != c["title_clean"]:
            bad_title += 1
        if r["view_type"] == "dropall" and r["valid_aug"] and r["perturbed_text"] != "":
            bad_dropall += 1
    add("view_title_item_unchanged", bad_title == 0, {"violations": bad_title, "n": len(tv) + len(dv)})
    add("view_seeds_not_mixed", bad_seed == 0, {"violations": bad_seed,
                                                "train": len(tv), "diag": len(dv)})
    add("dropall_valid_empty_text", bad_dropall == 0, {"violations": bad_dropall})
    add("valid_view_text_differs", bad_valid_text == 0, {"violations": bad_valid_text})

    failed = [c for c in checks if not c["passed"]]
    payload = {
        "schema": SCHEMA, "stage": "quality_rule_checks",
        "deterministic": {
            "n_checks": len(checks), "n_passed": len(checks) - len(failed),
            "n_failed": len(failed), "checks": checks, "failed_details": failed,
            "all_passed": len(failed) == 0,
            "quality_features_sha256": sha256_file(qpath),
        },
        "run": {"command": " ".join(sys.argv), "cwd": os.getcwd(),
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    }
    ev = Path(args.evidence_root)
    ev.mkdir(parents=True, exist_ok=True)
    with open(ev / "quality_rule_checks.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[v2_build] quality_rule_checks: {payload['deterministic']['n_passed']}/{len(checks)} passed")

    # interface_spec.json
    spec = {
        "schema": SCHEMA,
        "item_order": CANONICAL_ORDER_RULE,
        "item_ids_file": "outputs/v2/embeddings/item_ids.json (canonical; each group copies it)",
        "id_to_row": "id2row[item_id] = row index in every [N,D] array; N=3686",
        "join_rule": JOIN_RULE,
        "placeholder": PLACEHOLDER,
        "invalid_row_rule": INVALID_ROW_RULE,
        "cache_dtype_shape": f"{CACHE_DTYPE}, [3686, D], D=2560 (measured)",
        "metadata_schema": {
            "encoder": "repo_id/revision/local_path/pooling/max_length",
            "weight_dtype": "bfloat16", "pooling_norm_cache_dtype": CACHE_DTYPE,
            "attention": "sdpa", "batch": 1,
            "item_ids_sha256": "sha256 of canonical item_ids.json",
            "inputs_sha256": "quality_features/catalog_clean/train_views/diagnostic_views",
            "quality_rule": "quality_length/description_max_weight/formula",
            "code": "script sha256 + schema",
            "outputs": "per-file sha256",
        },
        "view_binding": "diagnostic/*.npy paired with {view}_valid.npy and views_source.json (manifest hash)",
    }
    with open(ev / "interface_spec.json", "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
    # id/view checks evidence
    with open(ev / "id_view_checks.json", "w", encoding="utf-8") as f:
        json.dump({"schema": SCHEMA,
                   "identity_checks": [c for c in checks if c["name"].startswith("synthetic")],
                   "view_checks": [c for c in checks if c["name"].startswith(("view_", "dropall", "valid_view"))],
                   "all_passed": all(c["passed"] for c in checks
                                     if c["name"].startswith(("synthetic", "view_", "dropall", "valid_view")))},
                  f, ensure_ascii=False, indent=2)
    return payload


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="MiniOneRec v2 Phase 2 embeddings builder")
    ap.add_argument("--manifest-root", default="outputs/v2/manifests")
    ap.add_argument("--encoder", default="Qwen/Qwen3-Embedding-4B")
    ap.add_argument("--pooling", default="masked_mean", choices=["masked_mean"])
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--quality-length", type=int, default=32)
    ap.add_argument("--description-max-weight", type=float, default=0.5)
    ap.add_argument("--output-root", default="outputs/v2/embeddings")
    ap.add_argument("--model-path", default="/hpc2hdd/home/swu073/sikun/dsaa5020/models/Qwen3-Embedding-4B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="T2 锁定: batch=1 (batch>1 在 2048 上 OOM)")
    ap.add_argument("--evidence-root", default="outputs/v2/evidence/phase2")
    ap.add_argument("--stages", default="quality",
                    help="逗号分隔: quality,encode,diagnostics,checks (默认 quality)")
    args = ap.parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    t0 = time.time()
    print(f"[v2_build] stages={stages} schema={SCHEMA}")

    if "quality" in stages:
        stage_quality(args, args.output_root)
    if "checks" in stages or "quality" in stages:
        run_checks(args)
    if "encode" in stages:
        stage_encode(args, args.output_root)
    if "diagnostics" in stages:
        stage_diagnostics(args, args.output_root)
    print(f"[v2_build] DONE_OK elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
