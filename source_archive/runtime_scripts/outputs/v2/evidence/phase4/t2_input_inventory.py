#!/usr/bin/env python
"""Phase4 T2: full consumption-input inventory (9 finals, 33-cache table, Phase1 manifests, 4 raw sources).
SCHEMA: phase4-t2_input_inventory-1.0"""
import os, sys, json, hashlib
os.environ.setdefault("OMP_NUM_THREADS", "8")
from pathlib import Path
import numpy as np
REPO = Path("/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec")
EV = REPO / "outputs/v2/evidence/phase4"
EV.mkdir(parents=True, exist_ok=True)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

res = {"stage": "phase4_t2_input_inventory", "schema": "phase4-t2_input_inventory-1.0", "date": "2026-11-11"}
all_groups_ok = True

# ---- 1. nine finals vs Phase3 run_inventory + per-run config.json binding ----
inv = json.load(open(REPO / "outputs/v2/evidence/phase3/run_inventory.json"))
CAND = inv["candidate_script_v13"]
finals = []
g_ok = True
for r in inv["runs"]:
    rid = r["run_id"]
    m, s = rid.split("-")[1], rid.split("-")[2]
    base = REPO / f"outputs/v2/rqvae/{m}/seed_{s}"
    fp = base / "final_checkpoint.pth"
    actual = sha(fp)
    ok = actual == r["final_sha256"]
    cfg = json.load(open(base / "config.json"))
    ent = {"run_id": rid, "path": f"outputs/v2/rqvae/{m}/seed_{s}/final_checkpoint.pth",
           "expected_sha256": r["final_sha256"], "actual_sha256": actual, "match": ok,
           "bytes": fp.stat().st_size,
           "config_json_script_sha256": cfg.get("script_sha256"),
           "config_script_is_v13": cfg.get("script_sha256") == CAND,
           "config_inputs_bound": {k: v["sha256"] for k, v in cfg.get("inputs", {}).items() if isinstance(v, dict)}}
    g_ok &= ok and ent["config_script_is_v13"]
    finals.append(ent)
    print(rid, "OK" if ok else "HASH_MISMATCH", flush=True)
res["nine_finals"] = {"baseline": "outputs/v2/evidence/phase3/run_inventory.json",
                      "candidate_script_v13": CAND, "entries": finals, "all_match": g_ok}
all_groups_ok &= g_ok

# ---- 2. Phase2 33-file cache table: re-hash every file ----
table = json.load(open(REPO / "outputs/v2/evidence/phase2/t5_cache_table.json"))
tf = table["files"]
rows = []
t_ok = True
if isinstance(tf, dict):
    items_iter = [(k, v) for k, v in tf.items()]
else:
    items_iter = [(e.get("path"), e) for e in tf]
for path, e in items_iter:
    expected = e["sha256"] if isinstance(e, dict) else str(e)
    fp = REPO / path
    if not fp.exists():
        rows.append({"path": path, "expected_sha256": expected, "actual_sha256": None, "match": False, "missing": True})
        t_ok = False
        continue
    actual = sha(fp)
    ok = actual == expected
    t_ok &= ok
    ent = {"path": path, "expected_sha256": expected, "actual_sha256": actual, "match": ok,
           "bytes": fp.stat().st_size}
    if path.endswith(".npy"):
        a = np.load(fp, mmap_mode="r")
        ent["dtype"] = str(a.dtype)
        ent["shape"] = list(a.shape)
        if path.endswith("_valid.npy"):
            ent["role"] = "mask"
            ent["structure_ok"] = a.dtype == np.bool_ and a.shape == (3686,)
            ent["n_valid"] = int(np.asarray(a).sum())
        else:
            ent["role"] = "vector"
            ent["structure_ok"] = a.dtype == np.float32 and a.shape == (3686, 2560)
            ent["all_finite"] = bool(np.isfinite(np.asarray(a)).all())
    rows.append(ent)
res["phase2_cache_table_33"] = {"baseline": "outputs/v2/evidence/phase2/t5_cache_table.json",
                                "n_files_listed": len(items_iter), "entries": rows, "all_match": t_ok}
all_groups_ok &= t_ok

# cross checks on the cache set
ids_paths = ["outputs/v2/embeddings/item_ids.json", "outputs/v2/embeddings/A/item_ids.json",
             "outputs/v2/embeddings/B/item_ids.json", "outputs/v2/embeddings/C/item_ids.json"]
ids_hashes = {p: sha(REPO / p) for p in ids_paths}
ids_obj = json.load(open(REPO / ids_paths[0]))
ids_list = [str(i) for i in (ids_obj["item_ids"] if isinstance(ids_obj, dict) else ids_obj)]
xc = {
    "item_ids_four_copies_same_hash": len(set(ids_hashes.values())) == 1,
    "item_ids_hashes": ids_hashes,
    "item_ids_order_rule": ids_obj.get("order_rule") if isinstance(ids_obj, dict) else None,
    "item_ids_len": len(ids_list),
    "item_ids_first_last": [ids_list[0], ids_list[-1]],
    "item_ids_contiguous_0_3685": sorted(ids_list, key=lambda x: (0, int(x)) if x.isdigit() else (1, x)) == [str(i) for i in range(3686)],
    "B_C_clean_byte_identical": sha(REPO / "outputs/v2/embeddings/B/clean.npy") == sha(REPO / "outputs/v2/embeddings/C/clean.npy"),
}
indep = []
for vt in ("drop30", "drop60", "dropall"):
    ha = sha(REPO / f"outputs/v2/embeddings/A/diagnostic/{vt}.npy")
    hb = sha(REPO / f"outputs/v2/embeddings/B/diagnostic/{vt}.npy")
    hc = sha(REPO / f"outputs/v2/embeddings/C/diagnostic/{vt}.npy") if (REPO / f"outputs/v2/embeddings/C/diagnostic/{vt}.npy").exists() else None
    # 预期共享规则：drop30/drop60 依赖方法（A 独立编码，B/C 共享复制件）；
    # dropall=仅标题视图，与方法无关 → A=B=C 字节相同（实测 d6bae025…）
    if vt == "dropall":
        share_ok = (ha == hb) and ((hc == hb) if hc is not None else True)
        note = "dropall 为仅标题视图，三方法输入相同 → 向量字节相同（哈希一致为预期，非缺陷）"
    else:
        share_ok = (ha != hb) and ((hc == hb) if hc is not None else True)
        note = "A 独立编码（与 B 哈希不同）；B/C 为共享复制件（哈希一致）"
    indep.append({"view": vt, "A_sha": ha, "B_sha": hb, "C_sha": hc,
                  "expected_sharing_rule": note, "sharing_matches_expected": share_ok})
xc["diagnostic_sharing"] = indep
meta = json.load(open(REPO / "outputs/v2/embeddings/A/metadata.json"))
xc["encoder_revision_from_metadata"] = meta.get("encoder", {}).get("revision")
xc["encoder_revision_expected"] = "5cf2132abc99cad020ac570b19d031efec650f2b"
xc["encoder_revision_match"] = xc["encoder_revision_from_metadata"] == xc["encoder_revision_expected"]
xc["views_source_summary"] = {}
for vp in ["outputs/v2/embeddings/A/diagnostic/views_source.json", "outputs/v2/embeddings/B/diagnostic/views_source.json"]:
    fp = REPO / vp
    xc["views_source_summary"][vp] = json.load(open(fp)) if fp.exists() else None
res["cache_cross_checks"] = xc
xc_ok = (xc["item_ids_four_copies_same_hash"] and xc["item_ids_len"] == 3686
         and xc["item_ids_contiguous_0_3685"] and xc["B_C_clean_byte_identical"]
         and all(d["sharing_matches_expected"] for d in indep)
         and xc["encoder_revision_match"])
res["cache_cross_checks"]["all_passed"] = xc_ok
all_groups_ok &= xc_ok

# ---- 3. Phase1 manifests + bases (11) + raw sources (4) ----
PH1 = [
    ("outputs/v2/environment/environment_lock.json", "b102c41efddd2b0970a1070ceeebc80ec32bf38d9830eb324422344db7a2cd87"),
    ("outputs/v2/environment/pip_freeze.txt", "f104eac93fce544d32ebc68ba19d2c63ed6394dcb8a24018fdadd3b2d10be727"),
    ("outputs/v2/manifests/file_hashes.json", "e8a801e84ae4ac50295051d3a05dae6e7c7370ab9ee1b08760774d9fab34be6b"),
    ("outputs/v2/manifests/data_audit.json", "a2db080e739f22bc0c0658c5dc6c185fb8553f1c2610ce2cf39ba34a3888f995"),
    ("outputs/v2/manifests/rows_manifest.jsonl", "b3e195c2c997b2b9a039c14bcaa0d4560395cfc76682d3140e3694feeb3ba1f3"),
    ("outputs/v2/manifests/catalog_clean.jsonl", "275f869fbd5b5cf9c731608b64a43ae5ee7fa3deb8ff8228ef2110759c1a5649"),
    ("outputs/v2/manifests/rl_tasks_10000.jsonl", "e77868cae2e4ffec6fda33dc1b2bee4b0f36d2800a2d270dfae0399bc5f11eb0"),
    ("outputs/v2/manifests/train_views.jsonl", "45540aec5e3f49c5b520e74682acd6e658cb48ce164f6c645025a570f6499510"),
    ("outputs/v2/manifests/diagnostic_views.jsonl", "17258e1d149c6937de9f97b08aee5461198bc87e8984b84ae9294175ccdea8d0"),
    ("outputs/v2/manifests/quality_features.jsonl", "9df37c0d775846890f08d8aed393ab29705fedf7e6b5f6c1a6c1911661a44fee"),
    ("outputs/v2/evidence/phase2/t5_cache_table.json", "639d2b897ae5a68654576eb7594708c9a466d92ac655c34e1f272fe1e3e7a547"),
]
ph = []
ph_ok = True
for rel, exp in PH1:
    actual = sha(REPO / rel)
    ok = actual == exp
    ph_ok &= ok
    ph.append({"path": rel, "expected_sha256": exp, "actual_sha256": actual, "match": ok})
res["phase1_manifests_and_bases_11"] = {"entries": ph, "all_match": ph_ok}
all_groups_ok &= ph_ok

RAW = [
    ("data/Amazon/index/Industrial_and_Scientific.item.json", "937600c7a03652f55864a1d12d6e9fe8ca3e24ecc4e255a88216b25c5f949496"),
    ("data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv", "da5f2af4f5047e18023dd4792f65a48bd7c8eab90447dd682fb48ca255d25f8c"),
    ("data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv", "12bbb02bced0bd754a8b4e0f9ca4086416e30d12b0b73a2c95e7b195dd67385a"),
    ("data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv", "973dd1db4fd549f3f2b40a0f039dbae157ebb92c716197cc246401e4e82c3a46"),
]
raw_rows = []
raw_ok = True
for rel, exp in RAW:
    actual = sha(REPO / rel)
    ok = actual == exp
    raw_ok &= ok
    raw_rows.append({"path": rel, "expected_sha256": exp, "actual_sha256": actual, "match": ok})
res["raw_sources_4"] = {"entries": raw_rows, "all_match": raw_ok}
all_groups_ok &= raw_ok

# ---- coverage statement ----
res["coverage_statement"] = {
    "checked_this_round": {
        "nine_finals": len(finals), "phase2_cache_files": len(rows),
        "phase1_manifests_and_bases": len(ph), "raw_sources": len(raw_rows),
        "total_files_rehashed": len(finals) + len(rows) + len(ph) + len(raw_rows),
    },
    "not_rechecked_statement": (
        "本轮未重哈希：Qwen3-Embedding-4B 权重文件本体（只以 metadata.json 内 revision "
        "5cf2132abc99cad020ac570b19d031efec650f2b 与 Phase 2 冻结记录追踪，不重称 ~8GB 权重）；"
        "models/Qwen3-0.6B-Base 仅 tokenizer/config 静态文件入 inventory（environment_lock），"
        "model.safetensors 未消费故不重哈希；outputs/v2/evidence/phase3 证据文件（Phase 3 终验已验，本轮仅引用其哈希基准）；"
        "历史废弃训练轮产物。以上均为显式未检项，不做全称声称。"),
}
res["summary"] = {"nine_finals_match": g_ok, "cache33_match": t_ok, "cross_checks_passed": xc_ok,
                  "phase1_bases_match": ph_ok, "raw_sources_match": raw_ok}
res["summary"]["all_passed"] = all(res["summary"].values())
out = EV / "input_inventory.json"
out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
print("SUMMARY:", json.dumps(res["summary"]))
print("INVENTORY_SHA256:", sha(out))
print("T2_INVENTORY_DONE")
