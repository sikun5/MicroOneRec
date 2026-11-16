#!/usr/bin/env python
"""Phase4 T2: support sets, exclusion counts, ID-summary hashes, shared perturbation sources.
SCHEMA: phase4-t2_support_checks-1.0"""
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

def ids_hash(ids):
    h = hashlib.sha256()
    h.update(json.dumps(sorted(str(i) for i in ids), ensure_ascii=False, separators=(",", ":")).encode())
    return h.hexdigest()

res = {"stage": "phase4_t2_support_checks", "schema": "phase4-t2_support_checks-1.0", "date": "2026-11-11"}

# ---- 1. raw item file: original missing description ----
raw = json.load(open(REPO / "data/Amazon/index/Industrial_and_Scientific.item.json"))
n_items = len(raw)
raw_missing = sorted([i for i, v in raw.items()
                      if ("description" not in v) or (v.get("description") is None)
                      or (str(v.get("description")).strip() == "")])
res["raw_items"] = {"n_items": n_items, "expected_3686": n_items == 3686,
                    "raw_missing_description_count": len(raw_missing),
                    "expected_229": len(raw_missing) == 229,
                    "definition": "description 字段缺失/为 None/strip 后为空（对原始 item JSON 独立实算）",
                    "missing_ids_sha256": ids_hash(raw_missing),
                    "missing_ids_first10": raw_missing[:10]}

# ---- 2. catalog_clean: both count definitions, kept separate ----
cat_missing, cat_clean_empty, cat_ids = 0, 0, []
desc_key = None
with open(REPO / "outputs/v2/manifests/catalog_clean.jsonl") as f:
    for line in f:
        r = json.loads(line)
        cat_ids.append(str(r["item_id"]))
        if r.get("original_description_missing") is True:
            cat_missing += 1
        if desc_key is None:
            for k in r:
                if k.startswith("description_clean"):
                    desc_key = k
                    break
        if desc_key and str(r.get(desc_key, "")).strip() == "":
            cat_clean_empty += 1
res["catalog_clean"] = {
    "n_rows": len(cat_ids), "expected_3686": len(cat_ids) == 3686,
    "original_description_missing_count": cat_missing, "expected_229": cat_missing == 229,
    "cleaned_description_field": desc_key,
    "cleaned_description_empty_count": cat_clean_empty, "expected_327": cat_clean_empty == 327,
    "two_definitions_disjoint_note": "原始缺失(229)=字段级缺失；清洗后空(327)=clean_text 后为空串，二者口径不同不混用",
    "raw_missing_ids_sha256": ids_hash(raw_missing),
    "catalog_missing_subset_of_clean_empty": None,
}

# clean-empty 但原始非缺失的补充集合（229 ⊆ 327 关系实测）
cat_raw_missing_ids, cat_clean_empty_ids = set(), set()
with open(REPO / "outputs/v2/manifests/catalog_clean.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r.get("original_description_missing") is True:
            cat_raw_missing_ids.add(str(r["item_id"]))
        if desc_key and str(r.get(desc_key, "")).strip() == "":
            cat_clean_empty_ids.add(str(r["item_id"]))
res["catalog_clean"]["raw_missing_subset_of_clean_empty"] = cat_raw_missing_ids <= cat_clean_empty_ids
res["catalog_clean"]["clean_empty_but_raw_present_count"] = len(cat_clean_empty_ids - cat_raw_missing_ids)
res["catalog_clean"]["clean_empty_but_raw_present_ids_sha256"] = ids_hash(cat_clean_empty_ids - cat_raw_missing_ids)

# ---- 3. diagnostic support sets: S_raw_nonempty ∩ mask ----
ids_raw = json.load(open(REPO / "outputs/v2/embeddings/item_ids.json"))
ids = [str(i) for i in (ids_raw["item_ids"] if isinstance(ids_raw, dict) else ids_raw)]
assert len(ids) == 3686
S_raw_nonempty = set(ids) - set(cat_raw_missing_ids)
res["raw_nonempty_set"] = {"n": len(S_raw_nonempty), "expected_3457": len(S_raw_nonempty) == 3457}

diag = {}
exp_counts = {"drop30": 3356, "drop60": 3356, "dropall": 3359}
ok_all = True
for grp in ("A", "B"):
    for vt in ("drop30", "drop60", "dropall"):
        m = np.asarray(np.load(REPO / f"outputs/v2/embeddings/{grp}/diagnostic/{vt}_valid.npy"))
        S_mask = set(ids[i] for i in range(3686) if bool(m[i]))
        inter = S_raw_nonempty & S_mask
        not_valid = sorted(S_raw_nonempty - S_mask)
        valid_but_missing = sorted(S_mask - S_raw_nonempty)
        ent = {"group": grp, "view": vt,
               "n_mask": len(S_mask), "expected_mask_count": exp_counts[vt],
               "mask_count_matches_expected": len(S_mask) == exp_counts[vt],
               "n_intersection_with_raw_nonempty": len(inter),
               "excluded_raw_nonempty_not_in_mask": {"n": len(not_valid), "ids_sha256": ids_hash(not_valid), "first10": not_valid[:10]},
               "mask_valid_but_raw_missing": {"n": len(valid_but_missing), "ids_sha256": ids_hash(valid_but_missing), "first10": valid_but_missing[:10],
                                              "note": "原始描述缺失但 mask 有效（dropall=仅标题视图可保留）——诊断分母按协议仍须排除，NPR/PPR 分母=交集"},
               }
        ok_all &= ent["mask_count_matches_expected"]
        diag[f"{grp}_{vt}"] = ent
res["diagnostic_support"] = diag

# ---- 4. diagnostic_views manifest vs masks ----
dv_counts, dv_seed, dv_valid_field = {}, set(), None
with open(REPO / "outputs/v2/manifests/diagnostic_views.jsonl") as f:
    for line in f:
        r = json.loads(line)
        vt = r["view_type"]
        dv_counts[vt] = dv_counts.get(vt, 0) + 1
        dv_seed.add(r.get("view_seed"))
        if dv_valid_field is None:
            dv_valid_field = "valid" if "valid" in r else ("is_valid" if "is_valid" in r else None)
res["diagnostic_views_manifest"] = {
    "counts_by_view_type": dv_counts, "view_seeds": sorted(dv_seed),
    "expected_seeds_2701_only": dv_seed == {2701},
    "valid_flag_field_in_manifest": dv_valid_field,
    "note": "mask 为有效集正本；manifest 提供视图文本与 seed 来源",
}

# ---- 5. train vs diagnostic view separation + shared sources ----
sep = {}
for grp in ("A", "B"):
    fp = REPO / f"outputs/v2/embeddings/{grp}/diagnostic/views_source.json"
    sep[f"{grp}_diagnostic_views_source"] = json.load(open(fp)) if fp.exists() else None
metaC = json.load(open(REPO / "outputs/v2/embeddings/C/metadata.json"))
sep["C_metadata_top_keys"] = sorted(metaC.keys())
sep["C_metadata_encoder_revision"] = metaC.get("encoder", {}).get("revision")
for k in metaC:
    if "train" in k.lower() or "aug" in k.lower() or "seed" in k.lower():
        sep[f"C_metadata_{k}"] = json.dumps(metaC[k], ensure_ascii=False)[:300]
res["view_separation_and_sharing"] = sep
res["shared_sources_statement"] = {
    "A_diagnostic_independent": True,
    "B_C_shared_copies": "B/C 诊断向量与 mask 为字节级共享复制件（哈希一致，见 input_inventory diagnostic_sharing）",
    "train_seed1701_not_used_for_diagnostics": "C/train_aug.npy 来自 seed1701 训练视图，仅 Phase 3 一致性项消费；诊断统一 seed2701",
}

res["summary"] = {
    "raw_missing_229": res["raw_items"]["expected_229"],
    "clean_empty_327": res["catalog_clean"]["expected_327"],
    "items_3686": res["raw_items"]["expected_3686"] and res["catalog_clean"]["expected_3686"],
    "mask_counts_match_3356_3356_3359": ok_all,
    "nonempty_3457": res["raw_nonempty_set"]["expected_3457"],
    "diag_seed_2701_only": res["diagnostic_views_manifest"]["expected_seeds_2701_only"],
}
res["summary"]["all_passed"] = all(v for k, v in res["summary"].items() if k != "all_passed")
out = EV / "support_checks.json"
out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
print("SUMMARY:", json.dumps(res["summary"]))
print("SUPPORT_SHA256:", sha(out))
print("T2_SUPPORT_DONE")
