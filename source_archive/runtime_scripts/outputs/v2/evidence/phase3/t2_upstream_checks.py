#!/usr/bin/env python
"""v2 Phase 3 T2 upstream checks: 33-file table re-verify + Phase 1 freeze + consumption deep reload.
Read-only. SCHEMA: phase3-t2_upstream_checks-1.0
"""
import json, hashlib, os, sys
import numpy as np

ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
os.chdir(ROOT)

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

res = {
    "stage": "phase3_t2_upstream_checks",
    "schema": "phase3-t2_upstream_checks-1.0",
    "date": "2026-11-10",
    "t1_gate": {
        "t1_commit": "e7517496fc6d216c51c32c36323853674de0132f",
        "t1_parent": "ded0bc13059b25c3c20f6cbc1a4c1a7c48d97649",
        "accepted_by_boss": True,
        "phase2_handover_doc": "docs/handover/phase2_artifacts.md",
    },
}

# ---------- E1a: 33-file cache table ----------
TBL_EXPECTED = "639d2b897ae5a68654576eb7594708c9a466d92ac655c34e1f272fe1e3e7a547"
tbl_path = "outputs/v2/evidence/phase2/t5_cache_table.json"
tbl_actual = sha256(tbl_path)
table = json.load(open(tbl_path))
entries, ok33 = [], True
for ent in table["files"]:
    rel = ent["path"]
    actual = sha256(rel)
    m = actual == ent["sha256"]
    ok33 &= m
    entries.append({"path": rel, "expected": ent["sha256"], "actual": actual, "match": m,
                    "kind": ent.get("kind")})
res["cache_table_33"] = {
    "table_file_sha256_expected": TBL_EXPECTED,
    "table_file_sha256_actual": tbl_actual,
    "table_hash_match": tbl_actual == TBL_EXPECTED,
    "files_total": len(entries),
    "files_matched": sum(1 for e in entries if e["match"]),
    "passed": ok33 and len(entries) == 33 and tbl_actual == TBL_EXPECTED,
    "entries": entries,
}

# ---------- E1b: Phase 1 frozen manifests ----------
P1 = {
    "outputs/v2/environment/environment_lock.json": "b102c41efddd2b0970a1070ceeebc80ec32bf38d9830eb324422344db7a2cd87",
    "outputs/v2/environment/pip_freeze.txt": "f104eac93fce544d32ebc68ba19d2c63ed6394dcb8a24018fdadd3b2d10be727",
    "outputs/v2/manifests/file_hashes.json": "e8a801e84ae4ac50295051d3a05dae6e7c7370ab9ee1b08760774d9fab34be6b",
    "outputs/v2/manifests/data_audit.json": "a2db080e739f22bc0c0658c5dc6c185fb8553f1c2610ce2cf39ba34a3888f995",
    "outputs/v2/manifests/rows_manifest.jsonl": "b3e195c2c997b2b9a039c14bcaa0d4560395cfc76682d3140e3694feeb3ba1f3",
    "outputs/v2/manifests/catalog_clean.jsonl": "275f869fbd5b5cf9c731608b64a43ae5ee7fa3deb8ff8228ef2110759c1a5649",
    "outputs/v2/manifests/rl_tasks_10000.jsonl": "e77868cae2e4ffec6fda33dc1b2bee4b0f36d2800a2d270dfae0399bc5f11eb0",
    "outputs/v2/manifests/train_views.jsonl": "45540aec5e3f49c5b520e74682acd6e658cb48ce164f6c645025a570f6499510",
    "outputs/v2/manifests/diagnostic_views.jsonl": "17258e1d149c6937de9f97b08aee5461198bc87e8984b84ae9294175ccdea8d0",
}
p1_entries, ok_p1 = [], True
for rel, exp in P1.items():
    actual = sha256(rel)
    m = actual == exp
    ok_p1 &= m
    p1_entries.append({"path": rel, "expected": exp, "actual": actual, "match": m})
res["phase1_freeze"] = {"files_total": len(p1_entries), "files_matched": sum(1 for e in p1_entries if e["match"]),
                        "passed": ok_p1, "entries": p1_entries}

# ---------- E2: consumption deep reload (read-only) ----------
def load_clean(group):
    return np.load(f"outputs/v2/embeddings/{group}/clean.npy")

A, B, C = load_clean("A"), load_clean("B"), load_clean("C")
vec_checks = {}
for name, x in (("A_clean", A), ("B_clean", B), ("C_clean", C)):
    norms = np.linalg.norm(x.astype(np.float64), axis=1)
    vec_checks[name] = {
        "shape": list(x.shape), "dtype": str(x.dtype),
        "finite": bool(np.isfinite(x).all()),
        "norm_min": float(norms.min()), "norm_max": float(norms.max()),
        "norm_in_band": bool((norms > 0.99).all() and (norms < 1.01).all()),
    }

# item_ids: canonical + 3 copies
CANON_EXPECTED = "f7c159fdaed280524878756f24e9271f916a9570dbdef3b965b50044c68ae791"
ids_paths = {g: f"outputs/v2/embeddings/{g}/item_ids.json" for g in ("A", "B", "C")}
ids_paths["canonical"] = "outputs/v2/embeddings/item_ids.json"
ids_hashes = {k: sha256(p) for k, p in ids_paths.items()}
canonical = json.load(open(ids_paths["canonical"]))
ids_list = canonical if isinstance(canonical, list) else canonical.get("item_ids")
qf_rows = [json.loads(l) for l in open("outputs/v2/manifests/quality_features.jsonl")]
qf_ids = [str(r["item_id"]) for r in qf_rows]
ids_can = [str(i) for i in ids_list]
order_match = ids_can == qf_ids
res["consumption"] = {
    "vectors": vec_checks,
    "b_equals_c_byte_equal": bool(np.array_equal(B, C)) and sha256(ids_paths["B"]) == sha256(ids_paths["C"]),
    "item_ids": {
        "hashes": ids_hashes,
        "all_equal_canonical": len(set(ids_hashes.values())) == 1 and ids_hashes["canonical"] == CANON_EXPECTED,
        "canonical_expected": CANON_EXPECTED,
        "count": len(ids_can),
        "order_matches_quality_features": order_match,
        "quality_features_rows": len(qf_rows),
    },
}

# C train_aug / valid_aug
valid = np.load("outputs/v2/embeddings/C/valid_aug.npy")
train_aug = np.load("outputs/v2/embeddings/C/train_aug.npy")
invalid_idx = np.where(~valid)[0]
invalid_fallback_ok = bool(np.array_equal(train_aug[invalid_idx], C[invalid_idx]))
res["consumption"]["C_views"] = {
    "train_aug_shape": list(train_aug.shape), "train_aug_dtype": str(train_aug.dtype),
    "train_aug_finite": bool(np.isfinite(train_aug).all()),
    "valid_aug_shape": list(valid.shape), "valid_aug_dtype": str(valid.dtype),
    "valid_count": int(valid.sum()), "invalid_count": int((~valid).sum()),
    "valid_count_expected_3356": int(valid.sum()) == 3356,
    "invalid_rows_byte_equal_clean": invalid_fallback_ok,
    "valid_rows_finite": bool(np.isfinite(train_aug[valid]).all()),
}

# metadata bindings
meta_bindings = {}
for g in ("A", "B", "C"):
    m = json.load(open(f"outputs/v2/embeddings/{g}/metadata.json"))
    ins = m.get("inputs_sha256", {})
    meta_bindings[g] = {
        "train_views_sha": ins.get("train_views.jsonl"),
        "diagnostic_views_sha": ins.get("diagnostic_views.jsonl"),
        "encoder_revision": (m.get("encoder") or {}).get("revision"),
        "schema": m.get("schema"),
    }
res["metadata_bindings"] = {
    "expected_train_views_sha": P1["outputs/v2/manifests/train_views.jsonl"],
    "expected_encoder_revision": "5cf2132abc99cad020ac570b19d031efec650f2b",
    "groups": meta_bindings,
    "all_train_views_bound": all(v["train_views_sha"] == P1["outputs/v2/manifests/train_views.jsonl"] for v in meta_bindings.values()),
    "all_encoder_revision_match": all(v["encoder_revision"] == "5cf2132abc99cad020ac570b19d031efec650f2b" for v in meta_bindings.values()),
}

# non-consumption manifest: diagnostic caches must NOT be on the training path
diag_paths = []
for g in ("A", "B", "C"):
    d = f"outputs/v2/embeddings/{g}/diagnostic"
    if os.path.isdir(d):
        diag_paths += [f"{d}/{f}" for f in sorted(os.listdir(d))]
res["non_consumption"] = {
    "training_reads": [
        "outputs/v2/embeddings/A/clean.npy", "outputs/v2/embeddings/B/clean.npy",
        "outputs/v2/embeddings/C/clean.npy", "outputs/v2/embeddings/C/train_aug.npy",
        "outputs/v2/embeddings/C/valid_aug.npy", "outputs/v2/embeddings/item_ids.json",
        "outputs/v2/embeddings/{A,B,C}/metadata.json",
    ],
    "training_must_not_read": diag_paths + ["outputs/v2/manifests/diagnostic_views.jsonl (provenance only)"],
    "diagnostic_file_count": len(diag_paths),
}

res["summary"] = {
    "cache_table_33_passed": res["cache_table_33"]["passed"],
    "phase1_freeze_passed": res["phase1_freeze"]["passed"],
    "vectors_passed": all(v["shape"] == [3686, 2560] and v["dtype"] == "float32" and v["finite"] and v["norm_in_band"] for v in vec_checks.values()),
    "b_equals_c_passed": res["consumption"]["b_equals_c_byte_equal"],
    "item_ids_passed": res["consumption"]["item_ids"]["all_equal_canonical"] and order_match and len(ids_can) == 3686,
    "C_views_passed": res["consumption"]["C_views"]["valid_count_expected_3356"] and invalid_fallback_ok,
    "metadata_passed": res["metadata_bindings"]["all_train_views_bound"] and res["metadata_bindings"]["all_encoder_revision_match"],
}
res["summary"]["all_passed"] = all(res["summary"].values())

out = "outputs/v2/evidence/phase3/upstream_checks.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("ALL_PASSED:", res["summary"]["all_passed"])
print(json.dumps(res["summary"], ensure_ascii=False))
print("OUT_SHA256:", sha256(out))
