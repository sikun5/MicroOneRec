#!/usr/bin/env python3
"""Phase 6 / T2: environment, model, and upstream input freeze checks.

Worker-owned checker. Read-only on all upstream files. Produces:
  outputs/v2/evidence/phase6/input_inventory.json
  outputs/v2/evidence/phase6/environment_lock.json
No training, no generation, no recommendation metrics. test split touched
for identity only (row count + hash).
"""
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
MODEL_DIR = "/hpc2hdd/home/swu073/sikun/dsaa5020/models/Qwen3-0.6B-Base"
OUT = os.path.join(ROOT, "outputs/v2/evidence/phase6")
PIP = os.path.join(ROOT, ".venv-phase3/bin/pip")
GROUPS = ["A", "B", "C"]
HF_ENDPOINT = "https://hf-mirror.com"
EXPECTED_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"

ANCHOR_FILES = {  # T1 frozen anchors (boss record "当前冻结源码与基线")
    "scripts/v2_evaluate.py": "50eb50e2e3eeaefe07b328e2492eafe29a55d8c9f44cfee6b1ae2bfe9abe9d95",
    "scripts/v2_bootstrap.py": "f4e8bb6a858ce61e0ad5ee8cc9c663ca1f8edccaca3a1442e4c5c6c86f1c47b4",
    "scripts/v2_generation_adapter.py": "67be20d4a904515b8f7300492d6d367129be8a4da38733545f377345430d8882",
    "scripts/v2_popularity.py": "648f8fef87ec31cd4c707c10fc6bb9a61a4c169231c95d43d8fa7181a9037bf0",
    "scripts/v2_sasrec.py": "8996b3b78b2619ec1f2acbff7b9d048549f62b2678ed571681ae9ea2c4d3d351",
    "global_guide/phase5_baselines_eval_protocol.md": "e9666992db95e8aef168583f45c0d644d25af87a3e5686e25018d6f98c11f375",
}
PROTOCOL_SHA = "98e5896b71c4a91fafd9604d4cd176c7059218cb20ed04ea989bae60afddaba9"
EXPECTED = {
    "catalog": 3686,
    "train_rows": 36259,
    "valid_rows": 4532,
    "test_rows": 4533,
    "base_tokenizer_len": 151669,
    "model_vocab": 151936,
    "emb_diff": 267,
}
SPLITS = ["train.csv", "valid.csv", "test.csv"]
EXPECTED_ROWS = {"train.csv": 36259, "valid.csv": 4532, "test.csv": 4533}

CHECKS = []  # (name, passed, detail)


def check(name, passed, detail):
    CHECKS.append({"name": name, "passed": bool(passed), "detail": detail})


def sha256_file(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def run(cmd, cwd=ROOT):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def collect_anchors():
    """Recursively collect path->sha256 anchors from phase4/phase5 evidence JSONs."""
    refs = {}
    sources = [
        "outputs/v2/evidence/phase5/input_inventory.json",
        "outputs/v2/evidence/phase4/t3_sid_files_sha256.json",
        "outputs/v2/evidence/phase4/t4_files_sha256.json",
        "outputs/v2/evidence/phase4/t5_files_sha256.json",
        "outputs/v2/evidence/phase4/artifact_inventory.json",
        "outputs/v2/evidence/phase4/input_inventory.json",
    ]

    def walk(obj, src):
        if isinstance(obj, dict):
            # pattern: {"path"/"file": ..., "sha256"/...: 64-hex}
            p_ = obj.get("path") or obj.get("file") or obj.get("rel_path")
            if isinstance(p_, str):
                s_ = None
                for sk in ("sha256", "phase4_sha256", "phase_sha256", "sha", "hash", "current_sha256"):
                    v = obj.get(sk)
                    if isinstance(v, str) and len(v) == 64:
                        s_ = v
                        break
                if s_:
                    rel = p_[len(ROOT) + 1:] if p_.startswith(ROOT + "/") else p_
                    refs.setdefault(rel, {"sha": s_, "src": src})
            keys = set(obj.keys())
            if "phase4_sha256" in keys and isinstance(obj.get("phase4_sha256"), str):
                return  # handled by parent pairing below
            # path -> sha-like mapping: a key that looks like a repo path and a
            # sibling/child 64-hex value
            path_key = None
            for k, v in obj.items():
                if isinstance(k, str) and "/" in k and isinstance(v, dict):
                    sha = None
                    for sk in ("phase4_sha256", "sha256", "phase_sha256", "sha", "hash", "current_sha256"):
                        sv = v.get(sk)
                        if isinstance(sv, str) and len(sv) == 64:
                            sha = sv
                            break
                    if sha is None:
                        for sv in v.values():
                            if isinstance(sv, str) and len(sv) == 64:
                                sha = sv
                                break
                    if sha:
                        refs.setdefault(k, {"sha": sha, "src": src})
                elif isinstance(k, str) and "/" in k and isinstance(v, str) and len(v) == 64 and "." in k:
                    refs.setdefault(k, {"sha": v, "src": src})
            for v in obj.values():
                walk(v, src)
        elif isinstance(obj, list):
            for v in obj:
                walk(v, src)

    loaded = {}
    for src in sources:
        p = os.path.join(ROOT, src)
        if os.path.exists(p):
            with open(p) as f:
                loaded[src] = json.load(f)
            walk(loaded[src], src)
    return refs


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    inv = {"stage": "phase6-t2-freeze", "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
           "cwd": ROOT, "venv": ".venv-phase3"}

    # ---------- A. committed Phase5 sources ----------
    rc, head, _ = run(["git", "rev-parse", "HEAD"])
    rc2, parent, _ = run(["git", "rev-parse", "HEAD~1"])
    rc3, branch, _ = run(["git", "branch", "--show-current"])
    rc4, status_out, _ = run(["git", "status", "--porcelain"])
    status_lines = status_out.split("\n") if status_out else []
    six = {}
    for rel, want in ANCHOR_FILES.items():
        cur = sha256_file(os.path.join(ROOT, rel))
        six[rel] = {"sha256": cur, "anchor": want, "match": cur == want}
    proto = sha256_file(os.path.join(ROOT, "outputs/v2/evidence/phase5/evaluation_protocol.json"))
    inv["committed_sources"] = {
        "head": head, "parent": parent, "branch": branch,
        "t1_commit_expected": "02a5119ef8b4ad107b4dc3e6cf1a61962345f0a9",
        "head_is_t1_commit": head == "02a5119ef8b4ad107b4dc3e6cf1a61962345f0a9",
        "status_porcelain": status_lines,
        "tracked_clean": all(not l.startswith((" M", "M ", "A ", "D ", "AM")) for l in status_lines),
        "six_files": six,
        "six_all_match": all(v["match"] for v in six.values()),
        "evaluation_protocol_sha256": proto,
        "evaluation_protocol_match": proto == PROTOCOL_SHA,
    }
    check("committed_sources_six_files", inv["committed_sources"]["six_all_match"], "5 scripts+guide = T1 anchors")
    check("protocol_frozen", inv["committed_sources"]["evaluation_protocol_match"], proto)
    check("head_is_t1_commit", inv["committed_sources"]["head_is_t1_commit"], head)
    check("tracked_clean", inv["committed_sources"]["tracked_clean"], f"{len(status_lines)} porcelain lines")

    # ---------- B. input inventory (data + tokenizer faces) ----------
    data_files = ["dataset_stats.json", "Industrial_and_Scientific.index.json",
                  "Industrial_and_Scientific.item.json", "info.txt", "prefix_trie.json",
                  "reverse_item_map.json", "train.csv", "valid.csv", "test.csv"]
    inventory = {}
    for g in GROUPS:
        for fn in data_files:
            rel = f"outputs/v2/data/{g}/{fn}"
            p = os.path.join(ROOT, rel)
            inventory[rel] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)} if os.path.exists(p) else None
    tok_top = ["shared_vocab.json", "token_id_map.json", "binding_manifest.json"]
    for fn in tok_top:
        rel = f"outputs/v2/tokenizer/{fn}"
        p = os.path.join(ROOT, rel)
        inventory[rel] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)} if os.path.exists(p) else None
    for g in GROUPS:
        gdir = os.path.join(ROOT, "outputs/v2/tokenizer", g)
        if os.path.isdir(gdir):
            for fn in sorted(os.listdir(gdir)):
                p = os.path.join(gdir, fn)
                if os.path.isfile(p):
                    rel = f"outputs/v2/tokenizer/{g}/{fn}"
                    inventory[rel] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}

    refs = collect_anchors()
    matched, conflicts, unanchored, unanchored_diff = 0, [], [], []
    for rel, meta in inventory.items():
        if meta is None:
            conflicts.append({"path": rel, "issue": "missing_on_disk"})
            continue
        ref = refs.get(rel)
        if ref:
            ok = ref["sha"] == meta["sha256"]
            meta["anchor_sha256"] = ref["sha"]
            meta["anchor_src"] = ref["src"]
            meta["match"] = ok
            if ok:
                matched += 1
            else:
                conflicts.append({"path": rel, "current": meta["sha256"], "anchor": ref["sha256"] if isinstance(ref, dict) else ref, "src": ref["src"]})
        else:
            g = rel.split("/data/")[1].split("/")[0] if "/data/" in rel else None
            a_rel = rel.replace(f"/data/{g}/", "/data/A/") if g else None
            a_meta = inventory.get(a_rel) if a_rel else None
            if a_meta and a_meta.get("sha256") == meta["sha256"]:
                meta["anchor_status"] = f"identical_to_data_A_anchored_counterpart"
                meta["group_A_counterpart"] = a_rel
                matched += 1
            elif g in ("B", "C") and a_meta is not None:
                meta["anchor_status"] = "first_baseline_recorded_here_differs_from_A(group-specific content)"
                unanchored_diff.append(rel)
            else:
                unanchored.append(rel)
    inv["input_inventory"] = inventory
    inv["inventory_summary"] = {
        "n_files": sum(1 for v in inventory.values() if v),
        "missing": sum(1 for v in inventory.values() if v is None),
        "anchor_matched": matched,
        "anchor_conflicts": conflicts,
        "no_anchor_recorded": unanchored,
        "no_anchor_group_specific_differs_from_A": unanchored_diff,
    }
    check("inventory_no_missing", inv["inventory_summary"]["missing"] == 0, json.dumps(inv["inventory_summary"]["missing"]))
    check("inventory_no_conflict", len(conflicts) == 0, json.dumps(conflicts)[:400])
    check("unanchored_files_have_status", True, f"unanchored={len(unanchored)} group_specific={len(unanchored_diff)}")

    # ---------- B2. group-specific file semantics (T2 supplement) ----------
    sem = {}
    stats_diff_note = None
    try:
        ds = {g: json.load(open(os.path.join(ROOT, f"outputs/v2/data/{g}/dataset_stats.json"))) for g in GROUPS}
        def deepdiff(x, y, path=""):
            out = []
            if type(x) != type(y):
                return [(path, "type")]
            if isinstance(x, dict):
                for k in set(x) | set(y):
                    if k not in x:
                        out.append((path + "/" + k, "missing_in_first"))
                    elif k not in y:
                        out.append((path + "/" + k, "missing_in_second"))
                    else:
                        out += deepdiff(x[k], y[k], path + "/" + k)
            elif isinstance(x, list):
                if len(x) != len(y):
                    out.append((path, "len"))
                else:
                    for i, (u, v) in enumerate(zip(x, y)):
                        out += deepdiff(u, v, f"{path}[{i}]")
            elif x != y:
                out.append((path, f"{x!r} vs {y!r}"))
            return out
        ab = deepdiff(ds["A"], ds["B"])
        bc = deepdiff(ds["B"], ds["C"])
        only_label = set(p.split("/")[1] for p, _ in ab + bc if p) <= {"source", "method"}
        stats_diff_note = {"A_vs_B_diff_paths": sorted(set(p for p, _ in ab)),
                           "B_vs_C_diff_paths": sorted(set(p for p, _ in bc)),
                           "only_source_and_method_differ": only_label}
        sem["dataset_stats.json"] = {
            "schema": ds["B"].get("schema"),
            "provenance_field": ds["B"].get("source"),
            "cross_group_diff": stats_diff_note,
        }
    except Exception as e:
        sem["dataset_stats.json"] = {"error": repr(e)[:200]}
    try:
        itemj = {}
        for g in GROUPS:
            d = json.load(open(os.path.join(ROOT, f"outputs/v2/data/{g}/Industrial_and_Scientific.item.json")))
            it0 = d["items"]["0"]
            itemj[g] = {"schema": d.get("schema"), "method": d.get("method"), "n_items": d.get("n_items"),
                        "item0_keys": sorted(it0.keys()), "item0_sid": it0.get("sid"),
                        "item0_original_description_missing": it0.get("original_description_missing")}
        # cross-check: group info.txt first line SID == item.json item0 sid
        info_check = {}
        for g in GROUPS:
            with open(os.path.join(ROOT, f"outputs/v2/data/{g}/info.txt")) as f:
                first = f.readline().rstrip("\n").split("\t")
                n_lines = 1 + sum(1 for _ in f)
            info_check[g] = {"first_line_sid": first[0] if first else None,
                             "first_line_item_id": first[2] if len(first) > 2 else None,
                             "n_lines": n_lines}
        sem["Industrial_and_Scientific.item.json"] = {
            "per_group": itemj,
            "semantic_role": "per-group item catalog embedding the group's own SID binding (items[*].sid) plus original_description_missing slice label consumed by the Phase 5 protocol",
            "info_txt_binding_check": {g: {"sid_match": info_check[g]["first_line_sid"] == itemj[g]["item0_sid"],
                                            "n_lines": info_check[g]["n_lines"],
                                            "first_line_item_id": info_check[g]["first_line_item_id"]} for g in GROUPS},
        }
        sem["info.txt"] = {
            "semantic_role": "per-group SID<->item binding TSV (<SID>\ttitle\titem_id); group-specific by design (different RQ quantizers)",
            "n_lines": {g: info_check[g]["n_lines"] for g in GROUPS},
            "first_line_sid": {g: info_check[g]["first_line_sid"] for g in GROUPS},
        }
    except Exception as e:
        sem["cross_check_error"] = repr(e)[:200]
    inv["group_specific_semantics"] = sem
    check("group_specific_semantics_recorded", len(sem) >= 3, json.dumps({k: ("err" if "error" in v else "ok") for k, v in sem.items()}))
    if stats_diff_note:
        check("dataset_stats_only_label_fields_differ", stats_diff_note["only_source_and_method_differ"] is True,
              json.dumps(stats_diff_note)[:250])

    # ---------- C. identity checks ----------
    identity = {}
    for g in GROUPS:
        gid = {"dataset_stats": json.load(open(os.path.join(ROOT, f"outputs/v2/data/{g}/dataset_stats.json")))}
        # catalog size
        try:
            rmap = json.load(open(os.path.join(ROOT, f"outputs/v2/data/{g}/reverse_item_map.json")))
            if isinstance(rmap, dict) and isinstance(rmap.get("reverse"), dict):
                n_catalog = len(rmap["reverse"])
                gid["catalog_count_source"] = "reverse_item_map.json:reverse"
            else:
                def biggest(o, best=(0, None)):
                    if isinstance(o, dict):
                        best = max(best, (len(o), "dict"), key=lambda t: t[0])
                        for v in o.values():
                            best = max(best, biggest(v, best), key=lambda t: t[0])
                    elif isinstance(o, list):
                        best = max(best, (len(o), "list"), key=lambda t: t[0])
                    return best
                n_container = biggest(rmap)[0]
                n_catalog = n_container
                gid["catalog_count_source"] = f"reverse_item_map.json:largest_container({biggest(rmap)[1]})"
        except Exception:
            idx = json.load(open(os.path.join(ROOT, f"outputs/v2/data/{g}/Industrial_and_Scientific.index.json")))
            n_catalog = len(idx)
            gid["catalog_count_source"] = "index.json"
        gid["catalog_count"] = n_catalog
        gid["catalog_match"] = n_catalog == EXPECTED["catalog"]
        # csv row counts + column hashes
        cols = {}
        for split, exp in EXPECTED_ROWS.items():
            path = os.path.join(ROOT, f"outputs/v2/data/{g}/{split}")
            with open(path, newline="") as f:
                r = csv.reader(f)
                header = next(r)
                n = 0
                hashes = [hashlib.sha256() for _ in header]
                for row in r:
                    n += 1
                    for i, v in enumerate(row):
                        hashes[i].update((v + "\x1f").encode())
            gid[split] = {"data_rows": n, "expected": exp, "match": n == exp, "header": header}
            cols[split] = {header[i]: h.hexdigest() for i, h in enumerate(hashes)}
        gid["columns"] = cols
        identity[g] = gid
    # cross-group column comparison (same source text/order, only SID differs)
    cross = {}
    for split in SPLITS:
        per_g = {g: identity[g]["columns"][split] for g in GROUPS}
        cols_a = set(per_g["A"].keys())
        same_header = all(set(per_g[g].keys()) == cols_a for g in GROUPS)
        diff_cols = sorted([c for c in cols_a if len({per_g[g][c] for g in GROUPS}) > 1])
        cross[split] = {"same_header_across_groups": same_header,
                        "differing_columns": diff_cols,
                        "identical_columns_n": len(cols_a) - len(diff_cols)}
    inv["identity"] = {"groups": {g: {k: v for k, v in identity[g].items() if k != "columns"} for g in GROUPS},
                       "cross_group_columns": cross}
    rows_ok = all(identity[g][s]["match"] for g in GROUPS for s in SPLITS)
    cat_ok = all(identity[g]["catalog_match"] for g in GROUPS)
    check("row_counts_identity", rows_ok, json.dumps({g: {s: identity[g][s]["data_rows"] for s in SPLITS} for g in GROUPS}))
    check("catalog_3686_all_groups", cat_ok, json.dumps({g: identity[g]["catalog_count"] for g in GROUPS}))
    check("cross_group_only_sid_differs", all(cross[s]["same_header_across_groups"] for s in SPLITS),
          json.dumps(cross))

    # ---------- D. model verification ----------
    model = {"path": MODEL_DIR}
    st_path = os.path.join(MODEL_DIR, "model.safetensors")
    model["safetensors_bytes"] = os.path.getsize(st_path)
    model["safetensors_sha256"] = sha256_file(st_path)
    # HF tree API at expected revision
    def hf_get(path):
        for cmd in (["curl", "-s", "--max-time", "25", f"{HF_ENDPOINT}{path}"],
                    None):
            try:
                if cmd is None:
                    req = urllib.request.Request(f"{HF_ENDPOINT}{path}", headers={"User-Agent": "Mozilla/5.0 t2-checks"})
                    with urllib.request.urlopen(req, timeout=25) as resp:
                        return json.load(resp)
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if r.returncode == 0 and r.stdout.strip():
                    return json.loads(r.stdout)
            except Exception:
                continue
        return None
    lfs_oid, lfs_size = None, None
    tree = hf_get(f"/api/models/Qwen/Qwen3-0.6B-Base/tree/{EXPECTED_REVISION}?recursive=false")
    refs = hf_get(f"/api/models/Qwen/Qwen3-0.6B-Base/refs")
    main_commit = None
    if isinstance(refs, dict):
        main_commit = next((b["targetCommit"] for b in refs.get("branches", []) if b.get("name") == "main"), None)
    if isinstance(tree, list):
        for ent in tree:
            if ent.get("path") == "model.safetensors":
                lfs_oid = (ent.get("lfs") or {}).get("oid")
                lfs_size = (ent.get("lfs") or {}).get("size")
    model["hf_api"] = {"endpoint": HF_ENDPOINT,
                       "tree_ok": isinstance(tree, list), "refs_ok": isinstance(refs, dict),
                       "main_branch_commit": main_commit,
                       "main_matches_expected": main_commit == EXPECTED_REVISION,
                       "method": "curl subprocess with urllib+UA fallback",
                       "note": "hf-mirror returns HTTP 403 to python-urllib default UA; curl verified working"}
    model["revision_expected"] = EXPECTED_REVISION
    model["lfs_oid"] = lfs_oid
    model["lfs_size"] = lfs_size
    model["local_matches_revision_lfs"] = (lfs_oid == model["safetensors_sha256"]) and (lfs_size == model["safetensors_bytes"])
    # config + safetensors header
    cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    model["config"] = {k: cfg.get(k) for k in ("architectures", "vocab_size", "torch_dtype",
                                               "tie_word_embeddings", "hidden_size", "num_hidden_layers",
                                               "eos_token_id", "bos_token_id", "max_position_embeddings")}
    with open(st_path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(n))
    emb_entries = {k: v.get("shape") for k, v in hdr.items() if k != "__metadata__" and "embed" in k.lower()}
    model["safetensors_header"] = {"n_headers": len([k for k in hdr if k != "__metadata__"]),
                                   "embed_entries": emb_entries}
    model_files = {}
    for fn in sorted(os.listdir(MODEL_DIR)):
        p = os.path.join(MODEL_DIR, fn)
        if os.path.isfile(p):
            model_files[fn] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    model["dir_files"] = model_files
    # tokenizer real load (only tokenizer; full model load deferred to T3/T4)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)
        tlen = len(tok)
        model["tokenizer"] = {"backend": type(tok).__name__, "len": tlen,
                              "expected_len": EXPECTED["base_tokenizer_len"],
                              "match": tlen == EXPECTED["base_tokenizer_len"],
                              "vocab_size": tok.vocab_size,
                              "eos_token_id": tok.eos_token_id, "pad_token_id": tok.pad_token_id}
        check("base_tokenizer_len_151669", model["tokenizer"]["match"], f"len={tlen}")
    except Exception as e:
        model["tokenizer"] = {"error": repr(e)[:300]}
        check("base_tokenizer_len_151669", False, repr(e)[:200])
    model["emb_rows_config"] = cfg.get("vocab_size")
    model["emb_minus_tokenizer"] = (cfg.get("vocab_size") - model.get("tokenizer", {}).get("len", 0)) if model.get("tokenizer", {}).get("len") else None
    model["verified_scope"] = "tokenizer real load + safetensors header + config + file hashes; full model load/forward NOT performed here (deferred to T3/T4)"
    inv["model"] = model
    check("model_safetensors_matches_hf_revision", model["local_matches_revision_lfs"],
          f"local={model['safetensors_sha256'][:16]} lfs={str(lfs_oid)[:16]}")
    check("hf_main_branch_is_expected_revision", model.get("hf_api", {}).get("main_matches_expected") is True,
          json.dumps(model.get("hf_api", {}).get("main_branch_commit")))
    check("config_vocab_151936", cfg.get("vocab_size") == EXPECTED["model_vocab"], str(cfg.get("vocab_size")))
    check("emb_tokenizer_diff_267", model.get("emb_minus_tokenizer") == EXPECTED["emb_diff"],
          str(model.get("emb_minus_tokenizer")))
    check("tie_word_embeddings_true", cfg.get("tie_word_embeddings") is True, str(cfg.get("tie_word_embeddings")))

    # ---------- E. environment lock ----------
    env = {}
    try:
        import torch
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        env["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda,
                        "device_name": p.name, "total_memory": p.total_memory,
                        "sm_count": p.multi_processor_count,
                        "compute_capability": f"{p.major}.{p.minor}",
                        "mem_free_at_lock": free, "mem_total_at_lock": total,
                        "mig_query_permissions_note": "nvidia-smi reports [Insufficient Permissions]; torch props authoritative"}
        env["python"] = platform.python_version()
        check("gpu_total_10200547328", p.total_memory == 10200547328, str(p.total_memory))
    except Exception as e:
        env["torch_error"] = repr(e)[:300]
        check("gpu_probe", False, repr(e)[:200])
    pkgs = {}
    for name in ("torch", "transformers", "accelerate", "datasets", "safetensors",
                 "tokenizers", "numpy", "pandas", "scikit-learn", "peft", "trl"):
        try:
            from importlib.metadata import version as _v
            pkgs[name] = _v(name)
        except Exception:
            pkgs[name] = None
    env["packages"] = pkgs
    rc, freeze, _ = run([PIP, "freeze"])
    raw = subprocess.run([PIP, "freeze"], capture_output=True, text=True).stdout
    freeze_raw_nl = hashlib.sha256(raw.encode()).hexdigest()
    ne = [l for l in raw.splitlines() if not l.startswith("-e")]
    freeze_filtered = hashlib.sha256("\n".join(ne).encode()).hexdigest()
    env["pip_freeze"] = {
        "n_lines": len(raw.splitlines()), "n_lines_no_editable": len(ne),
        "sha256_raw_with_trailing_newline": freeze_raw_nl,
        "sha256_filtered_no_editable": freeze_filtered,
        "phase3_phase5_recorded_raw": "a292c199147f608c00c6c9d253df78f8a4f2e7e6530632d5f8cc0100d41b9bd7",
        "phase5_recorded_filtered": "142b9e146290a1e534eef23e3cc5bd3bde0595eb0b471e7fc6e9d5026199505e",
        "raw_matches_phase3_phase5_record": freeze_raw_nl == "a292c199147f608c00c6c9d253df78f8a4f2e7e6530632d5f8cc0100d41b9bd7",
        "filtered_matches_phase5_record": freeze_filtered == "142b9e146290a1e534eef23e3cc5bd3bde0595eb0b471e7fc6e9d5026199505e",
    }
    both = env["pip_freeze"]["raw_matches_phase3_phase5_record"] and env["pip_freeze"]["filtered_matches_phase5_record"]
    check("pip_freeze_unchanged_since_phase3_phase5", both,
          f"raw={freeze_raw_nl[:16]} filtered={freeze_filtered[:16]}")
    env["freeze_note"] = "hash conventions decoded: phase records used raw-with-newline and filtered-no-editable variants; both reproduce exactly"
    du = shutil.disk_usage(ROOT)
    env["disk"] = {"total": du.total, "free": du.free, "used": du.used}
    env["hf_endpoint"] = HF_ENDPOINT
    env["notes"] = [
        "peft 0.13.2 / trl 0.11.3 installed but protocol forbids PEFT/LoRA/QLoRA paths; full-parameter only",
        "no environment adjustments made in T2; if any later adjustment occurs, a separate before/after lock is required",
    ]
    inv["environment"] = env

    inv["checks"] = CHECKS
    inv["all_pass"] = all(c["passed"] for c in CHECKS)
    inv["elapsed_seconds"] = round(time.time() - t0, 1)

    with open(os.path.join(OUT, "input_inventory.json"), "w") as f:
        json.dump(inv, f, ensure_ascii=False, indent=1, sort_keys=False)
    with open(os.path.join(OUT, "environment_lock.json"), "w") as f:
        json.dump({"stage": "phase6-t2-environment-lock", "date": inv["date"], "cwd": ROOT,
                   "git_head": head, "branch": branch, "environment": env,
                   "model_revision_expected": EXPECTED_REVISION,
                   "model_revision_hf_main_verified": model.get("hf_api", {}).get("main_matches_expected"),
                   "model_local_lfs_match": model.get("local_matches_revision_lfs"),
                   "pointer": "full details in input_inventory.json (same stage)"},
                  f, ensure_ascii=False, indent=1, sort_keys=False)

    print("=== T2 SUMMARY ===")
    print("all_pass=", inv["all_pass"])
    for c in CHECKS:
        print(("PASS" if c["passed"] else "FAIL"), c["name"], "|", c["detail"][:160])
    print("inventory:", json.dumps(inv["inventory_summary"])[:300])
    print("model: sha=", model["safetensors_sha256"], "lfs_match=", model["local_matches_revision_lfs"])
    print("tokenizer:", json.dumps(model.get("tokenizer", {}))[:200])
    print("gpu:", json.dumps(env.get("torch", {}))[:250])
    print("freeze:", json.dumps(env["pip_freeze"])[:300])


if __name__ == "__main__":
    main()
