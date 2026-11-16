#!/usr/bin/env python
"""Phase 5 T1: environment lock, upstream input inventory & implementation plan.
SCHEMA: phase5-t1-1.0. Read-only wrt upstream products; no training, no test metrics.
Independent recomputation (no Phase 4 generator internals reused)."""
import os, sys, json, hashlib, csv, ast, subprocess, shutil, socket
os.environ.setdefault("OMP_NUM_THREADS", "8")
from pathlib import Path
from collections import Counter

REPO = Path("/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec")
EV = REPO / "outputs/v2/evidence/phase5"
EV.mkdir(parents=True, exist_ok=True)
METHODS = ("A", "B", "C")
EXPECTED_HEADER = ["row_id", "user_id", "history_item_title", "item_title",
                   "history_item_id", "item_id", "history_item_sid", "item_sid"]
SPLIT_ROWS = {"train": 36259, "valid": 4532, "test": 4533}
LOCAL_SHA = {
    "sasrec.py": "05f0c85d1854f0c1ca41fb77b256db28a00f6b6c0127c3ab69236347ffe7cb69",
    "evaluate.py": "0951376c2135da7ef3305390997c9456bcb31d76ee1a01e03fe43c72e3685621",
    "calc.py": "3e5dc6bf10c2e23f50180f520993462c7756b717dd60d646d8b540274a3fce82",
    "SASRecModules_ori.py": "76ef521d9ad16dcf87f85f89604749103db6f087754a3210505bc22180a11bb3",
    "utility.py": "380115dd1e7abbb07b2a0b4e94f24e55786fc873907de3ecd02034f5b2faa1da",
    "data.py": "85edcb6b1e7d1124c204bbdcb312d70903920b3cb5f00bc3fc1414f3df03b555",
    "LogitProcessor.py": "13ab33ff0ed3f769370478a86624499ccc0f8813de9d63e3dc4e2fa25ec59eb8",
}

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def shatext(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))

FAILED = []
def need(key, ok, detail):
    ok = bool(ok)
    print(f"T1 {key}: {'OK' if ok else 'FAIL'} {detail if not ok else ''}", flush=True)
    if not ok:
        FAILED.append(key)
    return {"ok": ok, **({"detail": detail} if detail is not None else {})}

# =================== 1. environment lock ===================
env = {"schema": "phase5-environment-lock-1.0", "date": "2026-11-11"}
env["cwd"] = str(REPO)
env["host_container"] = run(["hostname"]).stdout.strip()
env["git"] = {
    "head": run(["git", "rev-parse", "HEAD"]).stdout.strip(),
    "log3": [l for l in run(["git", "log", "--format=%H %P %ad %s", "--date=iso", "-3"]).stdout.strip().splitlines()],
    "tracked_diff_empty": run(["git", "diff", "--quiet"]).returncode == 0,
    "diff_stat": run(["git", "diff", "--stat"]).stdout.strip() or "(empty)",
    "porcelain": run(["git", "status", "--porcelain"]).stdout.strip().splitlines(),
}
gpu_q = run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"])
env["nvidia_smi"] = gpu_q.stdout.strip() or f"query_failed: {gpu_q.stderr.strip()[:120]}"
import torch
p = torch.cuda.get_device_properties(0)
env["device"] = {"name": p.name, "total_memory_bytes": int(p.total_memory),
                 "multi_processor_count": p.multi_processor_count, "cc": f"{p.major}.{p.minor}",
                 "cuda_available": torch.cuda.is_available(), "note": "nvidia-smi memory shows [Insufficient Permissions] in container; torch props authoritative"}
env["python"] = sys.version.split()[0]
env["packages"] = {"torch": torch.__version__}
import numpy, pandas, sklearn
env["packages"].update({"numpy": numpy.__version__, "pandas": pandas.__version__, "sklearn": sklearn.__version__})
venv_pip = str(REPO / ".venv-phase3/bin/pip")
fr = run([venv_pip, "freeze"]).stdout
env["pip_freeze"] = {
    "sha256_filtered_no_editable": shatext("\n".join(l for l in fr.splitlines() if not l.startswith("-e"))),
    "sha256_full": shatext(fr), "n_lines": len(fr.splitlines()),
}
p4lock = json.load(open(EV.parent / "phase4/environment_lock.json"))
def find_freeze_sha(obj):
    out = []
    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, str) and len(v) == 64 and ("freeze" in k.lower()):
                    out.append((path + "/" + k, v))
                walk(v, path + "/" + k)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
    walk(obj, "")
    return out
env["phase4_freeze_comparison"] = {
    "phase4_lock_sha256": sha(EV.parent / "phase4/environment_lock.json"),
    "stored_freeze_shas": find_freeze_sha(p4lock),
}
du = shutil.disk_usage(REPO)
env["disk"] = {"total_bytes": du.total, "free_bytes": du.free}
(EV / "environment_lock.json").write_text(json.dumps(env, ensure_ascii=False, indent=1))
print("ENV_LOCK written", flush=True)

# =================== 2. code audit hashes ===================
code = {"schema": "phase5-code-audit-1.0", "date": "2026-11-11",
        "files": {}, "local_comparison_basis": "worker local clone of same repo, files unchanged since Phase3 baseline 44cb9ce"}
for f, lh in LOCAL_SHA.items():
    rh = sha(REPO / f)
    code["files"][f] = {"sha256": rh, "bytes": (REPO / f).stat().st_size,
                        "matches_local_clone": rh == lh}
code["audit_facts"] = {
 "sasrec.py": [
  "selection metric: ndcg_list[-1] with topk=[1,3,5,10,20] == NDCG@20 (protocol requires NDCG@10 -> adaptation)",
  "test leakage: evaluate_games(test) every eval round + final test eval + save_logits(test) -> all removed in adaptation",
  "wandb: init by default (only disabled when --debug) -> adaptation disables unconditionally",
  "DRO: propensity score computed from train targets each run; active only when alpha!=0; adaptation removes computation (recorded diff)",
  "negative sampling: per train target neg=np.random.randint(item_num), retry while ==target; numpy legacy global RNG (np.random.seed at setup_seed); negatives from [0,item_num) real items only, padding=item_num excluded by range",
  "padding: states padded with item_num; embedding rows item_num+1; mask=torch.ne(states,item_num)",
  "seq_size=10 hardcoded; item_num=len(info lines); Adam(lr, eps=1e-8, weight_decay=l2_decay=1e-5)",
  "loss: BCEWithLogitsLoss on cat(pos_scores,neg_scores) vs labels 1/0",
  "eval cadence: total_step % (num_batches*eval_num)==0 -> once per epoch at eval_num=1",
  "early stop: early_stop=0 on improvement else +=1; return when early_stop > early_stop_arg(20) -> stops at 21st consecutive non-improving eval (inherited semantics, patience param=20)",
  "tie rule: ndcg_last > ndcg_max strictly greater, first best kept (pre-declared)",
  "save: best model deepcopy in RAM, state_dict saved at end"],
 "SASRecModules_ori.py": [
  "causal future blinding: torch.tril mask applied to attention logits (verified present)",
  "key/query masking via torch.sign(abs(sum(embedding)))==0 padding detection",
  "scaling 1/sqrt(hidden_size); residual add; dropout on attention weights; FFN Conv1d 1x1 + LayerNorm residual"],
 "evaluate.py": [
  "LLM beam generation entry (not executed this phase; T3 adapts interface only)",
  "constraint: ConstrainedLogitsProcessor with hash-dict prefix trie from tokenized SID strings + EOS appended",
  "generation: num_beams/num_return_sequences=num_beams, length_penalty arg, left padding, pad=eos",
  "transformers>=4.50 guard: do_sample=False passed as generate() kwarg to override checkpoint generation_config",
  "gaps for adaptation: temperature not explicitly forced (protocol wants =1.0); row_id not visibly carried through dataset objects; group binding via --info_file arg (per-group info.txt)"],
 "calc.py": [
  "summary print only: first-hit break at minID, NDCG via 1/ln(minID+2) divided by (1/ln2) == 1/log2(rank+1)",
  "no user-macro average, no Coverage, no duplicate-rate, no illegal-placeholder positioning; illegal items only counted/printed -> cannot replace v2 statistics (guide-confirmed)"],
 "utility.py": ["pad_history/calculate_hit/extract_axis_1; current sasrec.py train path does not call calculate_hit (inline CUDA hit calc)"],
 "data.py": ["EvalD3Dataset/EvalSidDataset prompt builders consumed by evaluate.py (T3 scope)"],
 "LogitProcessor.py": ["ConstrainedLogitsProcessor used by evaluate.py beam path (T3 scope)"],
}
(EV / "code_audit.json").write_text(json.dumps(code, ensure_ascii=False, indent=1))
print("CODE_AUDIT written; local-match:", all(v["matches_local_clone"] for v in code["files"].values()), flush=True)

# =================== 3. inventory: consumed files ===================
inv = {"schema": "phase5-input-inventory-1.0", "date": "2026-11-11"}
consumed = [
    "outputs/v2/data/A/train.csv", "outputs/v2/data/A/valid.csv", "outputs/v2/data/A/test.csv",
    "outputs/v2/data/A/Industrial_and_Scientific.index.json", "outputs/v2/data/A/reverse_item_map.json",
    "outputs/v2/data/A/info.txt", "outputs/v2/data/A/Industrial_and_Scientific.item.json",
    "outputs/v2/data/A/dataset_stats.json",
    "outputs/v2/manifests/rows_manifest.jsonl", "outputs/v2/manifests/quality_features.jsonl",
    "outputs/v2/manifests/catalog_clean.jsonl", "outputs/v2/manifests/file_hashes.json",
    "outputs/v2/manifests/data_audit.json",
]
inv["consumed"] = {}
for rel in consumed:
    fp = REPO / rel
    inv["consumed"][rel] = {"sha256": sha(fp), "bytes": fp.stat().st_size}

# upstream comparison vs Phase4 accepted inventory (t5 keys are absolute paths -> normalize)
REPO_PREFIX = str(REPO) + "/"
p4 = json.load(open(EV.parent / "phase4/t5_files_sha256.json"))["files"]
p4_rel = {(k[len(REPO_PREFIX):] if k.startswith(REPO_PREFIX) else k): v for k, v in p4.items()}
inv["phase4_t5_comparison"] = {}
for rel, meta in inv["consumed"].items():
    if rel in p4_rel:
        inv["phase4_t5_comparison"][rel] = {"phase4_sha256": p4_rel[rel]["sha256"],
                                            "match": p4_rel[rel]["sha256"] == meta["sha256"]}
n_data_consumed = sum(1 for rel in inv["consumed"] if rel.startswith("outputs/v2/data/"))
inv["phase4_t5_comparison_all_match"] = (len(inv["phase4_t5_comparison"]) == n_data_consumed
                                         and all(v["match"] for v in inv["phase4_t5_comparison"].values()))
inv["phase4_t5_comparison_n"] = len(inv["phase4_t5_comparison"])

# identity-check-only (read once to verify A/B/C item-ID identity; not consumed by B0/B1)
inv["identity_check_read_only"] = {}
for m in ("B", "C"):
    for sp in ("train", "valid", "test"):
        rel = f"outputs/v2/data/{m}/{sp}.csv"
        inv["identity_check_read_only"][rel] = {"sha256": sha(REPO / rel), "bytes": (REPO / rel).stat().st_size}

# phase1 provenance: file_hashes.json summary
fh = json.load(open(REPO / "outputs/v2/manifests/file_hashes.json"))
def summarize_hashes(o, depth=0):
    if isinstance(o, dict):
        if depth >= 2:
            return f"<dict n={len(o)}>"
        return {k: summarize_hashes(v, depth + 1) for k, v in o.items()}
    if isinstance(o, list):
        return f"<list n={len(o)}>"
    return o
inv["phase1_file_hashes_summary"] = summarize_hashes(fh)
inv["phase1_file_hashes_sha256"] = sha(REPO / "outputs/v2/manifests/file_hashes.json")

# manifest provenance: search prior-phase evidence (phase1/phase2/phase4 json, files <5MB) for these manifests' hashes
MANIFESTS = ["rows_manifest.jsonl", "quality_features.jsonl", "catalog_clean.jsonl",
             "file_hashes.json", "data_audit.json"]
def find_sha_hits(o, targets, hits):
    if isinstance(o, dict):
        for k, v in o.items():
            ks = str(k)
            for t in targets:
                if ks == t or ks.endswith("/" + t):
                    if isinstance(v, dict) and "sha256" in v and isinstance(v["sha256"], str):
                        hits[t].append({"key": ks, "phase_sha256": v["sha256"]})
                    elif isinstance(v, str) and len(v) == 64:
                        hits[t].append({"key": ks, "phase_sha256": v})
            find_sha_hits(v, targets, hits)
    elif isinstance(o, list):
        for v in o:
            find_sha_hits(v, targets, hits)

prov = {m: [] for m in MANIFESTS}
scanned = []
for evdir in sorted((REPO / "outputs/v2/evidence").glob("phase*")):
    for jf in sorted(evdir.glob("*.json")):
        if jf.stat().st_size > 5_000_000:
            continue
        try:
            obj = json.load(open(jf))
        except Exception:
            continue
        scanned.append(str(jf.relative_to(REPO)))
        find_sha_hits(obj, MANIFESTS, prov)
inv["manifest_provenance"] = {}
for m in MANIFESTS:
    cur = inv["consumed"][f"outputs/v2/manifests/{m}"]["sha256"]
    entries = []
    for h in prov[m]:
        entries.append({**h, "match_current": h["phase_sha256"] == cur})
    inv["manifest_provenance"][m] = {
        "current_sha256": cur,
        "prior_hits": entries,
        "status": "prior_hash_match" if any(e["match_current"] for e in entries)
                  else ("prior_hash_MISMATCH" if entries else "no_prior_baseline_found_in_evidence_jsons")}
inv["manifest_provenance_scanned_files"] = scanned

inv["excluded"] = {
    "outputs/v2/rqvae/**": "9 RQ-VAE checkpoints+diagnostics; quantizer artifacts not consumed by traditional baselines",
    "outputs/v2/sid/**": "SID diagnostics/exports; not consumed by B0/B1",
    "outputs/v2/tokenizer/**": "shared vocab/tokenizers/binding manifest; LLM-side artifacts not consumed this phase",
    "outputs/v2/data/{B,C}/**": "per-group SID-differentiated copies; item-ID columns verified identical to A (identity check), B0/B1 consume only A",
    "outputs/v2/data/*/rl_tasks_10000_sid.jsonl": "RL prompt manifest for Phase 7/8; not consumed",
    "llm_checkpoints": "none exist; no LLM loaded anywhere in Phase 5",
    "data/Amazon/**": "Phase 1 original inputs; consumed transitively via Phase 1 manifests (hashes recorded), not read directly this phase",
}
(EV / "input_inventory.json").write_text(json.dumps(inv, ensure_ascii=False, indent=1))
print("INVENTORY written; phase4 match:", inv["phase4_t5_comparison_all_match"], flush=True)

# =================== 4. input checks (independent recompute) ===================
idx = json.load(open(REPO / "outputs/v2/data/A/Industrial_and_Scientific.index.json"))
catalog_ids = [int(x) for x in idx["item_ids"]]
catalog_set = set(catalog_ids)
chk = {}
chk["catalog_3686_static"] = need("catalog_3686_static", len(catalog_ids) == 3686 and len(catalog_set) == 3686,
    {"n": len(catalog_ids)})
chk["catalog_contiguous_0_3685"] = need("catalog_contiguous_0_3685",
    catalog_set == set(range(3686)) and min(catalog_ids) == 0 and max(catalog_ids) == 3685, None)

qf = {}
with open(REPO / "outputs/v2/manifests/quality_features.jsonl") as f:
    for line in f:
        d = json.loads(line)
        qf[str(d["item_id"])] = bool(d["original_description_missing"])
n_missing = sum(qf.values())
chk["missing_desc_229"] = need("missing_desc_229", n_missing == 229, {"count": n_missing, "total_items": len(qf)})

item_cat = json.load(open(REPO / "outputs/v2/data/A/Industrial_and_Scientific.item.json"))
items = item_cat["items"]
chk["item_json_3686"] = need("item_json_3686", item_cat.get("n_items") == 3686 and len(items) == 3686, None)
sample_fields = list(next(iter(items.values())).keys())
chk["item_json_orig_missing_field"] = need("item_json_orig_missing_field",
    "original_description_missing" in sample_fields, {"fields": sample_fields})
if "original_description_missing" in sample_fields:
    agree = all(bool(items[k].get("original_description_missing")) == qf[k] for k in items)
    im_n = sum(bool(v.get("original_description_missing")) for v in items.values())
    chk["item_json_vs_qf_agree"] = need("item_json_vs_qf_agree", agree and im_n == 229, {"item_json_missing": im_n})

splits = {}
for sp, n_exp in SPLIT_ROWS.items():
    rows = []
    with open(REPO / f"outputs/v2/data/A/{sp}.csv", newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        rows = list(rd)
    c = {}
    c["header_exact"] = header == EXPECTED_HEADER
    c["n_rows"] = len(rows)
    rid_ok = all(r[0] == f"{sp}:{i}" for i, r in enumerate(rows))
    c["row_id_positional"] = rid_ok
    c["row_id_unique"] = len(set(r[0] for r in rows)) == len(rows)
    users = set(r[1] for r in rows)
    c["n_users"] = len(users)
    bad_cat = 0; hist_bad = 0; max_len = 0; targets = []
    for r in rows:
        t = int(r[5]); targets.append(t)
        if t not in catalog_set: bad_cat += 1
        h = [int(x) for x in ast.literal_eval(r[4])]
        max_len = max(max_len, len(h))
        if any(x not in catalog_set for x in h): hist_bad += 1
    c["target_in_catalog"] = bad_cat == 0
    c["history_in_catalog"] = hist_bad == 0
    c["max_hist_len"] = max_len
    c["missing_target_rows"] = sum(1 for t in targets if qf[str(t)])
    splits[sp] = {"checks": c, "targets": targets, "users": users}
    ok_g = (c["header_exact"] and c["n_rows"] == n_exp and c["row_id_positional"] and
            c["row_id_unique"] and c["target_in_catalog"] and c["history_in_catalog"] and c["max_hist_len"] <= 10)
    print(f"T1 split_{sp}: {'OK' if ok_g else 'FAIL'} {json.dumps(c)}", flush=True)
    if not ok_g: FAILED.append(f"split_{sp}")

chk["counts_36259_4532_4533"] = need("counts_36259_4532_4533",
    splits["train"]["checks"]["n_rows"] == 36259 and splits["valid"]["checks"]["n_rows"] == 4532
    and splits["test"]["checks"]["n_rows"] == 4533, None)
chk["test_users_1987"] = need("test_users_1987", splits["test"]["checks"]["n_users"] == 1987,
    {"n": splits["test"]["checks"]["n_users"]})
chk["test_missing_target_447"] = need("test_missing_target_447", splits["test"]["checks"]["missing_target_rows"] == 447,
    {"n": splits["test"]["checks"]["missing_target_rows"]})
chk["test_identity_only"] = {"ok": True,
    "detail": "test used exclusively for row/user/missing-label identity audit; NO ranking metric computed on test in Phase 5"}

# data_audit cross-check of user counts
audit = json.load(open(REPO / "outputs/v2/manifests/data_audit.json"))
aud_users = {}
try:
    for sp in ("train", "valid", "test"):
        aud_users[sp] = audit["deterministic"]["splits"][sp]["n_users"]
except KeyError:
    aud_users = {"error": "structure"}
rec_users = {sp: splits[sp]["checks"]["n_users"] for sp in splits}
chk["user_counts_vs_audit"] = need("user_counts_vs_audit",
    all(aud_users.get(sp) == rec_users[sp] for sp in rec_users) if "error" not in aud_users else False,
    {"audit": aud_users, "recomputed": rec_users})

# A/B/C item-ID column identity
identity = {}
for m in ("B", "C"):
    same = True; detail = {}
    for sp in ("train", "valid", "test"):
        with open(REPO / f"outputs/v2/data/A/{sp}.csv", newline="") as f:
            ra = list(csv.reader(f))[1:]
        with open(REPO / f"outputs/v2/data/{m}/{sp}.csv", newline="") as f:
            rb = list(csv.reader(f))[1:]
        eq = len(ra) == len(rb) and all(
            a[0] == b[0] and a[1] == b[1] and a[4] == b[4] and a[5] == b[5]
            for a, b in zip(ra, rb))
        detail[sp] = eq
        same &= eq
    identity[m] = {"row_id_user_hist_target_identical": same, "per_split": detail}
    print(f"T1 identity_{m}: {'OK' if same else 'FAIL'}", flush=True)
    if not same: FAILED.append(f"identity_{m}")
chk["abc_itemid_columns_identical"] = {"ok": all(v["row_id_user_hist_target_identical"] for v in identity.values()),
                                       "groups": identity,
                                       "columns_compared": ["row_id", "user_id", "history_item_id", "item_id"]}

# rows_manifest join
man = {}
with open(REPO / "outputs/v2/manifests/rows_manifest.jsonl") as f:
    for line in f:
        d = json.loads(line)
        man[d["row_id"]] = d
join_ok = len(man) == 45324
manifest_first_fields = sorted(next(iter(man.values())).keys())
mism = 0
for sp in ("train", "valid", "test"):
    with open(REPO / f"outputs/v2/data/A/{sp}.csv", newline="") as f:
        rd = csv.reader(f); next(rd)
        for i, r in enumerate(rd):
            d = man.get(r[0])
            if d is None or d.get("split") != sp or d.get("row_index") != i \
               or [str(x) for x in d.get("history_ids", [])] != [str(x) for x in ast.literal_eval(r[4])] \
               or str(d.get("item_id_raw", d.get("item_id", ""))) != r[5]:
                mism += 1
chk["rows_manifest_join_45324"] = need("rows_manifest_join_45324", join_ok and mism == 0,
    {"manifest_lines": len(man), "mismatches": mism, "first_line_fields": manifest_first_fields})

chk["phase4_t5_hash_match"] = need("phase4_t5_hash_match", inv["phase4_t5_comparison_all_match"],
    {"n_compared": inv["phase4_t5_comparison_n"]})
mm = [m for m, v in inv["manifest_provenance"].items() if v["status"] == "prior_hash_MISMATCH"]
chk["manifest_provenance_consistent"] = need("manifest_provenance_consistent", len(mm) == 0,
    {"mismatched": mm, "statuses": {m: inv["manifest_provenance"][m]["status"] for m in MANIFESTS}})
print("T1 manifest_provenance:", json.dumps({m: inv["manifest_provenance"][m]["status"] for m in MANIFESTS}), flush=True)

# =================== 5. implementation_plan.md ===================
inv_sha = sha(EV / "input_inventory.json")
env_sha = sha(EV / "environment_lock.json")
code_sha = sha(EV / "code_audit.json")
plan = f"""# Phase 5 实现计划（implementation_plan.md）

> 生成：T1（2026-11-11，worker）。锁定对象：拟新增/适配入口与原实现差异。本期不训练 LLM、不评 test。
> 环境：HEAD `{env['git']['head']}`（tracked 零 diff）；device {env['device']['name']} {env['device']['total_memory_bytes']}B cc{env['device']['cc']}；Python {env['python']} torch {env['packages']['torch']}。
> 输入基线：消费文件 {len(inv['consumed'])} 个全部与 Phase4 t5 在档哈希一致（input_inventory.json=`{inv_sha}`）。

## 1. 共同接口冻结（B0/B1 及未来生成式共用）

- **目录与 ID 映射**：静态完整目录 = `outputs/v2/data/A/Industrial_and_Scientific.index.json` item_ids（3,686 项，整数 0..3685 连续，规范 item-ID 升序）。SASRec/评测内部索引 = 原始 item ID 本身（恒等映射）；padding 索引 = 3686（`item_num`），embedding 表 3,687 行；ID 0 是真实商品。
- **行集合**：A/B/C 三组 CSV 的 row_id/user_id/history_item_id/item_id 列逐行一致（本期复验）；B0/B1 单份基线消费 `outputs/v2/data/A/{{train,valid,test}}.csv`（36,259/4,532/4,533 行），row_id `{{split}}:行号` 唯一且随行序递增；不同组不产生三份基线。
- **自然切片标签**：`outputs/v2/manifests/quality_features.jsonl` 字段 `original_description_missing`（229 项缺失；与 item.json 同字段逐项一致）。按**目标**商品标签切片；清洗后空 327 不是切片标签。test 缺失目标行 447（仅身份审计）。
- **主口径**：相同完整目录、不屏蔽历史商品、非法候选保留原排名占位、重复目标按首次出现计命中；HR/NDCG@3/5/10 行平均 + 用户宏平均另列；NDCG@10 主指标；Coverage@10 = 前 10 唯一合法 item/3686；重复率@K 逐行 `(K-唯一数)/K`。

## 2. T2：`scripts/v2_evaluate.py`（新增）

- 行接口：`row_id/user_id/target_item_id/ranked_item_ids` + 合法目录 + 原始缺失标签；生成式原始 SID 串另行保留作非法诊断。
- 指标：HR@K/ND CG@K（1/log2(rank+1)，rank 为 1 基）、行平均、用户宏平均（先用户内行均值再对用户平均）、Coverage@10、非法率、重复率@K、自然切片（按目标标签）。
- 硬阻断：row_id 缺失/重复、预测行数≠评测行数、未知 target、ranked 列表类型错——抛异常退出，不以交集静默评测。
- 11 类合成样例手工答案对照（绝对误差 ≤1e-12）后才可用于 validation。

## 3. T3：聚类 Bootstrap 与生成式适配契约

- Bootstrap：seed42、2,000 轮、有放回抽 U 个用户簇（U=该评测集合实际唯一用户数，不硬编码 1987）、同用户全部行 + 重复抽中次数保留；方法间共享抽样清单；差值 = 每轮行加权均值差；CI = 2.5%/97.5% 分位；空切片轮标无效并计数。
- `evaluate.py` 适配契约（合成验证，不加载 LLM）：组绑定（--info_file 逐组 TSV）、row_id 全程保留、确定性 generation config 显式锁定（Beam 开发 10/正式 50 返回 50、length_penalty=0、do_sample=False、top_k=None、top_p=None、temperature=1.0）、四 token SID+EOS 路径、非法路径保位与拒绝。

## 4. T4：`scripts/v2_popularity.py`（新增）

- 仅计 train 目标频次（36,259 目标合计）；目录中未出现商品记 0；同频按规范数值 item ID 升序；输出完整 3,686 目录序列 + 频次向量 + 构建配置与输入/代码哈希；validation 经 v2_evaluate 评估；不按历史过滤。

## 5. T5：`scripts/v2_sasrec.py`（新增，自 sasrec.py 适配）

**继承**（SASRecModules_ori 结构不动）：单头 hidden32、state_size=10、因果 tril attention、零和 padding 掩码、`ff_out *= mask`、BCEWithLogitsLoss、每目标 1 负例 `np.random.randint(3686)` 重试 ≠目标（负例仅真商品，不含 padding，不用 test 目标排除；numpy legacy 全局 RNG，seed42 记录在案）、Adam(lr=1e-3, eps=1e-8, wd=1e-5)、batch1024 shuffle、每 epoch 验证一次、state_size 右 padding=item_num、tie 规则严格大于先到先得、早停继承语义（连续 21 次未改善返回，参数 patience=20）、最佳模型 deepcopy+state_dict 落盘。

**显式差异**（相对原 sasrec.py）：
1. 选模指标 NDCG@10（原代码取 ndcg_list[-1]=NDCG@20）；
2. 全程不评测 test（原代码每 eval 轮评 test + 结束再评 + 保存 logits）；
3. wandb 完全禁用；
4. DRO/propensity 计算整体移除（原 alpha=0 时已不生效，语义等价）；
5. 数据路径 outputs/v2/data/A/*.csv，列 history_item_id/item_id 解析 + row_id 保留；
6. checkpoint/日志/预测落盘 outputs/v2/baselines/B1/；
7. 训练样本 = 每 train 行 1 目标（36,259/epoch），不把同用户多行压成长序列。

**闸门**：先小样本合成检查因果 mask（扰动未来输入不影响过去输出）、padding 隔离、负例 ≠ 目标、loss 有限；validation NDCG@10 选模；test 不参与。

## 6. 证据产品（T2-T6）

metric_checks.json / bootstrap_checks.json / evaluation_protocol.json / validation_metrics.json / baselines/B0、B1 产物 / run_records.md / acceptance.md / artifact_inventory.json / resource_summary.json，全部在 outputs/v2/evidence/phase5 与 outputs/v2/baselines；正式 test 统一 Phase 8。

## 7. 审计基础

- 环境锁 `environment_lock.json`=`{env_sha}`；代码审计 `code_audit.json`=`{code_sha}`（7 个既有源文件现盘 SHA-256，与本地克隆逐一比对）；输入清单 `input_inventory.json`=`{inv_sha}`。
- 上游失效传播：data/A 文件变更→B0/B1 全部失效；quality_features 变更→切片标签失效；catalog 变更→目录/负采样/评测全部失效。
"""
(EV / "implementation_plan.md").write_text(plan)
print("IMPLEMENTATION_PLAN written", flush=True)

# =================== 6. summary ===================
summary = {
    "schema": "phase5-t1-summary-1.0", "date": "2026-11-11",
    "head": env["git"]["head"], "tracked_diff_empty": env["git"]["tracked_diff_empty"],
    "porcelain_n": len(env["git"]["porcelain"]),
    "checks": chk, "identity": identity,
    "split_users": rec_users,
    "split_missing_target_rows": {sp: splits[sp]["checks"]["missing_target_rows"] for sp in splits},
    "evidence_sha256": {
        "environment_lock.json": env_sha, "code_audit.json": code_sha,
        "input_inventory.json": inv_sha, "implementation_plan.md": sha(EV / "implementation_plan.md"),
        "t1_checks.py": sha(EV / "t1_checks.py"),
    },
    "all_passed": len(FAILED) == 0, "failed": FAILED,
    "boundary": "no training started, no test recommendation metrics, upstream products untouched",
}
(EV / "t1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
print("T1_SUMMARY_SHA256:", sha(EV / "t1_summary.json"), flush=True)
print("T1_ALL_PASSED:", len(FAILED) == 0, "FAILED:", FAILED, flush=True)
print("T1_DONE", flush=True)
