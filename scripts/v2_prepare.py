#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2_prepare.py — MiniOneRec v2 Phase 1 数据与环境准备入口。

阶段:
  environment : 环境锁 (git/Python/CUDA/GPU/依赖包/硬件/模型来源)
  file_hashes : 输入快照 SHA-256 清单 (目录元数据、三份 CSV、协议文件)
  audit       : 数据审计、rows_manifest、catalog_clean (T2)
  rl          : seed42 无放回 10,000 RL 任务清单 (T3)
  views       : 训练 drop30 (seed1701) 与诊断三类视图 (seed2701) (T3)

按 phase1_dataset_prep.md §2.4 的接口运行全部阶段; --only 可限定阶段子集
(分阶段验收用, 不改变默认全量行为)。

约束: 源文件只读; 不虚构时间戳; 不从 validation/test 构造训练任务;
不生成本阶段以外的产物; 不输出任何凭据。
"""
import argparse
import ast
import csv
import hashlib
import random
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from importlib.metadata import version as pkg_version, PackageNotFoundError
from pathlib import Path

SCHEMA_VERSION = "v2-phase1.4"


def canonical_id_key(s):
    """规范 item ID 排序键：数值 ID 按数值序，非数值 ID 按字符串序回退（v2 §5.3）。"""
    if s.lstrip("-").isdigit():
        return (0, int(s), "")
    return (1, 0, s)
STAGES = ("environment", "file_hashes", "audit", "rl", "views")

KEY_PACKAGES = [
    "torch", "transformers", "trl", "accelerate", "bitsandbytes",
    "numpy", "pandas", "huggingface_hub", "safetensors", "tokenizers",
]


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def run_cmd(args, cwd=None):
    try:
        out = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=60)
        return out.returncode, out.stdout.strip(), out.stderr.strip()
    except Exception as e:  # noqa: BLE001
        return 127, "", str(e)


def git_state(repo_dir):
    """git 分支/HEAD/tracked 差异哈希/untracked 清单及内容哈希。"""
    state = {"repo_dir": str(repo_dir), "is_git_repo": False}
    rc, _, _ = run_cmd(["git", "rev-parse", "--git-dir"], cwd=repo_dir)
    if rc != 0:
        state["error"] = "not a git repository"
        return state
    state["is_git_repo"] = True
    _, branch, _ = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir)
    _, head, _ = run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_dir)
    rc_diff, diff_text, _ = run_cmd(["git", "diff", "HEAD"], cwd=repo_dir)
    tracked_dirty = bool(diff_text)
    state["branch"] = branch
    state["head_commit"] = head
    state["tracked_diff_sha256"] = hashlib.sha256(diff_text.encode()).hexdigest()
    state["tracked_diff_empty"] = not tracked_dirty
    _, status_text, _ = run_cmd(["git", "status", "--porcelain", "-uall"], cwd=repo_dir)
    untracked = []
    for line in status_text.splitlines():
        if not line.startswith("??"):
            continue
        rel = line[3:].strip()
        fp = repo_dir / rel
        if fp.is_file():
            untracked.append({
                "path": rel,
                "size_bytes": fp.stat().st_size,
                "sha256": sha256_file(fp),
            })
        elif fp.is_dir():
            for p in sorted(fp.rglob("*")):
                if p.is_file():
                    untracked.append({
                        "path": str(p.relative_to(repo_dir)),
                        "size_bytes": p.stat().st_size,
                        "sha256": sha256_file(p),
                    })
    state["untracked_files"] = untracked
    _, desc, _ = run_cmd(["git", "describe", "--tags", "--always"], cwd=repo_dir)
    state["describe"] = desc
    return state


def env_state(model_root, model_specs):
    """Python / CUDA / 驱动 / GPU / 依赖 / 硬件 / 模型来源。"""
    info = {}
    info["python"] = {
        "executable": sys.executable,
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
    }
    # 依赖版本（经 importlib.metadata, 不实际 import 包）
    pkgs = {}
    for name in KEY_PACKAGES:
        try:
            pkgs[name] = pkg_version(name)
        except PackageNotFoundError:
            pkgs[name] = None
    info["packages"] = pkgs
    # torch runtime / GPU（导入 torch 以记录真实计算运行时）
    torch_info = {"imported": False}
    try:
        import torch  # noqa: PLS400
        torch_info["imported"] = True
        torch_info["version"] = torch.__version__
        torch_info["version_cuda"] = torch.version.cuda
        torch_info["version_cudnn"] = getattr(torch.version, "cudnn", None)
        torch_info["cuda_available"] = torch.cuda.is_available()
        devices = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                devices.append({
                    "index": i,
                    "name": p.name,
                    "total_memory_bytes": p.total_memory,
                    "multi_processor_count": p.multi_processor_count,
                    "compute_capability": f"{p.major}.{p.minor}",
                    "uuid": getattr(p, "uuid", None) and str(p.uuid),
                })
        torch_info["devices"] = devices
    except Exception as e:  # noqa: BLE001
        torch_info["error"] = repr(e)
    info["torch"] = torch_info
    # nvidia-smi: 驱动 / MIG 拓扑（框架实测之外的硬件事实）
    rc, smi, err = run_cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,uuid,mig.mode.current",
                            "--format=csv,noheader"])
    info["nvidia_smi"] = {"available": rc == 0}
    if rc == 0:
        gpus = []
        for line in smi.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "name": parts[0], "driver_version": parts[1],
                    "memory_total": parts[2], "uuid": parts[3],
                    "mig_mode_current": parts[4],
                })
        info["nvidia_smi"]["gpus"] = gpus
    else:
        info["nvidia_smi"]["error"] = err
    # 硬件
    cpu_model = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    mem_total = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                mem_total = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    info["hardware"] = {
        "hostname": socket.gethostname(),
        "user": os.environ.get("USER") or os.environ.get("LOGNAME"),
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "cpu_count_logical": os.cpu_count(),
        "mem_total_proc": mem_total,
        "home": str(Path.home()),
    }
    # 模型来源锁定
    info["models"] = {}
    for spec in model_specs:
        entry = {"repo_id": spec.repo_id, "revision": spec.revision,
                 "local_root": str(spec.local_dir), "present": False}
        if spec.local_dir.is_dir():
            entry["present"] = True
            entry["files"] = {}
            for fname in spec.hash_files:
                fp = spec.local_dir / fname
                if fp.is_file():
                    entry["files"][fname] = {
                        "size_bytes": fp.stat().st_size,
                        "sha256": sha256_file(fp),
                    }
                else:
                    entry["files"][fname] = None
        info["models"][spec.repo_id] = entry
    return info


def collect_file_hashes(data_root, dataset, repo_dir):
    """输入快照: 目录元数据 + 三份 CSV + 协议文件。"""
    entries = []

    def add(rel_path, role, source):
        fp = repo_dir / rel_path
        if not fp.is_file():
            entries.append({"path": rel_path, "exists": False, "role": role})
            return
        entries.append({
            "path": rel_path,
            "absolute_path": str(fp.resolve()),
            "exists": True,
            "size_bytes": fp.stat().st_size,
            "sha256": sha256_file(fp),
            "role": role,
            "source": source,
        })

    item_json = f"data/Amazon/index/{dataset}.item.json"
    add(item_json, "catalog_metadata",
        "MiniOneRec 官方预处理商品元数据, 随远端仓库 data/Amazon 提供; 只读")
    for split in ("train", "valid", "test"):
        matches = sorted((data_root / split).glob(f"{dataset}_*.csv"))
        if len(matches) != 1:
            entries.append({
                "path": f"data/Amazon/{split}/{dataset}_*.csv",
                "exists": False, "role": f"{split}_csv",
                "error": f"expected exactly 1 CSV, found {len(matches)}",
            })
            continue
        rel = str(matches[0].relative_to(repo_dir))
        add(rel, f"{split}_csv",
            "MiniOneRec 官方预处理切分 CSV, 保留原始行序/切分; 只读")
    for guide in ("global_guide/phase1_dataset_prep.md",
                  "global_guide/topic_minionerec_v2.md",
                  "global_guide/experiment_setting.md"):
        add(guide, "protocol_document",
            "Phase 1 指南/v2 协议/实验主入口; 远端副本与本地协议文件 SHA-256 一致; 只读")
    return entries


def load_common_cleaner(repo_dir):
    """复用仓库既有公共清洗函数 rq/text2emb/utils.py:clean_text。

    utils.py 模块级 import openai 等本项目范围外的依赖，故按 AST 逐字提取
    clean_text 函数源码并在仅含 html/re 的命名空间中执行——函数体与仓库文件
    字节一致（非复制改写），返回 (函数, 函数源码 sha256) 以供追溯。"""
    src_path = repo_dir / "rq" / "text2emb" / "utils.py"
    src = src_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn_node = next(n for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == "clean_text")
    fn_src = ast.get_source_segment(src, fn_node)
    fn_sha = hashlib.sha256(fn_src.encode("utf-8")).hexdigest()
    ns = {"html": __import__("html"), "re": __import__("re")}
    exec(compile(fn_src, str(src_path), "exec"), ns)  # noqa: S102 源码取自锁定仓库自身
    return ns["clean_text"], fn_sha


def parse_id_list(raw):
    """解析 CSV 中字符串化列表字段（如 "[117, 118]"）；失败返回 (None, 错误)。"""
    try:
        v = ast.literal_eval(raw)
    except Exception as e:  # noqa: BLE001
        return None, repr(e)
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v], None
    return [str(v)], None


def percentile(sorted_vals, q):
    """最近秩分位数（确定性）：q∈[0,1]，返回第 ceil(q*n) 个（1-based）。"""
    n = len(sorted_vals)
    if n == 0:
        return None
    import math
    idx = max(0, math.ceil(q * n) - 1)
    return sorted_vals[min(idx, n - 1)]


def hist_buckets(lengths):
    """历史长度分桶直方图（确定性边界）。"""
    buckets = {"1": 0, "2-3": 0, "4-5": 0, "6-10": 0, "11-20": 0,
               "21-50": 0, "51-100": 0, "101-200": 0, ">200": 0}
    for n in lengths:
        if n <= 1:
            buckets["1"] += 1
        elif n <= 3:
            buckets["2-3"] += 1
        elif n <= 5:
            buckets["4-5"] += 1
        elif n <= 10:
            buckets["6-10"] += 1
        elif n <= 20:
            buckets["11-20"] += 1
        elif n <= 50:
            buckets["21-50"] += 1
        elif n <= 100:
            buckets["51-100"] += 1
        elif n <= 200:
            buckets["101-200"] += 1
        else:
            buckets[">200"] += 1
    return buckets


def stage_audit(repo_dir, data_root, dataset, out_root, t0):
    """T2：数据审计、稳定 row_id、公共清洗目录。"""
    clean_text, cleaner_sha = load_common_cleaner(repo_dir)

    # ---------- 目录清洗 ----------
    item_path = data_root / "index" / f"{dataset}.item.json"
    with open(item_path, encoding="utf-8") as f:
        item2feature = json.load(f)
    catalog_ids = list(item2feature.keys())
    catalog_set = set(catalog_ids)

    type_anomalies = []
    missing_before = 0
    cleaned_empty_desc = 0
    empty_title_clean = 0
    id_numeric = all(s.lstrip("-").isdigit() for s in catalog_ids)
    id_ints = [int(s) for s in catalog_ids] if id_numeric else []
    missing_desc_ids = set()

    catalog_path = out_root / "manifests" / "catalog_clean.jsonl"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    with open(catalog_path, "w", encoding="utf-8", newline="\n") as out:
        for iid in catalog_ids:
            rec = item2feature[iid]
            title_raw = rec.get("title", "")
            desc_raw = rec.get("description", "")
            cat_raw = rec.get("categories", "")
            for fname, v in (("title", title_raw), ("description", desc_raw), ("categories", cat_raw)):
                if not isinstance(v, str):
                    type_anomalies.append({"item_id": iid, "field": fname, "type": type(v).__name__})
            title_raw = title_raw if isinstance(title_raw, str) else str(title_raw)
            desc_raw = desc_raw if isinstance(desc_raw, str) else str(desc_raw)
            cat_raw = cat_raw if isinstance(cat_raw, str) else str(cat_raw)
            # 清洗前固定缺失标签（v2 §3.2：原始描述去首尾空白后为空）
            desc_missing = desc_raw.strip() == ""
            if desc_missing:
                missing_before += 1
                missing_desc_ids.add(iid)
            title_clean = clean_text(title_raw).strip()
            desc_clean = "" if desc_missing else clean_text(desc_raw).strip()
            if desc_clean == "":
                cleaned_empty_desc += 1
            if title_clean == "":
                empty_title_clean += 1
            out.write(json.dumps({
                "item_id": iid,
                "original_description_missing": desc_missing,
                "title_raw": title_raw,
                "description_raw": desc_raw,
                "title_clean": title_clean,
                "description_clean": desc_clean,
                "categories_raw": cat_raw,
            }, ensure_ascii=False) + "\n")

    # ---------- 行清单 ----------
    rows_path = out_root / "manifests" / "rows_manifest.jsonl"
    split_stats = {}
    field_count_errors = []
    parse_errors = []
    all_row_ids = []
    rows_written = 0

    with open(rows_path, "w", encoding="utf-8", newline="\n") as out:
        for split in ("train", "valid", "test"):
            matches = sorted((data_root / split).glob(f"{dataset}_*.csv"))
            if len(matches) != 1:
                split_stats[split] = {"error": f"expected 1 CSV, found {len(matches)}"}
                continue
            csv_path = matches[0]
            with open(csv_path, encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                raw_rows = list(reader)
            users = set()
            lengths = []
            missing_target_rows = 0
            targets_not_in_catalog = set()
            history_not_in_catalog = set()
            seen_raw = {}
            dup_rows = 0
            for i, fields in enumerate(raw_rows):
                if len(fields) != 7:
                    field_count_errors.append({"split": split, "row_index": i,
                                               "n_fields": len(fields)})
                user_id = fields[0] if len(fields) > 0 else ""
                hist_raw = fields[3] if len(fields) > 3 else ""
                hist_ids, perr = (parse_id_list(hist_raw) if hist_raw else ([], None)) \
                    if len(fields) > 3 else ([], None)
                if perr:
                    parse_errors.append({"split": split, "row_index": i, "error": perr})
                target_id = fields[4] if len(fields) > 4 else ""
                hist_len = len(hist_ids) if hist_ids is not None else None
                if hist_ids:
                    lengths.append(len(hist_ids))
                    for h in hist_ids:
                        if h not in catalog_set:
                            history_not_in_catalog.add(h)
                users.add(user_id)
                if target_id not in catalog_set:
                    targets_not_in_catalog.add(target_id)
                if target_id in missing_desc_ids:
                    missing_target_rows += 1
                key = tuple(fields)
                if key in seen_raw:
                    dup_rows += 1
                else:
                    seen_raw[key] = i
                row_id = f"{split}:{i}"
                all_row_ids.append(row_id)
                out.write(json.dumps({
                    "row_id": row_id,
                    "split": split,
                    "row_index": i,
                    "user_id": user_id,
                    "history_item_id_raw": hist_raw,
                    "history_ids": hist_ids,
                    "history_len": hist_len,
                    "history_item_title_raw": fields[1] if len(fields) > 1 else "",
                    "item_title_raw": fields[2] if len(fields) > 2 else "",
                    "history_item_sid_raw": fields[5] if len(fields) > 5 else "",
                    "item_sid_raw": fields[6] if len(fields) > 6 else "",
                    "item_id": target_id,
                    "target_in_catalog": target_id in catalog_set,
                    "history_all_in_catalog": (hist_ids is not None and
                                               all(h in catalog_set for h in hist_ids)),
                }, ensure_ascii=False) + "\n")
                rows_written += 1
            sorted_lengths = sorted(lengths)
            split_stats[split] = {
                "csv_path": str(csv_path.relative_to(repo_dir)),
                "header": header,
                "n_rows": len(raw_rows),
                "n_users": len(users),
                "history_len": {
                    "n_with_parse": len(sorted_lengths),
                    "min": sorted_lengths[0] if sorted_lengths else None,
                    "max": sorted_lengths[-1] if sorted_lengths else None,
                    "mean": round(sum(sorted_lengths) / len(sorted_lengths), 6) if sorted_lengths else None,
                    "median": percentile(sorted_lengths, 0.5),
                    "p90": percentile(sorted_lengths, 0.9),
                    "p95": percentile(sorted_lengths, 0.95),
                    "p99": percentile(sorted_lengths, 0.99),
                    "buckets": hist_buckets(sorted_lengths),
                },
                "target_missing_desc_rows": missing_target_rows,
                "targets_not_in_catalog_count": len(targets_not_in_catalog),
                "targets_not_in_catalog_sample": sorted(targets_not_in_catalog)[:10],
                "history_items_not_in_catalog_count": len(history_not_in_catalog),
                "history_items_not_in_catalog_sample": sorted(history_not_in_catalog)[:10],
                "duplicate_raw_rows": dup_rows,
            }

    # ---------- 身份自检：写后重读 CSV 逐行比对 ----------
    identity_mismatches = []
    manifest_by_id = {}
    with open(rows_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            manifest_by_id[rec["row_id"]] = rec
    for split in ("train", "valid", "test"):
        matches = sorted((data_root / split).glob(f"{dataset}_*.csv"))
        if len(matches) != 1:
            continue
        with open(matches[0], encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for i, fields in enumerate(reader):
                rec = manifest_by_id.get(f"{split}:{i}")
                if rec is None:
                    identity_mismatches.append({"row_id": f"{split}:{i}", "issue": "missing_in_manifest"})
                    continue
                raw_map = {
                    "user_id": rec["user_id"], "history_item_id_raw": rec["history_item_id_raw"],
                    "history_item_title_raw": rec["history_item_title_raw"],
                    "item_title_raw": rec["item_title_raw"],
                    "history_item_sid_raw": rec["history_item_sid_raw"],
                    "item_sid_raw": rec["item_sid_raw"], "item_id": rec["item_id"],
                }
                expect = dict(zip(["user_id", "history_item_title_raw", "item_title_raw",
                                   "history_item_id_raw", "item_id",
                                   "history_item_sid_raw", "item_sid_raw"], fields))
                if raw_map != expect:
                    identity_mismatches.append({"row_id": rec["row_id"], "issue": "field_mismatch"})
                if rec["row_index"] != i:
                    identity_mismatches.append({"row_id": rec["row_id"], "issue": "row_index_drift"})

    # ---------- 源文件哈希与 T1 冻结清单一致性 ----------
    frozen = json.load(open(out_root / "manifests" / "file_hashes.json", encoding="utf-8"))
    frozen_map = {e["path"]: e.get("sha256") for e in frozen["deterministic"]["files"] if e.get("exists")}
    source_hash_check = {}
    for rel, frozen_sha in frozen_map.items():
        fp = repo_dir / rel
        live = sha256_file(fp) if fp.is_file() else None
        source_hash_check[rel] = {"frozen": frozen_sha, "live": live,
                                  "match": live == frozen_sha}

    measured = {
        "catalog_items": len(catalog_ids),
        "catalog_original_description_missing": missing_before,
        "catalog_categories_all_empty": all(
            not str(item2feature[i].get("categories", "")).strip() for i in catalog_ids),
        "train_rows": split_stats.get("train", {}).get("n_rows"),
        "valid_rows": split_stats.get("valid", {}).get("n_rows"),
        "test_rows": split_stats.get("test", {}).get("n_rows"),
        "test_users": split_stats.get("test", {}).get("n_users"),
        "test_target_missing_desc_rows": split_stats.get("test", {}).get("target_missing_desc_rows"),
    }
    expected = {
        "catalog_items": 3686,
        "catalog_original_description_missing": 229,
        "train_rows": 36259,
        "valid_rows": 4532,
        "test_rows": 4533,
        "test_users": 1987,
        "test_target_missing_desc_rows": 447,
    }
    reference_match = {k: (measured.get(k) == v) for k, v in expected.items()}

    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "audit",
        "deterministic": {
            "dataset": dataset,
            "data_root": str(data_root.relative_to(repo_dir)),
            "row_id_format": "{split}:{row_index}；row_index 为表头之后的数据行从零编号，行序即原 CSV 行序",
            "csv_schema": {"field_count_errors": field_count_errors},
            "catalog": {
                "n_items": len(catalog_ids),
                "original_description_missing": missing_before,
                "cleaned_description_empty": cleaned_empty_desc,
                "cleaned_title_empty": empty_title_clean,
                "field_type_anomalies": type_anomalies,
                "categories_constructed": False,
                "id_all_numeric": id_numeric,
                "id_min": min(id_ints) if id_ints else None,
                "id_max": max(id_ints) if id_ints else None,
                "id_contiguous_0_to_n": id_numeric and id_ints == list(range(len(id_ints))),
                "cleaner_source": "rq/text2emb/utils.py:clean_text (AST 逐字提取, 函数体字节一致)",
                "cleaner_source_sha256": cleaner_sha,
            },
            "splits": split_stats,
            "identity": {
                "total_rows": rows_written,
                "row_id_unique": len(set(all_row_ids)) == rows_written,
                "row_id_count_unique": len(set(all_row_ids)),
                "identity_check_mismatches": identity_mismatches,
                "identity_check_passed": len(identity_mismatches) == 0,
            },
            "reference_comparison": {"expected": expected, "measured": measured,
                                     "match": reference_match},
            "source_hash_check": source_hash_check,
            "anomalies": {
                "field_count_errors_n": len(field_count_errors),
                "history_parse_errors": parse_errors,
                "history_parse_errors_n": len(parse_errors),
            },
        },
        "run": {
            "command": " ".join(sys.argv),
            "cwd": os.getcwd(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": round(time.time() - t0, 3),
        },
    }
    audit_path = out_root / "manifests" / "data_audit.json"
    write_json(audit_path, payload)
    print(f"[v2_prepare] wrote {audit_path}")
    print(f"[v2_prepare] rows={rows_written} catalog={len(catalog_ids)} "
          f"missing_desc={missing_before} ref_match={all(reference_match.values())}")


def load_frozen_manifests(out_root):
    """读取 T2 冻结清单（catalog_clean / rows_manifest）。"""
    catalog = []
    with open(out_root / "manifests" / "catalog_clean.jsonl", encoding="utf-8") as f:
        for line in f:
            catalog.append(json.loads(line))
    rows = []
    with open(out_root / "manifests" / "rows_manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return catalog, rows


def stage_rl(repo_dir, out_root, args, t0):
    """T3-A：seed 42 无放回 10,000 RL 任务清单（不含任何组 SID）。"""
    catalog, rows = load_frozen_manifests(out_root)
    seq_target, title_target, text_target = args.rl_counts
    train_rows = [r for r in rows if r["split"] == "train"]  # rows_manifest 文件序 = row_index 序

    rng = random.Random(args.seed)
    checks = {}
    # 任务 1：训练序列→SID（来源=train 行 row_id）
    seq_sample = rng.sample(train_rows, seq_target)
    seq_ids = [r["row_id"] for r in seq_sample]
    # 任务 2：目录标题→SID（同标题按最小数值 item ID 保留一个）
    by_title = {}
    for c in catalog:
        t = c["title_clean"]
        if t:
            by_title.setdefault(t, []).append(c["item_id"])
    conflict_titles = {t: ids for t, ids in by_title.items() if len(ids) > 1}
    # 相同标题保留规范序最小的 item ID（不依赖文件序；数值 ID 按数值比较）
    title_candidates = [(t, min(ids, key=canonical_id_key)) for t, ids in by_title.items()]
    title_candidates.sort(key=lambda x: canonical_id_key(x[1]))
    if len(title_candidates) < title_target:
        raise SystemExit(
            f"[v2_prepare] BLOCK: 唯一清洗标题 {len(title_candidates)} < 需求 {title_target}，不补重复样本")
    title_sample = [title_candidates[i] for i in rng.sample(range(len(title_candidates)), title_target)]
    # 任务 3：训练文本历史→SID（与任务 1 独立抽样，允许跨任务共享行）
    text_sample = rng.sample(train_rows, text_target)
    text_ids = [r["row_id"] for r in text_sample]

    rl_path = out_root / "manifests" / "rl_tasks_10000.jsonl"
    with open(rl_path, "w", encoding="utf-8", newline="\n") as out:
        for i, r in enumerate(seq_sample):
            out.write(json.dumps({
                "task_id": f"seq2sid:{i:05d}", "task_type": "seq2sid",
                "source_id": r["row_id"], "source_split": "train",
                "user_id": r["user_id"], "history_item_id_raw": r["history_item_id_raw"],
                "item_id": r["item_id"],
            }, ensure_ascii=False) + "\n")
        for i, (t, iid) in enumerate(title_sample):
            out.write(json.dumps({
                "task_id": f"title2sid:{i:05d}", "task_type": "title2sid",
                "source_id": iid, "source_split": "catalog",
                "title_clean": t, "item_id": iid,
            }, ensure_ascii=False) + "\n")
        for i, r in enumerate(text_sample):
            out.write(json.dumps({
                "task_id": f"texthist2sid:{i:05d}", "task_type": "texthist2sid",
                "source_id": r["row_id"], "source_split": "train",
                "user_id": r["user_id"], "history_item_id_raw": r["history_item_id_raw"],
                "item_id": r["item_id"],
            }, ensure_ascii=False) + "\n")

    def conflict_stats(sample):
        groups = {}
        for r in sample:
            groups.setdefault(r["history_item_id_raw"], set()).add(r["item_id"])
        multi = {h: sorted(ts) for h, ts in groups.items() if len(ts) > 1}
        return {"distinct_prompts": len(groups), "prompt_conflict_groups": len(multi),
                "conflict_prompt_sample": list(multi.items())[:5],
                "rows_in_conflict": sum(len(multi[h]) for h in multi)}

    shared = len(set(seq_ids) & set(text_ids))
    summary = {
        "stage": "rl",
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "counts": {"seq2sid": len(seq_sample), "title2sid": len(title_sample),
                   "texthist2sid": len(text_sample)},
        "uniqueness": {"seq2sid_row_ids_unique": len(set(seq_ids)) == len(seq_ids),
                       "texthist2sid_row_ids_unique": len(set(text_ids)) == len(text_ids),
                       "title2sid_item_ids_unique": len({i for _, i in title_sample}) == len(title_sample)},
        "title_conflict_rule": {
            "titles_with_multiple_items": len(conflict_titles),
            "rule": "相同清洗标题保留最小数值 item ID（catalog 文件序=数值序首位）",
            "unique_title_candidates": len(title_candidates),
            "sufficient": len(title_candidates) >= title_target,
        },
        "prompt_conflicts": {"seq2sid": conflict_stats(seq_sample),
                             "texthist2sid": conflict_stats(text_sample)},
        "cross_task_row_reuse": {"seq2sid_and_texthist2sid_shared_rows": shared},
        "source_isolation": {"splits_used": sorted({r["split"] for r in seq_sample} |
                                                   {r["split"] for r in text_sample} |
                                                   {"catalog"}),
                             "valid_test_rows_used": 0},
        "output": str(rl_path.relative_to(repo_dir)),
    }
    ev_dir = out_root / "evidence" / "phase1"
    ev_dir.mkdir(parents=True, exist_ok=True)
    with open(ev_dir / "rl_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[v2_prepare] wrote {rl_path}")
    print(f"[v2_prepare] rl counts={summary['counts']} unique_titles={len(title_candidates)} "
          f"title_conflicts={len(conflict_titles)} shared_rows={shared}")
    return summary


def drop_words(tokens, p, rng):
    """按 k=min(n-1,max(1,floor(p*n))) 无放回选删除位置；n<2 返回 (None, None)。"""
    n = len(tokens)
    if n < 2:
        return None, None
    k = min(n - 1, max(1, int(p * n)))  # int() 即 floor（p>0, n>0）
    positions = sorted(rng.sample(range(n), k))
    kept = [t for i, t in enumerate(tokens) if i not in set(positions)]
    return k, {"positions": positions, "perturbed": " ".join(kept)}


def stage_views(repo_dir, out_root, args, t0):
    """T3-B/C：训练 drop30（seed 1701）与诊断三类视图（seed 2701）。"""
    catalog, _ = load_frozen_manifests(out_root)
    catalog = sorted(catalog, key=lambda c: canonical_id_key(c["item_id"]))

    def process(view_seed, view_type, p=None, only_eligible_draw=True):
        rng = random.Random(view_seed)
        recs = []
        reason_counts = {}
        draws = 0
        for c in catalog:
            iid = c["item_id"]
            desc = c["description_clean"]
            tokens = desc.split() if desc else []
            n = len(tokens)
            rec = {"item_id": iid, "view_type": view_type, "view_seed": view_seed,
                   "title_clean": c["title_clean"], "n_tokens": n,
                   "original_text": desc,
                   "original_text_sha256": hashlib.sha256(desc.encode("utf-8")).hexdigest(),
                   "k_deleted": 0, "deleted_positions": [],
                   "perturbed_text": "", "perturbed_text_sha256": "",
                   "valid_aug": False, "invalid_reason": None}
            if c["original_description_missing"]:
                rec["invalid_reason"] = "raw_missing"
            elif desc == "":
                rec["invalid_reason"] = "cleaned_empty"
            elif p is not None and n < 2:
                rec["invalid_reason"] = "insufficient_words"
            else:
                if p is not None:
                    k, res = drop_words(tokens, p, rng)
                    draws += 1 if k is not None else 0
                    rec["k_deleted"] = k
                    rec["deleted_positions"] = res["positions"]
                    rec["perturbed_text"] = res["perturbed"]
                else:  # dropall
                    rec["k_deleted"] = n
                    rec["deleted_positions"] = list(range(n))
                    rec["perturbed_text"] = ""
                if rec["perturbed_text"] == desc:
                    rec["invalid_reason"] = "view_unchanged"
                else:
                    rec["valid_aug"] = True
            if not rec["valid_aug"]:
                reason_counts[rec["invalid_reason"]] = reason_counts.get(rec["invalid_reason"], 0) + 1
            rec["perturbed_text_sha256"] = hashlib.sha256(
                rec["perturbed_text"].encode("utf-8")).hexdigest()
            recs.append(rec)
        return recs, reason_counts, draws

    train_recs, train_reasons, train_draws = process(args.train_view_seed, "drop30", p=0.3)
    d30, r30, dw30 = process(args.diagnostic_view_seed, "drop30", p=0.3)
    d60, r60, dw60 = process(args.diagnostic_view_seed, "drop60", p=0.6)
    dall, rall, _ = process(args.diagnostic_view_seed, "dropall", p=None)

    tv_path = out_root / "manifests" / "train_views.jsonl"
    with open(tv_path, "w", encoding="utf-8", newline="\n") as out:
        for rec in train_recs:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    dv_path = out_root / "manifests" / "diagnostic_views.jsonl"
    with open(dv_path, "w", encoding="utf-8", newline="\n") as out:
        for rec in (*d30, *d60, *dall):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def order_check(recs):
        bad = 0
        for rec in recs:
            if not rec["valid_aug"]:
                continue
            toks = rec["original_text"].split()
            kept = [t for i, t in enumerate(toks) if i not in set(rec["deleted_positions"])]
            if " ".join(kept) != rec["perturbed_text"]:
                bad += 1
            if min(rec["deleted_positions"], default=0) < 0 or \
               max(rec["deleted_positions"], default=-1) >= rec["n_tokens"]:
                bad += 1
        return bad

    def k_check(recs, p):
        bad = 0
        for rec in recs:
            if not rec["valid_aug"]:
                continue
            n = rec["n_tokens"]
            want = min(n - 1, max(1, int(p * n)))
            if rec["k_deleted"] != want:
                bad += 1
        return bad

    diag_all = [*d30, *d60, *dall]
    summary = {
        "stage": "views",
        "schema_version": SCHEMA_VERSION,
        "train_view": {"seed": args.train_view_seed, "view_type": "drop30",
                       "records": len(train_recs),
                       "valid_aug": sum(r["valid_aug"] for r in train_recs),
                       "invalid_reasons": train_reasons, "rng_draws": train_draws,
                       "k_formula_violations": k_check(train_recs, 0.3),
                       "order_violations": order_check(train_recs)},
        "diagnostic_views": {
            "seed": args.diagnostic_view_seed,
            "per_type": {
                vt: {
                    "records": sum(1 for r in diag_all if r["view_type"] == vt),
                    "valid_aug": sum(r["valid_aug"] for r in diag_all if r["view_type"] == vt),
                } for vt in ("drop30", "drop60", "dropall")
            },
            "invalid_reasons": {"drop30": r30, "drop60": r60, "dropall": rall},
            "k_formula_violations": {"drop30": k_check(d30, 0.3), "drop60": k_check(d60, 0.6)},
            "order_violations": order_check(diag_all),
            "items_covered_per_type": len(catalog),
        },
        "identity_unchanged": {"preserved": None},  # 下方逐条核验后覆盖
        "note": "invalid 原因分类: raw_missing(原始缺失229)/cleaned_empty(清洗置空98之外含全部清洗后为空者)/insufficient_words(n<2)/view_unchanged(理论不应出现)",
        "outputs": [str(tv_path.relative_to(repo_dir)), str(dv_path.relative_to(repo_dir))],
    }
    # 身份不变核验（逐条：item_id 与 title_clean 与目录一致）
    cat_map = {c["item_id"]: c for c in catalog}
    ident_bad = 0
    for rec in (*train_recs, *diag_all):
        c = cat_map.get(rec["item_id"])
        if c is None or rec["title_clean"] != c["title_clean"]:
            ident_bad += 1
    summary["identity_unchanged"] = {"violations": ident_bad,
                                     "preserved": ident_bad == 0}
    ev_dir = out_root / "evidence" / "phase1"
    ev_dir.mkdir(parents=True, exist_ok=True)
    with open(ev_dir / "views_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[v2_prepare] wrote {tv_path} and {dv_path}")
    print(f"[v2_prepare] views train valid={summary['train_view']['valid_aug']}/{len(train_recs)} "
          f"reasons={train_reasons}; diag valid per type={summary['diagnostic_views']['per_type']}")
    return summary


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


def not_implemented(stage):
    print(f"[v2_prepare] stage '{stage}' 属于后续子任务, 本版本未实现, 跳过。", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description="MiniOneRec v2 Phase 1 preparation (environment / hashes / audit / rl / views)")
    ap.add_argument("--data-root", default="data/Amazon")
    ap.add_argument("--dataset", default="Industrial_and_Scientific")
    ap.add_argument("--rl-counts", nargs=3, type=int, default=[8000, 1000, 1000],
                    metavar=("SEQ", "TITLE", "TEXTHIST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-view-seed", type=int, default=1701)
    ap.add_argument("--diagnostic-view-seed", type=int, default=2701)
    ap.add_argument("--output-root", default="outputs/v2")
    ap.add_argument("--model-root", default="/hpc2hdd/home/swu073/sikun/dsaa5020/models",
                    help="revision 锁定下载的模型根目录")
    ap.add_argument("--model-revision-base", default="da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
                    help="Qwen/Qwen3-0.6B-Base 冻结 revision (HF commit hash)")
    ap.add_argument("--model-revision-embedding", default="5cf2132abc99cad020ac570b19d031efec650f2b",
                    help="Qwen/Qwen3-Embedding-4B 冻结 revision (HF commit hash)")
    ap.add_argument("--only", default="all",
                    help="逗号分隔阶段子集: environment,file_hashes,audit,rl,views 或 all")
    args = ap.parse_args()

    t0 = time.time()
    repo_dir = Path(__file__).resolve().parent.parent
    data_root = (repo_dir / args.data_root).resolve()
    out_root = (repo_dir / args.output_root).resolve()
    only = STAGES if args.only == "all" else tuple(
        s.strip() for s in args.only.split(",") if s.strip())
    bad = [s for s in only if s not in STAGES]
    if bad:
        ap.error(f"unknown stage(s): {bad}; valid: {STAGES} or all")

    print(f"[v2_prepare] repo={repo_dir}")
    print(f"[v2_prepare] stages={only} schema={SCHEMA_VERSION}")

    if "environment" in only:
        from collections import namedtuple
        spec = namedtuple("ModelSpec", "repo_id revision local_dir hash_files")
        specs = [
            spec("Qwen/Qwen3-0.6B-Base", args.model_revision_base,
                 Path(args.model_root) / "Qwen3-0.6B-Base",
                 ["config.json", "model.safetensors", "generation_config.json"]),
            spec("Qwen/Qwen3-Embedding-4B", args.model_revision_embedding,
                 Path(args.model_root) / "Qwen3-Embedding-4B",
                 ["config.json", "model-00001-of-00002.safetensors",
                  "model-00002-of-00002.safetensors", "model.safetensors.index.json"]),
        ]
        env = env_state(Path(args.model_root), specs)
        env["git"] = git_state(repo_dir)
        # pip freeze 快照 (依赖证据, 与 lock 分离存放)
        pip_freeze_path = out_root / "environment" / "pip_freeze.txt"
        pip_freeze_path.parent.mkdir(parents=True, exist_ok=True)
        rc_fr, freeze_text, _ = run_cmd([sys.executable, "-m", "pip", "freeze"])
        pip_freeze_path.write_text(freeze_text + "\n", encoding="utf-8")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "stage": "environment",
            "deterministic": env,
            "run": {
                "command": " ".join(sys.argv),
                "cwd": os.getcwd(),
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "elapsed_seconds": round(time.time() - t0, 3),
                "pip_freeze_path": str(pip_freeze_path.relative_to(repo_dir)),
                "note": "run 小节含时间戳等易变信息, 与确定性内容分离",
            },
        }
        lock_path = out_root / "environment" / "environment_lock.json"
        write_json(lock_path, payload)
        print(f"[v2_prepare] wrote {lock_path}")

    if "file_hashes" in only:
        entries = collect_file_hashes(data_root, args.dataset, repo_dir)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "stage": "file_hashes",
            "deterministic": {
                "dataset": args.dataset,
                "data_root": args.data_root,
                "files": entries,
                "code_state": git_state(repo_dir),
                "config": {
                    "seed": args.seed,
                    "train_view_seed": args.train_view_seed,
                    "diagnostic_view_seed": args.diagnostic_view_seed,
                    "rl_counts": args.rl_counts,
                },
            },
            "run": {
                "command": " ".join(sys.argv),
                "cwd": os.getcwd(),
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "elapsed_seconds": round(time.time() - t0, 3),
            },
        }
        fh_path = out_root / "manifests" / "file_hashes.json"
        write_json(fh_path, payload)
        print(f"[v2_prepare] wrote {fh_path}")

    if "audit" in only:
        stage_audit(repo_dir, data_root, args.dataset, out_root, t0)

    if "rl" in only:
        stage_rl(repo_dir, out_root, args, t0)

    if "views" in only:
        stage_views(repo_dir, out_root, args, t0)

    print(f"[v2_prepare] DONE_OK elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
