#!/usr/bin/env python
"""Phase4 T2: environment freeze (versions, GPU, venv reuse, base tokenizer static check).
SCHEMA: phase4-t2_environment_lock-1.0"""
import os, sys, json, hashlib, subprocess
os.environ.setdefault("OMP_NUM_THREADS", "8")
from pathlib import Path
REPO = Path("/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec")
EV = REPO / "outputs/v2/evidence/phase4"
EV.mkdir(parents=True, exist_ok=True)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

import torch, numpy, sklearn, scipy, transformers, tqdm

info = {"stage": "phase4_t2_environment_lock", "schema": "phase4-t2_environment_lock-1.0",
        "date": "2026-11-11", "host": "swu073-a800-10g", "container": "34fb2691c7dd"}
info["python"] = {"version": sys.version.split()[0], "executable": str(sys.executable)}
info["venv_reuse"] = {
    "declaration": "复用已冻结 Phase 3 环境 .venv-phase3，无任何安装/升级操作",
    "executable_is_venv_phase3": str(Path(sys.executable).resolve()) == str((REPO / ".venv-phase3/bin/python").resolve()),
    "pip_freeze_phase3_sha256": sha(REPO / "outputs/v2/evidence/phase3/pip_freeze_phase3.txt"),
}
info["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda,
                 "cudnn": torch.backends.cudnn.version(),
                 "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32)}
info["libs"] = {"numpy": numpy.__version__, "scikit-learn": sklearn.__version__,
                "scipy": scipy.__version__, "transformers": transformers.__version__,
                "tqdm": tqdm.__version__}
g = {"cuda_available": bool(torch.cuda.is_available()), "device_count": int(torch.cuda.device_count())}
if g["cuda_available"]:
    p = torch.cuda.get_device_properties(0)
    g["name"] = p.name
    g["total_memory_bytes"] = int(p.total_memory)
    g["multi_processor_count"] = int(p.multi_processor_count)
    g["compute_capability"] = f"{p.major}.{p.minor}"
    try:
        g["uuid"] = str(p.uuid)
    except Exception:
        g["uuid"] = None
    g["note"] = "框架实测为准（MIG 切片）"
for label, args in (("nvidia_smi_L", ["nvidia-smi", "-L"]),
                    ("nvidia_smi_query", ["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"])):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=10)
        g[label] = (r.stdout.strip() or r.stderr.strip())[:300]
    except Exception as e:
        g[label] = f"unavailable: {type(e).__name__}"
info["gpu"] = g

# git state (T2 工作: 核对新 commit、现有差异)
def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO).stdout.strip()
info["git_state"] = {
    "head": run(["git", "rev-parse", "HEAD"]),
    "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    "porcelain": run(["git", "status", "--porcelain"]),
    "diff_stat": run(["git", "diff", "--stat"]) or "(empty)",
    "diff_cached_stat": run(["git", "diff", "--cached", "--stat"]) or "(empty)",
}

# base tokenizer static check（不下载、不加载 LLM）
TOK = Path("/hpc2hdd/home/swu073/sikun/dsaa5020/models/Qwen3-0.6B-Base")
tok = {"repo_id": "Qwen/Qwen3-0.6B-Base",
       "revision_pinned": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
       "revision_source": "boss 记录『已接受上游』；与 Phase 1 environment_lock 交叉核对（见 phase1_envlock_mentions）",
       "local_path": str(TOK),
       "mode": "static file check only；无下载、无 LLM 权重加载、无 tokenizer 运行时加载（运行时验证留 T4）"}
tok["files_sha256"] = {}
for fn in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
           "config.json", "generation_config.json", "README.md"]:
    fp = TOK / fn
    tok["files_sha256"][fn] = sha(fp) if fp.exists() else None
tok["all_present"] = all(tok["files_sha256"][f] is not None for f in
                         ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json"])
cfg = json.load(open(TOK / "config.json"))
tok["config_arch"] = {k: cfg.get(k) for k in ("architectures", "model_type", "vocab_size",
                                              "hidden_size", "num_hidden_layers", "max_position_embeddings")}
tj = json.load(open(TOK / "tokenizer.json"))
tok["tokenizer_json_valid"] = bool(tj.get("model"))
tok["tokenizer_json_model_type"] = (tj.get("model") or {}).get("type")
try:
    tok["tokenizer_json_vocab_size"] = len(tj.get("model", {}).get("vocab", {}))
except Exception:
    tok["tokenizer_json_vocab_size"] = None

def find_mentions(o, needles=("0.6B", "da87bfb", "Qwen3-0.6B")):
    hits = []
    def rec(x, path):
        if len(hits) >= 8:
            return
        if isinstance(x, dict):
            for k, v in x.items():
                rec(v, path + "/" + str(k))
        elif isinstance(x, list):
            for i, v in enumerate(x):
                rec(v, f"{path}[{i}]")
        elif isinstance(x, str) and any(n in x for n in needles):
            hits.append({"path": path, "value": x[:200]})
    rec(o, "")
    return hits

try:
    p1 = json.load(open(REPO / "outputs/v2/environment/environment_lock.json"))
    tok["phase1_envlock_mentions"] = find_mentions(p1)
except Exception as e:
    tok["phase1_envlock_mentions"] = f"read_error: {type(e).__name__}"
info["base_tokenizer"] = tok

# pip freeze snapshot（同一 venv、同一 pip freeze 口径）+ 与 phase3 freeze 逐行比对
r = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
freeze_lines = sorted(l for l in r.stdout.splitlines() if l and not l.startswith("-"))
fr = EV / "pip_freeze_phase4.txt"
fr.write_text("\n".join(freeze_lines) + "\n")
p3_lines = set(l for l in (REPO / "outputs/v2/evidence/phase3/pip_freeze_phase3.txt").read_text().splitlines() if l and not l.startswith("-"))
cur = set(freeze_lines)
info["pip_freeze_phase4"] = {"path": "outputs/v2/evidence/phase4/pip_freeze_phase4.txt",
                             "n_packages": len(freeze_lines), "sha256": sha(fr)}
info["env_change_boundary"] = {
    "identical_to_phase3_freeze": p3_lines == cur,
    "only_in_phase3": sorted(p3_lines - cur),
    "only_in_phase4": sorted(cur - p3_lines),
    "statement": "零包差异 → Phase 3 训练环境对 Phase 4 推理完全有效；任何未来差异即触发相应失效复核",
}

out = EV / "environment_lock.json"
out.write_text(json.dumps(info, ensure_ascii=False, indent=1))
print("GIT_HEAD:", info["git_state"]["head"])
print("SUMMARY:", json.dumps({"venv_ok": info["venv_reuse"]["executable_is_venv_phase3"],
                              "tok_all_present": tok["all_present"],
                              "tokenizer_json_valid": tok["tokenizer_json_valid"],
                              "freeze_identical": info["env_change_boundary"]["identical_to_phase3_freeze"],
                              "git_diff_empty": info["git_state"]["diff_stat"] == "(empty)"}, ensure_ascii=False))
print("ENVLOCK_SHA256:", sha(out))
print("T2_ENVLOCK_DONE")
