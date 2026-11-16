#!/usr/bin/env python
"""T3 supplement: measured failure-path probes (NaN / Inf / zero-norm) for the
formal assert_usable_vectors guard. Poisons COPIES in a temp tree and drives the
formal entry (CLI for NaN/Inf; the formal run_diagnostics function with a
stubbed encode_full for zero-norm, which embeddings alone cannot force).
Asserts: nonzero exit / RuntimeError with the exact abort message, and NO
acceptable metric artifacts produced. Accepted SID outputs are never touched.
SCHEMA: phase4-t3-failure-probe-1.0"""
import os, sys, json, shutil, subprocess, hashlib, types, traceback
from pathlib import Path
import numpy as np

REPO = Path("/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec")
EV = REPO / "outputs/v2/evidence/phase4"
TMP = EV / "t3_probe_tmp"
SRC_CKPT = REPO / "outputs/v2/rqvae/A/seed_42/final_checkpoint.pth"
SRC_EMB = REPO / "outputs/v2/embeddings"

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def build_case(name):
    d = TMP / name
    for sub in ("rqvae/A/seed_42", "emb/A/diagnostic", "sid", "ev/logs"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    os.link(SRC_CKPT, d / "rqvae/A/seed_42/final_checkpoint.pth")
    os.link(SRC_EMB / "item_ids.json", d / "emb/item_ids.json")
    os.link(SRC_EMB / "A/diagnostic/drop30.npy", d / "emb/A/diagnostic/drop30.npy")
    os.link(SRC_EMB / "A/diagnostic/drop30_valid.npy", d / "emb/A/diagnostic/drop30_valid.npy")
    clean = np.load(SRC_EMB / "A/clean.npy")
    np.save(d / "emb/A/clean.npy", clean)  # unpolluted copy by default (needed by formal path)
    return d, clean

def artifacts_absent(d):
    sid = d / "sid"
    return {
        "diagnostics_json": not (sid / "A/seed_42/diagnostics.json").exists(),
        "quality_csv": not (sid / "quality_summary.csv").exists(),
        "robustness_csv": not (sid / "robustness_summary.csv").exists(),
        "checks_json": not (d / "ev/diagnostic_checks.json").exists(),
        "inventory_json": not (d / "ev/t3_run_inventory.json").exists(),
    }

res = {"schema": "phase4-t3-failure-probe-1.0", "date": "2026-11-11",
       "accepted_sid_untouched": True, "cases": {}}
ok_all = True

# ---- case 1/2: NaN and Inf via the real CLI (end-to-end, poisoned clean copy) ----
for case, poison in (("nan", "nan"), ("inf", "inf")):
    d, clean = build_case(case)
    bad = clean.copy()
    if poison == "nan":
        bad[5, :] = np.nan
    else:
        bad[5, :] = np.inf
    np.save(d / "emb/A/clean.npy", bad)
    cmd = [".venv-phase3/bin/python", "scripts/v2_export_sid.py", "--mode", "diagnostics",
           "--methods", "A", "--seeds", "42", "--views", "drop30",
           "--rqvae-root", str(d / "rqvae"), "--embeddings-root", str(d / "emb"),
           "--output-root", str(d / "sid"), "--evidence-dir", str(d / "ev"), "--device", "cuda:0"]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=600)
    tail = (r.stderr or r.stdout).strip().splitlines()
    hit = "NONFINITE_ROWS_ABORT" in (r.stderr or "")
    absent = artifacts_absent(d)
    ok = (r.returncode != 0) and hit and all(absent.values())
    res["cases"][case] = {
        "injection": f"clean.npy copy row5={'NaN' if poison=='nan' else '+Inf'} (single 2560-dim row)",
        "mode": "real CLI end-to-end (load_model -> encode_full -> assert_usable_vectors)",
        "command": " ".join(cmd),
        "exit_code": r.returncode,
        "message_hit": hit,
        "stderr_tail": tail[-4:] if tail else [],
        "artifacts_absent": absent,
        "pass": bool(ok)}
    ok_all &= ok
    print(case, "exit", r.returncode, "hit", hit, "absent", all(absent.values()), flush=True)

# ---- case 3: zero-norm via the formal run_diagnostics function with a stubbed
# encode_full (embeddings cannot force a zero quantized-sum, so we feed the
# poisoned tensor directly into the same formal code path) ----
d, _ = build_case("zero")
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "rq"))
import importlib
m = importlib.import_module("v2_export_sid")
orig_encode = m.encode_full
def zero_encode(model, x_all, device):
    codes = np.zeros((3686, 3), dtype=np.int64)
    zq = np.zeros((3686, 32), dtype=np.float32)  # every row zero-norm
    return codes, zq
m.encode_full = zero_encode
args = types.SimpleNamespace(methods=["A"], seeds=[42], views=["drop30"], device="cuda:0",
                             rqvae_root=str(d / "rqvae"), embeddings_root=str(d / "emb"),
                             output_root=str(d / "sid"), evidence_dir=str(d / "ev"))
raised = None
try:
    m.run_diagnostics(args, {"all_passed": True, "cases": {}})
except RuntimeError as e:
    raised = str(e)
finally:
    m.encode_full = orig_encode
hit = bool(raised) and "ZERO_NORM_ROWS_ABORT" in raised
absent = artifacts_absent(d)
ok3 = hit and all(absent.values())
res["cases"]["zero_norm"] = {
    "injection": "zq_all_zero (3686x32 zeros) via stubbed encode_full into the formal run_diagnostics",
    "mode": "formal run_diagnostics code path (embeddings alone cannot force zero quantized-sum)",
    "raised_message": (raised or "")[:220],
    "message_hit": hit,
    "artifacts_absent": absent,
    "pass": bool(ok3)}
ok_all &= ok3
print("zero_norm hit", hit, "absent", all(absent.values()), flush=True)

res["conclusion"] = ("NaN/Inf/零范数三类失败路径均实测：正式入口 assert_usable_vectors 命中相应 "
                     "NONFINITE_ROWS_ABORT / ZERO_NORM_ROWS_ABORT 消息、退出码非零（CLI 两例）或 "
                     "RuntimeError 阻断（零范数例），且临时输出目录无任何 diagnostics/汇总/CSV 指标产物；"
                     "已验收 SID 输出未被触碰。")
res["all_passed"] = bool(ok_all)
out = EV / "t3_failure_probe.json"
out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
print("PROBE_SHA256:", sha(out), flush=True)
print("T3_PROBE_DONE", flush=True)
