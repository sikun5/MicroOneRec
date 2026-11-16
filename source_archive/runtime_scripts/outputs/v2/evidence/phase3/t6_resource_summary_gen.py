import os, sys, json, hashlib
os.environ.setdefault("OMP_NUM_THREADS", "8")
from pathlib import Path
REPO = Path("/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec")
sys.path.insert(0, str(REPO / "scripts"))
import v2_train_rqvae as V
def sha(p): return V.sha256_file(str(p))
inv = json.load(open(REPO / "outputs/v2/evidence/phase3/run_inventory.json"))

# ---- U3: frozen inputs unchanged (read-only, vs Phase 2 accepted values) ----
EXPECTED = {
    "outputs/v2/embeddings/A/clean.npy": "cb745f4cfed999ae2b17b5368de10192c24c6371e5d08925cb7c1538ddbd4e31",
    "outputs/v2/embeddings/B/clean.npy": "06d5797f590492f14a854785587096f54bb3321b3ae6c165dac3d7080b000c0f",
    "outputs/v2/embeddings/C/clean.npy": "06d5797f590492f14a854785587096f54bb3321b3ae6c165dac3d7080b000c0f",
    "outputs/v2/embeddings/C/train_aug.npy": "871b817e3f3a6038901f5a0d5ab19db75be392490aa389cd3387866757c80064",
    "outputs/v2/embeddings/C/valid_aug.npy": "7127c60951c3a23de6c9f13315f56d1b3d7992b557517e5a448e9d407f043c3b",
    "outputs/v2/embeddings/item_ids.json": "f7c159fdaed280524878756f24e9271f916a9570dbdef3b965b50044c68ae791",
    "outputs/v2/manifests/quality_features.jsonl": "9df37c0d775846890f08d8aed393ab29705fedf7e6b5f6c1a6c1911661a44fee",
    "outputs/v2/manifests/train_views.jsonl": "45540aec5e3f49c5b520e74682acd6e658cb48ce164f6c645025a570f6499510",
    "outputs/v2/manifests/catalog_clean.jsonl": "275f869fbd5b5cf9c731608b64a43ae5ee7fa3deb8ff8228ef2110759c1a5649",
}
inputs = []
all_ok = True
for p, exp in EXPECTED.items():
    a = sha(REPO / p)
    ok = a == exp
    all_ok &= ok
    inputs.append({"path": p, "expected_phase2_sha256": exp, "actual": a, "unchanged": ok})
res_inputs = {"frozen_inputs_unchanged": {"files_checked": len(inputs), "all_unchanged": bool(all_ok),
                                          "entries": inputs}}

# ---- U4: resource summary ----
runs = []
for r in inv["runs"]:
    runs.append({"run_id": r["run_id"], "method": r["method"], "seed": r["seed"],
                 "wall_seconds": r["resources"]["wall_seconds"],
                 "peak_alloc_MiB": round(r["resources"]["peak_vram_allocated_bytes"] / 2**20, 1),
                 "peak_res_MiB": round(r["resources"]["peak_vram_reserved_bytes"] / 2**20, 1),
                 "final_bytes": r["final_bytes"], "snapshot_bytes": r["resources"]["snapshot_bytes"],
                 "device": r["resources"]["device"]["name"]})
by_method = {}
for m in ("A", "B", "C"):
    g = [r for r in runs if r["method"] == m]
    by_method[m] = {"runs": 3, "wall_sum_s": sum(r["wall_seconds"] for r in g),
                    "peak_alloc_MiB_max": max(r["peak_alloc_MiB"] for r in g),
                    "final_bytes_sum": sum(r["final_bytes"] for r in g)}
tot_bytes = sum(r["final_bytes"] for r in runs)
res = {
    "stage": "phase3_t6_resource_summary", "schema": "phase3-t6_resource_summary-1.0", "date": "2026-11-12",
    "candidate_script_v13": sha(REPO / "scripts/v2_train_rqvae.py"),
    "per_run": runs,
    "by_method": by_method,
    "totals": {
        "run_wall_sum_seconds": sum(r["wall_seconds"] for r in runs),
        "run_wall_sum_human": "8h27m01s ≈ 8.4503 小时",
        "driver_span_seconds": 30551, "driver_span_human": "03:36:02→12:05:13 = 8h29m11s ≈ 8.4864 小时",
        "delta_seconds": 130, "delta_note": "进程启动/切换/记录等未细分跨度",
        "billing": "单 MIG 设备占用（A800-SXM4-80GB MIG 1g.10gb, 14SM）；driver 跨度≠GPU 忙碌时长",
        "final_bytes_sum": tot_bytes,
        "peak_alloc_MiB_max": max(r["peak_alloc_MiB"] for r in runs),
        "peak_res_MiB_max": max(r["peak_res_MiB"] for r in runs),
    },
    "smoke_and_debug_resources": {
        "formal_config_smoke_100ep": "A/B/C 实测峰值 574.1/574.1/684.4 MiB（见 smoke_checks.json）；已采集",
        "resume_noisefloor_runs": "120ep 进程级资源未单独采集（时间未仪器化），不冒充实测",
        "t4_probes": "CPU/GPU 探针耗时未仪器化采集（秒级观察），不补造",
    },
    "frozen_inputs": res_inputs,
}
out = REPO / "outputs/v2/evidence/phase3/resource_summary.json"
with open(out, "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("inputs_all_unchanged:", bool(all_ok))
print("totals:", json.dumps(res["totals"], ensure_ascii=False)[:400])
print("RES_SHA256:", sha(out))
print("RES_DONE")
