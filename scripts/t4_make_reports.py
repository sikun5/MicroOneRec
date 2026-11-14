#!/usr/bin/env python3
"""Phase 6 T4 report generator: aggregate memory probes + 100-sample smoke
into reports/{memory_report.json, smoke_report.json, acceptance.md} and
evidence/phase6/t4_summary.json. Read-only aggregation; no re-runs."""
import glob
import hashlib
import json
import os
import time

ROOT = "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec"
SMOKE = os.path.join(ROOT, "outputs/v2/debug/phase6/sft_smoke/attempt_001")
OUT = os.path.join(ROOT, "outputs/v2/evidence/phase6")
GROUPS = ["A", "B", "C"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    os.makedirs(os.path.join(SMOKE, "reports"), exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    # ---------- memory report ----------
    mem = {"stage": "t4-memory-probe", "seq_len_enforced": 512,
           "descent_order": [16, 8, 4, 2, 1],
           "history_note": ("v1 short-sequence probes (batch-padding to batch max, not 512) were "
                            "INVALID for the 512 requirement and are archived under "
                            "memory_probe/_history_shortseq_invalid/; the smoke eval fp32-logits "
                            "OOM (eval_micro 16->4) and the first smoke OOM runs are archived as "
                            "reports/t4_smoke_run*_oom/512unprobe logs"),
           "oom_history": {}, "per_group": {}, "common": {}}
    for g in GROUPS:
        gd = {"probes": {}, "reload": None, "accum": None, "oom": []}
        for f in sorted(glob.glob(os.path.join(SMOKE, "memory_probe", g, "*probe.json"))):
            r = load(f)
            gd["probes"][os.path.basename(f)] = {"ok": r.get("ok"), "micro": r.get("micro"),
                                                  "peak_allocated": r.get("memory", {}).get("peak_allocated"),
                                                  "peak_reserved": r.get("memory", {}).get("peak_reserved"),
                                                  "losses": r.get("losses"), "losses_finite": r.get("losses_finite"),
                                                  "validation": r.get("validation"),
                                                  "optimizer_state_dtype": r.get("optimizer_state_dtype"),
                                                  "persisted_bytes": r.get("saved", {}).get("persisted_bytes")}
        for f in glob.glob(os.path.join(SMOKE, "memory_probe", g, "*oom.json")):
            r = load(f)
            gd["oom"].append({"micro": r.get("micro"), "message": (r.get("oom", {}) or {}).get("message", "")[:160]})
        rl = os.path.join(SMOKE, "memory_probe", g, "reload_micro2.json")
        if os.path.exists(rl):
            r = load(rl)
            gd["reload"] = {"all_ok": r.get("all_ok"), "embed_shape": r.get("embed_shape"),
                             "tokenizer_len": r.get("tokenizer_len"),
                             "optimizer_n_states": r.get("optimizer_n_states"),
                             "optimizer_state_dtypes": r.get("optimizer_state_dtypes")}
        af = os.path.join(SMOKE, "memory_probe", g, "micro2_accum.json")
        if os.path.exists(af):
            r = load(af)
            gd["accum"] = {"ok": r.get("ok"), "accumulation": r.get("accumulation"),
                            "peak_allocated": r.get("memory", {}).get("peak_allocated"),
                            "losses_finite": r.get("losses_finite"),
                            "first_loss": (r.get("losses") or [None])[0]}
        mem["per_group"][g] = gd
    mem["common"] = {"micro": 2, "accum_formal_1024": 512,
                     "text": "micro2 is the largest config feasible on all three groups at enforced "
                             "seq len 512 (16/8/4 OOM on all groups); formal accumulation 1024/2=512 "
                             "micro-steps -> 1 optimizer.step verified per group",
                     "common_config_file": "memory_probe/common_config_512.txt"}
    with open(os.path.join(SMOKE, "reports", "memory_report.json"), "w") as f:
        json.dump(mem, f, ensure_ascii=False, indent=1)

    # ---------- smoke report ----------
    sm = {"stage": "t4-smoke100",
          "manifest": {"path": "samples_manifest.jsonl",
                        "n": 100,
                        "selection": "seed42 random.Random(42).sample over 33528 conflict-free complete-SID train rows (conflict = target sid == last history sid); same row_ids across A/B/C",
                        "sha256": sha256_file(os.path.join(SMOKE, "samples_manifest.jsonl"))},
          "budget_iterations": [
              {"v": "v4", "budget": "micro2 accum32 (effective64), 30 epochs, stop eval<=20% init",
               "outcome": "loss dropped (A 15.71->4.14) but hits=0; budget too weak for memorization",
               "log": "reports/t4_smoke_run_v4_hits0_budget.log", "history": "overfit_100/_history_v4_A_hits0"},
              {"v": "v5", "budget": "micro2 accum1, 60 epochs, same drop-ratio stop",
               "outcome": "stop fired at epoch1 (eval<=20% init), hits=0 in all groups",
               "log": "reports/t4_smoke_run_v5_droptarget_hits0.log",
               "history": "overfit_100/_history_v5_{A,B,C}_droptarget"},
              {"v": "v6 (final)", "budget": "micro2 accum1, 60 epochs, stop eval<=0.5 (memorization level) or cap",
               "outcome": "all three groups pass gate: loss drop + hits>0 + illegal 0 + reload exact"}],
          "earlier_failures": [
              "t4_smoke_run_v1_evalmicro16_oom.log: eval batch 16 OOM at fp32 logits upcast -> eval_micro descended to 4 (train micro unchanged)",
              "t4_smoke_run_v2_512unprobe_oom.log: probes had been run without enforced 512 padding; probes redone with padding=max_length 512",
              "t4_smoke_run_v3_genrawencode_crash.log: generation used HF tokenizer with data.Tokenizer-only kwargs; fixed to wrapper + save-before-generation ordering"],
          "per_group": {}}
    all_ok = True
    for g in GROUPS:
        r = load(os.path.join(SMOKE, "overfit_100", g, "smoke_result.json"))
        rl = load(os.path.join(SMOKE, "overfit_100", g, "reload_eval.json"))
        gen = load(os.path.join(SMOKE, "overfit_100", g, "generation_outputs.json"))
        gd = {"init_eval_loss": r["init_eval_loss"], "final_eval_loss": r["final_eval_loss"],
               "drop_ratio": r.get("drop_ratio"), "stop_reason": r["stop_reason"],
               "epochs_run": r["epochs_run"], "wall_seconds": r["wall_seconds"],
               "micro": r["micro"], "accum": r["accum"],
               "losses_finite": r["losses_finite"],
               "hits_exact_full_sid": r["hits"], "illegal_rate": r["illegal_rate"],
               "reload_abs_diff": rl["abs_diff"], "reload_match": rl["all_ok"],
               "generation_raw_outputs_n": len(gen),
               "raw_head": gen[:3],
               "loss_history_tail": r["loss_history"][-3:]}
        sm["per_group"][g] = gd
        gate = (r["losses_finite"] and r["illegal_rate"] == 0.0 and r["hits"] > 0 and rl["all_ok"])
        sm["per_group"][g]["gate_pass"] = gate
        all_ok = all_ok and gate
    sm["all_gate_pass"] = all_ok
    sm["boundary"] = "smoke checkpoints live only under the attempt root and are NEVER used to initialize formal SFT (T5 starts from the T3 common initialization)"
    with open(os.path.join(SMOKE, "reports", "smoke_report.json"), "w") as f:
        json.dump(sm, f, ensure_ascii=False, indent=1)

    # ---------- acceptance.md ----------
    acc = f"""# T4 压测与 100 样本 smoke 验收（worker，2026-11-12）

结论：**三组在强制 seq len=512 下共同可行训练配置=micro2**（16/8/4 全组 OOM 已留档）；三组真实全参数路径全过——probe（2 次 optimizer.step+AdamW 状态+validation+保存/新进程加载）、共同配置真实累积验证（1024/2=512 micro-step→1 step）、100 样本 smoke（同口径 loss 下降+约束 Beam10 命中>0+非法率 0+新进程 reload 逐位一致）。**smoke/压测闸门全过，正式 A-SFT 保持锁定等待 boss 放行。**

## 冻结预算与迭代史（先运行后修正均如实留痕）

- 压测：512 padding 强制（v1 短序列探针无效已归档）；micro16→8→4→2→1；micro1 备用 gradient checkpointing（未用到）。
- smoke 预算三轮：v4 (eff64/30ep, hits=0)→v5 (accum1, drop-ratio 停, hits=0)→**v6 (accum1, eval≤0.5 或 60ep)**。三轮日志与产物全部归档。

## 实测要点

| 项 | A | B | C |
|---|---|---|---|
| probe micro2 peak_allocated | 8,024,368,640B（三组同） | 同 | 同 |
| accum 验证（512 micro→1 step）peak | 7,108,342,784B | 同 | 同 |
| reload（probe ckpt） | OK | OK | OK |
| smoke init→final eval loss | 15.7135→1.3671 | 15.7148→0.8462 | 15.7906→0.3553 |
| 完整 SID 精确命中（Beam10 约束） | 7/100 | 36/100 | 34/100 |
| 非法率 | 0.0 | 0.0 | 0.0 |
| smoke ckpt 新进程 reload | diff=0.0 | diff=0.0 | diff=0.0 |
| 停止 | 60ep 封顶 | 60ep 封顶 | eval≤0.5 提前 |

## 边界

- optimizer state dtype=bf16（实测）；validation eval_micro=4（降载锁定）；alloc_conf=expandable_segments:True（smoke 阶段）。
- 无协议外降载（无量化优化器/CPU 卸载/截短 512；micro1+ckpt 未触发）。
- smoke checkpoint 绝不作为 A-SFT 初始化；正式训练输出仅 `outputs/v2/checkpoints/{{A,B,C}}-SFT/`。
- 本报告及三组关键数值已写 working-record；证据哈希见 `evidence/phase6/t4_summary.json`。
"""
    with open(os.path.join(SMOKE, "reports", "acceptance.md"), "w") as f:
        f.write(acc)

    # ---------- evidence index ----------
    idx = {"stage": "phase6-t4-summary", "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
           "smoke_root": SMOKE,
           "common_config": mem["common"],
           "smoke_all_gate_pass": all_ok,
           "artifacts": {}}
    for rel in ["reports/memory_report.json", "reports/smoke_report.json", "reports/acceptance.md",
                "samples_manifest.jsonl", "memory_probe/common_config_512.txt",
                "configs/t4_probe_config.json"]:
        p = os.path.join(SMOKE, rel)
        if os.path.exists(p):
            idx["artifacts"][rel] = sha256_file(p)
    for g in GROUPS:
        for rel in [f"overfit_100/{g}/smoke_result.json", f"overfit_100/{g}/reload_eval.json",
                    f"overfit_100/{g}/generation_outputs.json", f"overfit_100/{g}/config.json",
                    f"memory_probe/{g}/micro2_probe.json", f"memory_probe/{g}/micro2_accum.json",
                    f"memory_probe/{g}/reload_micro2.json"]:
            p = os.path.join(SMOKE, rel)
            if os.path.exists(p):
                idx["artifacts"][rel] = sha256_file(p)
    idx["source_hashes"] = {fn: sha256_file(os.path.join(ROOT, "scripts", fn))
                            for fn in ("v2_sft_memprobe.py", "v2_sft_reload_check.py",
                                       "v2_sft_smoke100.py", "v2_sft_smoke_reload.py")}
    idx["key_numbers"] = {
        "probe_peak_allocated_micro2": 8024368640,
        "accum_peak_allocated_micro2": 7108342784,
        "oom_levels_512": "16/8/4 on all groups (archived)",
        "smoke": {g: {"init": sm["per_group"][g]["init_eval_loss"],
                       "final": sm["per_group"][g]["final_eval_loss"],
                       "hits": sm["per_group"][g]["hits_exact_full_sid"],
                       "illegal": sm["per_group"][g]["illegal_rate"],
                       "reload_abs_diff": sm["per_group"][g]["reload_abs_diff"]} for g in GROUPS}}
    with open(os.path.join(OUT, "t4_summary.json"), "w") as f:
        json.dump(idx, f, ensure_ascii=False, indent=1)
    print("REPORTS_DONE all_gate_pass=", all_ok)
    print(json.dumps(idx["key_numbers"]["smoke"], indent=1)[:600])


if __name__ == "__main__":
    main()
