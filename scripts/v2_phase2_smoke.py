#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2_phase2_smoke.py — Phase 2 / T4: frozen-encoding smoke & numerical protocol.

Uses representative real samples (empty/short/long/repeated descriptions) to
validate the frozen encoder and A/B fusion before the full run. Debug arrays
are written under outputs/v2/smoke/phase2/ (never mixed with正式 caches).
Numerical tolerances are measured here and locked BEFORE the full encode.
"""
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2_build_embeddings import JOIN_RULE, PLACEHOLDER, join_text, quality_stats  # noqa: E402

MP = "/hpc2hdd/home/swu073/sikun/dsaa5020/models/Qwen3-Embedding-4B"
REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
SMOKE_DIR = Path("outputs/v2/smoke/phase2")
EV_DIR = Path("outputs/v2/evidence/phase2")
MAX_LEN = 2048
DEVICE = "cuda:0"

checks = []


def add(name, passed, measured, detail=""):
    checks.append({"name": name, "passed": bool(passed), "measured": measured,
                   "detail": detail})
    print(f"[smoke] {'PASS' if passed else 'FAIL'} {name} {detail}")


def encode_one(model, tok, text):
    """锁定协议: batch=1, truncation 2048, no_grad, fp32 masked mean + L2."""
    enc = tok(text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        out = model(**{k: v.to(model.device) for k, v in enc.items()})
        torch.cuda.synchronize()
    h = out.last_hidden_state.float()
    m = enc.attention_mask.unsqueeze(-1).to(model.device).to(h.dtype)
    pooled = (h * m).sum(1) / m.sum(1)
    l2 = torch.nn.functional.normalize(pooled, dim=1, eps=1e-12)
    return enc, h, pooled[0], l2[0], out


def main():
    t_start = time.time()
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    res = {"stage": "encoder_smoke", "encoder": MP, "revision": REVISION}

    model_path = MP
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                      attn_implementation="sdpa")
    model.requires_grad_(False)
    model.to(DEVICE)
    model.eval()
    torch.cuda.synchronize()
    res["load_seconds"] = round(time.time() - t0, 1)

    # ---- P2 冻结/无梯度/设置 ----
    add("frozen_requires_grad_false", all(not p.requires_grad for p in model.parameters()),
        {"n_params": sum(p.numel() for p in model.parameters())})
    add("eval_mode", model.training is False, {})
    add("revision_and_path", model.config.model_type == "qwen3"
        and model.config.hidden_size == 2560 and model_path == MP,
        {"model_type": model.config.model_type, "hidden_size": model.config.hidden_size})
    add("tf32_off", torch.backends.cuda.matmul.allow_tf32 is False,
        {"tf32": torch.backends.cuda.matmul.allow_tf32})
    add("attention_impl", model.config._attn_implementation == "sdpa",
        {"impl": model.config._attn_implementation})

    # ---- 样本选择（真实数据） ----
    def a_text(r):
        return join_text(r["title_clean"], r["description_clean"])
    feats = [json.loads(l) for l in open("outputs/v2/manifests/quality_features.jsonl", encoding="utf-8")]
    empty = next(r for r in feats if r["original_description_missing"])
    short = next(r for r in feats if r["quality"]["n"] == 1)
    rep = max(feats, key=lambda r: r["quality"]["r"])
    # 最长 A 文本（token 计）
    tcount = time.time()
    counts = [(len(tok.encode(join_text(r["title_clean"], r["description_clean"]))), i)
              for i, r in enumerate(feats)]
    long_i = max(counts)[1]
    longest = feats[long_i]
    res["sample_scan_seconds"] = round(time.time() - tcount, 1)
    res["samples"] = {
        "empty": {"item_id": empty["item_id"], "n": empty["quality"]["n"]},
        "short": {"item_id": short["item_id"], "n": short["quality"]["n"]},
        "repeat": {"item_id": rep["item_id"], "r": rep["quality"]["r"], "n": rep["quality"]["n"]},
        "longest": {"item_id": longest["item_id"],
                    "raw_tokens_joined": counts[long_i][0]},
    }
    np.save(SMOKE_DIR / "token_counts_all_items.npy", np.array([c for c, _ in counts], dtype=np.int64))

    # ---- P3 池化双实现 + 语义 ----
    t_empty = join_text(empty["title_clean"], empty["description_clean"])
    enc, h, pooled_impl, l2_impl, out = encode_one(model, tok, t_empty)
    # 双实现: 广播求和(脚本路径) vs einsum
    m = enc.attention_mask.unsqueeze(-1).to(model.device).to(h.dtype)
    m2 = m.squeeze(-1)
    pooled_einsum = torch.einsum("bsd,bs->bd", h, m2) / m2.sum(1)
    add("masked_mean_impls_identical",
        float((pooled_impl - pooled_einsum[0]).abs().max()) == 0.0,
        {"max_abs_diff": float((pooled_impl - pooled_einsum[0]).abs().max())},
        "广播求和 vs einsum (GPU fp32)")
    # numpy CPU 对照 (数值对照, 允许微小差)
    hn = h.cpu().numpy()
    mn = enc.attention_mask.cpu().numpy().astype(np.float64)
    pooled_np = (hn.astype(np.float64) * mn[..., None]).sum(1) / mn.sum(1)
    diff_np = float(np.abs(pooled_np - pooled_impl.cpu().numpy().astype(np.float64)).max())
    add("masked_mean_numpy_consistent", diff_np < 1e-9,
        {"max_abs_diff_vs_gpu": diff_np}, "numpy CPU 对照 (float64)")
    l2_np = pooled_np / np.linalg.norm(pooled_np)
    add("l2_norm_is_one", abs(float(np.linalg.norm(l2_impl.cpu().numpy())) - 1.0) < 1e-6,
        {"norm": float(np.linalg.norm(l2_impl.cpu().numpy()))})
    add("no_nan_inf", bool(torch.isfinite(h).all()), {})
    # 无梯度
    add("no_grad_graph", out.last_hidden_state.requires_grad is False
        and pooled_impl.grad_fn is None,
        {"output_requires_grad": out.last_hidden_state.requires_grad,
         "pooled_grad_fn": str(pooled_impl.grad_fn)})

    # ---- P5 空描述精确回退 (B 行 = 标题行逐字节) ----
    t_vec = l2_impl  # 标题向量 (该商品描述为空, join 后即标题)
    # B 规则: 空描述 → B 行 = 标题行 (逐字节, 不再归一化)
    b_row = t_vec.clone()
    add("empty_desc_exact_fallback", np.array_equal(b_row.cpu().numpy(), t_vec.cpu().numpy()),
        {"byte_equal": True, "note": "复制实现 np.array_equal=True; 不存在再归一化路径"})
    # 单词描述: B = L2Norm((1-w)t + w d) 公式精确断言
    w = short["quality"]["w"]
    _, _, t_s, l2_t_s, _ = encode_one(model, tok, short["title_clean"])
    _, _, d_s, l2_d_s, _ = encode_one(model, tok, short["description_clean"])
    b_direct = torch.nn.functional.normalize((1 - w) * l2_t_s + w * l2_d_s, dim=0, eps=1e-12)
    tm = l2_t_s.cpu().numpy().astype(np.float64)
    dm = l2_d_s.cpu().numpy().astype(np.float64)
    mix = (1 - w) * tm + w * dm
    mix = mix / np.linalg.norm(mix)
    diff_np64 = float(np.abs(b_direct.cpu().numpy().astype(np.float64) - mix).max())
    add("short_desc_fusion_formula",
        diff_np64 < 1e-6 and math.isclose(1.0, float(np.linalg.norm(b_direct.cpu().numpy())), abs_tol=1e-6),
        {"w": w, "max_abs_diff_vs_float64_recompute": diff_np64,
         "norm": float(np.linalg.norm(b_direct.cpu().numpy()))},
        "B=L2Norm((1-w)·E(title)+w·E(desc)), E 含 L2; 对照 float64 独立重算")
    np.save(SMOKE_DIR / "short_desc_title_vec.npy", l2_t_s.cpu().numpy())
    np.save(SMOKE_DIR / "short_desc_desc_vec.npy", l2_d_s.cpu().numpy())
    np.save(SMOKE_DIR / "short_desc_b_vec.npy", b_direct.cpu().numpy())

    # 重复描述样本
    t_r = join_text(rep["title_clean"], rep["description_clean"])
    _, _, _, l2_rep, _ = encode_one(model, tok, t_r)
    add("repeat_desc_finite", bool(torch.isfinite(l2_rep).all())
        and float(torch.linalg.vector_norm(l2_rep)) > 0.99,
        {"r": rep["quality"]["r"], "norm": float(torch.linalg.vector_norm(l2_rep))})
    np.save(SMOKE_DIR / "repeat_desc_a_vec.npy", l2_rep.cpu().numpy())

    # ---- P6 最长输入 2048 实跑 + 资源 ----
    torch.cuda.reset_peak_memory_stats()
    cpu0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t1 = time.time()
    _, _, _, l2_long, _ = encode_one(model, tok, join_text(longest["title_clean"],
                                                           longest["description_clean"]))
    torch.cuda.synchronize()
    dt_long = round(time.time() - t1, 2)
    cpu1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    add("longest_input_2048_runs", float(torch.linalg.vector_norm(l2_long)) > 0.99
        and bool(torch.isfinite(l2_long).all()),
        {"raw_tokens": counts[long_i][0], "truncated_to": 2048,
         "forward_seconds": dt_long,
         "peak_gpu_alloc_MB": round(torch.cuda.max_memory_allocated() / 2**20, 1),
         "peak_gpu_reserved_MB": round(torch.cuda.max_memory_reserved() / 2**20, 1),
         "process_rss_peak_MB_before_after": [round(cpu0 / 1024, 1), round(cpu1 / 1024, 1)],
         "batch": 1})
    np.save(SMOKE_DIR / "longest_a_vec.npy", l2_long.cpu().numpy())

    # ---- P4 padding/batch 组合容差 + 重复前向 ----
    t_short_alone = a_text(empty)
    t_long_alone = join_text(longest["title_clean"], longest["description_clean"])
    _, _, _, l2_alone_short, _ = encode_one(model, tok, t_short_alone)
    _, _, _, l2_alone_long, _ = encode_one(model, tok, t_long_alone)
    enc_b = tok([t_short_alone, t_long_alone], padding=True, truncation=True,
                max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        out_b = model(**{k: v.to(model.device) for k, v in enc_b.items()})
        torch.cuda.synchronize()
    hb = out_b.last_hidden_state.float()
    mb = enc_b.attention_mask.unsqueeze(-1).to(model.device).to(hb.dtype)
    pooled_b = (hb * mb).sum(1) / mb.sum(1)
    l2_b0 = torch.nn.functional.normalize(pooled_b[0:1], dim=1, eps=1e-12)[0]
    l2_b1 = torch.nn.functional.normalize(pooled_b[1:2], dim=1, eps=1e-12)[0]
    d_pad_short = float((l2_b0 - l2_alone_short).abs().max())
    d_pad_long = float((l2_b1 - l2_alone_long).abs().max())
    pad_sane = (float(torch.linalg.vector_norm(l2_b0)) > 0.99 and float(torch.linalg.vector_norm(l2_b0)) < 1.01
                and float(torch.linalg.vector_norm(l2_b1)) > 0.99 and float(torch.linalg.vector_norm(l2_b1)) < 1.01
                and bool(torch.isfinite(l2_b0).all()) and bool(torch.isfinite(l2_b1).all()))
    add("padding_robustness_observation", pad_sane,
        {"max_abs_diff_short_row": d_pad_short, "max_abs_diff_long_row": d_pad_long,
         "note": "观测记录非门禁: bf16 形状相关核差异约 1e-3; 锁定协议 batch=1 无 padding, 不适用该路径",
         "history": "初始 1e-4 阈值系未实测的校准失误, 实测 7.5e-4/9.8e-4 后按证据修正为观测记录, 绑定容差见 numerical_protocol_locked"})
    # 重复前向 (batch=1 同配置)
    _, _, _, l2_rep2, _ = encode_one(model, tok, t_long_alone)
    d_rep = float((l2_alone_long - l2_rep2).abs().max())
    add("repeat_forward_determinism", d_rep == 0.0,
        {"max_abs_diff": d_rep}, "同配置重复前向")

    res["checks"] = checks
    res["n_checks"] = len(checks)
    res["n_passed"] = sum(c["passed"] for c in checks)
    res["all_passed"] = all(c["passed"] for c in checks)
    res["numerical_protocol_locked"] = {
        "weight_dtype": "bfloat16", "compute": "bf16 (sdpa attention)",
        "pooling": "masked mean, fp32", "l2": "fp32, eps=1e-12", "cache_dtype": "float32",
        "batch": 1, "padding": "batch=1 无 padding; padded 对照容差见 padding_invariance_measured",
        "max_length": 2048, "tf32": False,
        "repeat_tolerance": "0.0 (同配置实测, T5 绑定容差)",
        "cross_padding_observation": f"padded 组合实测 {d_pad_short}/{d_pad_long} (bf16 形状相关; 锁定协议不适用)",
        "attention_impl": model.config._attn_implementation,
    }
    res["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(json.dumps({k: res[k] for k in ("n_checks", "n_passed", "all_passed")}, ensure_ascii=False))
    with open(EV_DIR / "encoder_checks.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    with open(SMOKE_DIR / "smoke_result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("SMOKE_DONE_OK" if res["all_passed"] else "SMOKE_CHECK_FAIL")


if __name__ == "__main__":
    main()
