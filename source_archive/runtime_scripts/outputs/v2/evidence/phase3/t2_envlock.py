#!/usr/bin/env python
"""v2 Phase 3 T2 environment lock: versions, device, locked-source hashes, RQVAE param count,
sklearn KMeans semantics probe, numerical config, static resource plan.
SCHEMA: phase3-t2_environment_lock-1.0
"""
import json, hashlib, os, sys, subprocess, inspect, time
sys.path.insert(0, "/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec/rq")
os.chdir("/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec")

import numpy as np
import torch
import sklearn
from sklearn.cluster import KMeans
import transformers
import tqdm

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

res = {"stage": "phase3_t2_environment_lock", "schema": "phase3-t2_environment_lock-1.0",
       "date": "2026-11-10", "cwd": os.getcwd()}

# ---------- versions ----------
p = torch.cuda.get_device_properties(0)
try:
    nvidia = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name,mig.mode.current",
                             "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
except Exception as e:
    nvidia = f"nvidia-smi unavailable: {e}; reference probe 2026-11-10 22:40 +0800: driver 610.57.04, CUDA UMD 13.3"
res["environment"] = {
    "venv": ".venv-phase3",
    "venv_base": "hpc2hdd/home/swu073/envs/dsaa (same base as .venv-phase2; torch/numpy/tqdm inherited)",
    "python": sys.version.split()[0],
    "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
    "cudnn": getattr(torch.backends.cudnn, "version", lambda: None)(),
    "numpy": np.__version__, "scikit_learn": sklearn.__version__,
    "transformers": transformers.__version__, "tqdm": tqdm.__version__,
    "pip_freeze": "outputs/v2/evidence/phase3/pip_freeze_phase3.txt",
    "pip_freeze_sha256": sha256("outputs/v2/evidence/phase3/pip_freeze_phase3.txt"),
    "phase12_locks_untouched": "outputs/v2/environment/* (Phase1) 与 .venv-phase2/pip_freeze_phase2.txt (11be5661…) 未修改",
    "sklearn_install_attempt": "attempt1 venv误建自 /opt/miniconda3（无torch）→ 删除重建于 dsaa env base（torch 2.14.0+cu130 可见）；两 attempt 均留档",
    "gpu": {
        "name": p.name, "total_memory_bytes": p.total_memory, "sm_count": p.multi_processor_count,
        "compute_capability": f"{p.major}.{p.minor}", "device_count": torch.cuda.device_count(),
        "cuda_available": torch.cuda.is_available(),
        "mig_note": "MIG 1g.10gb 切片 9,728MiB/14SM——单 MIG 设备计费，非 80GB 整卡",
    },
    "nvidia_smi": nvidia,
}

# ---------- locked source hashes ----------
RQ_FILES = ["rq/rqvae.py", "rq/trainer.py", "rq/datasets.py", "rq/utils.py",
            "rq/models/rqvae.py", "rq/models/rq.py", "rq/models/vq.py", "rq/models/layers.py"]
res["locked_source"] = {f: sha256(f) for f in RQ_FILES}

# ---------- RQVAE instantiation (protocol config) ----------
from models.rqvae import RQVAE
torch.manual_seed(42)
model = RQVAE(in_dim=2560, num_emb_list=[256, 256, 256], e_dim=32,
              layers=[2048, 1024, 512, 256, 128, 64], dropout_prob=0.0, bn=False,
              loss_type="mse", quant_loss_weight=1.0, beta=0.25,
              kmeans_init=True, kmeans_iters=100, sk_epsilons=[0.0, 0.0, 0.0], sk_iters=50)
total = sum(q.numel() for q in model.parameters())
enc = sum(q.numel() for q in model.encoder.parameters())
dec = sum(q.numel() for q in model.decoder.parameters())
cb = sum(q.numel() for q in model.rq.parameters())
res["model"] = {
    "in_dim": 2560, "num_emb_list": [256, 256, 256], "e_dim": 32,
    "layers": [2048, 1024, 512, 256, 128, 64],
    "param_total": total, "encoder_params": enc, "decoder_params": dec, "codebook_params": cb,
    "all_float32": all(q.dtype == torch.float32 for q in model.parameters()),
    "all_requires_grad": all(q.requires_grad for q in model.parameters()),
    "state_dict_bytes_fp32": total * 4,
    "adamw_extra_state_bytes_fp32": total * 4 * 2,
    "ckpt_estimate_bytes": total * 4 * 3,
    "seeded_construction_note": "xavier_normal_ 在模型构造时消耗 torch 全局 RNG——v2 入口必须保持 锁定顺序 seed→模型构造→DataLoader→训练",
}

# ---------- KMeans semantics probe ----------
sig = inspect.signature(KMeans.__init__).parameters
probe = {"installed_signature_defaults": {k: (str(sig[k].default) if k in sig else None)
                                          for k in ("init", "n_init", "max_iter", "random_state", "algorithm")}}
x = np.random.RandomState(0).randn(512, 32).astype(np.float32)
np.random.seed(42); c1 = KMeans(n_clusters=16, max_iter=10).fit(x).cluster_centers_
np.random.seed(42); c2 = KMeans(n_clusters=16, max_iter=10).fit(x).cluster_centers_
probe["probe_input"] = "[512,32] float32, K=16, max_iter=10"
probe["centers_dtype"] = str(c1.dtype)
probe["same_seed_twice_equal"] = bool(np.array_equal(c1, c2))
probe["max_abs_diff_same_seed"] = float(np.abs(c1 - c2).max())
probe["locked_call_semantics"] = {
    "call": "KMeans(n_clusters=256, max_iter=100).fit(x)  # rq/models/layers.py kmeans()",
    "input_path": "samples.cpu().detach().numpy()（GPU→CPU，torch tensor）",
    "centers_back": "torch.from_numpy(centers).to(device)（copy_ 入 float32 embedding）",
    "n_init_effective": "installed 默认（1.5.2: 'auto'→k-means++ 时 1 次运行）；未在锁定调用中显式设置",
    "random_state_effective": "None→使用全局 numpy RNG（入口 np.random.seed(seed) 后确定）",
    "threading": "OpenMP 线程数可能影响浮点归约顺序→运行环境固定 OMP_NUM_THREADS（T4/T5 启动统一）",
}
res["kmeans_semantics"] = probe

# ---------- numerical / determinism config ----------
res["numerical_config"] = {
    "dtype": "FP32 全程（模型参数/损失/缓存读入；不引入混合精度）",
    "tf32": "v2 入口显式 torch.backends.cuda.matmul.allow_tf32=False、torch.backends.cudnn.allow_tf32=False",
    "cudnn": "锁定入口已设 deterministic=True、benchmark=False（沿用）",
    "seeds": "v2 可配置 seed∈{42,43,44} 替代锁定 2024；设置顺序沿用锁定入口：random.seed→np.random.seed→torch.manual_seed→torch.cuda.manual_seed_all→cudnn 设置→模型构造",
    "optimizer": "AdamW lr=1e-3 weight_decay=0（trainer._build_optimizer 'adamw' 分支）",
    "scheduler": "transformers get_constant_schedule_with_warmup，warmup_steps=50×len(loader)=50（全量 batch 每 epoch 1 步）→ 前 50 epoch 线性升温后恒定；每 optimizer.step() 后 scheduler.step()",
    "grad_clip": "锁定 trainer 含 clip_grad_norm_(params, 1.0)——沿用锁定行为并如实记录",
    "sinkhorn": "sk_epsilons=[0,0,0]→VQ forward 走 argmin 分支（sk_epsilon<=0），Sinkhorn 数学上不执行",
    "kmeans_init_trigger": "各 VQ 层 initted=False；epoch1 首个训练 forward 内按层1→层3 顺序以当前残差触发 sklearn KMeans（数据依赖，方法间不要求一致；干净分支先触发）",
    "loss": "MSE(out, x) + 1.0×quant_loss；quant_loss=三层(codebook+0.25×commitment)均值；ST: x_q=x+(x_q-x).detach()",
    "eval": "锁定 trainer 每 50 epoch 全量 collision 评估（get_indices use_sk=False）+ best_loss/best_collision/rolling ckpt——v2 保留为历史诊断，正式 final 独立",
    "batch": "20480（实际单 batch=全目录 3686，每 epoch 1 次 optimizer 更新）",
    "batchsize_note": "DataLoader shuffle=True（torch 全局 RNG 决定 3686 排列）；num_workers 需固定并记录",
}

# ---------- static resource plan ----------
N = 3686
res["resource_plan"] = {
    "type": "静态估算（正式实测以 T4 smoke/T5 为准）",
    "device": "单 MIG 1g.10gb（9,728 MiB 上限 / 14 SM / cc 8.0）——占用按单 MIG 设备小时计",
    "vram_estimate": {
        "weights_fp32_bytes": total * 4,
        "optimizer_state_bytes": total * 4 * 2,
        "largest_activation": "3686×2048×4B ≈ 30.2 MB（encoder/decoder 最宽层）",
        "distance_matrices": "3686×256×4B×3 层 ≈ 11.3 MB",
        "torch_context_reference": "~0.3-0.6 GB（经验值，T4 实测为准）",
        "conclusion": "总量远低于 9,728 MiB 上限；batch=20480 全目录 3686 单 batch 可行",
    },
    "time_estimate": {
        "flops_per_epoch_approx": "≈3×2×param_total×3686（forward+backward）",
        "kmeans_init_once": "sklearn CPU 3686×32×256 簇×100 iters×3 层，秒级/次",
        "wall_per_run": "未实测——T4 smoke 以全量 batch 实测 per-epoch 耗时后线性外推；本任务不做正式训练",
        "storage_estimate": f"单 ckpt ≈ {total*4*3/1e6:.1f} MB（state_dict+AdamW 状态）×9 run ≈ {total*4*3*9/1e9:.2f} GB（估算）",
    },
    "consumption": {
        "training_reads": ["A/clean.npy", "B/clean.npy", "C/clean.npy", "C/train_aug.npy", "C/valid_aug.npy",
                           "item_ids.json(canonical)", "metadata×3"],
        "not_read_by_training": "{A,B,C}/diagnostic/*（27 组诊断缓存）、diagnostic_views.jsonl（仅 provenance）",
    },
}

out = "outputs/v2/evidence/phase3/environment_lock.json"
with open(out, "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("MODEL_PARAMS:", total, "| enc", enc, "dec", dec, "cb", cb)
print("KMEANS same-seed-equal:", probe["same_seed_twice_equal"], "| dtype:", probe["centers_dtype"])
print("NVIDIA:", nvidia[:120])
print("OUT_SHA256:", sha256(out))
print("ENVLOCK_DONE")
