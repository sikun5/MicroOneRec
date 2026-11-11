#!/usr/bin/env python
"""MiniOneRec v2 Phase 3 — RQ-VAE 训练入口（A/B/C × seed 42/43/44）。

SCHEMA: phase3-v2_train_rqvae-1.3
协议来源: MicroOneRec/global_guide/phase3_rqvae_training.md + topic_minionerec_v2.md §6
适配基线: 锁定仓库 rq/（rq/rqvae.py 入口、rq/models/* 模型、rq/datasets.py）——
本脚本不修改 rq/** 任何文件；以子类 V2RQVAE 扩展 forward 返回三层量化输出之和
（quantized_sum），straight-through 梯度由锁定 VQ 内部 `x + (x_q - x).detach()` 保留。

关键语义（与协议逐条对应，详见 working-record.md T3 COMMENTS）:
- A/B: 仅干净原损失（MSE + quant_loss_weight×rq_loss），扰动分支完全跳过
  （不加载数据、不前向、不消耗 RNG、无初始化副作用）。
- C: 干净原损失 + λ×mean_valid(1-cos(zq_clean, zq_aug))，λ 内部锁定 0.1；
  aug 仅对 valid 行（3,356）gather 后前向，无效行不进入前向与分母；
  cos 分母 clamp_min(1e-8)；无有效视图时一致性为有限 0。
- K-means 初始化由干净分支首个训练 forward 触发（锁定 vq.initted 机制）；
  final checkpoint 保存 vq_initted，恢复时回填（否则会重跑 K-means）。
- final_checkpoint.pth 含 model/optimizer/scheduler/RNG×4/epoch/step/初始化状态/
  完整 config 与输入+代码哈希；--resume 前对全部语义配置字段（protocol/environment/
  方法/seed/λ/输入哈希/源码哈希/脚本哈希）逐项全等校验，任一不符拒载，最后恢复 RNG。
- resume_snapshot.pth：每 --snapshot-every epoch 与循环结束时写入的完整恢复快照
  （与 final 同构）；正式输出仍只认第 10000 epoch 的 final_checkpoint.pth。
"""
import os

# 数值/确定性环境变量必须在 import torch/numpy/sklearn 之前设置
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import sklearn
import scipy
import joblib
import threadpoolctl
import tqdm
import transformers
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
RQ_DIR = REPO_ROOT / "rq"
sys.path.insert(0, str(RQ_DIR))

from models.rqvae import RQVAE          # noqa: E402  锁定模型（不修改）
from datasets import EmbDataset         # noqa: E402  锁定数据集（A/B 用）

from transformers import get_constant_schedule_with_warmup  # noqa: E402  锁定 scheduler

SCHEMA = "phase3-v2_train_rqvae-1.3"
METHOD_LAMBDA = {"A": 0.0, "B": 0.0, "C": 0.1}
RQ_SOURCE_FILES = ["rq/rqvae.py", "rq/trainer.py", "rq/datasets.py", "rq/utils.py",
                   "rq/models/rqvae.py", "rq/models/rq.py", "rq/models/vq.py", "rq/models/layers.py"]
COS_NORM_EPS = 1e-8
LOG = logging.getLogger("v2_train_rqvae")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class V2PairDataset(Dataset):
    """C 方法数据集：逐 item 返回 (clean, aug, valid)。aug 行取自 C/train_aug.npy，
    valid 取自 C/valid_aug.npy（Phase 2 seed1701 训练视图）。"""

    def __init__(self, clean_path, aug_path, valid_path):
        self.clean = np.load(clean_path)
        self.aug = np.load(aug_path)
        self.valid = np.load(valid_path)
        assert self.clean.shape == self.aug.shape, "clean/aug shape mismatch"
        assert self.valid.shape == (self.clean.shape[0],), "valid mask shape mismatch"
        self.dim = self.clean.shape[-1]

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        return (torch.FloatTensor(self.clean[index]),
                torch.FloatTensor(self.aug[index]),
                bool(self.valid[index]))


class V2RQVAE(RQVAE):
    """锁定 RQVAE 子类：仅扩展 forward 返回 quantized_sum（x_q）。
    state_dict 键与锁定 RQVAE 完全一致；get_indices 单表示推理原样继承。"""

    def forward(self, x, use_sk=True):
        h = self.encoder(x)
        x_q, rq_loss, indices = self.rq(h, use_sk=use_sk)
        out = self.decoder(x_q)
        return out, rq_loss, indices, x_q


def consistency_loss(zq_clean, zq_aug_valid, valid_mask):
    """λ·mean_valid(1-cos(zq_clean,zq_aug))；空 valid 集返回有限 0。
    范数下界 1e-8 分别作用于两个范数（非乘积统一截断）：
      denom = max(|zc|,1e-8) · max(|za|,1e-8)
    例：两个同向、范数 1e-5 的向量 → cos=1（乘积截断会错误地给 0.01）。
    zq_aug_valid 必须是仅对 valid 行 gather 前向后的输出（[n_valid, e]）——
    无效行不进入前向，也不进入本函数。"""
    idx = valid_mask.to(zq_clean.device)
    n_valid = int(idx.sum().item())
    if n_valid == 0:
        return torch.zeros((), device=zq_clean.device, dtype=zq_clean.dtype), 0
    zc = zq_clean[idx]
    if zq_aug_valid.shape[0] != n_valid:
        raise ValueError(
            f"zq_aug_valid rows {zq_aug_valid.shape[0]} != valid count {n_valid}; "
            f"aug must be gathered to valid rows before forward")
    za = zq_aug_valid
    denom = zc.norm(dim=1).clamp_min(COS_NORM_EPS) * za.norm(dim=1).clamp_min(COS_NORM_EPS)
    cos = (zc * za).sum(dim=1) / denom
    raw = (1.0 - cos).mean()
    return raw, n_valid


RESUME_COMPARE_KEYS = ["schema", "run_id", "method", "seed", "consistency_weight",
                       "inputs", "locked_source", "script_sha256", "protocol", "environment"]


def verify_resume_config(stored_cfg, current_cfg):
    """严格恢复配置校验：所有影响数值/轨迹的语义字段逐项全等比较（protocol 含结构/损失/
    optimizer/LR/wd/clip、scheduler/warmup、batch/worker/dtype/确定性、K-means/Sinkhorn、
    训练预算 epochs；environment 绑定 python/torch/numpy/cuda 版本；另含方法/seed/λ/
    输入哈希/锁定源码哈希/脚本自身哈希）。任何不符抛 ValueError——不能静默用不同
    LR/warmup/batch 继续。resume/日志路径不属于 config，无需豁免字段。"""
    mismatched = []
    for key in RESUME_COMPARE_KEYS:
        a = json.dumps(stored_cfg.get(key), sort_keys=True, default=str)
        b = json.dumps(current_cfg.get(key), sort_keys=True, default=str)
        if a != b:
            mismatched.append(key)
    if mismatched:
        raise ValueError(f"resume refused — config semantic mismatch in fields: {mismatched}")
    return True


def parse_args():
    ap = argparse.ArgumentParser(
        description="MiniOneRec v2 Phase 3 RQ-VAE training (A/B/C x seed 42/43/44); "
                    "adapts locked rq/ source without modifying it")
    ap.add_argument('--method', type=str, required=True, choices=["A", "B", "C"],
                    help="A=Concat, B=Quality Fusion, C=B+Consistency (lambda locked per method)")
    ap.add_argument('--seed', type=int, required=True, help='random seed (formal: 42/43/44)')
    ap.add_argument('--embedding-root', type=str, default="outputs/v2/embeddings",
                    help='root of Phase 2 caches (reads {root}/{method}/clean.npy; C also train_aug/valid_aug)')
    ap.add_argument('--epochs', type=int, default=10000, help='number of epochs')
    ap.add_argument('--batch-size', type=int, default=20480, help='batch size (>= catalog size -> full batch)')
    ap.add_argument('--learning-rate', type=float, default=1e-3, help='AdamW lr')
    ap.add_argument('--num-emb-list', type=int, nargs='+', default=[256, 256, 256],
                    help='codebook size of every vq layer')
    ap.add_argument('--consistency-weight', type=float, default=None,
                    help='must equal method-locked lambda (A/B:0, C:0.1); omitted = derived')
    ap.add_argument('--checkpoint-policy', type=str, default="final", choices=["final"],
                    help='only final-epoch checkpoint is the formal output (locked best_* kept as historical diagnostics)')
    ap.add_argument('--output-root', type=str, default="outputs/v2/rqvae",
                    help='output root; smoke runs MUST point this under outputs/v2/smoke/phase3/')
    ap.add_argument('--resume', type=str, default=None,
                    help='path to a v2 final-format checkpoint to resume strictly')
    ap.add_argument('--run-id', type=str, default=None, help='override run id (default RQ-{method}-{seed})')
    # ---- 锁定默认值（与 rq/rqvae.py + trainer.py 一致）----
    ap.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    ap.add_argument('--lr-scheduler-type', type=str, default="constant", help='scheduler (constant-with-warmup)')
    ap.add_argument('--warmup-epochs', type=int, default=50, help='warmup epochs (=steps, 1 step/epoch)')
    ap.add_argument('--weight-decay', type=float, default=0.0, help='l2 regularization weight')
    ap.add_argument('--dropout-prob', type=float, default=0.0, help='dropout ratio')
    ap.add_argument('--no-bn', dest='bn', action='store_false', default=False, help='disable BN (default off)')
    ap.add_argument('--loss-type', type=str, default="mse", choices=["mse", "l1"], help='loss_type')
    ap.add_argument('--kmeans-init', action=argparse.BooleanOptionalAction, default=True,
                    help='kmeans codebook init on first clean forward (locked default True)')
    ap.add_argument('--kmeans-iters', type=int, default=100, help='max kmeans iters')
    ap.add_argument('--sk-epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0],
                    help='sinkhorn epsilons; <=0 -> argmin path (locked [0,0,0])')
    ap.add_argument('--sk-iters', type=int, default=50, help='max sinkhorn iters')
    ap.add_argument('--e-dim', type=int, default=32, help='vq codebook embedding size')
    ap.add_argument('--quant-loss-weight', type=float, default=1.0, help='vq quant loss weight')
    ap.add_argument('--beta', type=float, default=0.25, help='beta for commitment loss')
    ap.add_argument('--layers', type=int, nargs='+', default=[2048, 1024, 512, 256, 128, 64],
                    help='hidden sizes of encoder/decoder MLP')
    ap.add_argument('--num-workers', type=int, default=4, help='dataloader workers (locked default 4)')
    ap.add_argument('--snapshot-every', type=int, default=500,
                    help='write full resume snapshot (model/optimizer/scheduler/RNG/init/config) '
                         'every N epochs and at loop end; formal output remains ONLY the '
                         '10000-epoch final_checkpoint.pth (non-semantic durability knob)')
    ap.add_argument('--eval-step', type=int, default=50, help='collision eval interval (locked; historical diagnostics)')
    ap.add_argument('--save-limit', type=int, default=5, help='rolling diagnostic ckpt limit (locked)')
    ap.add_argument('--device', type=str, default="cuda:0", help='device')
    ap.add_argument('--log-every', type=int, default=50, help='console log interval (epochs); JSONL logs every epoch')
    return ap.parse_args()


def build_config(args, input_hashes, rq_hashes):
    script_hash = sha256_file(str(Path(__file__).resolve()))
    lam = METHOD_LAMBDA[args.method]
    if args.consistency_weight is not None and abs(args.consistency_weight - lam) > 0:
        raise SystemExit(
            f"ERROR: --consistency-weight {args.consistency_weight} != method-locked lambda {lam} "
            f"(method {args.method}); formal runs must use the locked value or omit the flag")
    cfg = {
        "schema": SCHEMA,
        "run_id": args.run_id or f"RQ-{args.method}-{args.seed}",
        "method": args.method,
        "seed": args.seed,
        "consistency_weight_derived": lam,
        "consistency_weight_explicit": args.consistency_weight,
        "consistency_weight": lam,
        "protocol": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "learner": args.learner,
            "weight_decay": args.weight_decay, "lr_scheduler_type": args.lr_scheduler_type,
            "warmup_epochs": args.warmup_epochs,
            "num_emb_list": args.num_emb_list, "e_dim": args.e_dim,
            "layers": args.layers, "loss_type": args.loss_type,
            "quant_loss_weight": args.quant_loss_weight, "beta": args.beta,
            "kmeans_init": args.kmeans_init, "kmeans_iters": args.kmeans_iters,
            "sk_epsilons": args.sk_epsilons, "sk_iters": args.sk_iters,
            "dropout_prob": args.dropout_prob, "bn": args.bn,
            "eval_step": args.eval_step, "save_limit": args.save_limit,
            "num_workers": args.num_workers, "checkpoint_policy": args.checkpoint_policy,
            "grad_clip_norm": 1.0, "cos_norm_eps": COS_NORM_EPS,
            "fp32": True, "tf32": "off (matmul+cudnn explicit)",
            "deterministic_algorithms": True,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "environment": {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": np.__version__, "cuda": torch.version.cuda,
            "sklearn": sklearn.__version__, "scipy": scipy.__version__,
            "joblib": joblib.__version__, "threadpoolctl": threadpoolctl.__version__,
            "transformers": transformers.__version__, "tqdm": tqdm.__version__,
        },
        "inputs": input_hashes,
        "locked_source": rq_hashes,
        "script_sha256": script_hash,
    }
    return cfg


def collect_input_hashes(args):
    root = Path(args.embedding_root)
    m = args.method
    files = {
        "clean": root / m / "clean.npy",
        "item_ids_canonical": root / "item_ids.json",
        "metadata": root / m / "metadata.json",
    }
    if m == "C":
        files["train_aug"] = root / m / "train_aug.npy"
        files["valid_aug"] = root / m / "valid_aug.npy"
    for name, p in files.items():
        if not p.is_file():
            raise SystemExit(f"ERROR: required input missing: {p}")
    hashes = {k: {"path": str(p), "sha256": sha256_file(str(p))} for k, p in files.items()}
    forbidden = sorted((root / m / "diagnostic").glob("*")) if (root / m / "diagnostic").is_dir() else []
    hashes["diagnostic_not_read_count"] = len(forbidden)
    return files, hashes


def set_seed(seed):
    """锁定入口的 seed 设置顺序原样保留（在模型构造之前调用）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def vq_initted_flags(model):
    return [bool(vq.initted) for vq in model.rq.vq_layers]


def set_vq_initted(model, flags):
    assert len(flags) == len(model.rq.vq_layers)
    for vq, flag in zip(model.rq.vq_layers, flags):
        vq.initted = bool(flag)


def get_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def collision_eval(model, loader, device):
    """锁定 _valid_epoch 语义：全量 get_indices(use_sk=False) 去重计数。"""
    model.eval()
    seen = set()
    num_sample = 0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                data = batch[0]
            else:
                data = batch
            data = data.to(device)
            indices = model.get_indices(data)
            idx_np = indices.view(-1, indices.shape[-1]).cpu().numpy()
            num_sample += len(idx_np)
            for row in idx_np:
                seen.add("-".join(str(int(v)) for v in row))
    model.train()
    return (num_sample - len(seen)) / num_sample if num_sample else 0.0


def save_diagnostic_ckpt(path, epoch, best_loss, best_collision, model, optimizer, args):
    torch.save({"args": vars(args), "epoch": epoch, "best_loss": best_loss,
                "best_collision_rate": best_collision,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()}, path, pickle_protocol=4)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    run_id = args.run_id or f"RQ-{args.method}-{args.seed}"
    out_dir = Path(args.output_root) / args.method / f"seed_{args.seed}"
    is_smoke = "smoke" in str(Path(args.output_root))

    rq_hashes = {f: sha256_file(str(REPO_ROOT / f)) for f in RQ_SOURCE_FILES}
    input_paths, input_hashes = collect_input_hashes(args)
    cfg = build_config(args, input_hashes, rq_hashes)

    # ---- 数据（C 才加载 aug/valid；A/B 完全不触碰扰动数据）----
    t0 = time.time()
    if args.method == "C":
        dataset = V2PairDataset(input_paths["clean"], input_paths["train_aug"], input_paths["valid_aug"])
    else:
        dataset = EmbDataset(str(input_paths["clean"]))
    in_dim = dataset.dim
    LOG.info("dataset loaded: n=%d dim=%d (%.1fs)", len(dataset), in_dim, time.time() - t0)

    # ---- 锁定顺序：seed -> 模型构造 -> DataLoader -> 训练 ----
    set_seed(args.seed)
    model = V2RQVAE(in_dim=in_dim,
                    num_emb_list=args.num_emb_list, e_dim=args.e_dim, layers=args.layers,
                    dropout_prob=args.dropout_prob, bn=args.bn, loss_type=args.loss_type,
                    quant_loss_weight=args.quant_loss_weight, beta=args.beta,
                    kmeans_init=args.kmeans_init, kmeans_iters=args.kmeans_iters,
                    sk_epsilons=args.sk_epsilons, sk_iters=args.sk_iters)
    device = torch.device(args.device)
    model = model.to(device)

    loader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size,
                        shuffle=True, pin_memory=True)
    steps_per_epoch = len(loader)
    assert steps_per_epoch == 1, f"protocol expects ONE full-catalog batch per epoch, got {steps_per_epoch}"

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = get_constant_schedule_with_warmup(optimizer,
                                                  num_warmup_steps=args.warmup_epochs * steps_per_epoch)

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    best_collision = float("inf")
    best_loss_ckpt = "best_loss_model.pth"
    best_collision_ckpt = "best_collision_model.pth"

    # ---- 输出目录管理 ----
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.is_file():
            raise SystemExit(f"ERROR: --resume checkpoint not found: {ckpt_path}")
        if ckpt_path.parent.resolve() != out_dir.resolve():
            raise SystemExit(f"ERROR: resume checkpoint {ckpt_path} is not inside {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        if out_dir.exists() and any(out_dir.iterdir()):
            raise SystemExit(
                f"ERROR: output dir {out_dir} exists and is not empty; "
                f"interrupted runs must resume via --resume {out_dir}/final_checkpoint.pth "
                f"(or point a fresh run at an empty directory)")
        out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)

    log_path = out_dir / "training_log.jsonl"
    record_path = out_dir / "run_record.json"
    run_record = {
        "run_id": run_id, "schema": SCHEMA, "method": args.method, "seed": args.seed,
        "command": " ".join(sys.argv), "cwd": str(Path.cwd()),
        "attempts": [], "start": datetime.now().astimezone().isoformat(timespec="seconds"),
        "resume_used": bool(args.resume),
    }

    # ---- 恢复 ----
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        stored_cfg = state["config"]
        try:
            verify_resume_config(stored_cfg, cfg)
        except ValueError as e:
            raise SystemExit(f"ERROR: {e}")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch_completed"])
        global_step = int(state["global_step"])
        set_vq_initted(model, state["init_state"]["vq_initted"])
        set_rng_state(state["rng"])
        best_loss = float(state.get("best_loss", float("inf")))
        best_collision = float(state.get("best_collision_rate", float("inf")))
        run_record["attempts"].append({"type": "resume", "from": str(ckpt_path),
                                       "resumed_at_epoch_completed": start_epoch,
                                       "verified_fields": RESUME_COMPARE_KEYS,
                                       "at": datetime.now().astimezone().isoformat(timespec="seconds")})
        LOG.info("resumed from %s at epoch_completed=%d step=%d (vq_initted=%s, verified %d semantic fields)",
                 ckpt_path, start_epoch, global_step, state["init_state"]["vq_initted"], len(RESUME_COMPARE_KEYS))
    else:
        run_record["attempts"].append({"type": "fresh", "at": run_record["start"]})

    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    lambda_w = cfg["consistency_weight"]
    aug_forward_count = 0
    snapshots_written = 0
    last = {}

    def build_full_state(epoch_completed, epoch_last_index0, snapshot):
        return {
            "schema": SCHEMA, "run_id": run_id, "snapshot": bool(snapshot),
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": get_rng_state(),
            "epoch_completed": epoch_completed,
            "epoch_last_index0": epoch_last_index0,
            "global_step": global_step,
            "init_state": {"vq_initted": vq_initted_flags(model),
                           "kmeans_triggered_on_clean": True,
                           "in_dim": in_dim},
            "config": cfg,
            "final_losses": {k: last.get(k) for k in
                             ("recon", "quant", "consistency_raw", "consistency_weighted", "total", "lr")},
            "aug_forward_total": aug_forward_count,
            "best_loss_diagnostic": best_loss,
            "best_collision_diagnostic": best_collision,
        }

    with open(log_path, "a") as log_f:
        for epoch_idx in range(start_epoch, args.epochs):
            te0 = time.time()
            for batch in loader:
                if isinstance(batch, (tuple, list)):
                    x, aug, valid = batch
                    if not torch.is_tensor(valid):
                        valid = torch.tensor(valid, dtype=torch.bool)
                    aug = aug.to(device)
                    valid = valid.to(device)
                else:
                    x = batch
                x = x.to(device)
                optimizer.zero_grad()
                out, rq_loss, _indices, zq_clean = model(x, use_sk=True)
                loss_recon = F.mse_loss(out, x) if args.loss_type == "mse" else F.l1_loss(out, x)
                loss_quant = rq_loss
                cons_raw = torch.zeros((), device=device)
                n_valid = 0
                if lambda_w > 0:
                    aug_valid = aug[valid]
                    _, _, _, zq_aug = model(aug_valid, use_sk=True)
                    cons_raw, n_valid = consistency_loss(zq_clean, zq_aug, valid)
                    aug_forward_count += 1
                total = loss_recon + args.quant_loss_weight * loss_quant + lambda_w * cons_raw
                if torch.isnan(total) or torch.isinf(total):
                    raise ValueError(f"non-finite training loss at epoch {epoch_idx + 1}")
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                global_step += 1
            te1 = time.time()
            lr_now = scheduler.get_last_lr()[0]
            row = {
                "epoch": epoch_idx + 1, "global_step": global_step, "lr": lr_now,
                "recon": float(loss_recon.item()), "quant": float(loss_quant.item()),
                "consistency_raw": float(cons_raw.item()),
                "consistency_weighted": float(lambda_w * cons_raw.item()),
                "total": float(total.item()), "n_valid_consistency": n_valid,
                "aug_forward": 1 if lambda_w > 0 else 0,
                "epoch_time_s": te1 - te0,
            }
            last = row
            log_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log_f.flush()
            if (epoch_idx + 1) % args.log_every == 0 or (epoch_idx + 1) == args.epochs:
                LOG.info("epoch %d/%d lr=%.3e recon=%.6f quant=%.6f cons=%.6f total=%.6f %.2fs",
                         row["epoch"], args.epochs, lr_now, row["recon"], row["quant"],
                         row["consistency_raw"], row["total"], row["epoch_time_s"])

            # ---- 锁定评估行为（历史诊断，不作为正式输出）----
            if (epoch_idx + 1) % args.eval_step == 0:
                collision = collision_eval(model, loader, device)
                if row["total"] < best_loss:
                    best_loss = row["total"]
                    save_diagnostic_ckpt(out_dir / best_loss_ckpt, epoch_idx, best_loss,
                                         best_collision, model, optimizer, args)
                if collision < best_collision:
                    best_collision = collision
                    save_diagnostic_ckpt(out_dir / best_collision_ckpt, epoch_idx, best_collision,
                                         collision, model, optimizer, args)
                LOG.info("epoch %d eval collision=%.6f best_loss=%.6f best_collision=%.6f",
                         epoch_idx + 1, collision, best_loss, best_collision)

            # ---- 周期性完整恢复快照（与 final 同构；正式产物仍只认 10000-epoch final）----
            if args.snapshot_every > 0 and (epoch_idx + 1) % args.snapshot_every == 0:
                snap_path = out_dir / "resume_snapshot.pth"
                torch.save(build_full_state(epoch_idx + 1, epoch_idx, snapshot=True),
                           snap_path, pickle_protocol=4)
                snapshots_written += 1
                LOG.info("resume snapshot written at epoch_completed=%d -> %s", epoch_idx + 1, snap_path)

    # ---- 循环正常结束：先写末位快照（覆盖 loop 结束与 final 落盘之间的中断窗口），再写 final ----
    snap_path = out_dir / "resume_snapshot.pth"
    torch.save(build_full_state(args.epochs, args.epochs - 1, snapshot=True), snap_path, pickle_protocol=4)
    snapshots_written += 1

    # ---- final checkpoint（唯一正式产物）----
    final_state = build_full_state(args.epochs, args.epochs - 1, snapshot=False)
    final_path = out_dir / "final_checkpoint.pth"
    torch.save(final_state, final_path, pickle_protocol=4)

    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    run_record.update({
        "end": datetime.now().astimezone().isoformat(timespec="seconds"),
        "epochs_completed": args.epochs, "global_step": global_step,
        "final_losses": final_state["final_losses"],
        "aug_forward_total": aug_forward_count,
        "snapshots_written": snapshots_written,
        "snapshot_every": args.snapshot_every,
        "device": {"name": torch.cuda.get_device_name(device),
                   "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory},
        "peak_vram_allocated_bytes": int(peak_alloc),
        "peak_vram_reserved_bytes": int(peak_reserved),
        "checkpoint_bytes": {"final_checkpoint.pth": final_path.stat().st_size,
                             "resume_snapshot.pth": snap_path.stat().st_size},
        "final_checkpoint_sha256": sha256_file(str(final_path)),
        "config_sha256": sha256_file(str(out_dir / "config.json")),
        "training_log_lines": args.epochs,
    })
    with open(record_path, "w") as f:
        json.dump(run_record, f, ensure_ascii=False, indent=1)
    LOG.info("FINAL_SAVED %s sha256=%s bytes=%d peak_alloc=%.1fMiB peak_reserved=%.1fMiB",
             final_path, run_record["final_checkpoint_sha256"], final_path.stat().st_size,
             peak_alloc / 2**20, peak_reserved / 2**20)
    print("RUN_OK", run_id)


if __name__ == "__main__":
    main()
