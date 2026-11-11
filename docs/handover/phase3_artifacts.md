# Phase 3 已验收产物交接索引（RQ-A/B/C-42/43/44）

Phase 3（RQ-VAE 训练）已由 boss 终验通过（P3-1..P3-5 全部接受，携带已裁定的恢复限制交接）；运行产物位于
远端仓库根下 `outputs/v2/`（九个 193 MB 级 final checkpoint、训练日志、证据等，有意不纳入 git 版本库）。
本索引记录每项产物的仓库相对路径与 SHA-256 基准，供 Phase 4（SID 诊断、导出与重映射）及后续阶段核对。
验收依据与逐项证据见共享工作记录 `working-repo-phase3/working-record.md`（本地保管，不入库）。

- 基准建立时间：2026-11-11（Phase 3 终验接受；本索引随 Phase 3 提交写入）
- 代码：分支 `MicroOneRec`，Phase 3 起点 HEAD `e7517496fc6d216c51c32c36323853674de0132f`（Phase 2 提交）
- 节点/容器：`swu073-a800-10g`（MIG 1g.10gb 切片，9,728 MiB / 14 SM / cc8.0）/ `34fb2691c7dd`
- 工作目录：`/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec`
- 训练环境 `.venv-phase3`：Python 3.10.13 / torch 2.14.0+cu130 / numpy 2.2.6 / scikit-learn 1.5.2 /
  scipy 1.15.3 / transformers 4.51.3（freeze 见 `outputs/v2/evidence/phase3/pip_freeze_phase3.txt`，
  SHA-256 `a292c199147f608c00c6c9d253df78f8a4f2e7e6530632d5f8cc0100d41b9bd7`）

## 方法与实现概述

- 新增 3 个 worker 脚本（随本提交入库，见文末表）；MicroOneRec 既有源码（`rq/` 等）零修改
  （提交前 `git diff` 为空）；本提交不含任何 Phase 4 实现。
- `scripts/v2_train_rqvae.py`，SCHEMA=`phase3-v2_train_rqvae-1.3`（SHA-256
  `618f4ff18b244e6f49c76c94cc4ccea3e14f9920e8108c62ae525f5bf17ab5b8`，九次正式运行统一绑定，
  与每个 checkpoint 内 config.script_sha256 一致）：`V2RQVAE(RQVAE)` 前向返回四元组
  （out, rq_loss, indices, quantized_sum）；`V2PairDataset` 供 C 一致性分支；一致性损失采用
  每范数独立下限 max(|z|,1e-8)；λ 内部锁定 A/B=0、C=0.1（CLI 传入不一致即拒绝）；
  resume 快照每 500 epoch，加载前 10 组语义字段校验（协议+环境）；确定性栈：FP32 全程、
  TF32 off、cudnn deterministic、`torch.use_deterministic_algorithms(True)`、
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`、`OMP_NUM_THREADS=8`。
- 训练协议：AdamW lr 1e-3 / wd 0；transformers constant-with-warmup（warmup 50）；grad clip 1.0；
  K-means init 100 iters（sklearn 1.5.2，CPU 路径）；Sinkhorn [0,0,0]→argmin；
  batch 20480（实际全目录 3,686 行单批，每 epoch 1 步）；10,000 epochs / 10,000 steps；
  `--checkpoint-policy final`（第 10,000 epoch 独立落盘 final_checkpoint.pth，无最佳种子筛选）。
- 三方法输入（Phase 2 冻结缓存，全量 SHA-256）：
  - A=`outputs/v2/embeddings/A/clean.npy` `cb745f4cfed999ae2b17b5368de10192c24c6371e5d08925cb7c1538ddbd4e31`
  - B=`outputs/v2/embeddings/B/clean.npy` `06d5797f590492f14a854785587096f54bb3321b3ae6c165dac3d7080b000c0f`
  - C=`outputs/v2/embeddings/C/clean.npy`（与 B 逐字节相同，同上哈希）
    + `C/train_aug.npy` `871b817e3f3a6038901f5a0d5ab19db75be392490aa389cd3387866757c80064`
    + `C/valid_aug.npy` `7127c60951c3a23de6c9f13315f56d1b3d7992b557517e5a448e9d407f043c3b`（有效 3,356）
  - item 顺序：`outputs/v2/embeddings/item_ids.json`（canonical，A/B/C 副本同哈希）
    `f7c159fdaed280524878756f24e9271f916a9570dbdef3b965b50044c68ae791`
- 方法语义：A=Concat；B=QualityFusion；C=与 B 同一 clean 表示 + 训练一致性项
  （train_aug=seed1701 drop30 训练视图，valid_aug 掩码有效 3,356；扰动分支仅贡献一致性）。

## 九个正式 final（唯一正式 checkpoint；路径 `outputs/v2/rqvae/{M}/seed_{S}/final_checkpoint.pth`）

| run_id | epochs/step | bytes | final SHA-256 |
|---|---|---|---|
| RQ-A-42 | 10000/10000 | 193,394,919 | `a22716fcec92a92e676dd55effd69be7d962479b388bae5271fa225f42f44a9d` |
| RQ-A-43 | 10000/10000 | 193,394,855 | `d1810dd1c34c7fae40bfc492bd42a5d10fdf5f7989b980080a642ad52f06b9dc` |
| RQ-A-44 | 10000/10000 | 193,394,855 | `0cdb241c1660aba7d078fe2f7ef4e7dfaf8153a10f97ead521a84803d0d5db96` |
| RQ-B-42 | 10000/10000 | 193,394,919 | `cd026216bad73c05bded7c65f2766d014f1a253dfb558ef1cd5ac2234f0d3d78` |
| RQ-B-43 | 10000/10000 | 193,394,855 | `e180eb2dcfeef2c48be8ad7ac78d9673edd89fb8e4c709d3f783378475cfb586` |
| RQ-B-44 | 10000/10000 | 193,394,855 | `7aa6e36c736acb251f83fa840258d025a3426aa2c479563fa70e9cc474def7ab` |
| RQ-C-42 | 10000/10000 | 193,395,175 | `c36aab6b46b313ee6f81c927ddc80d2588bd92925d3009a9d1c75a71d439dde2` |
| RQ-C-43 | 10000/10000 | 193,395,111 | `96c5f2c318db6345ef56dcb9bff31489d23f7b1b9a4b76fd62c8f98137972313` |
| RQ-C-44 | 10000/10000 | 193,395,175 | `925bd1bda3f32d60e3c4bb2de11692d1d1233fcedb9a65016446c4a109c9aae2` |

九次全部 attempt1、fresh、`resume_used=false`（boss 裁定：正式路径仅接受不中断运行）；执行顺序
A42→B42→C42→A43→B43→C43→A44→B44→C44（2026-11-11 03:36:06→12:05:13 +0800）。每个 checkpoint 内含
完整 config（script_sha256=v1.3 绑定、逐文件输入哈希、locked_source）与 optimizer（31 条目，
step=10000）/scheduler/RNG 四套/vq_initted=[T,T,T] 状态，strict 加载可独立核验
（`outputs/v2/evidence/phase3/checkpoint_checks.json`）。

## 成本口径（双口径，单 MIG 设备计）

- run wall 之和：30,421 s = 8 h 27 m 01 s ≈ 8.4503 小时。
- driver 占用跨度：03:36:02→12:05:13 = 30,551 s = 8 h 29 m 11 s ≈ 8.4864 小时（差 130 s =
  启动/切换/记录开销，未细分）。
- 均非完整 A800 等效小时；smoke 100ep 峰值显存实测 A/B 574.1 MiB（reserved 662.0）、
  C 684.4 MiB（750.0）；恢复排错轮耗时未仪器化（如实注明，未补造）。

## T6 补证组成与 Phase 4 待重核边界

- Phase 3 T6 实测 24 文件 = 9 实际消费输入 + 4 原始源文件（item/train/valid/test CSV 原文重哈希）
  + 11 Phase 1/2 基准（`outputs/v2/evidence/phase3/t6_input_coverage.json`）。
- 21 个诊断缓存（`{A,B,C}/diagnostic/*`）Phase 3 训练不消费、未重哈希，沿用 Phase 2 基准
  `outputs/v2/evidence/phase2/t5_cache_table.json`（`639d2b89…`）。
- **Phase 4 T2 边界**：须对实际消费的干净/诊断向量、mask、item 顺序、metadata、诊断清单逐一实算
  并核对 Phase 2 完整 33 文件表，禁止仅复用上期子集检查结论。

## 恢复限制（P1 已裁定，携带交接）

- 跨进程训练轨迹未达预声明 bitwise=0 标准（`resume_checks.json`）：进程级 ~1e-9 数值噪声经
  VQ argmin 近并列离散放大（80 epoch 轨迹差 ~0.055）；resume 机制本身逐位保真
  （`snapshot_fidelity_checks.json` 逐字段核对全过）。
- 处置：正式路径限定不中断运行；限制保留披露，不声称根因完全证明。正式复现入口为
  `scripts/v2_phase3_resume_compare.py`；五件 /tmp 分层探针（E3–E7）未版本化，用途与来源见
  共享记录 Phase 3 T4 COMMENTS，输出存档于 `outputs/v2/evidence/phase3/t4_*`。

## 待 Phase 4 诊断/导出内容（本提交不含任何 Phase 4 实现）

- 九套 clean 原始质量统计：碰撞率 = 1 − 三层唯一数/3686、碰撞涉及比例、最大桶、三层利用率/256、
  归一化熵 $-\sum_k p_k \log p_k / \log 256$（p=0 项忽略），3,686 商品全覆盖、码域 0..255，
  保存 item 显式顺序、原始码与三层量化和向量。
- 27 个冻结扰动诊断：3 方法 × 3 种子 × drop30/drop60/dropall（seed2701 诊断视图）；
  严格 `eval()/no_grad()/use_sk=False`，无训练/无 K-means/无码本更新/无官方碰撞重分配；
  PPR@1/2/3 只比同模型前三层码，NPR@10 以该模型 clean 量化目录为固定候选（cosine、排除自身、
  按规范 item ID 破平局）；非有限/零范数不得静默填充。
- 三种子均值与样本标准差（ddof=1）按同指标同扰动汇总，原值全保存；负结果与坍塌信号保留。
- 下游仅 seed42：原始三层码 + sorted_item_suffix 桶内消歧的四 token 完整 SID；共同词表
  （A/B/C 每层全 256 语义 token + 三组 seed42 后缀并集，共同注册顺序）；index/item/info 与
  反向映射；实际商品合法前缀树；原 CSV 逐行重映射（非 SID 字段/行序/切分/缺失标签保持）；
  10k RL 任务接口（8,000/1,000/1,000，保留 task/source ID 与顺序，同 prompt 多目标不被字典覆盖）。
- 拟新增入口 `scripts/v2_export_sid.py`（Phase 4 指南模板，尚未实现，本提交未包含）。

## 证据文件哈希（`outputs/v2/evidence/phase3/`，核心 14 件）

| 文件 | SHA-256 |
|---|---|
| `environment_lock.json` | `d89f149db341fd4773274da6c6222943128b0f1e76c0cdec118c0d4ff8c5b279` |
| `pip_freeze_phase3.txt` | `a292c199147f608c00c6c9d253df78f8a4f2e7e6530632d5f8cc0100d41b9bd7` |
| `upstream_checks.json` | `eeb1643f878b4734c4ce698c6b80e9f361371ea6d359c1443f35f6bb0cded443` |
| `loss_gradient_checks.json` | `c5cb608437d910197da92512145e037606bf5f7fa925ffd9438c510895cc1d15` |
| `smoke_checks.json` | `3a3fb65abb86dbae8b17a6e625196db74b0520dbff539db872e279895ddd5a32` |
| `resume_checks.json` | `a9881c7d41b7da94fb1bb200d0ac93dccf0c5b5391d91954a77499492b98a644` |
| `snapshot_fidelity_checks.json` | `759f56f61519c7b805d58a721d5bd285662df854c7ff7727807c950b32cf56b3` |
| `run_inventory.json` | `78f5318407a99870cdc34b03b44558ab239be6db90a7fc3b01d493b182c450a3` |
| `checkpoint_checks.json` | `46a1001ba4b5c170fb5c7cdce184aaa8014d022d83c1ceb098b8a03ead824620` |
| `resource_summary.json` | `74c996e36ee5bfb24684a688e23634e36f64e956b053c01948a85ef3b401d905` |
| `t5_record_extract.json` | `e51a9cd4c3338fcf003b60403e2ccec06b31a70e7f9efbc2b1133d137c642b57` |
| `t6_input_coverage.json` | `4925771a7c624360548cf185b97efc19be54b1a7d476e99f838e02bc63a3d833` |
| `run_records.md` | `85dcb877567d0337c17f89d0a3beee3672280767731000671bfe240c44551463` |
| `acceptance.md` | `b18dcb6c74f50773c7b050803d97a77b886e6425a3d093a7cf2681fc96945dd6` |

同目录另有 36 件 T2–T6 过程存档（探针脚本与输出、λ 拒绝日志、接口探针、逐 run 生成器脚本，
`t2_*`…`t6_*`、`v2_train_rqvae_help.txt`），哈希已在 Phase 3 共享记录登记，未逐一列入本表。

## 本提交入库文件（4 个，全部新增）

| 文件 | 用途 |
|---|---|
| `scripts/v2_train_rqvae.py` | Phase 3 正式训练入口（v1.3，哈希与九 checkpoint config 绑定一致） |
| `scripts/v2_phase3_checks.py` | P3-1 λ=0 退化/双分支梯度证据生成器 |
| `scripts/v2_phase3_resume_compare.py` | T4 恢复对照证据生成器（限制披露的正式复现入口） |
| `docs/handover/phase3_artifacts.md` | 本索引（自身哈希不入本表，记录于共享工作记录） |

排除项及理由：`outputs/`（权重/缓存/证据/数据，Phase 1/2 先例有意不入库）、`.venv-phase2/`、
`.venv-phase3/`（虚拟环境）、`global_guide/topic_minionerec_v2.md`（用户自有协议文档，保持
untracked）、`scripts/archive/`（Phase 1 期旧归档副本+__pycache__）、`scripts/outputs/`
（临时复算输出）。本提交不含原始数据、模型权重、缓存、环境或任何凭证。
