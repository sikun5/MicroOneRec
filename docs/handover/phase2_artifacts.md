# Phase 2 已验收产物交接索引（P2-EMB-A / P2-EMB-BC / P2-EMB-DIAG）

Phase 2（质量特征与冻结文本表示）已由 boss 终验通过；运行产物位于远端仓库根下 `outputs/v2/`
（缓存/清单/证据等，有意不纳入 git 版本库）。本索引记录每项产物的仓库相对路径与 SHA-256 基准，
供 Phase 3 及后续阶段核对。验收依据与逐项证据见共享工作记录
`working-repo-phase2/working-record.md`（本地保管，不入库）。

- 基准建立时间：2026-11-10（Phase 2 终验）
- 代码：分支 `MicroOneRec`，Phase 2 起点 HEAD `ded0bc13059b25c3c20f6cbc1a4c1a7c48d97649`（Phase 1 提交）
- 节点/容器：`swu073-a800-10g`（MIG 1g.10gb 切片，9,728 MiB）/ `34fb2691c7dd`
- 工作目录：`/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec`

## 方法与实现概述

- 新增 3 个 worker 脚本（随本提交入库，见下表）；MicroOneRec 既有源码（`rq/`、编码器调用等）零修改；
  本提交不含任何 Phase 3 实现。
- 编码器：`Qwen/Qwen3-Embedding-4B` @ revision `5cf2132abc99cad020ac570b19d031efec650f2b`
  （本地 `/hpc2hdd/home/swu073/sikun/dsaa5020/models/Qwen3-Embedding-4B/`，4,021,774,336 参数全 BF16）。
- 编码环境 `.venv-phase2`：Python 3.10.13 / torch 2.14.0+cu130 / transformers 4.51.3 /
  tokenizers 0.21.4 / huggingface_hub 0.36.2（freeze 见
  `outputs/v2/evidence/phase2/pip_freeze_phase2.txt` `11be5661…`）。
- 冻结编码协议（T4 锁定）：BF16 权重 / sdpa / batch=1 无 padding / FP32 池化·L2·缓存 /
  tf32 off / max_length 2048（全量零截断）/ requires_grad False + eval / expandable_segments。
- 质量规则与融合（B/C）：`q=1[n>0]·min(n/32,1)·(1-r)`，`w=0.5q`；
  `L2Norm((1-w)·E(title)+w·E(desc))`；空描述精确回退标题（327 行）。
- 三方法：A=拼接编码（title+"\n"+desc）；B=质量融合；C=与 B 同一干净表示，
  另存 seed1701 训练视图 drop30 缓存与有效掩码（供 Phase 3 一致性项）。
  原损失仅干净输入；C 扰动分支仅贡献一致性（Phase 3 实现）。

## 33 个正式缓存（组成及生成方式）

**13 向量**（全部 `[3686,2560]` float32、L2 归一化）：A/B/C clean ×3 + C train_aug ×1 +
9 诊断向量（A 独立 3 + B/C 共享 3，各 drop30/drop60/dropall）。
**10 mask**（`[3686]` bool）：C valid_aug + 9 诊断 valid，有效数 3,356/3,356/3,356/3,359。
**10 JSON**：item_ids ×4（canonical + A/B/C 副本，哈希同为 `f7c159fd…`）+ metadata ×3 + views_source ×3。
生成方式：`scripts/v2_build_embeddings.py` v1.1 单进程顺序编码（A→B→C→诊断），
B/C clean 与 7 个 B/C 诊断文件字节级共享（各只编码一次，C 侧为复制件）；
dropall 输入=仅标题；中断按 memmap 进度续跑（本次全程零恢复事件）。
完整 33 文件逐项哈希表：`outputs/v2/evidence/phase2/t5_cache_table.json`（`639d2b89…`）。

## 真实资源成本（实测，详见 `cost_accounting.json` `3206edf4…`）

- 文本 34,229 条 / forward 34,229 次（batch=1）；tokens raw=post=3,040,709（零截断）
- wall 2,473.5s（41.2 分钟）= **0.687 单 MIG 设备占用小时**（非完整 A800 等效）
- 33 正式文件 491,257,061 bytes = 468.5 MiB；embeddings 目录全量 841,796,519 bytes
  （含 `.progress` 工作缓存 350.5 MB）
- distinct 文本 26,873/34,229（重复提交 7,356 次：空描述同标题 + dropall 三组同文本 + 目录原生重复）
- **限制：T5 正式运行峰值显存未采集**（如实披露）；T2/T4 探测参考 batch=1 峰值 8,122/7,994 MiB

## 验收定位与勘误

- P2-1..P2-6 全部通过（boss 终验 2026-11-10）；无未解决 P0。
- item 1631 融合 1.49e-8 差异定向定位：**生产路径（numpy 融合）复算 diff=0.0、byte_equal=True**；
  1.49e-8 为 T6 检查器 torch GPU `F.normalize` 独立实现的舍入差（检查器近似误差），
  非生产链非确定性——初判"跨进程核差异"已勘误（`item1631_localization.json` `a0299bbd…`）。
- `outputs/v2/evidence/phase2/acceptance.md` 终版 = `dfcad2c07b711484dd2f82736bc8374b613b90118508dacd5f95783c6e7f78be`
  （2026-11-10 20:32 终验哈希回填后版本；早期版本哈希不同，以本哈希为准）。
- `run_records.md` = `c10320f65603a77cf7658de1ac7c712f43e9a894cb075002d4b2cd32735294a5`。
- 未生成正式 test 推荐指标；未运行 RQ-VAE/SFT/GRPO；未启动 Phase 3/4。

## Phase 3 输入表（实测哈希，2026-11-10 复核）

| 仓库相对路径 | SHA-256 |
|---|---|
| `outputs/v2/embeddings/A/clean.npy` | `cb745f4cfed999ae2b17b5368de10192c24c6371e5d08925cb7c1538ddbd4e31` |
| `outputs/v2/embeddings/B/clean.npy` = `C/clean.npy` | `06d5797f590492f14a854785587096f54bb3321b3ae6c165dac3d7080b000c0f` |
| `outputs/v2/embeddings/C/train_aug.npy` | `871b817e3f3a6038901f5a0d5ab19db75be392490aa389cd3387866757c80064` |
| `outputs/v2/embeddings/C/valid_aug.npy` | `7127c60951c3a23de6c9f13315f56d1b3d7992b557517e5a448e9d407f043c3b`（valid=3,356） |
| `outputs/v2/embeddings/item_ids.json`（canonical，A/B/C 副本同哈希） | `f7c159fdaed280524878756f24e9271f916a9570dbdef3b965b50044c68ae791` |
| `outputs/v2/manifests/quality_features.jsonl` | `9df37c0d775846890f08d8aed393ab29705fedf7e6b5f6c1a6c1911661a44fee` |
| `outputs/v2/embeddings/A/metadata.json` | `9b3e65ca75ea6667949894217a38db94317c673428d880d6d63f6e23144406ee` |
| `outputs/v2/embeddings/B/metadata.json` | `5db063945bd1a5e8227add7f5a6c9b8c608f28e9e9c27b2afa49c52d8a7f73d3` |
| `outputs/v2/embeddings/C/metadata.json` | `d6b6e680e317dc9763846a334cd4a9d2a2d8abc7e751818f74a0e45e74c2e6fc` |
| `outputs/v2/evidence/phase2/t5_cache_table.json`（33 文件全表） | `639d2b897ae5a68654576eb7594708c9a466d92ac655c34e1f272fe1e3e7a547` |

诊断向量组（`{A,B,C}/diagnostic/{vt}.npy` + `{vt}_valid.npy`）：dropall 三组同值
`d6bae025…`；drop30 B/C `3cc9609e…`、A `4449c10d…`；drop60 B/C `da62caf0…`、A `2544fd8b…`；
drop30/60 valid `7127c609…`、dropall valid `89b93037…`（逐文件哈希见 33 文件全表）。

## 未纳入版本控制项及理由

- `outputs/v2/**`：全部运行产物（缓存/清单/证据/日志），按上表哈希追踪，不入库（Phase 1 同例）。
- `outputs/v2/smoke/phase2/`、`outputs/v2_repro_tmp/`、`scripts/outputs/`：smoke/调试/复核副本，非正本。
- `.venv-phase2/`：虚拟环境；依赖以 `pip_freeze_phase2.txt` 哈希引用。
- `global_guide/topic_minionerec_v2.md`：用户添加的 untracked 文件（研究协议），保留原状。
- `scripts/archive/`：Phase 1 期已存在的旧版归档，非 Phase 2 源码，保留原状。

## 新增脚本（随本提交入库）

| 路径 | SHA-256 | 作用 |
|---|---|---|
| `scripts/v2_build_embeddings.py` | `83678bd245d84bb6bb564a2a5d132fb0729662be1e84fae99152620301597214` | v1.1 正式编码（quality/encode/diagnostics 三阶段） |
| `scripts/v2_phase2_smoke.py` | `6587596e39d62ccf0fef7430219a23db84ef703bc1b86cb8f1f36aba97f1acdc` | T4 冻结协议 smoke（16/16） |
| `scripts/v2_phase2_final_checks.py` | `088062662e9238e964ded3f5ce32032740f59cfc4910e9f37eac6a0291cd2724` | T6 独立检查（cache/view/成本） |

## 证据索引（`outputs/v2/evidence/phase2/`，22 文件）

| 文件 | SHA-256 |
|---|---|
| `acceptance.md` | `dfcad2c07b711484dd2f82736bc8374b613b90118508dacd5f95783c6e7f78be` |
| `run_records.md` | `c10320f65603a77cf7658de1ac7c712f43e9a894cb075002d4b2cd32735294a5` |
| `upstream_checks.json` | `4729937c583581b98151cf73f17cdb2b26b4d1212e92d8331a5fd77b5220f4cb` |
| `environment_lock.json` | `194f114a49b39e7daa7d9268602720830d1e05e0979800a83c8c748d7b7d6c06` |
| `model_probe.json` | `ecd5f4039587d8378f6566053fa71e64f5ba2ebb8e1935b7b006d8bd0373b91e` |
| `model_probe.log` | `32cd657097a0630f76089d02f27f84b633d2bca21cfcebe0f2b4a08ab0203ccb` |
| `dsaa_load_attempt.json` | `e87ed860653c132bad8a77969643bcb56e6122234b29075435071ec867e06079` |
| `pip_freeze_phase2.txt` | `11be5661f75d8154dd38294d72eca783fe0e753498b3ef82afd0cd14fc9a3835` |
| `interface_spec.json` | `1a2b3386e442ed1daf571871b8561df388db3f05af593750c3fd8f1b61f33548` |
| `quality_rule_checks.json` | `b6473d6b0fac41a7ff489f996a35c0398fa16be5c97d2498b876c9062f962e1a` |
| `id_view_checks.json` | `d499a129b504155c4e1e85e8e4e0e00fc10cd038577a32b0c496bbcf5043f203` |
| `encoder_checks.json` | `0beccfee49033d2db03e8232d4677745cf53829857af4a0ddcf2a8f0ed52d014` |
| `encoder_smoke.log` | `8d2c5d6140bb9b416b3de2cf5e6279204c37e086b78c6311d4e3f10f8d10043f` |
| `param_memory_check.json` | `cf37ce787c9996c0618391a20eb5b9ef43bb910e7aa050e3256b0161d22e43a4` |
| `encode_run.log` | `4d3d4bacbd4b24398c47d7d83bb12ecc0c80b63acc2b0d2c20071f9552d3ea8a` |
| `t5_verify.json` | `cd250f0c0b5c5f6432d694d5d84b388cc69199a46c75e5f2523317826b33da3b` |
| `t5_cache_table.json` | `639d2b897ae5a68654576eb7594708c9a466d92ac655c34e1f272fe1e3e7a547` |
| `cache_checks.json` | `f6158d9287ec8863f0e54ed269ea5e87ef36dc40709196620baeab7c2ba27967` |
| `view_checks.json` | `0041e5ebe08205f049fb9824049cf82d65ba10c53ec799b0a9263c84cc0fd710` |
| `cost_accounting.json` | `3206edf4f33b0eb62525bd557e0a545ef8bfb5f08e553b6a0a8691b312e6d747` |
| `item1631_localization.json` | `a0299bbde7952f53265506cfbafadd5d026d246c658aa8e309a1d4e5c5c38fbd` |
| `final_checks.log` | `3b8302959e9f731751d51141c7d45ddfbe25c17f4601309e7b08bf3bae28b6ce` |
