# Phase 1：数据与环境准备

> **状态：已完成并验收（2026-11-09，run_id `P1-PREP-42`）。**审计参考值来自v2，已在本期实测复核，全部通过；验收证据见 `outputs/v2/evidence/phase1/acceptance.md`。  
> 导航：[主入口](experiment_setting.md) · 上一期：无 · [下一期：质量与表示](phase2_quality_embeddings.md)  
> 协议：[v2 数据与环境协议](../../topic_minionerec_v2.md#3-数据与环境协议)。运行路径均相对 `project/MiniOneRec/`；本期实际执行目录为远端 `/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec`（经用户确认为指定工作目录的物理路径）。

## 1. 实验目标

固定所有方法共同的数据、模型来源与随机清单，为RQ1/RQ2/RQ3提供可追踪输入。本期只审计test结构和身份，不生成test推荐指标。

## 2. 实验内容

### 2.1 前置输入

- `data/Amazon/index/Industrial_and_Scientific.item.json`；
- `data/Amazon/{train,valid,test}/` 的Industrial CSV；
- 当前代码commit与未提交差异；`topic_minionerec_v2.md`；
- 后续GPU环境信息和两个模型的本地路径/revision；本地无GPU时只完成CPU数据部分，GPU锁定项保持待验收。

### 2.2 所需实现与执行顺序

1. **环境冻结：**记录commit、源码差异哈希、Python/CUDA/驱动、PyTorch/Transformers/TRL/Accelerate/bitsandbytes版本、硬件、模型revision与路径，不输出访问令牌。
2. **输入快照：**保存每个源文件SHA-256、大小和来源；源文件只读。
3. **数据审计：**逐切分统计行数、用户、历史长度、目标与历史商品覆盖；检查原CSV字段，不虚构时间戳。
4. **身份固定：**为每行建立 `split + 原始从零行号` 的row_id，保留原始用户、历史、目标、标题与行序；不得最长序列去重或重新切分。
5. **公共目录：**输出清洗后的标题/描述与原item ID，并在清洗前固定 `original_description_missing`；不使用全空categories构造类别标签。
6. **RL来源清单：**seed42，无放回抽8,000个train序列样本、1,000个目录标题任务、1,000个train文本历史任务；各任务内来源唯一，跨任务可共享交互。标题冲突按规范item ID保留一个，数量不足则阻断，不补重复样本。
7. **扰动清单：**训练seed1701只生成描述drop30；诊断seed2701生成drop30/drop60/dropall。部分删除保序、保留至少一个词；n<2标无有效部分删除。清单记录item ID、删除位置、原/扰动文本哈希，标题和目标不变。
8. **可重现性：**重复运行清单构建，验证相同输入、seed和版本的输出哈希一致；记录任务与prompt冲突，后续映射阶段仍需复查。

所有清洗采用共同函数，不为了某组修正文本。完整目录元数据共同可见属于transductive设置；禁止从validation/test交互构造训练样本或质量规则。

### 2.3 预期产物

| 路径 | 内容 |
|---|---|
| `outputs/v2/environment/environment_lock.json` | 环境、硬件和模型revision |
| `outputs/v2/manifests/file_hashes.json` | 原文件与协议哈希 |
| `outputs/v2/manifests/data_audit.json` | 计数、覆盖、历史长度、异常 |
| `outputs/v2/manifests/rows_manifest.jsonl` | 稳定row_id与原始身份、切分 |
| `outputs/v2/manifests/catalog_clean.jsonl` | 公共清洗文本与原始缺失标签 |
| `outputs/v2/manifests/rl_tasks_10000.jsonl` | task/source/target来源，尚未注入某组SID |
| `outputs/v2/manifests/train_views.jsonl` | 训练drop30视图 |
| `outputs/v2/manifests/diagnostic_views.jsonl` | 独立三类诊断视图 |
| `outputs/v2/evidence/phase1/` | 验收、重复构建和运行记录 |

### 2.4 待新增命令接口

`scripts/v2_prepare.py` 当前是待新增入口，不是现成脚本；先完成上述实现再运行接口模板：

```bash
python scripts/v2_prepare.py \
  --data-root data/Amazon --dataset Industrial_and_Scientific \
  --rl-counts 8000 1000 1000 --seed 42 \
  --train-view-seed 1701 --diagnostic-view-seed 2701 \
  --output-root outputs/v2
```

> **实现状态（2026-11-09）：**`scripts/v2_prepare.py` 已实现并验收（含 environment/file_hashes/audit/rl/views 五阶段与 `--only` 过滤器）；检查入口 `scripts/v2_phase1_checks.py`；同版本重建副本 `scripts/v2_prepare_v13.py`。

## 3. 验收且可推进标准

| 编号 | 必须核验的条件 | 未来证据位置 | 不满足时 |
|---|---|---|---|
| P1-1 | 3,686个目录商品、229个原始缺失描述 | `manifests/data_audit.json` | 核对数据版本，不静默改参考值 |
| P1-2 | train/valid/test为36,259/4,532/4,533行，test为1,987用户、447缺失目标行 | 同上 | 阻断，检查文件和解析 |
| P1-3 | 所有历史/目标在目录中、row_id唯一、源行与源文件不变 | `evidence/phase1/identity_checks.json` | 阻断，禁止跳行 |
| P1-4 | RL恰为8,000/1,000/1,000，交互来源无validation/test | RL清单与抽样审计 | 阻断，修复抽样 |
| P1-5 | 两类seed、词删除规则正确，标题/item/目标不变，重复生成同哈希 | `evidence/phase1/reproducibility.json` | 阻断 |
| P1-6 | 模型revision、代码状态和后续计算环境记录齐全 | environment lock | 缺项保持待验收，不下载/训练冒充完成 |

数据版本真实变化只有经协议变更明确记录后才能采用；本指南本身不授权重定义计数或切分。自然缺失和不均衡是研究事实，不是错误。

### 3.1 验收结果（2026-11-09 实测，run_id `P1-PREP-42`）

| 编号 | 实测值 | 证据（sha256 前 16 位，完整值见 acceptance.md） | 结论 |
|---|---|---|---|
| P1-1 | 3,686 个目录商品；229 个原始缺失描述（清洗前固定）；另 98 条被官方 ≥2,000 字符清洗规则置空，清洗后空描述共 327；categories 全空未构造标签 | `data_audit.json` `a2db080e739f22bc` | 通过 |
| P1-2 | train/valid/test = 36,259/4,532/4,533 行；test 1,987 用户、447 缺失目标行；train 7,034 用户/1,919 缺失、valid 2,011/417（与 v2 §3.1 独立审计一致） | `data_audit.json` 同上 | 通过 |
| P1-3 | 历史/目标目录覆盖 100%（三切分不在目录数均 0）；row_id 45,324 全局唯一、行序保持；逐行身份差异 0；源文件哈希前后一致 | `identity_checks.json` `7a0a135acf05e861`（16/16 实数据不变量） | 通过 |
| P1-4 | 恰 10,000 任务 = 8,000/1,000/1,000；任务内来源唯一；valid/test 泄漏 0；目标与行清单一致 0 差异；同 prompt 多目标冲突如实记录（296/18 组） | `rl_tasks_10000.jsonl` `e77868cae2e4ffec`、`rl_summary.json` `636d402d659d1bd3` | 通过 |
| P1-5 | seed1701/2701 独立随机流；k=min(n−1,max(1,⌊p·n⌋)) 与保序违规 0；标题/item 身份不变；**同版本重建**（脚本 sha256 = T3 正本 `3d4c5ba1…`）5 个 jsonl 清单整文件哈希逐字节一致；data_audit 确定性节规范化哈希相等（全文件因 run 元数据不同，两种口径未混淆） | `reproducibility.json` `d303d447998836f9`（含 same_version_rebuild 节）、`views_summary.json` `ebb26ef8a355b859` | 通过 |
| P1-6 | `environment_lock.json`：git `c6a9ffd3`/分支 MicroOneRec/tracked 无差异；Python 3.10.13、torch 2.14.0+cu130、transformers 4.45.2、trl 0.11.3、accelerate 1.0.1、bitsandbytes 0.50.2、驱动 610.57.04；GPU 实测 A800-80GB MIG 1g.10gb（9.5 GiB、14 SM）；双模型本地路径与不可变 revision `da87bfb6…` / `5cf2132a…`（4 个大文件 SHA-256 = HF LFS oid） | `environment_lock.json` `b102c41efddd2b09`、`pip_freeze.txt` `f104eac93fce544d` | 通过 |

## 4. 推进、恢复与交接

P1-1至P1-6通过后，将环境锁和全部manifest的路径/哈希交给Phase 2。中断后复核源哈希和已生成清单，仅补缺失部分。源数据或清单发生变化时，后续依赖缓存和训练全部标记需重新验收。

> **交接状态（2026-11-09）：**P1-1至P1-6已全部通过并完成交接；完整路径/哈希交接表、恢复规则（哈希基准、全阶段重建法、失效传播、revision 锁定、复核副本定位）与问题披露见 `outputs/v2/evidence/phase1/acceptance.md`（sha256 `8ad3963540b939b2`）。注意：本节点可用 GPU 为 MIG 1g.10gb（9.5 GiB）切片，Phase 2 编码资源预算须在开工前决策。

## 5. 运行记录

运行记录实际文件：`outputs/v2/evidence/phase1/run_records.md`（sha256 `5f240aa55a330bdc`，run_id `P1-PREP-42`，含 R1–R5 全部运行与失败 attempt 留档）；验收文件：`outputs/v2/evidence/phase1/acceptance.md`（sha256 `8ad3963540b939b2`）。下方为原模板：

| 字段 | 初始值 |
|---|---|
| run_id / status | `P1-PREP-42` / 未开始 |
| code/model/environment revision | 待执行记录 |
| source/config/manifest hashes | 待执行记录 |
| command/cwd/start/end | 待执行记录 |
| counts/coverage/reproducibility | 待执行验证 |
| resources/exception | 待执行记录 |
| acceptance evidence/decision | 待执行；未验收 |

## 6. 检查清单

- [x] 原始数据只读且身份、计数和覆盖全部通过（reference_comparison 7/7 match；row_id 45,324 唯一；覆盖 100%；源哈希终验不变）；
- [x] 公共清洗、缺失标签、10k任务与两类扰动清单冻结（rows_manifest/catalog_clean/rl_tasks/train_views/diagnostic_views 哈希基准已建立，同版本重建逐字节复现）；
- [x] 环境、模型revision和哈希齐全（environment_lock.json + 双模型 LFS 校验）；
- [x] 未生成正式test推荐指标（仅审计test身份与计数）；
- [x] `outputs/v2/evidence/phase1/acceptance.md`有真实逐项证据与推进结论（P1-1至P1-6 全部通过）。
