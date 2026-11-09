# Phase 1：数据与环境准备

> **初始状态：未开始。**审计参考值来自v2，须在本期重新验证。  
> 导航：[主入口](experiment_setting.md) · 上一期：无 · [下一期：质量与表示](phase2_quality_embeddings.md)  
> 协议：[v2 数据与环境](../../topic_minionerec_v2.md#3-数据与环境协议)。运行路径均相对 `project/MiniOneRec/`。

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

## 4. 推进、恢复与交接

P1-1至P1-6通过后，将环境锁和全部manifest的路径/哈希交给Phase 2。中断后复核源哈希和已生成清单，仅补缺失部分。源数据或清单发生变化时，后续依赖缓存和训练全部标记需重新验收。

## 5. 运行记录模板

未来填写 `outputs/v2/evidence/phase1/run_records.md`：

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

- [ ] 原始数据只读且身份、计数和覆盖全部通过；
- [ ] 公共清洗、缺失标签、10k任务与两类扰动清单冻结；
- [ ] 环境、模型revision和哈希齐全；
- [ ] 未生成正式test推荐指标；
- [ ] `outputs/v2/evidence/phase1/acceptance.md`有真实逐项证据与推进结论。
