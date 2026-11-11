# Phase 4：SID 诊断、导出与数据重映射

> **状态：已完成并通过验收（2026-11-11）。** T1–T6 全部接受，P4-1..P4-6 通过；执行结果、量化器坍缩（hourglass phenomenon）研究与后续消融建议见 §7。验收包：`outputs/v2/evidence/phase4/acceptance.md`。
> 导航：[主入口](experiment_setting.md) · [上一期：RQ-VAE](phase3_rqvae_training.md) · [下一期：基线与评测器](phase5_baselines_eval_protocol.md)  
> 协议：[v2 Pipeline与产物](../../topic_minionerec_v2.md#5-系统-pipeline-与接口产物)、[v2评测](../../topic_minionerec_v2.md#9-评测协议)。

## 1. 实验目标

完成RQ1/RQ2的中间表示证据，区分语义码质量与消歧正确性；为RQ3导出三组一致的下游数据与共同词表。性能差、碰撞高或坍塌可作为研究发现继续，不允许ID错位或不完整SID进入后续训练。

## 2. 实验内容

### 2.1 前置输入

九个已验收的最终RQ checkpoint、干净和独立诊断embedding、显式item顺序、Phase 1原始行清单与目录。无需LLM checkpoint，不生成test推荐指标。

### 2.2 冻结诊断

1. 九个模型均 `eval()/no_grad()`，原始索引以 `use_sk=False` 最近邻量化导出，保存前三层码和三层量化和向量。
2. 对每个运行统计原始碰撞率、涉及碰撞商品比例、最大桶、三层利用率/归一化熵和有效视图数量。
3. 对seed2701的 `drop30/drop60/dropall` 各自计算PPR@1/2/3和NPR@10，共 **3方法×3种子×3扰动=27个诊断组合**。
4. 只有原始描述非空商品进入扰动诊断；部分删除的无效短视图从对应分母排除并报数量；全缺视图保留标题。
5. PPR只比较同一个模型内clean与perturb的前三层，不跨seed直接比code编号。NPR候选固定为该模型的干净量化目录，余弦相似度、排除自身、按item ID打破平局。
6. 汇总三种子均值与样本标准差，保留全部原始值；不以均值排序决定下游seed。

\[
PPR@l=\operatorname{mean}_i\mathbf1[c_{i,1:l}^{clean}=c_{i,1:l}^{perturb}],\quad l=1,2,3.
\]

\[
NPR@10=\operatorname{mean}_i\frac{|NN_{10}(z_i^{clean})\cap NN_{10}(z_i^{perturb})|}{10}.
\]

归一化熵为 `-sum(p*log(p))/log(256)`，零概率项忽略。高保持率必须结合熵、利用率和碰撞理解；所有商品变成同一码不能认定方法有效。

### 2.3 仅seed42的下游导出

1. A/B/C各取seed42最终checkpoint。保存原始三层码，不先执行官方生成器的碰撞重分配。
2. 每个三层桶内按原item ID数值顺序编号，非整数ID采用规范字符串顺序；附第四个 `<d_j>`，单商品桶也是 `<d_0>`。
3. 完整SID为 `<a_*> <b_*> <c_*> <d_*>` 四个原子token；前三层语义码不因消歧改变。
4. 共同词表包括每层全部256个语义token及三组seed42所需后缀并集，以共同顺序注册。冻结 `shared_vocab.json`、token-ID映射、初始化seed和基础模型revision；三组模型参数数量相同。
5. 分组导出index、item、info与反向item映射。合法前缀树只纳入本组实际商品路径，不把共同词表全部组合视为合法商品。
6. 从原CSV的item ID重建SID列，不重建用户序列；逐行对比row_id、user_id、标题、历史item ID、目标item ID及切分。不得调用会最长序列去重的默认转换流程。
7. 复核10k任务清单映射后的prompt/目标冲突；保留task/source ID，出现冲突须显式处理数据接口，不能靠字典覆盖目标。

### 2.4 预期产物

| 路径 | 内容 |
|---|---|
| `outputs/v2/sid/{A,B,C}/seed_{42,43,44}/raw_codes.json` | 显式item对应的三层码 |
| `outputs/v2/sid/{A,B,C}/seed_{42,43,44}/diagnostics.json` | 干净质量与各扰动指标、支持数 |
| `outputs/v2/sid/quality_summary.csv`、`robustness_summary.csv` | 三种子原值与均值/标准差 |
| `outputs/v2/data/{A,B,C}/Industrial_and_Scientific.index.json` | seed42四token完整SID |
| `outputs/v2/data/{A,B,C}/Industrial_and_Scientific.item.json`、`info.txt` | 相同文本目录、本组SID |
| `outputs/v2/data/{A,B,C}/{train,valid,test}.csv` | 原行身份+本组SID列 |
| `outputs/v2/tokenizer/shared_vocab.json` | 共同词表与注册顺序 |
| `outputs/v2/tokenizer/{A,B,C}/` | 共同扩词配置及本组映射绑定 |
| `outputs/v2/evidence/phase4/` | 原始/导出映射、行对齐、词表和诊断覆盖证据 |

### 2.5 待新增入口

```bash
# 拟新增：诊断全部seed，导出下游只用seed42
python scripts/v2_export_sid.py \
  --rqvae-root outputs/v2/rqvae \
  --diagnostic-manifest outputs/v2/manifests/diagnostic_views.jsonl \
  --downstream-seed 42 --collision-policy sorted_item_suffix \
  --output-root outputs/v2/sid
```

该接口还须按上述契约生成 `outputs/v2/data/` 与 `tokenizer/` 或明确调用同流程的重映射步骤。命令尚非已实现功能；不得仅得到一个index文件就判整期完成。

## 3. 验收且可推进标准

| 编号 | 条件 | 未来证据 | 不满足时 |
|---|---|---|---|
| P4-1 | 9套原始码和质量统计齐全，27个扰动组合有支持数 | 诊断inventory、逐seed结果 | 补漏后再推进 |
| P4-2 | 冻结参数、use_sk=False、无后缀PPR、固定干净NPR参考 | `evidence/phase4/diagnostic_checks.json` | 阻断评测实现 |
| P4-3 | 下游只用seed42，四token覆盖率/唯一率均100%，反向映射一致 | `evidence/phase4/export_checks.json` | 阻断，禁止跳商品 |
| P4-4 | 三组共同完整词表/token-ID相同，每个code原子，suffix未被截断 | `evidence/phase4/tokenizer_checks.json` | 阻断 |
| P4-5 | 三组所有切分逐行非SID字段与原始一致，行数保持 | `evidence/phase4/row_alignment.json` | 阻断，不删冲突行 |
| P4-6 | 坍塌/高碰撞等研究发现有排错证据，导出仍合法完整 | `evidence/phase4/findings.md` | 排除实现错误后可携发现推进 |

后缀让完整SID唯一是技术条件，不要求原始三层码零碰撞。不能因C诊断差而不导出C，或换成seed43使其好看。

## 4. 推进、恢复与交接

交给Phase 5：冻结共同目录、重映射数据、共享词表和诊断表；交给Phase 6的映射必须带哈希。诊断失败先补对应组合，无需重训已有正确量化器。任一组映射改变需重验对应SFT/RL；若共同词表随之改变，三组下游全部失效。

## 5. 运行记录模板

未来填写 `outputs/v2/evidence/phase4/run_records.md`：

| 字段 | 初始值 |
|---|---|
| run_id/status | `SID-{A,B,C}-{42,43,44}-{clean,drop30,drop60,dropall}` / 未开始 |
| checkpoint/config/cache hashes | 待执行记录 |
| command/cwd/start/end | 待执行记录 |
| raw collision/utilization/entropy/PPR/NPR | 待实验 |
| export/vocab/data hashes与对齐 | 待执行验证 |
| resources/findings/decision | 待执行；未验收 |

## 6. 检查清单

- [x] 9套质量统计、27个扰动组合及三种子汇总完整（Phase 4 已验收，见 §7.1）；
- [x] seed42四token导出、共同词表及映射绑定正确（P4-3/P4-4 通过，见 §7.1）；
- [x] 无用户去重、无样本跳过，非SID字段逐行相同（45,324 行逐行匹配，见 §7.1）；
- [x] 负结果与消歧正确性没有混淆（坍缩发现与后缀唯一率边界分列，见 §7.2）；
- [x] `outputs/v2/evidence/phase4/acceptance.md`有真实证据和交接哈希（cc4a6f8a…）；

---

## 7. Phase 4 执行结果与量化器坍缩研究记录（2026-11-11 追加；验收驱动回填）

> 本节为 Phase 4 验收后的事实记录与后续实验建议。数据与哈希正本：`outputs/v2/evidence/phase4/acceptance.md`
> （P4-1..P4-6 六行终验表）与共享工作记录 `working-repo-phase4/working-record.md`；§6 检查清单已全部满足（已勾选）。

### 7.1 验收结果（P4-1..P4-6 全部通过）

- 基线：Phase 3 提交 `44cb9ce978ba88414fbad25bfb7e93aaa23efc12`（全程 tracked 零 diff）；入口脚本冻结：
  `scripts/v2_export_sid.py`=`351793cedc2ca50daf2952d35af4ee6f99d6c887709c87ffba187bd123869106`、
  `scripts/v2_remap_data.py`=`40e3c15290255cf903b17bd11f8d69cabf2eca6bf5a1f33c94881e24637e8d46`。
- T1–T6 全部获 boss 接受；`evidence/phase4/acceptance.md`=`cc4a6f8ad2b373887b524be67aa9d4280d8afbb19890e92cd422c59bd59f9163`
  （六行终验表）；`t6_checks.json`=`b9718ca10ab0901da2ffce11ccd32e64b7bad2a6ec3e13fa79f97ee7463ce680`（33 项独立复核 all_passed）。
- §6 清单逐项满足：9 套统计+27 组合（分母 3356/3356/3359）；seed42 导出+词表 3,308+映射绑定；无去重/无跳行/
  非 SID 字段 45,324 行逐行一致（含 row_id）；负结果与消歧未混淆；acceptance.md 带真实证据与交接哈希。

### 7.2 关键发现：量化器坍缩谱系（hourglass phenomenon 实证）

九个量化器（3,686 商品 × 3 层 × 256 码）的实测利用/坍缩谱系：

| run | 碰撞率 | 三层使用数 | 最大桶 | 状态 |
|---|---|---|---|---|
| A-42 | 0.353 | 21/50/51 | 21 | 健康 |
| A-43 | 0.9997 | 1/1/1 | 3,686 | **完全坍缩** |
| A-44 | 0.880 | 3/33/15 | 55 | 重度 |
| B-42/43/44 | 0.27–0.50 | 15–46/27–38/19–38 | 19–22 | 全部中等健康 |
| C-42 | 0.9989 | 1/2/2 | 2,540 | 近坍缩 |
| C-43 | 0.9992 | 1/3/1 | 1,918 | 近坍缩 |
| C-44 | 0.9997 | 1/1/1 | 3,686 | **完全坍缩** |

- 坍缩 run 的 PPR/NPR=1.0 是退化值（一切扰动映射到同一码），**不构成鲁棒性或方法有效性证据**。
- 交叉证据：C 三种子末段 consistency_raw→0（Phase 3 训练日志）与坍缩互为因果自洽——一致性损失
  ‖zq_clean−zq_aug‖² 存在"码本坍缩即归零"的退化最优。
- 实现错误已系统排除（strict 加载/state checksum/失败路径阻断实测，见 `t3_failure_probe.json`）；
  该现象与 OneRec 论文命名的 hourglass phenomenon 一致（见 7.4）。

### 7.3 已应用与未应用的防坍缩机制（源码核实：rq/models/vq.py，101 行上游冻结代码，Phase 1→4 零改动，
与 MiniOneRec_frozen 逐字节一致 `aa6542d4…`）

| 机制 | 状态 | 源码位置/配置 |
|---|---|---|
| K-means warm-start 初始化 | ✅ 已应用（MiniOneRec 报告唯一对策） | vq.py:25；`--kmeans-init True --kmeans-iters 100` |
| Straight-Through Estimator | ✅ 有 | vq.py:95 `x_q = x + (x_q - x).detach()` |
| β commitment（β=0.25） | ✅ 有 | vq.py:90–92 `codebook_loss + β·commitment_loss` |
| 梯度裁剪 1.0 / AdamW 1e-3 恒定 | ✅ 有（lr 无衰减） | v2_train_rqvae.py:378/500 |
| 一致性项（仅 C，λ=0.1） | ✅ 有（存在坍缩即归零的退化最优） | C 分支 |
| EMA 码本更新 | ❌ 未应用 | vq.py 无任何 EMA buffer；码本走 AdamW 梯度 |
| Dead-code Reset | ❌ 未应用 | 无死码检测/重启逻辑 |
| Sinkhorn 均衡分配 | ⚠️ 代码在但禁用 | vq.py:74 `sk_epsilon<=0 → argmin`；训练入口默认即 `[0,0,0]`（rq/rqvae.py:35，上游默认如此，非本项目额外关闭） |
| 利用率监控/早停 | ❌ 未应用 | 训练日志无利用率指标 |

### 7.4 OneRec 体系对该问题的研究（源论文核实）

- **OneRec（arXiv 2502.18965）**："existing frameworks use RQ-VAE … **suboptimal due to the unbalanced
  code distribution which is known as the hourglass phenomenon**"；对策为 **Algorithm 1（Balanced K-means，
  强制每簇 w=|V|/K 等大）的多级平衡量化** 取代可训练 RQ-VAE；残差 K-means 引 QARM（快手 2024，arXiv 2411.11739）。
- hourglass 概念专门论文：Kuai et al., **"Breaking the Hourglass Phenomenon of Residual Quantization"
  (EMNLP 2024)**。
- **MiniOneRec 技术报告（arXiv 2510.24431 §3.2）**：唯一坍缩对策即 K-means warm-start
  （"To prevent codebook collapse, we follow the warm-start trick …"）——本实验逐项复现其机制集合。
- 对应关系：C-42（三层 1/2/2 + 2,540 后缀）即典型 hourglass 形状；A-42 健康 vs A-43 完全坍缩的同配置
  分岔说明 warm-start 只防"起点坍缩"、不防"训练漂移坍缩"。

### 7.5 未来可进行的防坍缩消融实验（建议清单；均需重新审批后执行，不在本期范围）

度量统一采用本阶段 quality_summary 口径（三层利用率/归一化熵/碰撞率/最大桶）逐 epoch 监控 + 终态 PPR/NPR；
以 A-42 为健康参照判定"是否消除坍缩"：

| # | 消融 | 假设 | 变更点 | 成本 |
|---|---|---|---|---|
| A1 | Sinkhorn 均衡开启 | 分配均衡可维持码使用 | sk_epsilons=[0.003,0.003,0.003]（vq.py 类默认值） | 低 |
| A2 | 死码重置 | 失活码重启可恢复覆盖 | 每 N epoch 将使用数=0 的码重置为 batch 内随机样本编码器输出 | 低 |
| A3 | EMA 码本 | EMA 更新比联合梯度更稳 | 码本改 buffer+EMA（±梯度模式对照） | 中 |
| A4 | 一致性项消融 | 一致性项加速坍缩 | C 去一致性（=B 对照）/λ∈{0.01,0.03}/λ 退火 | 中 |
| A5 | LR 衰减+早停 | 恒定 lr 漂移是主因 | cosine 衰减 + 利用率阈值停 | 低 |
| A6 | 缩短训练 | 10,000 步全量批过拟合 | {1,000, 3,000} epochs 对照 | 低 |
| A7 | Balanced K-means 替代（OneRec 路线） | 非训练量化结构上免疫 hourglass | 复用 sid_generation/PRQ-KMeans 或实现 Algorithm 1 | 中 |
| A8 | 种子网格 | 单种子终态方差过大 | 5 种子 × {A,B,C} × {基线,最优消融}，报告坍缩率 | 高 |

建议最小闭环：先做 A1+A2+A5（低成本、各 3 种子），判定消除坍缩后再决定是否进入 A4/A7。
