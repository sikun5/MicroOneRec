# Phase 4：SID 诊断、导出与数据重映射

> **初始状态：未开始。**  
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

- [ ] 9套质量统计、27个扰动组合及三种子汇总完整；
- [ ] seed42四token导出、共同词表及映射绑定正确；
- [ ] 无用户去重、无样本跳过，非SID字段逐行相同；
- [ ] 负结果与消歧正确性没有混淆；
- [ ] `outputs/v2/evidence/phase4/acceptance.md`有真实证据和交接哈希。
