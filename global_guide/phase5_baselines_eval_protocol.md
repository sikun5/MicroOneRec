# Phase 5：传统基线与评测器

> **初始状态：未开始。**  
> 导航：[主入口](experiment_setting.md) · [上一期：SID诊断导出](phase4_sid_diagnostics_export.md) · [下一期：SFT](phase6_sft_training.md)  
> 协议：[v2主矩阵](../../topic_minionerec_v2.md#8-主实验矩阵与受控比较)、[v2推荐评测](../../topic_minionerec_v2.md#93-推荐排序评测)。

## 1. 实验目标

在昂贵的LLM训练前，验证所有模型共同使用的item-ID排序、自然切片和统计口径；训练并冻结B0/B1。此时可以评测validation，不运行正式test推荐指标。基线与后续模型对比是系统参照，不据此推断生成式架构在等知识条件下必然更优。

## 2. 实验内容

### 2.1 前置输入

Phase 4验收后的共同目录、row_id、重映射数据和词表；Phase 1原始缺失标签与输入哈希。纯合成测试可提前开发，但正式B0/B1训练和本期验收依赖冻结输入。

### 2.2 评测接口与原脚本适配

- 拟新增 `scripts/v2_evaluate.py`，统一接受每行 `row_id/user_id/target_item_id/ranked_item_ids`、合法目录与原始缺失标签。原始生成SID另行保留以诊断非法项。
- 适配 `evaluate.py`：显式确定性生成、逐行身份、四token路径、本组info/映射；不能把原 `calc.py` 的汇总打印直接替代完整v2统计。
- 主口径不屏蔽历史商品。非法候选保留原排名占位；重复目标按首次出现计命中，不删除后前移。
- HR/NDCG在相同test行上按行平均；用户宏平均另列；Coverage以共同完整目录为分母。
- Bootstrap按user簇抽样、保留全部行与重复次数；方法之间使用同一抽样清单。
- 自然切片依据目标商品原始描述缺失，不按清洗/扰动后文本或历史是否缺失分类。

### 2.3 必须手工核对的合成样例

| 样例 | 预期 |
|---|---|
| target在第1位 | HR@3/5/10=1，NDCG@3/5/10=1 |
| target在第3位 | HR@3=1，NDCG@3=`1/log2(4)=0.5` |
| target不在前K | 对应HR/NDCG=0 |
| 第1位非法、第2位target | 不移位；NDCG@3=`1/log2(3)` |
| target在第2与第3位重复 | 使用第2位，不重复计收益，重复率按唯一数计算 |
| 历史包含target | 不因已见而删除target |
| 同用户2行、另一个用户1行 | 行平均与用户宏平均可不同；抽中前用户时2行一起出现 |
| 一个用户簇被抽中两次 | 该用户全部行按两份权重进入差值，不去重 |
| 两模型预测完全相同 | 所有配对差值为0，区间端点也为0 |
| 切片一轮无有效行 | 标无效并计数，不填0冒充区间样本 |
| 多方法重复推荐同一商品 | Coverage只计唯一合法商品，分母不随模型改变 |

合成HR、NDCG、Coverage的纯数值检查默认绝对误差不超过 `1e-12`；这不要求GPU训练逐位可复现。禁止在本期以真实test结果检查指标是否“看起来合理”。

### 2.4 B0：Popularity

- 只计train目标商品频次，不使用validation/test频次；目录中无train目标次数的商品记0。
- 同频按规范item ID顺序排序；不按用户历史过滤。
- 保存频次向量、完整目录顺序、构建配置与哈希。无训练checkpoint不等于不需要可追踪产物。

### 2.5 B1：SASRec

承接既有SASRec架构，适配原始row集合及item-ID输出。用原train序列，validation选模，test不参与。作为执行默认值固定seed42、hidden32、单head、dropout0.3、batch1024、Adam学习率`1e-3`、weight decay`1e-5`、最多500epochs、每epoch验证一次、patience20。

采用既有正/负样本BCE方式；每训练目标抽一个非目标目录商品作为负例，随机种子固定并记录采样实现，不将test目标加入排除集合。保留现有因果attention和padding处理，关闭与本研究无关的DRO项。以validation NDCG@10选择checkpoint，无超参网格搜索。若适配后使用的模型细节与既有实现不同，先记录差异并使基线配置显式可追踪，不借机扩大实验矩阵。

### 2.6 预期产物与入口

| 路径 | 内容 |
|---|---|
| `outputs/v2/baselines/B0/popularity.json` | 训练频次、目录和排序 |
| `outputs/v2/baselines/B1/best_checkpoint.pth` | 最佳validation NDCG@10权重 |
| `outputs/v2/baselines/{B0,B1}/config.json` | 数据/代码/参数哈希 |
| `outputs/v2/evidence/phase5/metric_checks.json` | 合成样例手工答案对照 |
| `outputs/v2/evidence/phase5/bootstrap_checks.json` | 聚类、配对和切片分母检查 |
| `outputs/v2/evidence/phase5/validation_metrics.json` | 仅validation结果 |
| `outputs/v2/evidence/phase5/evaluation_protocol.json` | 正式评价协议及代码哈希 |

`sasrec.py`和`evaluate.py`已有，但需适配；统一评测统计入口待新增。只在接口实现和合成验证后运行validation。最终test统一等到Phase 8。

## 3. 验收且可推进标准

| 编号 | 条件 | 未来证据 | 不满足时 |
|---|---|---|---|
| P5-1 | 合成排序、非法占位、重复与Coverage均等于手工答案 | metric_checks | 阻断，先修评价器 |
| P5-2 | 行平均/宏平均分开，用户簇及配对抽样正确 | bootstrap_checks | 阻断统计实现 |
| P5-3 | 同目录、不屏蔽历史、目标与缺失切片标签正确 | evaluation_protocol | 阻断协议不一致 |
| P5-4 | B0只用train频次，B1只用validation选模 | 基线配置、日志、checkpoint | 阻断数据泄漏 |
| P5-5 | B0/B1产物和评价代码冻结、可恢复，test未评测 | evidence与哈希 | 补齐或说明违规后暂停 |

SASRec validation效果低但实现检查通过，可携带研究发现推进；若loss非有限、负例等于目标或attention错误则属于阻断。

## 4. 推进、恢复与交接

把已冻结的基线、评测协议和合成测试报告交给Phase 6/8。本期可生成基线validation预测，但test预测留空。中断只恢复未完成的基线训练；指标代码修复后对所有合成样例和validation方法统一重算，不只修有利项。

## 5. 运行记录模板

未来填写 `outputs/v2/evidence/phase5/run_records.md`：

| 字段 | 初始值 |
|---|---|
| run_id/status | `P5-METRIC-CHECKS`、`B0`、`B1` / 未开始 |
| code/data/catalog/protocol hashes | 待执行记录 |
| command/cwd/start/end | 待执行记录 |
| baseline config/selected epoch/checkpoint | 待执行记录 |
| validation metrics/synthetic checks | 待实验/待验证 |
| resource/exception/decision | 待执行；未验收 |

## 6. 检查清单

- [ ] 合成样例全部通过，聚类配对和切片口径正确；
- [ ] B0/B1只使用允许数据，配置与最佳checkpoint冻结；
- [ ] item-ID接口和生成式解码适配说明完整；
- [ ] 没有生成或查看正式test推荐指标；
- [ ] `outputs/v2/evidence/phase5/acceptance.md`有真实检查证据和结论。
