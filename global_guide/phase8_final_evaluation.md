# Phase 8：最终评测与统计分析

> **初始状态：未开始。**本期是统一生成和查看正式test推荐指标的阶段。  
> 导航：[主入口](experiment_setting.md) · [上一期：GRPO](phase7_grpo_training.md) · 下一期：无，交付课程报告与答辩  
> 协议：[v2评测协议](../../topic_minionerec_v2.md#9-评测协议)、[v2结果与成本模板](../../topic_minionerec_v2.md#11-结果表与资源记录模板)。

## 1. 实验目标

以相同test行、目录和统计口径，完成B0/B1及六个LLM模型的效果评测，将数据质量、SID机制与推荐结果连接成RQ1/RQ2/RQ3证据链。保留预定比较、非显著差异和负结果，不在看到test后调模型。

## 2. 实验内容

### 2.1 前置输入与最终冻结

1. 核对Phase 1至7验收记录；必须有9个RQ最终checkpoint、三组诊断与三个seed42映射、B0/B1、三个SFT与三个GRPO权重。
2. 冻结模型、tokenizer、映射、共同目录、test row_id、代码、配置和评测器哈希，生成 `evaluation_lock.json`。
3. 确认此前只审计test数据结构，没有用test指标选模型；若发生提前查看，须明确记录，不能继续声称完全未触碰测试集。
4. 所有模型使用同一4,533行test、1,987用户；目标缺失描述行参考447，实际以冻结manifest核验。不得只保留各模型交集中的“容易样本”。

### 2.2 正式预测与解码

| 方法 | 预测方式 | 统一要求 |
|---|---|---|
| B0 | 固定训练频次排序 | 相同目录，不屏蔽历史 |
| B1 | 冻结SASRec完整目录打分 | 相同目录，不屏蔽历史 |
| A/B/C-SFT | 本组映射 constrained beam search | Beam=50、四token路径、确定性 |
| A/B/C-RL | 同上，使用最终两epoch权重 | 同一评测协议 |

生成式配置：返回50候选，`length_penalty=0`、`do_sample=False`、`top_k=None`、`top_p=None`、`temperature=1.0`。原CLI不支持所有参数，必须先通过Phase 5适配的generate调用强制，而非假定模型默认配置正确。

预测保留原始SID、row_id、user_id、target item与ranked item IDs。逐组用本组info反向映射；非法候选保留位置，重复候选使用首次命中，不删后前移。四token+EOS应在16新token内结束；无法结束、候选数不足或缺行先诊断，不能静默过滤。

batch从8开始，OOM可降4/2/1；Beam始终50。记录每方法耗时、峰值显存与test行/秒，保留中断日志。预测恢复时逐row_id检查，不能因合并结果重复计行。

### 2.3 正式评测命令接口

既有 `evaluate.py` 在Phase 5完成身份/生成适配后，逐模型使用模板：

```bash
python evaluate.py \
  --base_model outputs/v2/checkpoints/C-RL/final_checkpoint \
  --info_file outputs/v2/data/C/info.txt \
  --category Industrial_and_Scientific \
  --test_data_path outputs/v2/data/C/test.csv \
  --result_json_data outputs/v2/predictions/C-RL.json \
  --batch_size 8 --num_beams 50 --max_new_tokens 16 \
  --length_penalty 0.0 --seed 42

# 拟新增统计入口，须在Phase 5完成合成验证后使用
python scripts/v2_evaluate.py \
  --prediction-root outputs/v2/predictions \
  --row-manifest outputs/v2/manifests/rows_manifest.jsonl \
  --quality-manifest outputs/v2/manifests/quality_features.jsonl \
  --bootstrap-unit user --bootstrap-repeats 2000 --seed 42 \
  --output-root outputs/v2/results
```

### 2.4 排序指标与自然切片

- 主要指标NDCG@10；补充HR/NDCG@3/5/10、Coverage@10、非法SID率、重复率和吞吐。
- 单目标命中第r位（从1起）时，NDCG@K为 `1/log2(r+1)`，未命中为0；主表对测试行平均。
- 用户宏平均另报：先算每用户行均值，再对用户平均，不能与行平均混名。
- Coverage为前10合法候选唯一商品数/共同完整目录数；非法率和重复率同时报告。
- 自然切片按目标商品清洗前的描述缺失标记划分；整体、缺失、非缺失各报行数、用户数、NDCG@10、HR@10和差值区间。
- 目标缺失不代表历史全部缺失，不作“缺失造成性能变化”的因果陈述。

### 2.5 配对用户聚类 Bootstrap

seed42，2,000次。每轮有放回抽1,987个用户簇，保留用户全部test行和重复次数；同一重采样用于全部方法，按行加权重新计算指标差。

正式预定比较共有9项：

| 阶段 | 配对比较 |
|---|---|
| SFT | B-SFT−A-SFT、C-SFT−B-SFT、C-SFT−A-SFT |
| RL | B-RL−A-RL、C-RL−B-RL、C-RL−A-RL |
| 后训练增量 | A-RL−A-SFT、B-RL−B-SFT、C-RL−C-SFT |

对主要NDCG@10及辅助HR@10报告2.5%/97.5%分位区间。自然切片使用相同用户抽样后再筛目标行；无有效行轮次剔除并报告有效次数，不填0。缺失切片支持较少，不能只给点估计。

区间只反映固定模型下用户采样不确定性；量化器三种子均值/标准差与LLM单种子区间单独报告。不得把它们合并解释成端到端多种子稳定性。

### 2.6 汇总SID机制、成本与RQ结论

导入Phase 4的9个量化器质量统计及27个扰动组合，保留逐seed数据，按扰动类型报均值和样本标准差。对照推荐结果分析：

- **RQ1：**B−A是字段分离和质量融合整体变化，不能独立证明权重优于等权。
- **RQ2：**C−B若保持率提高但熵下降/碰撞升高，应讨论过度平滑或坍塌；如果SID更稳而推荐无增量，也要明确写出。
- **RQ3：**分别看SFT和RL的差异及自然切片；RL−SFT含额外训练预算，不能只归功于目标函数。
- 不声称完成扰动目录下的推荐重训练、冷启动、线上CTR/CVR或全球首次创新。

资源汇总覆盖公共编码、诊断编码、9次RQ、3次SFT、3次GRPO、基线和正式推理。共享缓存只计一次，C额外视图/前向显式计费；实际wall time、GPU-hours、显存与存储分开。

### 2.7 预期产物

| 路径 | 内容 |
|---|---|
| `outputs/v2/evidence/phase8/evaluation_lock.json` | 最终冻结来源与协议 |
| `outputs/v2/predictions/{B0,B1,A-SFT,B-SFT,C-SFT,A-RL,B-RL,C-RL}.json` | 原预测及完整row_id/item-ID排名 |
| `outputs/v2/results/per_row_metrics.jsonl` | 逐模型逐行HR/NDCG与诊断 |
| `outputs/v2/results/main_results.csv`、`user_macro_results.csv` | 行主表与用户宏平均 |
| `outputs/v2/results/slice_results.csv`、`paired_differences.csv` | 三切片、9项差值与区间 |
| `outputs/v2/results/bootstrap/` | 配对用户抽样设置、分布及有效轮次 |
| `outputs/v2/results/sid_quality.csv`、`sid_robustness.csv` | 三种子质量、27组合诊断 |
| `outputs/v2/results/resources.csv` | 全流程成本与共享账目 |
| `outputs/v2/results/research_findings.md` | 按RQ组织的正/负结果与限制 |
| `outputs/v2/evidence/phase8/` | 完整性、复核、运行和最终验收记录 |

报告可用三组图：SID稳定性/坍塌诊断、推荐整体及切片对照、效果—成本关系。图必须来自上述表格，不从论文复制成个人结果。

## 3. 验收且可推进标准

本期通过即进入课程报告/答辩交付，没有新的模型训练阶段。

| 编号 | 条件 | 未来证据 | 不满足时 |
|---|---|---|---|
| P8-1 | 八方法各覆盖同4,533行，row_id不缺不重，目录和标签一致 | `evidence/phase8/prediction_inventory.json` | 补正确预测，不过滤样本 |
| P8-2 | 六LLM均Beam50/确定性/完整四token，非法率0 | `evidence/phase8/decoding_checks.json` | 先修技术错误，暂停效果结论 |
| P8-3 | 主表行平均、用户宏平均、切片支持数与指标复核一致 | `evidence/phase8/metric_review.json` | 修复并重算所有受影响方法 |
| P8-4 | 2,000次用户聚类、同簇全行、方法配对及9项比较完整 | bootstrap数据与checks | 阻断统计结论 |
| P8-5 | SID9运行/27组合、成本和所有正式模型记录齐全 | inventory与资源表 | 补缺项，不缩小矩阵冒充完成 |
| P8-6 | RQ解释、负结果和限制完整，无测试调参或因果/线上外推 | research_findings与审阅记录 | 修正文案/披露偏差 |

结果无显著提升、RL下降或C坍塌不影响技术验收通过；完整报告这类结果即形成有效研究交付。非法候选、数据错位或未完成正式实验不能用“负结果”替代。

## 4. 恢复、修复与最终交接

推理中断后按row_id补未完成部分，再验证总行数；保留attempt和生成配置，不将同一行的多个预测择优合并。评测代码错误允许修复后对全部受影响方法统一重算，记录修复及已查看test的事实；不得据新结果调训练参数。模型/词表变化必须回溯相关阶段，不在本期临时更换checkpoint。

交付报告与答辩时附：研究协议版本、完整矩阵完成情况、主表与区间、SID机制图、切片支持数、成本表、可复现命令和贡献归属。checkpoint可外部存储，但其路径、哈希和生成方法必须可追踪。

## 5. 运行记录模板

未来填写 `outputs/v2/evidence/phase8/run_records.md`：

| 字段 | 初始值 |
|---|---|
| run_id/status | `TEST-{八个方法}`、`P8-BOOTSTRAP-42` / 未开始 |
| checkpoint/vocab/data/protocol/code hashes | 待执行记录 |
| command/cwd/start/end | 待执行记录 |
| prediction rows/decoding config | 待执行验证 |
| metrics/slices/paired CI | 待实验 |
| timing/VRAM/throughput/storage | 待执行记录 |
| repair/findings/evidence/decision | 待执行；未验收 |

## 6. 检查清单

- [ ] 八方法完整、同目录同测试行，六LLM生成合法；
- [ ] 主指标、切片和用户宏平均分开，9个预定比较均有结果；
- [ ] 2,000次配对用户聚类Bootstrap实现正确；
- [ ] 量化器三种子、全部扰动组合和资源账目齐全；
- [ ] 负结果、修复与解释边界完整；
- [ ] `outputs/v2/evidence/phase8/acceptance.md`有真实证据，交付课程报告与展示。
