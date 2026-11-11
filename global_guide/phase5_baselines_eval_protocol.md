# Phase 5：传统基线与评测器

> **状态：已完成并通过验收（2026-11-12）。** boss 终验宣告 T1–T6 全部接受、P5-1 至 P5-5 通过（43 项 worker 检查项完成，无未解决 P0）；实测见 §7 验收记录，证据包在 `outputs/v2/evidence/phase5/`。  
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

> 已填写（2026-11-12）：`outputs/v2/evidence/phase5/run_records.md`=`2897368f…`（T1–T5 全部运行史，废弃缺陷尝试单列不计正式方法）。

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

- [x] 合成样例全部通过，聚类配对和切片口径正确——`metric_checks.json` 21/21（max_abs_error=0.0≤1e-12，`4cd455f5…`）、`bootstrap_checks.json` 9/9（`75921c7d…`），详见 §7.2；
- [x] B0/B1只使用允许数据，配置与最佳checkpoint冻结——B0 频次和=36,259 且 builder 零 valid/test 路径；B1 validation NDCG@10 选模 best@epoch197、resume_used=false（`best_checkpoint.pth`=`08a7c730…`），详见 §7.2；
- [x] item-ID接口和生成式解码适配说明完整——`v2_evaluate.py`=`50eb50e2…`、`v2_generation_adapter.py`=`67be20d4…`、`evaluation_protocol.json`=`98e5896b…`，详见 §7.4/§7.5；
- [x] 没有生成或查看正式test推荐指标——T6 零产物双查（文件名+row_id 前缀语义）通过，详见 §7.3/§7.5；
- [x] `outputs/v2/evidence/phase5/acceptance.md`有真实检查证据和结论——`9be225a0…`，P5-1..P5-5 全过，详见 §7.1。

## 7. 验收记录（2026-11-12）

### 7.1 结论与工作区

boss RESULTS：“完成：T1–T6全部已接受，Phase5通过P5-1至P5-5验收。43个worker检查项完成，最终证据支持交接Phase6/8，无未解决P0。”进入时 HEAD=`fcdec290e59833da53ff05166b90f6ddf71db959`，全程 tracked 零 diff；五个 v2 脚本（`v2_evaluate/v2_bootstrap/v2_generation_adapter/v2_sasrec/v2_popularity`）以 untracked 新文件交付。验收包 `outputs/v2/evidence/phase5/acceptance.md`=`9be225a03efb8682c86e829c5de72c5226493e6f1cc34e632c9bbfc46ca67958`。

### 7.2 P5-1..P5-5 实测回填（对应 §3）

| 编号 | 实测要点 | 证据（SHA-256 前缀） |
|---|---|---|
| P5-1 | 21/21 场景全过，max_abs_error=0.0≤1e-12；首非法占位 NDCG=`1/log2(3)`、重复取首次、Coverage 分母固定 | `metric_checks.json`=`4cd455f5…81cf`（T6 复跑逐字节一致） |
| P5-2 | 9/9：multiplicity 手算、相同预测 CI=[0,0]、独立枚举 CI=[0,0.25] 吻合、混合切片 1502 有效/498 无效、手算分位公式≤1e-12 | `bootstrap_checks.json`=`75921c7d…`、`shared_sampling_summary.json`=`9753e8b1…` |
| P5-3 | 严格 reference 绑定（同 row_id 改 user/换合法 target/缺标签均拒绝）；B0/B1 同 4,532 行/2011 用户；切片按原始缺失标签 417/4115 | `evaluation_protocol.json`=`98e5896b…`（含 33 文件三组绑定与 B1 来源绑定） |
| P5-4 | B0 仅 train 频次（频次和=36,259，builder 零 valid/test 路径）；B1 seed42 单次不中断（resume_used=false）、validation NDCG@10 选模 best@epoch197、patience=20 实测生效、零 test 接触 | B0 `popularity.json`=`79e81576…`、B1 `train_summary.json`=`8a5bddc6…`、`best_checkpoint.pth`=`08a7c730…`、`t5_smoke_checks.json`=`4f3efb7f…` |
| P5-5 | 六件 B0/B1 产物=记录接受值；checkpoint 独立重推理 4,532 行逐位全等；指标独立重算零差异；test 零产物双查（文件名+row_id 前缀） | `artifact_inventory.json`=`b1995962…`、`t6_checks.json`=`68e90bc1…`、`t6_protocol_check.json`=`fb7cb89b…`（9/9，含旧 `[dict,null]` 形状负例拒绝） |

### 7.3 validation 实测要点（完整表=`validation_metrics.json`=`d57d542e…`；无任何 test 指标）

| 基线 | 行 HR@10 | 行 NDCG@10 | 用户宏 NDCG@10 | Coverage@10 | 全局非法率 |
|---|---|---|---|---|---|
| B0 Popularity | 0.054060 | 0.038595 | 0.026999 | 10/3686（0.002713） | 0.0 |
| B1 SASRec | 0.113857 | 0.090282 | 0.064929 | 3084/3686（0.836679） | 0.0 |

资源：B1 正式 wall 80.9s、峰值显存 83,430,400B；废弃缺陷尝试 70.5s 单列、不计正式方法（`resource_summary.json`=`ae233998…`）。

### 7.4 受控偏差与终版更正（均经 boss 裁定/确认）

1. **MultiHeadAttentionV2 mask 修复**：原 `SASRecModules_ori.py` 以 `sign(abs(sum(row)))` 猜 padding——LN 后 fp32 行和恰为 0.0 的行实测 327/2000，真实 item 的注意力被静默清零。boss 判 blocker 并授权最小修复：V2 改用显式 padding 布尔掩码追溯 key/query 有效性，**设计上即不与原版行为等价**；原文件未动（`76ef521d…`）。
2. **patience=20 语义裁定**：第 20 次连续无提升处停止（首次计 1）；原实现在第 21 次才返回，按裁定更正后正式运行。
3. **evaluation_protocol 形状缺陷修复**：bootstrap 边界误写成 `[dict, null]` 元组序列化；修复后终版=`98e5896b…`（pre-fix 历史=`482494c9…` 已被取代），`t6_protocol_check` 负例回归通过。
4. **v2_sasrec 终版哈希**：正式运行版=`8996b3b7…`；坏版 `ef474ab5…`（缺 `import torch`，未通过 smoke）从未进入正式路径，记录误写已更正。
5. **终版脚本哈希**：`v2_evaluate.py`=`50eb50e2…`、`v2_bootstrap.py`=`f4e8bb6a…`、`v2_generation_adapter.py`=`67be20d4…`、`v2_sasrec.py`=`8996b3b7…`、`v2_popularity.py`=`648f8fef…`。
6. **acceptance.md 转写 slips（正文未改，注记在案）**：metric_checks 63 位哈希漏“7”（文内已自附更正注）；`t6_protocol_check.py` 现盘=`0abb52dd…`、`.json` 现盘=`fb7cb89b…`（文内句式归属相反，本节按 2026-11-12 远端 sha256sum 实测更正）。

### 7.5 交接状态（→ Phase 6/8）

- **基线加载**：B0=读 `outputs/v2/baselines/B0/popularity.json` 的 `ordering`（3,686 完整排序，与历史无关）；B1=`torch.load(best_checkpoint.pth)` → `scripts/v2_sasrec.py:build_model()` + `load_state_dict` → `forward_eval`（重载推理已验证逐位一致）。
- **评价契约**：`v2_evaluate.py` 行接口 + 严格 reference 模式（真值只来自冻结源，禁止从预测自建）；切片标签=quality_features `original_description_missing`；不得退回仅 row_id 兼容模式。
- **adapter 契约已锁**：组绑定/四原子 token+EOS/trie/非法保位/Beam10 开发·Beam50 正式返回 50/确定性生成参数；真实 LLM 未加载，生成非法率闸门（=0）留 Phase 6/8。
- **版本失效范围**：`data/A` 变更→B0/B1 失效；quality_features 变更→切片标签失效；catalog/index 变更→目录/负采样/评测全失效；`v2_evaluate` 变更→所有指标按 T2 规则统一重算。
- **正式路径纪律**：仅接受不中断运行（resume_used=false）；加载推理可恢复 ≠ 训练精确续训（B1 checkpoint 仅 state_dict+meta，Phase 3 恢复限制不外推）。
