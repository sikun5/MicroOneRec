# Phase 6：三组全参数 SFT（已完成并通过验收，2026-11-15）

> **状态：已完成（A/B/C 三组正式训练与独立终验通过，P6-1..P6-6 有界证据验收）。**
> 导航：[主入口](experiment_setting.md) · [上一期：基线与评测器](phase5_baselines_eval_protocol.md) · [下一期：GRPO](phase7_grpo_training.md)
> 协议：[v2 Tokenizer与SFT](../../topic_minionerec_v2.md#7-tokenizersft-与-grpo)。

## 1. 实验目标与完成结论

完成 `A-SFT/B-SFT/C-SFT` 三次正式全参数训练，建立RQ1/RQ2下游比较及RQ3后训练的起点。正式test留到Phase 8；本期无任何正式test预测/指标（终验双通道扫描核验）。

| 组 | 停止点 | best validation loss（选模依据） | wall（单 MIG 切片） |
|---|---|---|---|
| A-SFT | step 546/770（epoch 7.09），exit 0，patience=3 | **1.239202**@checkpoint-429 | 51,240.86s（14.23h） |
| B-SFT | step 546/770（epoch 7.0898），exit 0，patience=3 | **1.223916**@checkpoint-429 | 48,045.17s（13.35h） |
| C-SFT | step 468/770（epoch 6.077），exit 0，patience=3 | **1.253765**@checkpoint-351 | 41,202.33s（11.45h） |

三组顶层 `model.safetensors` 与 best checkpoint 权重逐字节相等（A=`f753205d…`、B=`90fe4eb3…`、C=`2d5ac0be…`），完整 SHA 见 `selected_checkpoints.json`=`e902a6f4…`。

## 2. 实际执行记录

### 2.1 前置输入（已核验）

Phase 4三组 seed42 SID映射、共同完整词表（base 151669+共同 3308=154977）、重映射CSV和row清单；Phase 5通过验证的item-ID评价器与目录协议；同一 Qwen3-0.6B-Base revision=`da87fb608c14b7cf20ba1ce41287e8de496c0cd`（safetensors=`cd2a5120…`）。三组初始完整权重（进程内实测 B/C、同种子同路径重构核验 A）全等=`820b20708d8554bee82f153f1d8f93454e369cd378c9ab6a20ee8ea33384e161`。

### 2.2 训练入口适配（实际采用 `scripts/v2_sft.py` 受控入口）

1. 共同词表按冻结注册序（shared_vocab=`95779752…`）扩充三组tokenizer并 `resize_token_embeddings(154977)`；逐项对账 `token_id_map.json` 0 失配；模型 vocab 151936 的 267 legacy 行未动+新增 3041 行。初始化 seed42 三组一致（state_dict SHA 全等，见 2.1）。
2. 数据面用 `scripts/v2_sft_data.py`（canonical=`9bf546d9…`）三类 V2 数据类替换 legacy 类：三组同任务来源、同原文本、同样本顺序，仅 SID 列不同（跨组列级对照：仅 history_item_sid/item_sid 两列不同）；title2sid 采用 canonical（每唯一标题一条、数值最小 item ID）=3646 条；三任务计数=36259/7332/36259×3。
3. labels：prompt/padding=-100、目标四 token+EOS 参与损失；cutoff 512 截断不产生空监督（`label_integrity` 全过）。
4. 全参数：`requires_grad=True`×3，trainable=total=**599,163,904**，tie_word_embeddings=true，无 PEFT/LoRA/QLoRA。
5. **历史入口版本差异披露**：A-SFT 用 T5 版入口（SHA=`6c786573…`，fire→argparse 等价接线）；B/C 用 T6 终版（SHA=`1038df67…`，新增运行时 init SHA 直写 run_meta、全量 log_history 落盘、VRAM 峰值实测三项持久化改善，训练语义不变）。提交入库版本=终版 `1038df67…`；A 运行时为旧版的事实由 Phase6 evidence 锚定。

### 2.3 独立100样本调试（已通过）

- 压测：512 强制长度探针下 micro16/8/4 全组 OOM，**三组共同可行=micro2**（probe 峰值=8,024,368,640B，仅独立参考）；reload 三组通过。
- smoke（60ep/eval≤0.5 停）：三组 loss 15.71→1.367/0.846/0.355 下降；约束 Beam10 完整 SID 命中 7/36/34；非法率 0×3；reload abs_diff=0.0×3。
- **严格生成契约（冻结）**：原始输出恰为 [4 原子 SID token, EOS]，无换行、无后处理；三组 100/100 全过 Phase5 冻结 validator（`validate_sid_sequence`）。调试变体（nl-allowed）不用于正式声明。
- smoke/调试全部产物隔离于 `outputs/v2/debug/phase6/sft_smoke/attempt_001/`，未进入正式路径。

### 2.4 实际正式配置（三组相同）

| 参数 | 实际值 |
|---|---|
| Base/seed | 本地路径 Qwen3-0.6B-Base（revision 同上），seed42 |
| 参数更新 | 全参数，BF16，AdamW（adamw_torch），LR `3e-4`，warmup20+默认线性调度 |
| 有效batch | 1024；**micro=2**（T4 冻结三组共同可行），梯度累积 512 |
| 验证batch | eval_micro=4（eval fp32 logits 降载后锁定） |
| 长度/上限 | cutoff 512 / 10 epochs |
| 验证/保存 | 5% 计划进度（实测间隔恒 39 步）；save_total_limit=1；load_best_model_at_end（metric=eval_loss） |
| 停止 | EarlyStopping patience=3；最低 validation SFT loss 选模 |
| 数据顺序 | 三任务 ConcatDataset 一次 shuffle（seed42）；验证=主 SID 推荐任务 |
| 计划步 | 770（77 updates/epoch floor，尾部部分累积窗丢弃——**源码推导 trainer.py:5303**，逐批未插桩=未知） |

命令入口（实际）：`.venv-phase3/bin/python scripts/v2_sft.py --train_file outputs/v2/data/{G}/train.csv --eval_file outputs/v2/data/{G}/valid.csv --output_dir outputs/v2/checkpoints/{G}-SFT --batch_size 1024 --micro_batch_size 2 --eval_micro 4 --num_epochs 10 --learning_rate 3e-4 --cutoff_len 512 --seed 42`（G=A/B/C；其余参数同上表默认）。

### 2.5 输出与证据（实际产物）

- 正式 checkpoint：`outputs/v2/checkpoints/{A,B,C}-SFT/`（权重+tokenizer+manifest；顶层权重=best，逐字节）。
- 严格生成独立核验：valid head(100) 确定性清单，strict=100/100×3、hit@1=**7/4/5**、换行=0（`final_check.json` A=`5121e500…`/B=`8ed74d7d…`/C=`49a18cdf…`）；T8-R2 补持久化逐条 raw（`formal_valid_raw_outputs.json` A=`32ae2110…`/B=`1a0fb1f8…`/C=`e928d37e…`，独立 reference 交叉一致）。T4 smoke train100 命中 7/24/32 为另一套独立证据，不可与 valid100 数值混用。
- 证据五件（`outputs/v2/evidence/phase6/`）：`selected_checkpoints.json`=`e902a6f4…`、`resource_summary.json`=`986a20ba…`、`artifact_inventory.json`=`da0bc73f…`（含 r2_supplement 段）、`run_records.md`=`e9f5e94b…`、`acceptance.md`=`e562ff39…`。独立终验 `t8_checks.json`=`8a73a0c8…`（27/27 全过）。

## 3. 验收标准与实际结果（P6-1..P6-6 全过）

| 编号 | 条件 | 证据 | 实际结果 |
|---|---|---|---|
| P6-1 | 三组共同词表/token-ID/初始权重、revision、参数量一致 | `initialization_checks.json`=`766f8331…` | 全等 820b2070…；599,163,904×3 |
| P6-2 | 全参数、label/四token完整、任务来源一致 | `data_gradient_checks.json`=`4c7ba0b4…` | requires_grad 全 True；跨组仅 SID 两列不同 |
| P6-3 | 100样本调试 loss 下降/可学习回放/合法生成 | `smoke_report.json`=`1fc5cb1e…` | 全 gate 过；严格形态 100/100×3 |
| P6-4 | 三组相同上限/停止政策 | 训练日志+trainer_state | 全 exit0 早停，同政策 |
| P6-5 | 最低 val loss 选模且可加载 | `selected_checkpoints.json`=`e902a6f4…` | 顶层=best 逐字节×3；strict 100/100×3 |
| P6-6 | 步数/成本/异常齐全，test 未评测 | run_records/资源表 | 齐全；test 零接触 |

## 4. 已接受限制（继续披露，不得由他组填补）

1. **A 组统计缺项**：运行时进程内 init hash 未采集（同种子同路径重构核验=820b2070 等值）；运行 VRAM 峰值未知；best 后三次 eval 中一次数值不可恢复。B/C 运行时实测 init=820b2070…（进程内）、VRAM 峰值=6,539,923,456B（实测）。
2. **尾窗/消费口径**：77 updates/epoch 与尾部部分累积窗丢弃为源码推导（transformers trainer.py:5303）；逐批直接计数未插桩=未知。
3. **C 组 seed42 语义码近坍塌**：保留研究发现；C 的 loss/hit 不与 A/B 比较优劣，不因 SFT loss 排序解释推荐优势。
4. **资源口径**：GPU=A800 MIG 1g.10gb 切片（torch 实测 total=10,200,547,328B），**非完整 80GB**；wall 为单 MIG 切片实测，非完整卡等效算力；T4 调试≈1.5h 混合估计（逐 attempt GPU-busy 未知）。
5. **test 零接触**：终验为文件名级双模式扫描（范围=checkpoints/evidence/smoke 三根），非内容级数据流证明——如实披露。

## 5. 交接（Phase 7 输入，已就绪）

- 三组选定 checkpoint+tokenizer：`outputs/v2/checkpoints/{A,B,C}-SFT/`（完整 SHA 见 `selected_checkpoints.json`=`e902a6f4…`）。
- SID/词表/映射：各组 data 面 6 文件+shared_vocab/token_id_map/binding_manifest。
- 任务 manifest：`task_manifests/` 9 件（canonical 计数 36259/7332/36259）。
- 严格生成形态=4 原子 token+EOS；Beam50 正式效果测试留 Phase 8；validation 仅诊断，正式 test 统一 Phase 8。
- 单组 SFT checkpoint 变化将使其对应 RL 失效；共同词表改动影响全部 SFT/RL。

## 6. 检查清单（全部完成）

- [x] 三组同源文本与任务，仅SID不同；
- [x] 全参数、共同扩词、label及smoke检查通过；
- [x] 三次正式SFT完成且最佳validation loss checkpoint可核对（顶层=best 逐字节）；
- [x] 实际训练量/成本记录完整，无正式test指标；
- [x] `outputs/v2/evidence/phase6/acceptance.md`=`e562ff39…` 有真实证据和三个交接checkpoint。

> 验收主体与整改全史见 `outputs/v2/evidence/phase6/{run_records.md,acceptance.md}` 与 Phase7 工作记录锚点；本文仅回写已验收事实。
