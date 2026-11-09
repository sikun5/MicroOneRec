# Phase 6：三组全参数 SFT

> **初始状态：未开始。**  
> 导航：[主入口](experiment_setting.md) · [上一期：基线与评测器](phase5_baselines_eval_protocol.md) · [下一期：GRPO](phase7_grpo_training.md)  
> 协议：[v2 Tokenizer与SFT](../../topic_minionerec_v2.md#7-tokenizersft-与-grpo)。

## 1. 实验目标

完成 `A-SFT/B-SFT/C-SFT` 三次正式全参数训练，建立RQ1/RQ2下游比较及RQ3后训练的起点。正式test留到Phase 8，不按test挑模型。

## 2. 实验内容

### 2.1 前置输入

Phase 4三个seed42 SID映射、共同完整词表、重映射CSV和row清单；Phase 5通过验证的item-ID评价器与目录协议；同一Qwen3-0.6B-Base revision。不能用RQ seed43/44替换表现差的seed42。

### 2.2 训练入口适配

1. 从共同词表按相同顺序扩充三组tokenizer和模型embedding，初始化seed42一致；不仅注册本组实际出现的token，以免参数数量不同。
2. 保留 `SidSFTDataset/SidItemFeatDataset/FusionSeqRecDataset`，三组同任务来源、同原文本、同样本顺序，只替换SID。
3. 保存SFT任务来源清单，检查四token（含suffix）在主任务和对齐任务中均完整；合法文本→SID对齐不等于泄漏，但不能把目标SID提前放进其预测prompt。
4. 校验labels：prompt和padding为-100，目标及正确EOS参与损失；最大长度512不得把整个目标截空。
5. 确认所有模型参数 `requires_grad=True`，`freeze_LLM=False`，无PEFT/LoRA/QLoRA；保存trainable/total参数数。

### 2.3 独立100样本调试

从train中固定选100个无冲突的推荐样本，清单单独保存，不涉及validation/test。先验证全量标签、梯度、保存、加载和完整SID回放；至少记录初始与结束的同一批样本loss、训练样本完整SID命中和非法率。对固定batch的loss必须下降，完整SID生成须有可验证的可学习样例且非法率为0；若无法过拟合，应先检查mask、词表、目标截断和数据冲突，而不是直接启动完整训练。

smoke允许为调试增加小数据epochs，但不进入正式checkpoint、选模或结果表，不能把smoke当成三次正式运行的一部分。A/B/C至少各做输入、词表、label和梯度检查，避免只验证A就假定其他映射正确。

### 2.4 固定正式配置

| 参数 | 共同值 |
|---|---|
| Base/seed | `Qwen/Qwen3-0.6B-Base`，seed42 |
| 参数更新 | 全参数，BF16，AdamW |
| 有效batch | 1024，初始micro16、梯度累积64 |
| LR/长度 | `3e-4` / 512 |
| 最长训练 | 10epochs |
| 验证/停止 | 沿用既有5%计划进度验证规则，patience3次；三组同规则 |
| 选择 | 最低validation SFT loss对应checkpoint |

记录实际验证间隔的optimizer steps、停止epoch、训练样本和token数。相同上限与early stopping规则不意味着实际FLOPs相等。

### 2.5 命令模板与输出

`sft.py`参数已有，但共同词表/完整SID与任务适配是前置要求；不是未经适配即可满足协议。分别将C替换为A/B/C：

```bash
python sft.py \
  --base_model Qwen/Qwen3-0.6B-Base \
  --train_file outputs/v2/data/C/train.csv \
  --eval_file outputs/v2/data/C/valid.csv \
  --sid_index_path outputs/v2/data/C/Industrial_and_Scientific.index.json \
  --item_meta_path outputs/v2/data/C/Industrial_and_Scientific.item.json \
  --category Industrial_and_Scientific \
  --batch_size 1024 --micro_batch_size 16 \
  --num_epochs 10 --learning_rate 3e-4 --cutoff_len 512 \
  --freeze_LLM False --train_from_scratch False --seed 42 \
  --output_dir outputs/v2/checkpoints/C-SFT
```

每组在 `outputs/v2/checkpoints/{A,B,C}-SFT/` 保存权重、tokenizer和最终选模manifest。原入口的 `final_checkpoint/` 若在 `load_best_model_at_end=True` 后写出，须核实其对应最佳validation loss权重；不能只看目录名推断是最后一个epoch。

在 `outputs/v2/evidence/phase6/` 保存SFT任务清单检查、扩词初始化记录、smoke、训练/validation曲线、checkpoint选择表、参数和资源。只生成validation预测进行合法性检查。

## 3. 验收且可推进标准

| 编号 | 条件 | 未来证据 | 不满足时 |
|---|---|---|---|
| P6-1 | 三组共同词表/token-ID/初始权重、模型revision和参数数量一致 | `evidence/phase6/initialization_checks.json` | 阻断，重验受影响组 |
| P6-2 | 全参数，label mask/四token完整、任务来源一致 | `evidence/phase6/data_gradient_checks.json` | 阻断 |
| P6-3 | 100样本调试显示loss下降、可学习回放、合法生成 | `evidence/phase6/smoke_report.json` | 排查后再正式训练 |
| P6-4 | A-SFT/B-SFT/C-SFT各完成相同上限/停止政策 | 正式训练日志和配置 | 补未完成组，不因差而取消 |
| P6-5 | 各组按最低validation loss选模，权重/词表/映射一致可加载 | `evidence/phase6/selected_checkpoints.json` | 阻断选模或映射错误 |
| P6-6 | 实际步数、成本、异常齐全，test未评测 | run records与资源表 | 补齐后推进 |

C的validation指标未提升或SFT组间差异小，可携研究发现推进；NDCG不作为本期必须超过某阈值的闸门。数值非有限或非法SID必须先修复。

## 4. 推进、恢复与交接

交给Phase 7三个选定SFT checkpoint、tokenizer、映射、任务清单及哈希。OOM降低micro-batch并增加累积维持1024；记录变化，必要时三组统一检查配置。smoke checkpoint不得混入正式初始化。改动共同词表影响全部SFT/RL；单组checkpoint变化使对应RL重新验收。

## 5. 运行记录模板

未来填写 `outputs/v2/evidence/phase6/run_records.md`：

| 字段 | 初始值 |
|---|---|
| run_id/status/attempt | `A-SFT`、`B-SFT`、`C-SFT` / 未开始 |
| model/code/data/vocab/config hashes | 待执行记录 |
| command/cwd/start/end | 待执行记录 |
| trainable/total、actual batch/steps | 待执行记录 |
| best validation loss/epoch/checkpoint hash | 待实验 |
| GPU-hours/wall time/VRAM/bytes | 待执行记录 |
| evidence/findings/decision | 待执行；未验收 |

## 6. 检查清单

- [ ] 三组同源文本与任务，仅SID不同；
- [ ] 全参数、共同扩词、label及smoke检查通过；
- [ ] 三次正式SFT完成且最佳validation loss checkpoint可核对；
- [ ] 实际训练量/成本记录完整，无正式test指标；
- [ ] `outputs/v2/evidence/phase6/acceptance.md`有真实证据和三个交接checkpoint。
