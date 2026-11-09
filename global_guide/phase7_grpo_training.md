# Phase 7：三组全参数 GRPO

> **初始状态：未开始。**  
> 导航：[主入口](experiment_setting.md) · [上一期：SFT](phase6_sft_training.md) · [下一期：最终评测](phase8_final_evaluation.md)  
> 协议：[v2全参数GRPO](../../topic_minionerec_v2.md#74-全参数-grpo)。

## 1. 实验目标

分别从A-SFT/B-SFT/C-SFT进入同协议后训练，完成A-RL/B-RL/C-RL，为RQ3提供比较。质量规则和一致性只作用于SID构建，本期不新增reward、提示或训练任务。

## 2. 实验内容

### 2.1 前置输入

- Phase 6选定的本组SFT权重、tokenizer和manifest；
- Phase 4本组四token SID、info和合法目录；
- Phase 1固定的10,000个任务来源清单；
- Phase 5验证后的validation评测器；冻结环境与单卡配置。

核对每组checkpoint、tokenizer、info、SID和来源哈希；A不能加载B的词表绑定或C的info。

### 2.2 必需代码适配

1. 拟新增 `scripts/v2_train_grpo.py`，按task/source ID消费全局10k manifest，而不是对现有三个dataset各自截取10k。
2. 三组任务组成与原始文本一致，仅SID不同；在映射后复查相同prompt的目标冲突。奖励从稳定任务身份恢复目标，不能靠覆盖式history字典悄悄选一个答案。
3. 复用原 `ReReTrainer`、correctness与ranking函数；不增加前缀/语义/SASRec奖励，也不改为LoRA。
4. 政策模型全参数更新；参考模型独立完整加载，不进optimizer，但 `sync_ref_model=True` 按锁定版本回调同步。记录实际 `ref_model_sync_steps` 和混合参数，三组相同。
5. 检查额外模型副本与生成cache，避免原入口仅为读取device而留下无用LLM；清理不应影响reward、参考模型或优化目标。

### 2.3 Reward与分组控制实验

只使用训练小样本或人工SID控制组，不查看test。每个prompt恰有G=16个候选，记录组边界和target来源。

| 控制组 | 预期行为 |
|---|---|
| 一个合法候选精确命中 | 命中correctness=1、其rank项0；其他候选为原位置对应负ranking项 |
| 16个候选均未命中 | correctness与ranking均为0，组内reward std和奖励优势为0 |
| 相同候选/相同reward | 检查重复率；不能通过不同prompt误归组制造方差 |
| 同历史但不同task/target的样本 | 目标按稳定身份恢复，无字典覆盖或跨组混淆 |
| 完整后缀不同、前三层相同 | 只前三层相同不算精确命中 |
| 已知合法SID与格式边界 | 规范化/EOS处理一致，受约束路径可到达目标，无非法token |

原rank函数可用以下手工检查：令第j位（j从0起）\(a_j=1/\log_2(j+2)\)。组内有真实目标时，非目标的rank项为 \(-a_j/\sum_{m=0}^{15}a_m\)，目标为0；组内无真实目标时所有rank项为0。再逐候选加correctness，不引入新的系数或按质量加权。

在独立smoke中执行至少一个完整G=16的生成—参考前向—policy loss—反向更新，确认参数改变、loss有限、参考无梯度、suffix保留、资源可承受。开发G=4/8只能作为调试，正式运行仍G=16。

若真实rollout持续全零奖励，先用上述已知命中控制组验证代码和目录可达性。控制正确且只是小模型未命中，则记录为稀疏信号/负结果，仍可按原配置完成正式训练；不得偷偷注入目标或更换reward来“通过”。奖励优势为0时仍可能存在KL梯度。

### 2.4 固定正式配置

| 参数 | 值 |
|---|---|
| 运行ID | A-RL、B-RL、C-RL |
| 初始化 | 本组最低validation loss的SFT checkpoint |
| 任务数 | 10,000：8,000/1,000/1,000 |
| 训练 | 全参数，BF16，`paged_adamw_32bit` |
| Epochs/seed | 2 / 42 |
| G / per-device batch / 累积 | 16 / 16 / 32 |
| 每次更新有效候选 | 512，约32个prompt组 |
| LR / KL beta | `1e-5` / `1e-3` |
| Completion上限 | 128 |
| Rollout | 原 constrained beam search，G=16 |
| 参考同步 | 开启、实际回调参数锁定并记录 |
| 关闭项 | dynamic sampling、DAPO、GSPO、训练中额外Top-K生成 |
| 正式checkpoint | 完整2epochs后的最终权重 |

记录每个任务类型的命中率、reward均值/方差、零优势组比例、KL、候选唯一率、completion长度与有效prompt数量；task混合数据下不能只给一个无法解释的总体reward。

### 2.5 待新增入口与产物

```bash
# 拟新增适配入口；先验证单进程配置没有继承八卡或ZeRO-2默认设置
accelerate launch --num_processes 1 scripts/v2_train_grpo.py \
  --model-path outputs/v2/checkpoints/C-SFT/final_checkpoint \
  --method C --data-root outputs/v2/data/C \
  --task-manifest outputs/v2/manifests/rl_tasks_10000.jsonl \
  --train-batch-size 16 --eval-batch-size 16 \
  --gradient-accumulation-steps 32 --num-generations 16 \
  --num-train-epochs 2 --learning-rate 1e-5 --beta 1e-3 \
  --max-completion-length 128 --sync-ref-model True \
  --seed 42 --output-dir outputs/v2/checkpoints/C-RL
```

按A/B/C分别执行，保留各自 `outputs/v2/checkpoints/{A,B,C}-RL/final_checkpoint/`。在 `outputs/v2/evidence/phase7/` 保存人工reward、group/target检查、trainable ratio、参考同步、smoke、每组训练诊断与成本，以及正式checkpoint清单。validation仅用于诊断，不替代已固定的两epoch最终选模规则。

## 3. 验收且可推进标准

| 编号 | 条件 | 未来证据 | 不满足时 |
|---|---|---|---|
| P7-1 | 人工命中/未命中组reward等于手工答案、任务目标不覆盖 | `evidence/phase7/reward_target_checks.json` | 阻断，不能当成稀疏奖励 |
| P7-2 | G=16分组正确，完整后缀、合法路径和KL/梯度正确 | `evidence/phase7/rollout_gradient_checks.json` | 阻断 |
| P7-3 | 每组恰用同源10k任务，样本顺序/奖励/有效batch一致 | 清单与配置哈希 | 阻断不公平输入 |
| P7-4 | 三组均全参数、参考独立无梯度且同步设置一致 | 参数/optimizer/回调审计 | 阻断训练范式变化 |
| P7-5 | A-RL/B-RL/C-RL完整2epochs、最终权重可加载 | `evidence/phase7/final_checkpoints.json` | 只补未完成部分 |
| P7-6 | 零优势、KL、reward、多样性、实际步数和资源齐全 | 训练诊断与run records | 补齐再交接 |

GRPO后validation下降、reward不涨或真实组大量零优势，在技术检查通过时均可“携研究发现推进”。非有限loss、错误target、非法路径、空checkpoint属于阻断。

## 4. 推进、恢复与交接

交给Phase 8六个LLM checkpoint（保留三个SFT）与B0/B1清单，冻结推理配置。G=16时当前trainer的per-device batch不能降到16以下；OOM先checkpointing、关闭额外生成、清理无用副本。若缩短completion须覆盖所有真实任务且三组同改；否则暂停，不改LoRA或G。

恢复时核对policy、optimizer、scheduler、RNG、参考模型及同步进度；仅恢复policy权重不能声称原过程无缝续跑。某组SFT或manifest变化后对应RL失效，不能复用旧RL进最终表。不得因结果下降选择性重跑更好的attempt。

## 5. 运行记录模板

未来填写 `outputs/v2/evidence/phase7/run_records.md`：

| 字段 | 初始值 |
|---|---|
| run_id/status/attempt | `A-RL`、`B-RL`、`C-RL` / 未开始 |
| SFT/model/code/manifest/reward hashes | 待执行记录 |
| command/cwd/start/end | 待执行记录 |
| policy/ref/optimizer/sync/batch | 待执行验证 |
| epochs/steps/final checkpoint | 待执行记录 |
| reward/zero-advantage/KL/diversity | 待实验 |
| GPU-hours/wall time/VRAM/bytes | 待执行记录 |
| evidence/findings/decision | 待执行；未验收 |

## 6. 检查清单

- [ ] 已知答案reward、分组、目标映射与合法路径通过；
- [ ] 三组同10k来源、同G16与全参数、同参考同步协议；
- [ ] 三次正式GRPO达到2epochs，未按test或validation效果挑其他RL checkpoint；
- [ ] 真实零优势和负结果如实记录，无正式test指标；
- [ ] `outputs/v2/evidence/phase7/acceptance.md`和最终权重清单齐全。
