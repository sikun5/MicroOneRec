# Phase 7：三组全参数 GRPO

> **执行状态：2026-11-17 已完成并验收。** T1–T8 与 P7-1..6 按已批准限制闭环；Phase8 正式评测尚未执行。
>
> 导航：[主入口](experiment_setting.md) · [上一期：SFT](phase6_sft_training.md) · [下一期：最终评测](phase8_final_evaluation.md)
>
> 协议：[v2 全参数 GRPO](topic_minionerec_v2.md#74-全参数-grpo) · [已执行核心源码归档](../source_archive/phase7_completed/README.md)

## 1. 实验目标与完成范围

分别从 A-SFT/B-SFT/C-SFT 进入同协议后训练，完成 A-RL/B-RL/C-RL，为 RQ3 提供比较。质量规则和一致性只作用于 SID 构建，本期未新增 reward、提示、模型规模或训练任务。

三组均完成 625 个真实 optimizer 更新；10,000 个基础任务各 copy0/copy1 一次构成 20,000 组的确定性流，一次物理遍历等价两轮逻辑 epochs。RL 均保留事前指定的最终权重，没有按 validation 或训练效果挑中间 checkpoint。A 在旧 A800 MIG 上完成；迁移后 B、C 在 H100 上顺序完成，B 验收后才启动 C。

以下第 2–4 节保留执行协议和闸门；第 5–7 节总结真实运行、证据与限制。验收支持课程离线研究交接，不代表推荐效果已经提升。

## 2. 已实现的 Pipeline 与固定协议

### 2.1 前置输入

- Phase6 最低 validation loss 的本组 SFT 可加载**顶层目录**：`outputs/v2/checkpoints/{A,B,C}-SFT/`，选定 step 为 429/429/351。
- Phase4 本组 seed42 四 token SID、info 与 3,686 商品目录；共同词表 154,977、原子 SID token 3,308。
- Phase1 固定 10,000 个任务：8,000 序列→SID、1,000 标题→SID、1,000 文本历史→SID；交互来源仅 train。
- `outputs/v2/manifests/rl_stream_20000_{A,B,C}.jsonl` 保留来源、任务、copy 和目标身份；三组 20,000 行十项非 SID 字段逐行同序。
- Phase5 冻结的 raw 合法性契约及 validation 技术核验接口。

### 2.2 实际代码职责

1. `scripts/v2_grpo_stream.py` 物化同源两逻辑 epoch 流，避免旧 Trainer 尾窗造成预算/权重不一致。
2. `scripts/v2_train_grpo.py` 已实现统一来源任务适配；**正式使用 `scripts/v2_train_grpo_audit.py`**，保留真实参考同步事件。审计入口 SHA 为 `fda1ee5b525a9d49667effab51c5dc9d5493cb3c20f521c889da03c8f5797ab1`。
3. 原 ReReTrainer constrained beam search 与 correctness/rank-aware reward 保留，目标按稳定身份恢复；不增加语义、前缀或 SASRec reward。
4. 策略全部参数更新；完整 reference 独立存储、不进入 optimizer，但按真实 512 步回调更新权重。
5. B/C 被动遥测仅观察原始候选、分组、loss/KL、梯度、参数变化、显存峰值及原生同步快照；不增加 forward、rollout 或更新，不改变随机流/训练数学。独立 supervisor 捕获真实起止与 OS exit。
6. 后验 CPU 审计独立核对原始来源、奖励公式、raw SID、有限权重、词表绑定及全模型 EMA；新进程仅运行原始 validation 前 10 行技术加载检查。

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

### 2.4 最终批准配置

| 参数 | 实际值与解释 |
|---|---|
| 基座 | `Qwen/Qwen3-0.6B-Base`，revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |
| 策略 | 599,163,904 参数全部参与训练；FP32 policy/master/grad，BF16 autocast |
| Reference | 完整独立 BF16、无梯度；FP32 中间 EMA 后单次转 BF16 |
| Optimizer / LR / beta | `paged_adamw_32bit` / `1e-5` / `1e-3` |
| G / batch / 累积 | 16 / 16 / 32；512 候选=32 prompt 组/更新 |
| 预算 / seed | 10k × 2 逻辑 epochs；20k 流一次物理遍历、625 整窗更新；seed42 |
| 实际 CLI epochs | `num_train_epochs=1` 作用于已展开的 20k 流，不是只训练一轮基础 10k |
| Prompt / completion | 512 / 128；目标必须完整四 token + EOS |
| Reference 同步 | `sync_ref_model=True`、interval512、alpha0.6；每组真实发生一次 |
| Scheduler / warmup / clip | cosine / 0.03 / `max_grad_norm=0.3` |
| 保存 | `save_steps=0.05`、实际间隔 32 更新；`save_total_limit=6` 为批准的存储例外 |
| 数值与资源适配 | gradient checkpointing；沿已验收 SDP 设置；无额外 policy/ref CPU offload |
| 关闭项 | LoRA/QLoRA、dynamic sampling、DAPO、GSPO、训练中额外 Top-K 生成 |
| 最终权重 | 固定完整预算后的 final，不按 validation/KL 或 attempt 效果选择 |

原 v4 锁 SHA `1f421c03b2d7b670ddc034ced8fef46700e20cb3c7bce6abbe15e053fb876b09` 保留；其中原计划保留 4 份，实际批准为 6，差异用 addendum 记录，未回写旧锁。Phase6 选择器的 revision 旧文本少一个 `b`；新交接按原环境锁使用上述 40 字符值，旧选择器及模型哈希未改。

### 2.5 已执行入口、配置与产物

[源码归档](../source_archive/phase7_completed/README.md)包含被动遥测、训练/后验 supervisor、CPU 审计、同步/尖峰诊断、成功的 T8 attempt002 检查器及最终包构建/核验源码。B/C 的完整实际 argv 与 environment 分别保存在 [B 配置](../source_archive/phase7_completed/configs/B-RL.recorded.json)和 [C 配置](../source_archive/phase7_completed/configs/C-RL.recorded.json)。原训练入口已在 `scripts/` 版本化，本次未改。

这些 JSON 记录已完成的运行，包含原证据/输出路径；不能将它们当作新的空目录训练模板直接重放。下列命令只查看已执行命令，不启动训练：

```bash
# 从 MicroOneRec 仓库根执行；只打印记录的 argv。
python - <<'PY_CONFIG'
import json
import shlex
from pathlib import Path
for method in ("B", "C"):
    path = Path("source_archive/phase7_completed/configs") / f"{method}-RL.recorded.json"
    cfg = json.loads(path.read_text())
    print(method, shlex.join(cfg["command"]))
PY_CONFIG
```

B/C 实际 cwd 为 `/data/user/swu073/dsaa5020/MicroOneRec`，使用 `.venv-phase3`；旧绝对 ROOT 通过兼容链接解析。归档保留原路径与日期依赖，不构成通用入口重构。历史最终核验器固定检查实验时 HEAD `1f5c01c`，在新的源码归档提交上原样运行会拒绝新 HEAD；不能据此宣称归档后重新通过该历史检查。

## 3. 验收且可推进标准

| 编号 | 保留的验收条件 | 运行证据（相对 outputs/v2/） | 技术失败处理 |
|---|---|---|---|
| P7-1 | 人工命中/未命中组reward等于手工答案、任务目标不覆盖 | `evidence/phase7/reward_target_checks.json` | 阻断，不能当成稀疏奖励 |
| P7-2 | G=16分组正确，完整后缀、合法路径和KL/梯度正确 | `evidence/phase7/rollout_gradient_checks.json` | 阻断 |
| P7-3 | 每组恰用同源10k任务，样本顺序/奖励/有效batch一致 | 清单与配置哈希 | 阻断不公平输入 |
| P7-4 | 三组均全参数、参考独立无梯度且同步设置一致 | 参数/optimizer/回调审计 | 阻断训练范式变化 |
| P7-5 | A-RL/B-RL/C-RL完整2epochs、最终权重可加载 | `evidence/phase7/final_checkpoints.json` | 只补未完成部分 |
| P7-6 | 零优势、KL、reward、多样性、实际步数和资源齐全 | 训练诊断与run records | 补齐再交接 |

GRPO后validation下降、reward不涨或真实组大量零优势，在技术检查通过时均可“携研究发现推进”。非有限loss、错误target、非法路径、空checkpoint属于阻断。

## 4. 恢复与交接边界

交给 Phase8 六个 LLM 完整推理目录、Phase5 两个基线及 Phase4 诊断来源。G16 时当前 Trainer 的 per-device batch 不能降到 16 以下；未改 LoRA、G、奖励或数据种子。C seed42 近坍塌 SID 始终保留。

本期未验证中断后的精确续训；最终推理目录不承诺含完整 optimizer、scheduler、RNG 和参考同步状态。仅有 policy 权重不能称无缝恢复。运行中断须保留失败 attempt/起态并裁定，不能自动 policy-only resume、热改或按效果重跑。上游 SFT、SID 或来源改变时，对应 RL 需要重新判断有效性。

Phase8 的 Beam50、确定性解码和统一 item-ID 评测协议仅作交接冻结；本期没有执行正式 test 推荐推理、指标或选模。少量 test 文件仅按既有索引核字节身份。

## 5. 真实结果与证据范围

### 5.1 运行和独立审计

| 项目 | A-RL | B-RL | C-RL |
|---|---|---|---|
| 更新/逻辑预算 | 625 / 20k 组 | 625 / 20k 组 | 625 / 20k 组 |
| 双 reward 记录 | 640,000 行 | 640,000 行 | 640,000 行 |
| 真实 512 同步 | 1 次，独立 CPU EMA 与 after 一致 | 1 次，原生前/策略/后复算差异 0 | 1 次，原生前/策略/后复算差异 0 |
| before 证据 | 从冻结 SFT 重建初始参考，非原生 before | 原生 before | 原生 before |
| 全训练 raw token 独立审计 | 未保存同等全流，保持已接受范围 | 320,000/320,000 合法 | 320,000/320,000 合法 |
| 新进程 validation 技术检查 | 合法 10/10，命中 0/10 | 合法 10/10，命中 0/10 | 合法 10/10，命中 0/10 |
| 直接捕获正式 OS exit | 历史未知，以 DONE/625/产物/进程等联合证据接受 | 0 | 0 |

Validation 使用原始前 10 行、Beam10，每行只返回 1 候选；不是 HR@10，也不能据此推断正式推荐质量。零命中、稀疏/零优势及 RL 退化均如实保留，没有预写提升数字。

T4 三组全尺寸 smoke 各实测 64 微批/2 更新、每批 16 候选、32 累积、310 张量/599,163,904 元素 FP32 CUDA 梯度有限且参数变化。独立展开原始嵌套 `stream_index:[i]` 得 0..63；旧派生布尔误判保留说明，未修改旧报告。9,992,640 参数 tiny fixture 只属机制控制，不冒称全尺寸训练实测。

T8 成功 CPU attempt002 核验六模型各 310 张量/599,163,904 元素全部有限、SFT BF16/RL FP32、同结构和词表/SID 绑定；333 个当前路径及三组 20k 流的非 SID 字段顺序一致。独立最终包检查覆盖 366 路径、21,132,551,471 逻辑 bytes，逐 size/SHA 一致、failures 为空。它只覆盖选定权重与证据，不代表重哈全部历史 runtime。本次源码归档只做静态/字节检查，没有重跑这些实验。

### 5.2 预选六模型

| 模型 | 仓库相对目录 | model.safetensors SHA-256 |
|---|---|---|
| A-SFT | `outputs/v2/checkpoints/A-SFT` | `f753205d43dfb2240b96d5a9b0de8f6e375c65bddc313bdf19ca7aab55db405c` |
| B-SFT | `outputs/v2/checkpoints/B-SFT` | `90fe4eb374afd5d3b7a9a1c3dbc2cf5ed6ac695792f35a8cf005605e67296aef` |
| C-SFT | `outputs/v2/checkpoints/C-SFT` | `2d5ac0befc8093c727ec5ffd9b8c2db4957b81f6de27bf263647975c3206f538` |
| A-RL | `outputs/v2/debug/phase7/audit_fresh_attempt002/final_checkpoint` | `bbd1aacfb343788296e3af6441018c199474184999f40985a8152c42457f66fe` |
| B-RL | `outputs/v2/checkpoints/B-RL/final_checkpoint` | `9f38d1f230f6dddfa313c6f4667431d50440e75dc92a4aaa30444c8921bba973` |
| C-RL | `outputs/v2/checkpoints/C-RL/final_checkpoint` | `33afe7d008f54c644d79afe2d08b0b333db9bf9e71b29130719a1468c6d37d1f` |

主 A 必须保持审计 fresh 路径，即使旧未选 A 与它最终模型字节相同，也不替换为旧目录。三 SFT 存储为 BF16，三 RL 为 FP32；共同词表与各组 SID/info 绑定均通过。最终交接里的加载结论复用已接受的真实加载证据，T8 没有新跑六模型 GPU 推理。

### 5.3 资源与额外成本

| 运行 | 硬件 | HF train_runtime 秒 | 正式 PyTorch allocated/reserved 峰值 bytes |
|---|---|---:|---|
| 主 A 审计 fresh | A800 MIG 约 10GB | 26982.2607 | 历史未知 |
| 原未选 A | A800 MIG 约 10GB | 26895.1772 | 历史未知 |
| B | H100 80GB HBM3 | 5420.217 | 8,539,622,400 / 9,353,297,920 |
| C | H100 80GB HBM3 | 5647.0415 | 8,539,622,400 / 9,353,297,920 |

A 两轮 HF 时间合计 53877.4379 秒；三份现存 OOM 日志不等于完整失败启动次数，缺失次数/耗时不记为 0。B launcher/scope wall 为 5512.517137/5502.073165 秒；C 为 5709.752646/5701.027757 秒。各时间范围分列，异构硬件耗时不能当作受控加速比。

B/C 峰值来自同一正式进程模型加载前至最终保存，未连续测量 PyTorch 外 native paged 分配或全设备总峰值。CPU 审计、validation、C 早期同步核验与 S1、T8 失败/成功 CPU 检查另列，不能混成训练 GPU-hours。T8 v1 检查器误把 SFT 存储要求为 FP32，exit1/62.661 秒留档；v2 仅修正阶段 dtype 条件，模型未改、真实 exit0。

### 5.4 研究发现与保留限制

- C 最大 KL 在 step514 为 10997.427993829，日志 grad_norm 为 10472.1572265625；step623 的 1004.23 只是末段例子。step316 同步前已有尖峰，时间相关性不能证明 512 同步导致尖峰。
- 实际 clip 配置为 0.3；Trainer→Accelerator→torch 源码返回裁剪前 total_norm，不能把大日志 norm 当作实测裁剪后范数。逐步 post-clip norm 和逐 token policy/reference logps 未采集，保持未知。目标/分组/reward/raw/mask、最终有限权重及全量 EMA 审计未发现技术错误，保留优化不稳定负结果。
- A 的 per-task KL/token diversity、完整训练峰值和直接 OS exit 等未知继续未知，不能用 B/C 新采字段补造。
- 单个 LLM seed42 未测训练随机性；kernel4.18 警告、旧路径兼容链接及内部死分支 EOS-only fallback 保留。B/C 最终选中 raw 全流审计合法，不外推为未来所有生成绝无非法候选。

## 6. 验收状态

- [x] 人工 reward/目标与来源协议通过，tiny 控制和全尺寸训练证据分开。
- [x] 三组同源两逻辑 epochs、G16、全参数与独立完整 ref，真实 512 同步复核通过。
- [x] 三组预选最终权重及三个 SFT 的推理目录/词表/SID 交接齐全。
- [x] 负结果、有限尖峰、A 历史未知及各次额外成本均披露。
- [x] P7-1..6 与五份 canonical 交接完成；自有计算任务和 SSH 连接已结束，GPU 已释放。

## 7. 交接与可追踪锚点

运行产物保存在远端仓库的 `outputs/v2/evidence/phase7/`，模型、数据、环境、日志、原始预测与执行回执不随本次源码提交。五份 canonical 文件及完整证据锚点同时记录于[源码来源清单](../source_archive/phase7_completed/manifest.json)：

| canonical 文件 | SHA-256 |
|---|---|
| `final_checkpoints.json` | `17f72a0186f781676722a9e79a1e35671670ce872c26128e52c65fef8891946b` |
| `artifact_inventory.json` | `46c6ff655dac932f65a169381bf904f3b9d26ccc32bb5ce1bbffe589ee57c0b8` |
| `resource_summary.json` | `6058bf0b260dc30604d5023abf9c82579b678917ff78492c58900a10f54fb259` |
| `run_records.md` | `a91e8739a9dbe71c2b5ed2ebd4ae3defd971dc97c0d2227038c6ba6c02ee73d0` |
| `acceptance.md` | `617bf67ab7c62d90f02714043ccb089214dd22c48e4109dbba4547887c24b602` |

Phase8 另接收 Phase5 的 `baselines/B0/popularity.json` 与 `baselines/B1/best_checkpoint.pth`、Phase4 九个干净量化器及 27 扰动组合。主 A/B/C 的更细证据在各自运行包，S1 用独立补充索引，旧索引不覆写；最终索引自身和后写回执由权威记录锚定，避免自哈希循环。

完整验收与工作历史留在项目工作记录目录；本指南汇总已接受事实，归档 README 解释源码原始位置与使用限制。后续正式评测须作为 Phase8 单独执行。
