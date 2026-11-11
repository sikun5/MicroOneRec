# Phase 3：RQ-VAE 训练

> **状态：已完成并验收（2026-11-11，run_id `RQ-{A,B,C}-{42,43,44}`）。**P3-1至P3-5全部实测通过（携带已裁定的恢复限制交接），无未解决P0；九次正式训练全部attempt1×fresh、`resume_used=false`、10,000 epochs/10,000 steps；验收证据见 `outputs/v2/evidence/phase3/acceptance.md`。  
> 导航：[主入口](experiment_setting.md) · [上一期：质量与表示](phase2_quality_embeddings.md) · [下一期：SID诊断导出](phase4_sid_diagnostics_export.md)  
> 协议：[v2量化器训练](../../topic_minionerec_v2.md#6-量化器训练与种子协议)。所有产物路径相对仓库根目录；本期实际执行目录为远端 `/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec`（Phase 1 已确认为指定工作目录的物理路径）。

## 1. 实验目标

完成RQ1/RQ2的九次受控量化训练，证明一致性项和梯度实现正确。保留全部方法与最终checkpoint，不按碰撞率或下游效果挑选运行。

## 2. 实验内容

### 2.1 前置输入

Phase 2验收通过的A/B/C缓存、有效视图标记、相同item顺序与metadata；Phase 1冻结环境。B/C干净缓存必须相同。

### 2.2 必需代码适配

- 原 `rq/rqvae.py` 的硬编码seed改为可配置并完整记录；不能假定原入口已有 `--seed`。
- 训练dataset接受clean/aug/valid_aug，推理仍接受单表示。
- RQ-VAE可选返回三层量化输出之和 `quantized_sum`，保留straight-through梯度。
- A/B只优化干净重构和量化损失；C新增有效视图余弦一致性，权重0.1，两个分支均有梯度。
- 导出最终epoch checkpoint和恢复所需状态；原best_loss/best_collision文件可保留诊断，但不作为正式输出。

### 2.3 正式训练前的smoke与梯度检查

1. 在FP32、同一小批输入和完全相同初始参数下，令C的λ=0；与B比较总损失和各参数梯度，默认 `atol=1e-6, rtol=1e-5`，记录最大误差。
2. λ=0时跳过或隔离扰动分支的状态副作用，保证K-means初始化先发生于干净输入；不让新增分支改变基线初始化。
3. λ=0.1时分别验证干净与扰动量化表示的梯度路径；检查不是只对离散indices或detach后的表示求损失。
4. 缺失视图不进入分母；全无有效视图时consistency=0且总损失有限。
5. 运行短程smoke，检查finite loss、参数更新、日志分量、保存与恢复；smoke放在 `outputs/v2/smoke/phase3/`，不计正式九次。

### 2.4 固定配置与正式顺序

| 配置 | 值 |
|---|---|
| codebook/维度/MLP | 三层×256；e_dim32；`[2048,1024,512,256,128,64]` |
| 基础损失 | MSE + quant weight1，commitment beta0.25 |
| 一致性 | A/B为0；C为0.1，只对有效视图 |
| 优化 | AdamW，LR=`1e-3`，weight decay0，50epoch warmup+constant |
| 初始化/其他 | K-means100 iterations；Sinkhorn `[0,0,0]`；dropout0、BN关闭 |
| batch/epochs | 20,480（实际目录全量）；10,000 epochs |
| checkpoint | 最终epoch，所有方法相同 |

推荐先依次完成 `RQ-A-42 → RQ-B-42 → RQ-C-42`，验证完整保存后再执行43、44；这只是运行顺序，不允许看到指标后取消某组。

| 方法 | seed42 | seed43 | seed44 |
|---|---|---|---|
| A | `RQ-A-42` | `RQ-A-43` | `RQ-A-44` |
| B | `RQ-B-42` | `RQ-B-43` | `RQ-B-44` |
| C | `RQ-C-42` | `RQ-C-43` | `RQ-C-44` |

同seed初始化和干净输入顺序一致；训练扰动seed始终1701，不随量化器seed改变。输入维度从缓存读取。

### 2.5 产物与命令接口

每个运行保存于 `outputs/v2/rqvae/{A,B,C}/seed_{42,43,44}/`：`final_checkpoint.pth`、完整配置/输入哈希、训练loss各分量、实际更新步数、资源记录与恢复状态。

```bash
# 已实现入口：scripts/v2_train_rqvae.py v1.3（sha256 618f4ff18b244e6f49c76c94cc4ccea3e14f9920e8108c62ae525f5bf17ab5b8，SCHEMA=phase3-v2_train_rqvae-1.3）
# λ按方法内部锁定（A/B=0，C=0.1，--consistency-weight 显式传入必须与锁定值一致）；A/B使用consistency-weight 0，C为0.1
python scripts/v2_train_rqvae.py \
  --method C --seed 42 --embedding-root outputs/v2/embeddings \
  --epochs 10000 --batch-size 20480 --learning-rate 1e-3 \
  --num-emb-list 256 256 256 --consistency-weight 0.1 \
  --checkpoint-policy final --output-root outputs/v2/rqvae
```

## 3. 验收且可推进标准

| 编号 | 条件 | 未来证据 | 不满足时 |
|---|---|---|---|
| P3-1 | λ=0损失与梯度退化通过，λ>0两分支有正确梯度 | `evidence/phase3/loss_gradient_checks.json` | 阻断全部正式C运行 |
| P3-2 | 9个正式run ID齐全且各达10,000epochs | `evidence/phase3/run_inventory.json` | 只补缺失/中断运行 |
| P3-3 | 均使用固定最终checkpoint，无最佳种子筛选 | checkpoint元数据与哈希 | 阻断选模偏差 |
| P3-4 | loss/参数有限，保存可加载，缓存和配置匹配 | load/finite检查、日志 | 修复后重验相关运行 |
| P3-5 | 各分量、实际步数与额外C前向成本记录完整 | 资源表与日志 | 补齐记录 |

重构loss未下降到预期、C更差或初步出现坍塌，不自动判实现失败；先排查P3-1/P3-4，通过后记录为研究发现，交由Phase 4完整诊断。不能降低λ或换seed后只保留更好的结果。

### 3.1 验收结果（2026-11-11 实测，run_id `RQ-{A,B,C}-{42,43,44}`）

| 编号 | 实测值 | 证据（sha256 前 16 位，完整值见 acceptance.md） | 结论 |
|---|---|---|---|
| P3-1 | B vs C(λ=0) 同实例恢复初始态后损失差 0.0（逐位）、16,112,032 元素梯度 max_err=0.0/失败张量 0，RNG 与 step 后参数逐位一致；λ=0.1 双分支 zq_clean/zq_aug 梯度范数 0.411/0.403 均非零；indices 无 grad_fn、ST 路径实证；混合批分母=504 与掩码精确相等、仅扰动无效行 aug 一致性逐位不变、全无效批（8 行）一致性=有限 0 | `loss_gradient_checks.json` `c5cb608437d910197`（容差 atol=1e-6/rtol=1e-5 预声明） | 通过 |
| P3-2 | 9/9 ID 齐全；每 run epoch=10000、step=10000；training_log.jsonl 各 10,000 行且 epoch/step 序列 1..10000 连续无跳步；全部 attempt1×fresh、`resume_used=false`（正式路径限定不中断运行） | `run_inventory.json` `78f5318407a99870`、`checkpoint_checks.json` `46a1001ba4b5c170` | 通过 |
| P3-3 | 九次 final 均为第 10,000 epoch 独立落盘（epoch=10000/step=10000 内嵌于 checkpoint）；best_loss/best_collision 为锁定行为历史诊断，未替代正式 final；下游固定 seed42，三种子全部保留 | `checkpoint_checks.json` `46a1001ba4b5c170`（final_not_best_substitute） | 通过 |
| P3-4 | 9/9 文件 SHA-256 与 inventory 逐字一致；strict 加载成功；31 参数张量 16,112,032 元素逐元素有限；state 字段逐位在位（optimizer 31 条目 step=10000、scheduler、RNG 四套、vq_initted=[True,True,True]、config 绑定 v1.3 与逐文件输入哈希）；eval-only 8 行前向 out [8,2560]/zq [8,32]/indices [8,3] 全有限且与 get_indices 一致；前向前后参数 checksum 不变 + initted 不变（无参数更新/无 K-means 重跑实测）；日志全分量逐行有限 | `checkpoint_checks.json` `46a1001ba4b5c170`（9/9 run_passed） | 通过 |
| P3-5 | 逐 epoch 11 字段（lr/recon/quant/consistency_raw/weighted/total/n_valid=3,356/aug_forward/epoch_time 等）×10,000 行；C 三 run 额外前向各 10,000（合计 30,000）、A/B=0；逐 run wall 2,843–4,403 s；峰值显存 A/B 574.1、C 684.4 MiB；final 193,394,855–193,395,175 bytes | `resource_summary.json` `74c996e36ee5bfb2`、`run_records.md` `85dcb877567d0337` | 通过 |

重构loss未下降到预期、C更差或初步出现坍塌，不自动判实现失败；先排查P3-1/P3-4，通过后记录为研究发现，交由Phase 4完整诊断。不能降低λ或换seed后只保留更好的结果。九次正常完成/损失有限不构成"无坍塌"或"方法优劣"的证明，相关判定留Phase 4与后续评测。

## 4. 推进、恢复与交接

九次均完整且数值/梯度检查通过后，把final checkpoint清单交Phase 4。严格恢复需optimizer/scheduler/RNG和进度状态；仅有权重时不得声称精确续跑，应记录新的attempt。对应缓存或训练代码改变将使该方法量化器及其下游重新验收；不擅自删除未提升的方法。

> **交接状态（2026-11-11）：**P3-1至P3-5全部通过并完成交接。基线：Phase 1 commit `ded0bc13059b25c3c20f6cbc1a4c1a7c48d97649`、Phase 2 提交 `e7517496fc6d216c51c32c36323853674de0132f`；正式代码 `scripts/v2_train_rqvae.py` v1.3 `618f4ff18b244e6f49c76c94cc4ccea3e14f9920e8108c62ae525f5bf17ab5b8`（九次统一绑定；相对T4验证的v1.2 `47b6b6ef…` 仅SCHEMA/docstring/`resume_used`记录字段变化）。九个final全文SHA-256以 `outputs/v2/evidence/phase3/run_inventory.json`（`78f53184…`）与本指南T5交接正本为准，路径 `outputs/v2/rqvae/{A,B,C}/seed_{42,43,44}/final_checkpoint.pth`；下游仅用各方法seed42，三种子全部保留。
> **恢复限制（已裁定，携带交接）：**跨进程训练轨迹未达预声明逐位等价（同进程机制逐位保真——`snapshot_fidelity_checks.json` `759f56f6…`；跨进程 ~1e-9 噪声经 VQ argmin 近并列离散放大，`resume_checks.json` `a9881c7d…`）。boss裁定：正式结果仅接受每个ID从初始状态一次不中断完成的运行（`resume_used=false`）；完整恢复状态仍随 checkpoint 保存与加载核验，供检查/追溯，不宣称已验证跨进程精确续跑。K-means未重跑的证据=守卫+标志回填+权重不变推断（非调用计数）。
> **输入覆盖（24 文件实算 unchanged）：**消费并集9（A/B/C clean、C train_aug/valid_aug、canonical item_ids、A/B/C metadata）+ 原始四数据源（item.json/train/valid/test csv，实算重校）+ Phase1/2 基准11（manifests/evidence）；`t6_input_coverage.json` `4925771a…`。21个诊断缓存训练不消费，沿用 Phase 2 基准未重测。完整交接表与限制披露见 `outputs/v2/evidence/phase3/acceptance.md`（sha256 前 16 位 `b18dcb6c74f50773`）。

## 5. 运行记录

运行记录实际文件：`outputs/v2/evidence/phase3/run_records.md`（sha256 前 16 位 `85dcb877567d0337`；run_id `RQ-{A,B,C}-{42,43,44}`，2026-11-11 于节点 `swu073-a800-10g`/容器 `34fb2691c7dd`，九 run 全部 attempt1×fresh、无失败轮，历史废弃轮（T4 恢复排错/60ep 击杀过晚轮）已留档不冒充正式）；验收文件：`outputs/v2/evidence/phase3/acceptance.md`（sha256 前 16 位 `b18dcb6c74f50773`）。下方为原模板：

| 字段 | 初始值 |
|---|---|
| run_id/status/attempt | `RQ-{A,B,C}-{42,43,44}` / 未开始 / 待执行 |
| code/config/input hashes | 待执行记录 |
| command/cwd/start/end | 待执行记录 |
| final epoch/step/checkpoint hash | 待执行记录 |
| recon/quant/consistency/finite | 待实验 |
| GPU-hours/wall time/VRAM/bytes | 待执行记录 |
| evidence/exception/decision | 待执行；未验收 |

## 6. 检查清单

- [x] 梯度、有效分母和λ=0检查通过（loss_gradient_checks：同实例逐位 0.0/失败 0；λ=0.1 双分支梯度非零；混合批分母=504、全无效批=有限 0）；
- [x] 9次正式训练覆盖完整，smoke独立存放（run_inventory 9/9 attempt1×fresh、resume_used=false、10,000 epochs；smoke 100ep 于 `outputs/v2/smoke/phase3/` 实测峰值 574.1–684.4 MiB）；
- [x] 使用最终checkpoint，下游固定seed42（final=第 10,000 epoch 独立落盘，非 best_* 替代；三种子全部保留）；
- [x] 数值正确性、恢复状态和资源记录齐全（checkpoint_checks 9/9 strict 加载+31 参数张量有限+eval sanity+无更新实测；resource_summary 双口径成本 30,421s/30,551s 分列）；
- [x] `outputs/v2/evidence/phase3/acceptance.md`有真实证据，负结果未被隐藏（P3-1至P3-5 全部通过；T4 跨进程轨迹逐位限制携带披露；坍塌/方法优劣留 Phase 4）。
