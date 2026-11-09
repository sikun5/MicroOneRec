# Phase 3：RQ-VAE 训练

> **初始状态：未开始。**  
> 导航：[主入口](experiment_setting.md) · [上一期：质量与表示](phase2_quality_embeddings.md) · [下一期：SID诊断导出](phase4_sid_diagnostics_export.md)  
> 协议：[v2量化器训练](../../topic_minionerec_v2.md#6-量化器训练与种子协议)。

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
# 待新增入口；A/B使用consistency-weight 0，C为0.1
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

## 4. 推进、恢复与交接

九次均完整且数值/梯度检查通过后，把final checkpoint清单交Phase 4。严格恢复需optimizer/scheduler/RNG和进度状态；仅有权重时不得声称精确续跑，应记录新的attempt。对应缓存或训练代码改变将使该方法量化器及其下游重新验收；不擅自删除未提升的方法。

## 5. 运行记录模板

未来填写 `outputs/v2/evidence/phase3/run_records.md`，每个正式run一条：

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

- [ ] 梯度、有效分母和λ=0检查通过；
- [ ] 9次正式训练覆盖完整，smoke独立存放；
- [ ] 使用最终checkpoint，下游固定seed42；
- [ ] 数值正确性、恢复状态和资源记录齐全；
- [ ] `outputs/v2/evidence/phase3/acceptance.md`有真实证据，负结果未被隐藏。
