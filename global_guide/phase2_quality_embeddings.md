# Phase 2：质量特征与文本表示

> **初始状态：未开始。**  
> 导航：[主入口](experiment_setting.md) · [上一期：数据与环境](phase1_dataset_prep.md) · [下一期：RQ-VAE](phase3_rqvae_training.md)  
> 协议：[v2 方法定义](../../topic_minionerec_v2.md#4-abc-方法定义)。所有产物路径相对仓库根目录。

## 1. 实验目标

为RQ1提供A与B的受控数据构建，为RQ2提供与B同一干净表示的C训练视图。本期验证规则、冻结编码和缓存，不运行量化器或推荐训练。

## 2. 实验内容

### 2.1 前置输入

Phase 1已通过的环境锁、公共目录、item/row/task清单、训练与诊断视图及哈希。编码器固定 `Qwen/Qwen3-Embedding-4B` revision，不使用推荐标签确定权重。

### 2.2 所需实现与步骤

1. 适配既有 `rq/text2emb/amazon_text2emb.py` 的冻结编码能力，新增显式item ID缓存与字段融合入口；不把已有word-drop参数当作全部新方法。
2. 为全部目录商品固定 `item_ids.json`，每份数组按此顺序保存；打乱输入文件顺序后仍能按item ID恢复对应，不依赖数组行号等于item ID。
3. 计算清洗描述的词数n和重复三元组比例r；导出q和w、缺失标记、有效视图标记，人工核对空/短/重复描述样例。
4. 编码器始终 `eval()/no_grad()`，masked mean pooling排除padding，每次上限2,048；L2归一化、非有限和零范数检查一致。可降低编码batch，不更换pooling或截断规则。
5. A：编码拼接文本。B：独立编码标题与非空描述并按权重融合。C：复用B的干净缓存，再编码训练drop30视图并重新计算其质量权重。
6. 对诊断drop30/drop60/dropall，为A及B/C分别准备扰动表示，使用Phase 1的独立清单。B/C共享相同诊断输入表示；三组看到相同商品和扰动文本。
7. 重跑一小段冻结编码检查差异，锁定dtype与允许的数值容差；同一共享缓存应直接复用/复制验证哈希，不靠浮点重算制造B/C不一致。
8. 记录文本调用数、有效token量、wall time、显存和产物大小；公共缓存成本只计一次。

### 2.3 固定公式与边界

\[
q=\mathbf1[n>0]\min(n/32,1)(1-r),\quad w=0.5q,
\quad x=\operatorname{L2Norm}((1-w)t+wd).
\]

r采用小写词三元组的 `1-唯一数/(n-2)`，n<3时为0。空描述q=w=0，不调用空文本编码；标题/描述都空使用共同 `unknown item` 占位并记录。训练视图仅对有效商品计算，原始缺失标签不因清洗或扰动变化。

字段分离增加调用数或实际可见文本预算属于A→B的整体变化，不能声称只验证权重或编码成本相等。

### 2.4 预期产物

| 路径 | 内容 |
|---|---|
| `outputs/v2/manifests/quality_features.jsonl` | item ID、n/r/q/w、原始缺失、有效标记 |
| `outputs/v2/embeddings/{A,B,C}/item_ids.json` | 相同目录身份与顺序 |
| `outputs/v2/embeddings/{A,B,C}/clean.npy` | `[N,D]`干净表示；B/C同哈希 |
| `outputs/v2/embeddings/C/train_aug.npy`、`valid_aug.npy` | 训练视图与有效分母标记 |
| `outputs/v2/embeddings/{A,B,C}/diagnostic/` | 三种扰动表示、对应有效标记和清单哈希 |
| `outputs/v2/embeddings/{A,B,C}/metadata.json` | 模型revision、规则、dtype、pooling、输入输出哈希 |
| `outputs/v2/evidence/phase2/` | 规则、冻结编码、顺序和资源检查 |

### 2.5 待新增入口

```bash
# 拟新增，完成实现与检查后运行；不是原脚本已经支持的参数
python scripts/v2_build_embeddings.py \
  --manifest-root outputs/v2/manifests \
  --encoder Qwen/Qwen3-Embedding-4B \
  --pooling masked_mean --max-length 2048 \
  --quality-length 32 --description-max-weight 0.5 \
  --output-root outputs/v2/embeddings
```

## 3. 验收且可推进标准

| 编号 | 条件 | 未来证据 | 不满足时 |
|---|---|---|---|
| P2-1 | 空描述精确回退、q/w范围正确、重复描述按公式降权 | `evidence/phase2/quality_rule_checks.json` | 阻断规则实现 |
| P2-2 | 所有缓存N=3,686，维度一致、显式ID对应，无NaN/Inf/零范数 | `evidence/phase2/cache_checks.json` | 阻断，不填零掩盖 |
| P2-3 | B/C干净表示及item顺序相同，训练/诊断清单没有混用 | metadata与哈希 | 阻断 |
| P2-4 | encoder无梯度、eval，padding不入池化，统一长度/归一化 | `evidence/phase2/encoder_checks.json` | 阻断 |
| P2-5 | 视图不改变标题/item/目标；缺失和极短描述有效标记正确 | `evidence/phase2/view_checks.json` | 阻断 |
| P2-6 | 各缓存可追溯到Phase 1，编码成本与共享部分单独记录 | metadata、资源表 | 补齐证据后推进 |

权重集中或短描述被低估可作为方法局限记录，不要求编码结果使某组获胜。数值异常或ID错位属于技术错误，不能当作负结果推进。

## 4. 推进、恢复与交接

将质量表和所有缓存哈希交给Phase 3/4。编码中断只恢复未完成item块；核对模型revision、item顺序与dtype，不能拼接不同模型产生的分块。缓存改变后，相关量化器与所有下游产物失效；只修改统计展示不应重训模型。

## 5. 运行记录模板

未来填写 `outputs/v2/evidence/phase2/run_records.md`：

| 字段 | 初始值 |
|---|---|
| run_id / status | `P2-EMB-A`、`P2-EMB-BC`、`P2-EMB-DIAG` / 未开始 |
| encoder/code revision、dtype/batch | 待执行记录 |
| manifest/config/cache hashes | 待执行记录 |
| command/cwd/start/end | 待执行记录 |
| item/view counts、norm/finite checks | 待执行验证 |
| token calls/wall time/GPU-hours/VRAM | 待执行记录 |
| evidence/exception/decision | 待执行；未验收 |

## 6. 检查清单

- [ ] 固定质量公式和冻结编码检查全部通过；
- [ ] A/B/C及所有训练/诊断视图缓存齐全；
- [ ] B/C干净表示、目录顺序和共享来源一致；
- [ ] 无推荐训练、无正式test指标；
- [ ] `outputs/v2/evidence/phase2/acceptance.md`记录实测证据和推进结论。
