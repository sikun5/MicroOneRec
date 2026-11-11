# Phase 2：质量特征与文本表示

> **状态：已完成并验收（2026-11-10，run_id `P2-EMB-A`/`P2-EMB-BC`/`P2-EMB-DIAG`）。**P2-1至P2-6全部实测通过（携带已披露限制交接），无未解决P0，正式缓存33文件齐全；验收证据见 `outputs/v2/evidence/phase2/acceptance.md`。  
> 导航：[主入口](experiment_setting.md) · [上一期：数据与环境](phase1_dataset_prep.md) · [下一期：RQ-VAE](phase3_rqvae_training.md)  
> 协议：[v2 方法定义](../../topic_minionerec_v2.md#4-abc-方法定义)。所有产物路径相对仓库根目录；本期实际执行目录为远端 `/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec`（Phase 1 已确认为指定工作目录的物理路径）。

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

> **实现状态（2026-11-10）：**`scripts/v2_build_embeddings.py` v1.1 已实现并验收（sha256 前 16 位 `83678bd245d84bb6`；阶段 quality/encode/diagnostics，memmap 进度恢复签名绑定 chunk_tag/max_length/模型/文本集 SHA）；smoke 入口 `scripts/v2_phase2_smoke.py`（`6587596e…`）；独立检查入口 `scripts/v2_phase2_final_checks.py`（`08806266…`）。实测冻结协议：独立环境 `.venv-phase2`（transformers 4.51.3）、revision `5cf2132a…` 实测 D=2560、BF16 权重、batch=1 无 padding、sdpa、FP32 masked mean/L2/缓存、tf32 off、max_length 2048 全量零截断。

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

### 3.1 验收结果（2026-11-10 实测，run_id `P2-EMB-A`/`P2-EMB-BC`/`P2-EMB-DIAG`）

| 编号 | 实测值 | 证据（sha256 前 16 位，完整值见 acceptance.md） | 结论 |
|---|---|---|---|
| P2-1 | 公式样例 18/18 + 独立重算 20 样本全对；q∈[0,1]、w∈[0,0.5] 全量成立；重复描述按公式降权实证（32 词重复 r=0.9333→q=0.0667）；B 空描述 327 行逐字节回退（np.array_equal）；229/327/3 三种缺失口径并存 | `quality_rule_checks.json` `b6473d6b0fac41a7`（18/18）、`cache_checks.json` `f6158d9287ec8863` | 通过 |
| P2-2 | 正式缓存恰 33 文件 = 13 向量数组 [3686,2560] float32 + 10 bool mask [3686] + 10 JSON；item_ids 三副本+canonical 一致且与 quality_features 同序；13 数组全局范数 ∈ [0.99999990, 1.00000011]，max\|norm−1\| = 1.052e-07；无 NaN/Inf/零范数 | `cache_checks.json` `f6158d9287ec8863`（12/12）、`t5_verify.json` `cd250f0c…`（14/14）、`t5_cache_table.json` `639d2b89…`（33 文件全表） | 通过 |
| P2-3 | B/C clean 整文件 SHA-256 相等（`06d5797f…`），B/C 诊断 7 文件逐字节相等；train 清单全 seed1701（3,686 条）、diagnostic 全 seed2701（11,058 条）零混用；views_source 绑定 `17258e1d…`；乱序输入按 item ID 逐字节恢复 | `cache_checks.json`（shared_cache_sha_proof）、`view_checks.json` `0041e5eb…` | 通过 |
| P2-4 | 4,021,774,336 参数全 BF16、全 requires_grad=False + eval、输出无 grad_fn；masked mean 双实现 diff=0.0（numpy 对照 <1e-9）；batch=1 无 padding（锁定协议，padding 对照 7.5e-4/9.8e-4 仅观测）；2048 上限全量零截断（最长 1,108 tokens）；生产路径小段重编码逐字节复现（历史 1.49e-8 经定向定位为检查器独立实现舍入差，非生产链非确定性） | `encoder_checks.json` `0beccfee…`（16/16）、`param_memory_check.json` `cf37ce78…`、`item1631_localization.json` `a0299bbd…` | 通过 |
| P2-5 | 视图 title/item 与目录 0 违规（3,686+11,058 条）；有效标记 3,356/3,356/3,356/3,359 与冻结清单一致；无效行（229+98+3+330）复制本组干净向量 0 违规；dropall 有效行输入=仅标题；扰动质量独立重算 | `view_checks.json` `0041e5eb…`、`id_view_checks.json` `d499a129…` | 通过 |
| P2-6 | metadata（A/B/C）含 revision/规则/dtype/join/占位/无效行/输入输出哈希且内部复算通过，可追溯 Phase 1（上游 22/22）；token 只读重建 34,229 条 = 3,040,709 tokens（raw=post，零截断）；wall 2,473.5s（阶段 2,452.88s + 未细分 20.62s）≈ 41.2 分钟 = 0.687 单 MIG 设备占用小时；33 正式文件 491,257,061 bytes，目录 841,796,519 bytes 含 .progress 工作缓存；B/C 共享编码只计一次（distinct 26,873 / 重复提交 7,356）；**T5 正式峰值显存未采集，如实披露** | `cost_accounting.json` `3206edf4…`、`run_records.md` `c10320f6…`、`upstream_checks.json` `4729937c…` | 通过（携带"峰值未采集"记录限制） |

## 4. 推进、恢复与交接

将质量表和所有缓存哈希交给Phase 3/4。编码中断只恢复未完成item块；核对模型revision、item顺序与dtype，不能拼接不同模型产生的分块。缓存改变后，相关量化器与所有下游产物失效；只修改统计展示不应重训模型。

> **交接状态（2026-11-10）：**P2-1至P2-6全部通过并完成交接；基线为 Phase 1 commit `ded0bc13059b25c3c20f6cbc1a4c1a7c48d97649`，33 个正式缓存逐文件哈希以 `outputs/v2/evidence/phase2/t5_cache_table.json`（`639d2b89…`）为基准（关键正本：quality_features `9df37c0d…`、canonical item_ids `f7c159fd…`、A clean `cb745f4c…`、B/C clean `06d5797f…`、C train_aug `871b817e…`、C valid_aug `7127c609…`）。恢复规则：中断按 memmap 进度签名（绑定 chunk_tag/max_length/模型/文本集 SHA）续跑；上游清单变化则全部缓存与下游失效；模型 revision 锁定不跟随 main。披露：T5 正式峰值显存未采集（T2/T4 探测 8,122/7,994 MiB 仅参考）；节点 GPU 为 MIG 1g.10gb（约 9.5 GiB）切片，后续全参数训练资源不由此保证。完整交接表与问题披露见 `outputs/v2/evidence/phase2/acceptance.md`（sha256 前 16 位 `dfcad2c07b711484`）。

## 5. 运行记录

运行记录实际文件：`outputs/v2/evidence/phase2/run_records.md`（sha256 前 16 位 `c10320f65603a77c`；run_id `P2-EMB-A`/`P2-EMB-BC`/`P2-EMB-DIAG`，2026-11-10 于节点 `swu073-a800-10g`/容器 `34fb2691c7dd`，含 R-P2-0/R-P2-SMOKE/R-P2-VRF 与失败 attempt 留档）；验收文件：`outputs/v2/evidence/phase2/acceptance.md`（sha256 前 16 位 `dfcad2c07b711484`）。下方为原模板：

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

- [x] 固定质量公式和冻结编码检查全部通过（quality_rule_checks 18/18、encoder_checks 16/16；生产路径小段重编码逐字节复现）；
- [x] A/B/C及所有训练/诊断视图缓存齐全（正式缓存 33 文件 = 13 向量 [3686,2560] + 10 mask [3686] + 10 JSON；train_aug 有效 3,356）；
- [x] B/C干净表示、目录顺序和共享来源一致（clean 与 7 个诊断文件字节一致；seed1701/2701 清单零混用；乱序 ID 按身份恢复）；
- [x] 无推荐训练、无正式test指标（未运行 RQ-VAE/SFT/GRPO；Phase 3/4 未放行未启动）；
- [x] `outputs/v2/evidence/phase2/acceptance.md`记录实测证据和推进结论（P2-1至P2-6 全部通过，携带已披露限制）。
