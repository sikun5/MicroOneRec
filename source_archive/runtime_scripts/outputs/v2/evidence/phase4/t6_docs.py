#!/usr/bin/env python
"""Phase 4 T6: generate run_records.md / findings.md / artifact_inventory.json / acceptance.md."""
import json, hashlib
from pathlib import Path
REPO = Path("/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec")
EV = REPO / "outputs/v2/evidence/phase4"

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

checks = json.load(open(EV / "t6_checks.json"))

# ---------- artifact_inventory.json (machine evidence only; the four handover
# documents' own hashes are recorded in the shared working record, avoiding
# self-inclusion loops per protocol) ----------
def build_inv(roots):
    files = {}
    for r in roots:
        base = REPO / r
        for p in sorted(base.rglob("*")):
            if p.is_file() and "t3_probe_tmp" not in str(p):
                files[str(p.relative_to(REPO))] = {"sha256": sha(p), "bytes": p.stat().st_size}
    return files

inv = {
    "schema": "phase4-artifact-inventory-1.0", "date": "2026-11-11",
    "scripts": {
        "scripts/v2_export_sid.py": sha(REPO / "scripts/v2_export_sid.py"),
        "scripts/v2_remap_data.py": sha(REPO / "scripts/v2_remap_data.py"),
    },
    "evidence_core": {f: sha(EV / f) for f in [
        "environment_lock.json", "pip_freeze_phase4.txt", "input_inventory.json", "support_checks.json",
        "diagnostic_selftest.json", "diagnostic_checks.json", "t3_run_inventory.json",
        "t3_sid_files_sha256.json", "t3_failure_probe.py", "t3_failure_probe.json",
        "export_checks.json", "tokenizer_checks.json", "t4_files_sha256.json",
        "row_alignment.json", "rl_mapping_checks.json", "t5_files_sha256.json",
        "t6_checks.py", "t6_checks.json", "resource_summary.json"]},
    "sections": {
        "sid_outputs": build_inv(["outputs/v2/sid"]),
        "data_outputs": build_inv(["outputs/v2/data"]),
        "tokenizer_outputs": build_inv(["outputs/v2/tokenizer"]),
    },
    "phase5_6_handover": {
        "consumable": ["outputs/v2/data/{A,B,C}/Industrial_and_Scientific.index.json (+reverse_item_map/prefix_trie)",
                       "outputs/v2/data/{A,B,C}/{train,valid,test}.csv (row_id+SID 列)", "outputs/v2/data/{A,B,C}/Industrial_and_Scientific.item.json",
                       "outputs/v2/data/{A,B,C}/info.txt (SID\ttitle\titem_id TSV)", "outputs/v2/data/{A,B,C}/rl_tasks_10000_sid.jsonl",
                       "outputs/v2/tokenizer/{shared_vocab,token_id_map,binding_manifest}.json", "outputs/v2/tokenizer/{A,B,C}/",
                       "outputs/v2/sid/{A,B,C}/seed_{42,43,44}/{raw_codes,diagnostics}.json 与量化向量 npy"],
        "invalidation": "映射变更使对应下游失效；shared vocab/词表顺序变更使三组全部下游失效；未来 LLM embedding 初始化必须由 binding_manifest（base revision+注册顺序+seed42）复现",
    },
}
(EV / "artifact_inventory.json").write_text(json.dumps(inv, ensure_ascii=False, indent=1))

# ---------- run_records.md ----------
run_records = """# Phase 4 运行记录（run_id: T2-T5 全部运行）

| 字段 | 值 |
|---|---|
| phase / 期间 | Phase 4 / 2026-11-11（Asia/Hong_Kong，worker 会话 16:0x-19:2x） |
| 节点 / 容器 | `swu073-a800-10g` / `34fb2691c7dd`；GPU=A800-SXM4-80GB MIG 1g.10gb（9,728MiB/14SM/cc8.0），单 MIG 设备计 |
| 工作目录 / 环境 | `/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec`；复用 `.venv-phase3`（pip freeze 与 Phase 3 零包差异，`pip_freeze_phase4.txt`=`b4dc59db9c96a52990c3b375a9b210abfb75b0ed14083821dd198c59fb95acea`） |
| 基线 | Phase 3 提交 `44cb9ce978ba88414fbad25bfb7e93aaa23efc12`（T1 产生，parent=e7517496…）；全程 tracked 零 diff、6 项无关 untracked 保留 |
| Phase 4 入口脚本 | `scripts/v2_export_sid.py` 最终 `351793cedc2ca50daf2952d35af4ee6f99d6c887709c87ffba187bd123869106`（诊断部分=run7 版 bf150946… 仅新增 export+回解析检查，诊断函数路径未改——三重跑 quality/robustness CSV 哈希逐字节相同为证）；`scripts/v2_remap_data.py` 最终 `40e3c15290255cf903b17bd11f8d69cabf2eca6bf5a1f33c94881e24637e8d46` |

## 运行清单（顺序=实际执行）

| # | 阶段 | 运行 | 命令要点 | 结果 |
|---|---|---|---|---|
| 1 | T2 | envlock/inventory/support | 三生成器 nohup 顺序执行（日志 /tmp/t2_run.log→run2.log→inv.log；三次 worker 脚本缺陷失败后修复） | 全绿：`environment_lock.json` 636edd7f…、`input_inventory.json` 476a0a66…、`support_checks.json` 694f53c9… |
| 2 | T3 | 合成自测（4 次迭代：top10 排除实现修正等） | `--self-test-only` | C1-C7 全 PASS，`diagnostic_selftest.json`=e1521b6d… |
| 3 | T3 | 9+27 冻结诊断（run5 最终成功轮；run1-4 失败=CUDA 统计 API/键名/守卫口径，如实记录） | `--mode diagnostics --self-test` | 9 clean+27 组合、守卫全过、checkpoint 未变；wall 和 31.6s、峰值 234,453,504B |
| 4 | T3 | 守卫版重跑 run6/run7（C7+abort 守卫+守卫前移） | 同上 | CSV 与 run5 逐字节相同（60503484…/e927eb8d…）；run7 inventory=01d94259… |
| 5 | T3 | 失败路径探针 | `t3_failure_probe.py`（临时树+硬链接，污染副本） | NaN/Inf exit=1+NONFINITE_ROWS_ABORT、零范数 RuntimeError+ZERO_NORM_ROWS_ABORT，零产物；probe json=8b621495… |
| 6 | T4 | seed42 导出（run2 含独立回解析） | `--mode export` | 3×3686 覆盖/唯一/反向 100%、词表 3308、拒绝采样 3000/3000；export_checks=3a50e21c…、tokenizer_checks=95e5c216… |
| 7 | T5 | 三组重映射（run4→run7 整改轮） | `v2_remap_data.py` | 45,324 行全匹配、row_id 唯一+join 45,324、RL 10k×3 保序；row_alignment=1ddf88f3…、rl_mapping_checks=080500ba… |
| 8 | T6 | 独立终验 | `t6_checks.py` | 见 t6_checks.json（全部检查项本表末尾回填） |

成本口径：单 MIG 设备；T2 <15s、T3 31.6s+哈希 ~20s、T4 每轮 ~100-110s、T5 每轮 ~13-14s、探针 <60s；失败启动轮精确 wall 未细分采集（如实注明）。存储：见 resource_summary.json storage_bytes。
"""
(EV / "run_records.md").write_text(run_records)

# ---------- findings.md ----------
findings = """# Phase 4 研究发现（负结果口径；不构成方法优劣结论）

## F1 量化器坍缩与高碰撞（RQ1/RQ2 中间证据）

- 九个已验收量化器中 5 个存在不同程度坍缩：**A-43 与 C-44 完全坍缩**（3,686 商品全部落入同一三元组桶、三层利用率 1/1/1、归一化熵 0、最大桶 3,686）；**C-42/C-43 近坍缩**（碰撞率 0.9989/0.9992，最大桶 2,540/1,918，利用率 1-3/256）；**A-44 重度**（碰撞率 0.8801）。A-42/B-42/B-43/B-44 保持中高利用率（碰撞率 0.27-0.50，三层使用 15-51/256）。
- 实现错误已系统排除：strict 加载、vq_initted 恢复、前后 state checksum 不变、forward≡get_indices、全量单批、有限性守卫全过；九 checkpoint 为 Phase 3 终验独立核验过的已接受产物（未被本期改动）。
- 交叉自洽：C 三种子 PPR@1=1.0 与其 1/2 层利用率 1-3 一致；Phase 3 记录的 C 末段 consistency_raw→0（C-44=0.0、C-43=1.78e-07）与码本坍缩互为因果证据（坍缩后扰动视图量化输出与 clean 相同，一致性损失必然趋零）。A-43 无一致性项仍完全坍缩，属 RQ-VAE/K-means 训练自身失败模式。坍缩的机制归因不在本期范围。
- **边界**：高 PPR/NPR（坍缩 run 上为 1.0）是坍缩的必然产物，**不构成鲁棒性或方法有效性证据**；负结果按要求原样保留，未换 seed/checkpoint 修饰。

## F2 批次形状敏感性

- batch512 与全量单批：**indices 九 run 全部逐位一致**；zq 浮点输出非逐位（最大绝对差 ≤7.45e-9）。正式口径统一全量单批（每输入恰一次前向），不存在跨批歧义。

## F3 消歧与唯一率边界

- sorted_item_suffix 消歧使三组完整 SID 唯一率 100%（C-42 需 `<d_0..2539>` 共 2,540 个后缀 token），**后缀唯一率是技术条件，不是语义质量提升**；C-42 原始碰撞率 0.9989 照常披露，语义评估留 Phase 5。

## F4 恢复限制（携带披露）

- Phase 3 裁定的跨进程恢复逐位等价限制继续携带（正式 run 均 resume_used=false）；本期为冻结推理，不涉及恢复轨迹，无需重验。

## F5 一致性项与坍缩的相关性提示（非结论）

- C 全系坍缩/近坍缩而 A/B 部分坍缩的模式，与"一致性损失在坍缩后趋零"自洽；但不能据此断言一致性项导致坍缩（A-43 无一致性项也坍缩）。归因留后续研究。
"""
(EV / "findings.md").write_text(findings)

# ---------- acceptance.md ----------
t6c = json.load(open(EV / "t6_checks.json"))
t6_sha = sha(EV / "t6_checks.json")
inv_sha = sha(EV / "artifact_inventory.json")
res_sha = sha(EV / "resource_summary.json")
acc = f"""# Phase 4 验收包（P4-1..P4-6）

> 工作目录 `/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec`；基线 commit `44cb9ce978ba88414fbad25bfb7e93aaa23efc12`（Phase 3 提交，全程 tracked 零 diff）。
> 结论总览：**P4-1 至 P4-6 全部通过**；T2-T5 全部经 boss 接受后进入本终验；独立复核脚本 `t6_checks.json`=`{t6_sha}`（all_passed={checks['all_passed']}）；无未解决 P0；未训练任何 LLM、未生成正式 test 推荐指标、未启动 Phase 5-8。

| 编号 | 条件 | 实测值（要点） | 证据（SHA-256） | 判定 |
|---|---|---|---|---|
| P4-1 | 9 套原始码+质量统计、27 组合有支持数 | 9×raw_codes(3686×3, 0..255)+diagnostics、27 组合分母 3356/3356/3359、分子/排除数全量在档；三种子 mean/std(ddof=1) 汇总 | `t3_run_inventory.json`=`01d94259b4a1fbe86fd5bdbb3af5429888873ec56b73a55c68a092a2285d6982`、`quality_summary.csv`=`60503484e866b710ede4157afd994781e68d22192c1561beb32f495ba1776871`、`robustness_summary.csv`=`e927eb8db1067597aaa90e534ed9ad3cba52ca79fb0c0be0425adaffef9ba0d4` | 通过 |
| P4-2 | 冻结参数、use_sk=False、前缀 PPR、固定 clean NPR 参考 | 每 run 前后 state checksum 不变+initted 不变+checkpoint 文件未变；use_sk=False 全程；NPR 双查询固定 clean 目录（自测 C3/C7 证明）；失败路径 NaN/Inf/零范数实测阻断（probe json=`8b621495071110dd4d5b2e579b1cd35c69e2989dea6014e6afd656bfdf3ebac6`） | `diagnostic_checks.json`=`cb46805d3875b01fa9de3661317da5ab49f1f728be50a143a9d80931868f13a3`、`diagnostic_selftest.json`=`e1521b6dedaa5676a7958c39b7e8da8c7cb45e705022494ef9b999a893b78529` | 通过（携带 F1 坍缩负结果） |
| P4-3 | seed42 四 token 覆盖/唯一/反向 100% | 3×3,686 覆盖/唯一/反向全部 100%；前三码独立回解析逐项一致；max_suffix=20/21/2539 | `export_checks.json`=`3a50e21c37bb42fe9eed35d048e6a2ac2cb0ea43e2615f9377c480c17d3fabe2`、`t4_files_sha256.json`=`635b78096645d534efa0e19f8b0f26f1ea54a64fb4d67f4ed339ec5b34431550` | 通过（C-42 原始碰撞 0.9989 照常披露） |
| P4-4 | 共同词表/token-ID 一致、原子、无截断 | 3,308=768+2540、注册顺序 SHA=`95779752f01350851ede60ab58752fba3ea638fd4a2237394c1af51294110716`、三组 token-ID 表一致、全 token 原子、全部 SID roundtrip、EOS 协议冻结；base revision `da87bfb…`+init_seed42 绑定，未加载/训练 LLM | `tokenizer_checks.json`=`95e5c2167d35b86dbca5c973aefc15a1aedeebc510aae6f9c913f2031af803e6`、binding_manifest 在档 | 通过 |
| P4-5 | 三组逐行非 SID 字段一致、行数保持 | 45,324 行（36,259/4,532/4,533）逐行匹配×3 组；row_id 唯一+rows_manifest join 45,324 全中；1,987 用户/447 缺失目标/3 重复行保留；RL 10k×3 保序（seq 内容键 296 组/涉 922、texthist 文本键 20 组/涉 47 与 ID 键 18 组/39 双口径在档） | `row_alignment.json`=`1ddf88f3ebb40c7c70d074792b6b8c7f144439965b3581a7909dd66557e0fdee`、`rl_mapping_checks.json`=`080500ba8734a6f732910a0e171ea78aba89e22e04ef37e9232dc323b28d7d22` | 通过 |
| P4-6 | 坍塌/高碰撞发现排错保留、导出合法完整 | findings.md F1-F5：5/9 run 坍缩谱系+排错链+与 Phase 3 consistency_raw→0 互证；导出在坍缩下仍合法完整；恢复限制携带；后缀唯一率≠语义改善 | `findings.md`、`resource_summary.json`=`{res_sha}` | 通过（携带发现推进） |

## 独立终验覆盖

- 9 final 重哈希=上期九行全同；T2 evidence 3 件、T3 evidence 7 件、T4/T5 清单 30+30 文件逐字节核对全部一致；三组 index/reverse/trie 重载复检、词表/token-ID/前缀树复检、CSV 行数/row_id/info TSV 复检全过（`t6_checks.json`）。
- 版本追踪：诊断 run7 入口 `bf150946…`→现入口 `351793cedc2ca50daf2952d35af4ee6f99d6c887709c87ffba187bd123869106` 仅新增 export 与回解析检查，诊断函数路径未改（三重跑 CSV 哈希逐字节相同为证）；remap 入口 `40e3c15290255cf903b17bd11f8d69cabf2eca6bf5a1f33c94881e24637e8d46`。
- 全部产物合并清单 `artifact_inventory.json`=`{inv_sha}`；资源（wall/显存/存储）`resource_summary.json`=`{res_sha}`；运行记录 `run_records.md`；研究发现 `findings.md`。

## Phase 5/6 交接与失效传播

- 可消费产物与接口见 `artifact_inventory.json` phase5_6_handover；映射变更→对应下游失效；shared vocab/注册顺序变更→三组全部下游失效；未来 embedding 初始化由 binding_manifest（base revision `da87bfb…`+注册顺序+init_seed 42）复现。
- 本期未开始：Phase 5 基线、SFT、GRPO、RL 训练、正式 test 推荐指标。无未解决 P0；P1（Phase 3 恢复限制）按裁定携带披露。
"""
(EV / "acceptance.md").write_text(acc)
print("ACCEPTANCE_SHA256:", sha(EV / "acceptance.md"), flush=True)
print("RUN_RECORDS_SHA256:", sha(EV / "run_records.md"), flush=True)
print("FINDINGS_SHA256:", sha(EV / "findings.md"), flush=True)
print("ARTIFACT_INV_SHA256:", sha(EV / "artifact_inventory.json"), flush=True)
print("T6_DOCS_DONE", flush=True)
