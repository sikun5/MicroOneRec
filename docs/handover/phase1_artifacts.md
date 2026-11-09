# Phase 1 已验收产物交接索引（P1-PREP-42）

Phase 1（数据与环境准备）已验收；运行产物位于远端仓库根下 `outputs/v2/`（大型清单/缓存，
有意不纳入 git 版本库）。本索引记录每项产物的仓库相对路径与 SHA-256 基准，供后续阶段核对。
验收依据与逐项证据见共享工作记录 `working-repo-phase1/working-record.md`（本地保管，不入库）。

- 基准建立时间：2026-11-09（Phase 1 终验）
- 代码：分支 `MicroOneRec`，Phase 1 起点 HEAD `c6a9ffd3c8e9cc4a4d6527d1b0d66601e21b1189`
- 数据源（只读，哈希见 `outputs/v2/manifests/file_hashes.json`）：
  `data/Amazon/index/Industrial_and_Scientific.item.json`、
  `data/Amazon/{train,valid,test}/Industrial_and_Scientific_5_2016-10-2018-11.csv`

## outputs/v2 产物（17 项）

| 路径（仓库相对） | SHA-256 |
|---|---|
| `outputs/v2/environment/environment_lock.json` | `b102c41efddd2b0970a1070ceeebc80ec32bf38d9830eb324422344db7a2cd87` |
| `outputs/v2/environment/pip_freeze.txt` | `f104eac93fce544d32ebc68ba19d2c63ed6394dcb8a24018fdadd3b2d10be727` |
| `outputs/v2/manifests/file_hashes.json` | `e8a801e84ae4ac50295051d3a05dae6e7c7370ab9ee1b08760774d9fab34be6b` |
| `outputs/v2/manifests/data_audit.json` | `a2db080e739f22bc0c0658c5dc6c185fb8553f1c2610ce2cf39ba34a3888f995` |
| `outputs/v2/manifests/rows_manifest.jsonl` | `b3e195c2c997b2b9a039c14bcaa0d4560395cfc76682d3140e3694feeb3ba1f3` |
| `outputs/v2/manifests/catalog_clean.jsonl` | `275f869fbd5b5cf9c731608b64a43ae5ee7fa3deb8ff8228ef2110759c1a5649` |
| `outputs/v2/manifests/rl_tasks_10000.jsonl` | `e77868cae2e4ffec6fda33dc1b2bee4b0f36d2800a2d270dfae0399bc5f11eb0` |
| `outputs/v2/manifests/train_views.jsonl` | `45540aec5e3f49c5b520e74682acd6e658cb48ce164f6c645025a570f6499510` |
| `outputs/v2/manifests/diagnostic_views.jsonl` | `17258e1d149c6937de9f97b08aee5461198bc87e8984b84ae9294175ccdea8d0` |
| `outputs/v2/evidence/phase1/rl_summary.json` | `636d402d659d1bd3482e445009b66961af57a017ea79ec00924ef34f0e6594ce` |
| `outputs/v2/evidence/phase1/views_summary.json` | `ebb26ef8a355b859ebe8088de5fe55ed18639c43e358499b51a1bc46728cbf71` |
| `outputs/v2/evidence/phase1/identity_checks.json` | `7a0a135acf05e861e9d7d24451ac9d93298970dcb2198d5758a06c401fe7a8a0` |
| `outputs/v2/evidence/phase1/reproducibility.json` | `d303d447998836f97252891e5b6962e3c531b480a8959e55d33b7e19e302d335` |
| `outputs/v2/evidence/phase1/run_records.md` | `5f240aa55a330bdc6cb959273238542238d6d9d7e8a5e2d824c454ae540b9fa6` |
| `outputs/v2/evidence/phase1/acceptance.md` | `8ad3963540b939b2822313f86e540ee5bf05c11b1687389c4fe5779de3452210` |
| `scripts/v2_prepare.py` | `2140447dcf000b5ad1e16a17e26309553fc914df68560d00f8f961e1f1f5662d` |
| `scripts/v2_prepare_v13.py` | `3d4c5ba1176bd3fa797c99fe164a30c88a38631d360694885ed7179799618955` |

（`scripts/v2_phase1_checks.py` = `f5346b98ce5de9b89a2c9c13ef4c86e96e7d5b82b780849f6f2130419521318e`，
与上述两脚本一并随本提交入库，见提交文件清单。）

## 模型来源（仓库外，revision 锁定）

| 模型 | 本地路径（仓库外） | revision |
|---|---|---|
| Qwen/Qwen3-0.6B-Base | `/hpc2hdd/home/swu073/sikun/dsaa5020/models/Qwen3-0.6B-Base` | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |
| Qwen/Qwen3-Embedding-4B | `/hpc2hdd/home/swu073/sikun/dsaa5020/models/Qwen3-Embedding-4B` | `5cf2132abc99cad020ac570b19d031efec650f2b` |

## 计数基准（Phase 1 实测）

- 目录 3,686 商品；原始缺失描述 229；清洗后空描述 327（含长度规则置空 98）
- train/valid/test = 36,259/4,532/4,533 行；test 1,987 用户、447 缺失目标行；45,324 唯一 row_id
- RL 任务 10,000（8,000/1,000/1,000）；训练有效视图 3,356；诊断有效视图 3,356/3,356/3,359

## 其他

- 未入库项（有意排除）：`outputs/v2/**` 运行产物（大文件，以上表哈希追踪）、
  模型权重与原始数据、`.env` 及一切凭据、`outputs/v2_repro_tmp/` 复核副本（非正本）。
- `global_guide/topic_minionerec_v2.md` 为用户添加的 untracked 文件，属用户已有修改，
  本提交不包含（保留原状）。
