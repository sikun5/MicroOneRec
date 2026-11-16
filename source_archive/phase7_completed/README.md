# Phase7 completed experiment source archive

Phase7 was accepted on 2026-11-17. This archive preserves the executed experiment
sources without changing training mathematics or the original working-record
copies. [manifest.json](manifest.json) records each source, original runtime
location, full SHA-256, role, and the accepted evidence anchors. The training
entries under `scripts/` were already versioned and remain unchanged.

## Contents and interfaces

| Source | Historical interface / purpose |
|---|---|
| `h100_telemetry.py` | `--evidence DIR --entry scripts/v2_train_grpo_audit.py` followed by original training arguments; `--self-test` is a CPU control. |
| `h100_supervisor.py` | One JSON config argument; captures actual exit and refuses duplicate/existing outputs. |
| `h100_final_audit.py` | `--method B\|C --evidence DIR`; audits existing final weights, raw/reward records and native EMA snapshots on CPU. |
| `post_supervisor.py` | One JSON config argument containing `repo`, `evidence`, and ordered `stages`; requires the formal supervisor to have exited successfully. |
| `sync_only.py` | One evidence-directory argument; compares existing native step512 snapshots on CPU. |
| `spike_diag.py` | One C evidence-directory argument; inspects existing logs and installed clipping source. |
| `phase7_final_check.py` | `--out DIR`; successful attempt002 checker, requiring BF16 SFT and FP32 RL storage. |
| `package_phase7.py` | Uses repository cwd and completed date-specific evidence; creates canonical summaries and index with no-overwrite guards. |
| `verify_phase7_package.py` | Historical byte/index/process verification; deliberately pinned to experiment HEAD `1f5c01c`. |
| `preflight/probe.py`, `preflight/supervisor.py` | Executed M3 environment, native CUDA and model-load checks; the supervisor invokes both existing venvs. |
| `configs/B-RL.recorded.json`, `configs/C-RL.recorded.json` | Exact accepted remote configuration bytes, including actual command/environment and original paths. These describe completed runs. |

## Historical execution context

B/C ran on H100 at `/data/user/swu073/dsaa5020/MicroOneRec`, using `.venv-phase3`.
The immutable audit entry also imports the old
`/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec` prefix through its compatibility
link. The preflight sources use `.venv-phase2` and `.venv-phase3`. These snapshots
retain their original date-specific paths and assumptions; they are not portable
replacements for the current entrypoints.

To inspect a completed invocation, read its recorded JSON `command` and
`environment`. To reconstruct original source placement, use the manifest's
`original_runtime_paths` and verify the copied SHA. Restore only into an explicit
fresh runtime or an already verified matching file; never overwrite unrelated
files. Several tools write evidence reports, while the package builder refuses
existing summaries. Do not recursively run this archive or replay completed
training as part of source installation.

The historical package verifier checks HEAD `1f5c01c` and old task identities.
After this archive commit it will intentionally reject the new HEAD; its original
successful result remains an experiment-time statement, not a claim that the
unchanged verifier passes at arbitrary later commits. Updating that interface is
outside this snapshot.

## Results and boundaries

[Phase7 actual summary](../../global_guide/phase7_grpo_training.md) records the
approved protocol, selected checkpoints, resource scopes, accepted limitations and
Phase8 handoff. Model weights, data, environments, raw predictions, metrics,
process receipts and logs remain under ignored `outputs/v2/` or the original
working-record directory. Their selected hashes are anchored by the manifest;
the payloads are not included here. SSH, migration and polling orchestration are
also excluded. No model loading, inference, training or historical package
verification was run while making this source archive.
