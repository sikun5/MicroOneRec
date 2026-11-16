#!/bin/bash
cd /hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec
PY=.venv-phase3/bin/python
SMOKE=outputs/v2/debug/phase6/sft_smoke/attempt_001
for g in A B C; do
  $PY scripts/v2_sft_smoke_reload.py --group $g --smoke_root $SMOKE --seed 42
done
echo "T4_RELOADS_DONE"
