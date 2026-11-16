#!/bin/bash
cd /hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec
PY=.venv-phase3/bin/python
SMOKE=outputs/v2/debug/phase6/sft_smoke/attempt_001
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for g in A B C; do
  echo "== STRICTREGEN $g =="
  $PY scripts/v2_sft_smoke_regen.py --group $g --smoke_root $SMOKE --seed 42
  echo "REGEN_RC $g $?"
done
echo "T4_REGEN_ALL_DONE"
