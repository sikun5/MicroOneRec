#!/bin/bash
cd /hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec
PY=.venv-phase3/bin/python
SMOKE=outputs/v2/debug/phase6/sft_smoke/attempt_001
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "eval_micro=4 (descended from 16 after OOM at fp32 logits upcast; train micro unchanged=8); alloc_conf=expandable_segments:True"
for g in A B C; do
  echo "== SMOKE $g =="
  $PY scripts/v2_sft_smoke100.py --stage train --group $g --micro 8 --seed 42
  rc=$?
  echo "SMOKE_RC $g $rc"
  if [ $rc -ne 0 ]; then echo "SMOKE_FAIL $g"; exit $rc; fi
done
echo "T4_SMOKE_ALL_DONE"
