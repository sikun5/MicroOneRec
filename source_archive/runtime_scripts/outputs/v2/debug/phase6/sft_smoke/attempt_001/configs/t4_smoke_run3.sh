#!/bin/bash
cd /hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec
PY=.venv-phase3/bin/python
SMOKE=outputs/v2/debug/phase6/sft_smoke/attempt_001
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "smoke micro=2 (common feasible at enforced 512 per probes); accum=64/2=32 (effective 64); eval_micro=4; alloc_conf=expandable_segments:True"
for g in A B C; do
  echo "== SMOKE512 $g =="
  $PY scripts/v2_sft_smoke100.py --stage train --group $g --micro 2 --seed 42
  rc=$?
  echo "SMOKE_RC $g $rc"
  if [ $rc -ne 0 ]; then echo "SMOKE_FAIL $g"; exit $rc; fi
done
echo "T4_SMOKE_ALL_DONE"
