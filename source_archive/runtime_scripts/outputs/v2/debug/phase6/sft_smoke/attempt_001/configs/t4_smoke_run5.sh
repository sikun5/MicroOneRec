#!/bin/bash
cd /hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec
PY=.venv-phase3/bin/python
SMOKE=outputs/v2/debug/phase6/sft_smoke/attempt_001
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "smoke v5 budget (amended after v4 hits=0): micro=2 accum=1 (effective 2, ~50 optimizer steps/epoch), max 60 epochs, stop eval<=20% init; memorization needs far more steps than v4's ~50 total"
for g in A B C; do
  echo "== SMOKE512 $g =="
  $PY scripts/v2_sft_smoke100.py --stage train --group $g --micro 2 --accum 1 --epochs 60 --seed 42
  rc=$?
  echo "SMOKE_RC $g $rc"
  if [ $rc -ne 0 ]; then echo "SMOKE_FAIL $g"; exit $rc; fi
done
echo "T4_SMOKE_ALL_DONE"
