#!/bin/bash
cd /hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec
PY=.venv-phase3/bin/python
SMOKE=outputs/v2/debug/phase6/sft_smoke/attempt_001
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "smoke v4 micro=2; save-before-generation ordering; generation uses data.Tokenizer wrapper"
for g in A B C; do
  echo "== SMOKE512 $g =="
  $PY scripts/v2_sft_smoke100.py --stage train --group $g --micro 2 --seed 42
  rc=$?
  echo "SMOKE_RC $g $rc"
  if [ $rc -ne 0 ]; then echo "SMOKE_FAIL $g"; exit $rc; fi
done
echo "T4_SMOKE_ALL_DONE"
