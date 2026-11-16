#!/bin/bash
cd /hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec
PY=.venv-phase3/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for g in A B C; do
  echo "== FORMALRAW $g =="
  $PY scripts/t8_formal_raw.py --group $g --out_dir outputs/v2/checkpoints/$g-SFT --seed 42
  echo "RAW_RC $g $?"
done
echo "T8_RAW_ALL_DONE"
