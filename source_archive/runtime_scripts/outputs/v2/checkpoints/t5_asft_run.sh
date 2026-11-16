#!/bin/bash
cd /hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec
PY=.venv-phase3/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p outputs/v2/checkpoints/A-SFT
$PY scripts/v2_sft.py \
  --base_model /hpc2hdd/home/swu073/sikun/dsaa5020/models/Qwen3-0.6B-Base \
  --train_file outputs/v2/data/A/train.csv \
  --eval_file outputs/v2/data/A/valid.csv \
  --output_dir outputs/v2/checkpoints/A-SFT \
  --seed 42 --batch_size 1024 --micro_batch_size 2 --eval_micro 4 \
  --num_epochs 10 --learning_rate 3e-4 --cutoff_len 512 \
  --eval_step 0.05 --save_total_limit 1 --warmup_steps 20 --early_stopping_patience 3
echo "A_SFT_EXIT_CODE=$?"
