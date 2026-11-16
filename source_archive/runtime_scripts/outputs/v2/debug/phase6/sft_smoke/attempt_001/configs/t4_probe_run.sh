#!/bin/bash
cd /hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec
PY=.venv-phase3/bin/python
SMOKE=outputs/v2/debug/phase6/sft_smoke/attempt_001
mkdir -p $SMOKE/memory_probe $SMOKE/configs $SMOKE/reports
cat > $SMOKE/configs/t4_probe_config.json <<'CFG'
{
 "stage": "t4-memory-probe-frozen",
 "date": "2026-11-12",
 "descent_order": [16, 8, 4, 2, 1],
 "stop_rules": "OOM -> descend one level, keep OOM json; micro1 runs with gradient checkpointing; micro1+ckpt fail = resource block P0; no quantized optimizers, no CPU offload, no truncation below 512",
 "probe_content": "seed42 common expanded model (init 820b2070...), real SidSFTDatasetV2 samples, seq<=512 full supervision, 2 real optimizer.step (AdamW delayed state), validation pass, save model+optimizer+tokenizer, separate-process reload check",
 "accum_content": "at common feasible micro: exact accumulation 1024/micro micro fwd/bwd -> 1 optimizer.step, all micro steps counted",
 "eval_micro": 4,
 "lr": 3e-4,
 "source_binding": {"init_state_dict_sha256": "820b20708d8554bee82f153f1d8f93454e369cd378c9ab6a20ee8ea33384e161"}
}
CFG
declare -A FEAS
for g in A B C; do
  FEAS[$g]=ERR
  for m in 16 8 4 2 1; do
    extra=""
    [ $m -eq 1 ] && extra="--grad-ckpt"
    echo "== PROBE $g micro=$m =="
    $PY scripts/v2_sft_memprobe.py --group $g --micro $m --stage probe --smoke_root $SMOKE $extra
    rc=$?
    if [ $rc -eq 0 ]; then FEAS[$g]=$m; break; fi
    if [ $rc -eq 3 ]; then echo "OOM $g micro$m"; continue; fi
    echo "HARD_ERROR $g micro$m rc=$rc"; break
  done
  m=${FEAS[$g]}
  if [ "$m" != "ERR" ]; then
    $PY scripts/v2_sft_reload_check.py --group $g --save_dir $SMOKE/memory_probe/$g/micro$m --out $SMOKE/memory_probe/$g/reload_micro$m.json
  fi
done
COMMON=$(printf '%s\n' "${FEAS[A]}" "${FEAS[B]}" "${FEAS[C]}" | sort -n | head -1)
echo "FEAS A=${FEAS[A]} B=${FEAS[B]} C=${FEAS[C]} COMMON=$COMMON"
if ! [[ "$COMMON" =~ ^[0-9]+$ ]]; then echo "NO_COMMON_CONFIG"; exit 9; fi
echo "$COMMON" > $SMOKE/memory_probe/common_config.txt
for g in A B C; do
  echo "== ACCUM $g micro=$COMMON =="
  $PY scripts/v2_sft_memprobe.py --group $g --micro $COMMON --stage accum --smoke_root $SMOKE
done
echo "T4_PROBES_ALL_DONE"
