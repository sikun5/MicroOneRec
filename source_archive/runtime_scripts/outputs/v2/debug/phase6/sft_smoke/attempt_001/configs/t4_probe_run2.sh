#!/bin/bash
cd /hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec
PY=.venv-phase3/bin/python
SMOKE=outputs/v2/debug/phase6/sft_smoke/attempt_001
# archive short-sequence probe history (invalid for the 512 requirement)
HIST=$SMOKE/memory_probe/_history_shortseq_invalid
mkdir -p $HIST
for g in A B C; do
  [ -d $SMOKE/memory_probe/$g ] && mv $SMOKE/memory_probe/$g/* $HIST/ 2>/dev/null && mkdir -p $SMOKE/memory_probe/$g
done
mv $SMOKE/memory_probe/common_config.txt $HIST/ 2>/dev/null
echo "PROBE LENGTH ENFORCED TO 512 (max_length padding); short-seq results archived"
declare -A FEAS
for g in A B C; do
  FEAS[$g]=ERR
  for m in 16 8 4 2 1; do
    extra=""
    [ $m -eq 1 ] && extra="--grad-ckpt"
    echo "== PROBE512 $g micro=$m =="
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
echo "FEAS512 A=${FEAS[A]} B=${FEAS[B]} C=${FEAS[C]} COMMON=$COMMON"
if ! [[ "$COMMON" =~ ^[0-9]+$ ]]; then echo "NO_COMMON_CONFIG"; exit 9; fi
echo "$COMMON" > $SMOKE/memory_probe/common_config_512.txt
for g in A B C; do
  echo "== ACCUM512 $g micro=$COMMON =="
  $PY scripts/v2_sft_memprobe.py --group $g --micro $COMMON --stage accum --smoke_root $SMOKE
done
echo "T4_PROBES512_ALL_DONE"
