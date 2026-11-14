#!/bin/bash
# Train -> generate -> evaluate every arm on the 10-image subset, one at a time.
# Resumable: a stage whose artifacts pass scripts/ops/check_fewshot_arm.py is
# skipped. Stops (HALT in campaign.log) instead of moving on from an OOM, a NaN
# loss, or artifacts that fail the check; I/O deaths are retried.
set -u -o pipefail
cd /workspace/Improving-CA-LoRA
export HF_HOME=/workspace/.cache/huggingface
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONUNBUFFERED=1
PY=/workspace/venv-calora/bin/python
#
# The defaults below (CONFIG, RUN, SOURCE, ARMS) are those of the earlier few-shot
# campaign (2026-11-02), whose CA arms used selections measured on real Cityscapes
# frames; SOURCE=runs/sdxl_cityscapes is the selection directory of that earlier
# experiment. Of that campaign only E_random, E_all and control_base are kept
# (results/sdxl_cityscapes_fewshot10/): they do not depend on CA and were not rerun.
#
# The reported experiment (self-generated CA; A_t81 / B_top3 / C_t201 x style,
# viewpoint; 2026-11-10) ran with the overrides below. They are reconstructed from
# its campaign.log (arm order, no control stage), its config.yaml and the selection
# files it copied; the command line itself was not logged.
#   CONFIG=configs/sdxl_cityscapes_selfgen_fewshot10.yaml RUN=runs/sdxl_cityscapes_selfgen_fewshot10 \
#   SOURCE=runs/ca_selfgen ARMS="B_top3_style B_top3_viewpoint A_t81_style A_t81_viewpoint \
#   C_t201_style C_t201_viewpoint" CONTROL=0 bash scripts/ops/run_fewshot10.sh
CONFIG=${CONFIG:-configs/sdxl_cityscapes_fewshot10.yaml}
RUN=${RUN:-runs/sdxl_cityscapes_fewshot10}
SOURCE=${SOURCE:-runs/sdxl_cityscapes}
CONTROL=${CONTROL:-1}
if [ -n "${ARMS:-}" ]; then read -r -a ARMS <<< "$ARMS"; else
ARMS=(A_t81_style A_t81_viewpoint B_top3_style B_top3_viewpoint C_t41_style C_t41_viewpoint
      D_t1_style D_t1_viewpoint E_random E_all)
fi
LOGS="$RUN/logs"
mkdir -p "$LOGS"
CAMPAIGN="$RUN/campaign.log"

say() { echo "$(date -u +%FT%TZ) $*" >> "$CAMPAIGN"; }
halt() { say "HALT $*"; exit 1; }

check() { "$PY" scripts/ops/check_fewshot_arm.py --run-dir "$RUN" --arm "$1" --stage "$2" >> "$LOGS/check.log" 2>&1; }

# stage <arm> <stage> <attempts> <cmd...>
stage() {
  local arm=$1 name=$2 attempts=$3 log rc before
  shift 3
  log="$LOGS/${name}_${arm}.log"
  if check "$arm" "$name"; then say "SKIP $arm $name"; return 0; fi
  for attempt in $(seq 1 "$attempts"); do
    say "START $arm $name attempt=$attempt"
    # The log is appended across attempts and relaunches; only this attempt counts.
    before=$(cat "$log" 2>/dev/null | wc -l)
    "$@" >> "$log" 2>&1
    rc=$?
    say "END $arm $name rc=$rc"
    if tail -n +"$((before + 1))" "$log" | grep -aqE "CUDA out of memory|OutOfMemoryError"; then
      halt "$arm $name out of memory"
    fi
    if tail -n +"$((before + 1))" "$log" | grep -aqE "loss=nan|loss=inf"; then
      halt "$arm $name non-finite loss"
    fi
    if [ "$rc" = 0 ]; then
      # Twice: the check reads every artifact back from a volume that throws EIO.
      check "$arm" "$name" || { sleep 60; check "$arm" "$name"; } \
        || halt "$arm $name artifacts failed the check (see logs/check.log)"
      return 0
    fi
    say "RETRY $arm $name rc=$rc"
    sleep 60
  done
  halt "$arm $name failed $attempts times"
}

for arm in "${ARMS[@]}"; do
  src="$SOURCE/selection_$arm.json"; dst="$RUN/selection_$arm.json"
  for attempt in 1 2 3; do
    cp "$src" "$dst"
    [ "$(sha256sum < "$src")" = "$(sha256sum < "$dst")" ] && break
    [ "$attempt" = 3 ] && halt "selection copy of $arm does not verify"
  done
done
say "CAMPAIGN START"

for arm in "${ARMS[@]}"; do
  stage "$arm" train 2 "$PY" scripts/train_lora.py --config "$CONFIG" \
    --selection "selection_$arm.json" --run-name "$arm"
  stage "$arm" generate 4 "$PY" scripts/generate_images.py --config "$CONFIG" \
    --lora "$PWD/$RUN/$arm/lora-final.safetensors" --run-name "$arm" --mode both --skip-existing
  stage "$arm" evaluate 3 "$PY" scripts/evaluate.py --config "$CONFIG" --run-name "$arm"
done

# The 0% control: nothing to train, sampled and scored like every other run.
if [ "$CONTROL" = 1 ]; then
stage control_base generate 4 "$PY" scripts/generate_images.py --config "$CONFIG" \
  --run-name control_base --mode both --skip-existing
stage control_base evaluate 3 "$PY" scripts/evaluate.py --config "$CONFIG" --run-name control_base
fi

say "CAMPAIGN END"
