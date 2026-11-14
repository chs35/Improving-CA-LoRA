#!/bin/bash
# Stage 2 of the self-generated CA re-measurement (2026-11-07), on one A100:
#   generate -> precision gate -> measure (20 t) -> verify -> score (13-grid)
#   -> fixed-t selections -> HANDOFF.md
# Resumable: every stage is skipped when its output is already complete
# (generation and encoding are sha256-verified, measure_ca.py --skip-existing
# re-checks each existing file's provenance). Stops with HALT in campaign.log
# on an OOM, a failed verification or a stage that keeps failing.
#
#   nohup bash scripts/ops/run_ca_selfgen.sh > runs/ca_selfgen/nohup.out 2>&1 &
#   python scripts/ops/campaign_status.py ...   # or: tail runs/ca_selfgen/campaign.log
#
# Env overrides: CONFIG, PY, GATE_SAMPLES (32), FORCE_PRECISION (skip the gate,
# e.g. fp32), ROOT (repo).
set -u -o pipefail
ROOT=${ROOT:-/workspace/Improving-CA-LoRA}
cd "$ROOT"
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} MKL_NUM_THREADS=${MKL_NUM_THREADS:-16}
export PYTHONUNBUFFERED=1
PY=${PY:-/workspace/venv-calora/bin/python}
CONFIG=${CONFIG:-configs/ca_selfgen.yaml}
GATE_SAMPLES=${GATE_SAMPLES:-32}
SETS=("$@")                      # extra --set overrides (tests: tiny config)

RUN=$("$PY" - "$CONFIG" "${SETS[@]}" <<'EOF'
import sys
sys.path.insert(0, "src")
from calora.config import load_config, parse_cli_overrides
cfg = load_config(sys.argv[1], parse_cli_overrides(sys.argv[2:]))
print(cfg.run_dir)
EOF
) || { echo "cannot resolve the run dir of $CONFIG"; exit 1; }
TIMESTEPS=$("$PY" - "$CONFIG" "${SETS[@]}" <<'EOF'
import sys
sys.path.insert(0, "src")
from calora.config import load_config, parse_cli_overrides
print(" ".join(map(str, load_config(sys.argv[1], parse_cli_overrides(sys.argv[2:])).measure.timesteps)))
EOF
)
# The fixed 13-point grid arm B is scored on (intersected with what is
# measured, so a reduced test config still runs).
GRID13="1 41 81 121 201 301 401 481 601 701 801 901 981"
SCORE_T=$(for t in $GRID13; do for m in $TIMESTEPS; do [ "$t" = "$m" ] && echo -n "$t "; done; done)

mkdir -p "$RUN/logs"
CAMPAIGN="$RUN/campaign.log"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$CAMPAIGN"; }
halt() { say "HALT $*"; exit 1; }

# stage <name> <attempts> <cmd...>; OOM is fatal (retrying will not help)
stage() {
  local name=$1 attempts=$2 log rc before
  shift 2
  log="$RUN/logs/$name.log"
  for attempt in $(seq 1 "$attempts"); do
    say "START $name attempt=$attempt"
    before=$(cat "$log" 2>/dev/null | wc -l)
    "$@" >> "$log" 2>&1
    rc=$?
    say "END $name rc=$rc"
    if tail -n +"$((before + 1))" "$log" | grep -aqE "CUDA out of memory|OutOfMemoryError"; then
      halt "$name out of memory (see $log)"
    fi
    [ "$rc" = 0 ] && return 0
    say "RETRY $name rc=$rc"
    sleep 60
  done
  halt "$name failed $attempts times (see $log)"
}

say "CAMPAIGN START config=$CONFIG run=$RUN timesteps=[$TIMESTEPS] score=[$SCORE_T]"

# 1. The self-generated set (once; later runs verify and generate nothing).
stage generate 4 "$PY" scripts/generate_selfgen.py --config "$CONFIG" --set "${SETS[@]}"

# 2. Precision gate (once). FORCE_PRECISION bypasses it and is recorded.
GATE="$RUN/precision_gate.json"
if [ -n "${FORCE_PRECISION:-}" ]; then
  PRECISION=$FORCE_PRECISION; GC=true
  say "GATE bypassed: FORCE_PRECISION=$PRECISION"
else
  if [ ! -s "$GATE" ]; then
    stage precision_gate 2 "$PY" scripts/ops/check_measure_precision.py --config "$CONFIG" \
      --set "${SETS[@]}" --num-samples "$GATE_SAMPLES"
  fi
  read -r PRECISION GC < <("$PY" -c "
import json; d = json.load(open('$GATE'))['adopted']
print(d['precision'], str(d['gradient_checkpointing']).lower())") || halt "unreadable $GATE"
  say "GATE adopted precision=$PRECISION gradient_checkpointing=$GC ($GATE)"
fi

# 3. Measurement, every timestep in one process (samples prepared once).
stage measure 3 "$PY" scripts/measure_ca.py --config "$CONFIG" --skip-existing \
  --set "${SETS[@]}" "measure.precision=$PRECISION" "measure.gradient_checkpointing=$GC"

# 4. Read every file back (twice on failure: the volume throws transient EIO).
"$PY" scripts/ops/verify_measurements.py --run-dir "$RUN" --timesteps $TIMESTEPS \
  --expect-source selfgen >> "$RUN/logs/verify.log" 2>&1 \
  || { sleep 60; "$PY" scripts/ops/verify_measurements.py --run-dir "$RUN" \
         --timesteps $TIMESTEPS --expect-source selfgen >> "$RUN/logs/verify.log" 2>&1; } \
  || halt "measurement verification failed (see logs/verify.log)"
say "VERIFIED $(wc -l < "$RUN/measurements/SHA256SUMS") files"

# 5. Timestep scores on the fixed 13-point grid only.
stage score 2 "$PY" scripts/score_timesteps.py --config "$CONFIG" \
  --set "${SETS[@]}" --timesteps $SCORE_T

# 6. Fixed-timestep selections (A_t81, C_t41, D_t1 x both axes). B_top3 needs the
#    ranking, which is reviewed first; E_all / E_random do not depend on CA
#    and stay as they are.
for spec in A_t81:81 C_t41:41 D_t1:1; do
  arm=${spec%%:*}; t=${spec##*:}
  case " $TIMESTEPS " in *" $t "*) ;; *) continue ;; esac
  for axis in style viewpoint; do
    stage "select_${arm}_${axis}" 2 "$PY" scripts/select_projections.py --config "$CONFIG" \
      --set "${SETS[@]}" "select.axis=$axis" --strategy single --timesteps "$t" \
      --out "selection_${arm}_${axis}.json"
  done
done

# 7. Handoff note for the other projects (paths + hashes; nothing is copied).
"$PY" - "$RUN" "$PRECISION" "$GC" <<'EOF' || halt "HANDOFF.md could not be written"
import hashlib, json, sys
from pathlib import Path
run, precision, gc = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
summary = json.loads((run / "measure_summary.json").read_text())
prov = summary.get("provenance", {})
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
sels = sorted(run.glob("selection_*.json"))
lines = [
    "# Self-generated CA measurement: handoff", "",
    f"- measurements: `{(run / 'measurements').resolve()}` (verify: `cd` there and "
    "`sha256sum -c SHA256SUMS`)",
    f"- timesteps: {sorted(int(t) for t in summary['summaries'])}",
    f"- x0: data_source={prov.get('data_source')}, selfgen manifest sha256 "
    f"`{prov.get('selfgen_manifest_sha256')}`",
    f"- precision: {precision}, gradient checkpointing: {gc} "
    f"(gate: `{(run / 'precision_gate.json').resolve()}`)",
    f"- timestep scores (13-point grid): `{(run / 'timestep_scores.json').resolve()}`",
    "", "## Selections", "",
    *[f"- `{p.resolve()}` sha256 `{sha(p)}`" for p in sels],
    "", "## Consumers", "",
    "- Hierarchical-CFG: copy `selection_A_t81_style.json` to "
    "`configs/selections/selfgen_style_t81.json`.",
    "- CA-LoRA-TSM: `bash scripts/ops/import_measurements.sh <measurements dir> "
    "sdxl_cityscapes_selfgen`, then `python -m catsm.cli select`.",
    "- Improving-CA-LoRA: B_top3 from `timestep_scores.json` after review "
    "(`select_projections.py --strategy aggregate --from-scores --set select.top_timesteps=3`).",
]
(run / "HANDOFF.md").write_text("\n".join(lines) + "\n")
EOF
say "CAMPAIGN END (see $RUN/HANDOFF.md)"
