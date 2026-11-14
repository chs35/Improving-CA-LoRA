#!/bin/bash
# supervise.sh <success_marker> <logfile> <cmd...>
# Retries (up to 12, 60s pause) on: no marker within 420s, or post-marker log
# freeze for 480s while the child lives. All log inspection streams the file
# with cat (real EOF) because mfs FUSE attr caching makes stat sizes stale.
MARKER="$1"; LOG="$2"; shift 2
touch "$LOG"
logsize() { cat "$LOG" 2>/dev/null | wc -c; }
newpart() { cat "$LOG" 2>/dev/null | tail -c +$(( $1 + 1 )); }
for attempt in $(seq 1 12); do
  base=$(logsize)
  "$@" >> "$LOG" 2>&1 &
  child=$!
  ok=0
  for i in $(seq 1 "${MARKER_TRIES:-42}"); do
    sleep 10
    kill -0 "$child" 2>/dev/null || break
    if newpart "$base" | tr '\r' '\n' | grep -aq "$MARKER"; then ok=1; break; fi
  done
  if [ "$ok" = 0 ] && ! kill -0 "$child" 2>/dev/null; then
    wait "$child"; rc=$?
    if newpart "$base" | tr '\r' '\n' | grep -aq "$MARKER"; then
      echo "[supervise] attempt $attempt: child finished rc=$rc with marker" >&2
      exit $rc
    fi
    echo "[supervise] attempt $attempt: child exited rc=$rc without marker, retrying" >&2
    sleep 60; continue
  fi
  if [ "$ok" = 1 ]; then
    echo "[supervise] attempt $attempt: marker seen, watching for stalls" >&2
    prev=$(logsize); still=0
    while kill -0 "$child" 2>/dev/null; do
      sleep 60
      cur=$(logsize)
      if [ "$cur" = "$prev" ]; then still=$((still+1)); else still=0; prev=$cur; fi
      if [ "$still" -ge "${STALL_MINUTES:-8}" ]; then
        echo "[supervise] attempt $attempt: log frozen 480s, killing $child, retrying" >&2
        kill -9 "$child" 2>/dev/null; sleep 60; continue 2
      fi
    done
    wait "$child"; rc=$?
    echo "[supervise] child exited rc=$rc after marker" >&2
    exit $rc
  fi
  echo "[supervise] attempt $attempt: no marker in 420s, killing $child, retrying" >&2
  kill -9 "$child" 2>/dev/null; sleep 60
done
echo "[supervise] giving up after 12 attempts" >&2
exit 97
