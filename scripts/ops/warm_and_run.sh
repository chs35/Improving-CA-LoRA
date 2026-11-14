#!/bin/bash
# Warm the whole venv into page cache (per-package bounded retries), then exec.
# Usage: warm_and_run.sh <command...>
SP=/workspace/venv-calora/lib/python3.10/site-packages
for d in "$SP"/*/ "$SP"/*.py; do
  [ -e "$d" ] || continue
  for i in 1 2 3; do
    timeout 240 tar -cf /dev/null "$d" 2>/dev/null && break
    sleep 3
  done
done
exec "$@"
