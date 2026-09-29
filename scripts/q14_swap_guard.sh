#!/usr/bin/env bash
# Restarts the resumable Qwen2.5-14B run when swap exceeds 7 GB (clears swap; cached answers replay instantly).
# Exits when the run is no longer active (finished or stopped by the user).
cd "$(dirname "$0")/.."
S=scripts/qwen14b_rest.sh
while true; do
  sleep 120
  [ -f results/raw/qwen14b_rest.pid ] && kill -0 "$(cat results/raw/qwen14b_rest.pid)" 2>/dev/null || exit 0
  mb=$(sysctl -n vm.swapusage | awk '{print $6}' | tr -d 'M' | cut -d. -f1)
  if [ "$mb" -gt 7000 ]; then
    echo "=== $(date +%T) SWAP GUARD: ${mb} MB swap -> restarting run" >> results/raw/qwen14b_rest_log.txt
    bash $S stop > /dev/null; sleep 20; bash $S start > /dev/null
  fi
done
