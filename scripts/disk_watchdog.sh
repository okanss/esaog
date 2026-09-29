#!/usr/bin/env bash
# Stops the Ollama experiment queue if the system disk falls below 2.5 GB free (results are cached; resumable).
cd "$(dirname "$0")/.."
while true; do
  mb=$(df -m / | tail -1 | awk '{print $4}')
  if [ "$mb" -lt 2560 ]; then
    echo "=== $(date +%T) WATCHDOG: ${mb} MB free -> stopping Ollama queue (resumable from cache)" >> results/raw/real_queue_log.txt
    pkill -f run_real_llm_queue6.sh; pkill -KILL -f "run_experiments.py --profile llm:ollama"
    exit 0
  fi
  sleep 30
done
