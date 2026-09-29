#!/usr/bin/env bash
# Resumable Qwen2.5-14B run on the 144 instances outside the 48-instance subset.
#   bash scripts/qwen14b_rest.sh start    # start, or resume after a stop
#   bash scripts/qwen14b_rest.sh stop     # stop now; nothing is lost
#   bash scripts/qwen14b_rest.sh status   # progress
# Work is split into 18 chunks (one base workflow x 8 variants). Each finished chunk is saved as
# results/raw/q14_parts/<base>.jsonl and skipped on restart; inside an unfinished chunk, every model
# answer is already cached, so a restart replays it instantly. When all chunks are done they are merged
# into results/raw/real_qwen2.5-14b_rest.jsonl. Uses the main Ollama (port 11434); the external drive
# holding the models must be mounted.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
M=B1,B2,B3,B4,B6,B7,B8,P,P10,M1,M2,M3,M4,M5,M7
BASES="lit-01 lit-02 lit-04 lit-05 lit-06 lit-07 sw-01 sw-02 sw-04 sw-05 sw-06 sw-07 ent-01 ent-02 ent-04 ent-05 ent-06 ent-07"
PARTS=results/raw/q14_parts
LOG=results/raw/qwen14b_rest_log.txt
PIDF=results/raw/qwen14b_rest.pid
mkdir -p "$PARTS"

worker() {
  for b in $BASES; do
    [ -f "$PARTS/$b.jsonl" ] && continue
    g=$(df -g / | tail -1 | awk '{print $4}')
    if [ "$g" -lt 4 ]; then echo "=== $(date +%T) STOP: only ${g} GB free on system disk (resume later)"; exit 1; fi
    echo "=== $(date +%T) chunk $b"
    ESAOG_OLLAMA_CTX=4096 ESAOG_OLLAMA_HOST=http://localhost:11434 $PY scripts/run_experiments.py \
        --profile llm:ollama:qwen2.5:14b --methods $M --seeds 0 --workers 1 --chunksize 15 \
        --bases "$b" --out "q14_tmp_$b" 2>&1 | grep -v Owlready
    if grep -q '"error"' "results/raw/q14_tmp_$b.jsonl" 2>/dev/null; then
      echo "=== $(date +%T) chunk $b had errors; kept as q14_tmp_$b.jsonl for inspection, will retry on next start"
    elif [ -f "results/raw/q14_tmp_$b.jsonl" ]; then
      mv "results/raw/q14_tmp_$b.jsonl" "$PARTS/$b.jsonl"; rm -f "results/raw/q14_tmp_$b.csv"
      echo "=== $(date +%T) chunk $b saved"
    fi
  done
  n=$(ls "$PARTS"/*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -eq 18 ]; then
    cat "$PARTS"/*.jsonl > results/raw/real_qwen2.5-14b_rest.jsonl
    echo "=== $(date +%T) ALL 18 CHUNKS DONE -> results/raw/real_qwen2.5-14b_rest.jsonl ($(wc -l < results/raw/real_qwen2.5-14b_rest.jsonl) runs)"
  fi
  rm -f "$PIDF"
}

case "${1:-status}" in
  start)
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "already running (pid $(cat "$PIDF"))"; exit 0; fi
    curl -s -m 5 http://localhost:11434/api/tags | grep -q 'qwen2.5:14b' || { echo "Ollama is not serving qwen2.5:14b (is the app running and the external drive mounted?)"; exit 1; }
    ( worker >> "$LOG" 2>&1 ) & echo $! > "$PIDF"
    echo "started (pid $(cat "$PIDF")); log: $LOG" ;;
  stop)
    [ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null
    pkill -f "run_experiments.py --profile llm:ollama:qwen2.5:14b"; rm -f "$PIDF"
    ollama stop qwen2.5:14b >/dev/null 2>&1
    echo "=== $(date +%T) STOPPED by user (resume with: bash scripts/qwen14b_rest.sh start)" >> "$LOG"
    echo "stopped; model unloaded. Resume any time with: bash scripts/qwen14b_rest.sh start" ;;
  status)
    n=$(ls "$PARTS"/*.jsonl 2>/dev/null | wc -l | tr -d ' ')
    c=$(sqlite3 results/llm_cache/ollama_qwen2.5_14b.sqlite "select count(*) from r" 2>/dev/null)
    run="stopped"; [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null && run="running"
    echo "$run | chunks done: $n/18 | cached answers: $c | last: $(grep '^===' "$LOG" 2>/dev/null | tail -1)" ;;
esac
