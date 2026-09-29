#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
M=B1,B2,B3,B4,B6,B7,B8,P,M1,M2,M3,M4,M5,M7
SUB=lit-00,lit-03,sw-00,sw-03,ent-00,ent-03
REST=lit-01,lit-02,lit-04,lit-05,lit-06,lit-07,sw-01,sw-02,sw-04,sw-05,sw-06,sw-07,ent-01,ent-02,ent-04,ent-05,ent-06,ent-07
Q3=hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
L3=hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M
run() { echo "=== $(date +%T) $1 $2"; $PY scripts/run_experiments.py --profile "llm:ollama:$2" --methods $M --seeds 0 --workers 1 --bases "$3" --out "$1" 2>&1 | grep -v Owlready; }


run real_qwen3-4b_sub $Q3 $SUB
run real_llama3.2-3b_sub $L3 $SUB
run real_qwen3-4b_rest  $Q3 $REST
run real_llama3.2-3b_rest $L3 $REST
run real_qwen2.5-14b_sub qwen2.5:14b $SUB
echo "=== $(date +%T) queue finished"
