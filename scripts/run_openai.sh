#!/usr/bin/env bash
# OpenAI rerun (gpt-6-luna) of the 14 LLM-dependent configurations on all 192 SOST instances.
# Requires OPENAI_API_KEY in the environment. Hard spend cap (USD) via ESAOG_SPEND_CAP_USD.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
M=B1,B2,B3,B4,B6,B7,B8,P,M1,M2,M3,M4,M5,M7
export ESAOG_OPENAI_REASONING=low
ESAOG_SPEND_CAP_USD=20 $PY scripts/run_experiments.py --profile llm:openai:gpt-6-luna --methods $M --seeds 0 \
    --workers 4 --chunksize 14 --out real_gpt-6-luna_all 2>&1 | grep -v Owlready
echo "=== $(date +%T) openai runs finished"
