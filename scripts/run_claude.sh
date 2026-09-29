#!/usr/bin/env bash
# Claude Haiku 4.5 rerun of the 14 LLM-dependent configurations (ESAOG v1.1) on all 192 SOST instances.
# Requires ANTHROPIC_API_KEY in the environment. Hard spend cap (USD) via ESAOG_SPEND_CAP_USD.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
M=B1,B2,B3,B4,B6,B7,B8,P,P10,M1,M2,M3,M4,M5,M7
ESAOG_SPEND_CAP_USD=${ESAOG_SPEND_CAP_USD:-20} $PY scripts/run_experiments.py --profile llm:anthropic:claude-haiku-4-5 \
    --methods $M --seeds 0 --workers 4 --chunksize 15 --out real_claude-haiku-4-5_all 2>&1 | grep -v Owlready
echo "=== $(date +%T) claude runs finished"
