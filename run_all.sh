#!/usr/bin/env bash
# One-command reproduction: environment check, benchmark, all runs, sensitivity, coverage, tables & figures.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
$PY -m pip install -r requirements.txt >/dev/null
PYTHONPATH=src $PY -m esaog.ontology                          # release domain TTL modules
PYTHONPATH=src $PY -m esaog.benchmark                         # SOST benchmark JSON (deterministic)
$PY -m pytest -q tests                                          # semantic-core unit tests
$PY scripts/run_experiments.py --out runs                       # 17 configs x 192 instances x 5 seeds
for prof in weak strong; do                                     # SimLLM competence sensitivity
  $PY scripts/run_experiments.py --out sens_$prof --profile $prof --seeds 0,1 \
      --methods B1,B4,B6,B8,P,M6
done
$PY scripts/coverage.py                                         # ontology/SHACL coverage + evaluator tests
$PY scripts/trace_example.py                                    # failure-to-recomposition trace
$PY scripts/analysis.py                                         # tables (LaTeX) + figures
$PY scripts/sensitivity.py
