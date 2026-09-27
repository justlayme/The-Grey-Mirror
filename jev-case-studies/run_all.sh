#!/usr/bin/env bash
# Reproduce both Jev case studies end to end.
#
#   export AI_GATEWAY_API_KEY=...      # Vercel AI Gateway key (Jev = typesafe-ai/jev)
#   ./run_all.sh
#
# Options (environment):
#   LLM_COMPARATOR=anthropic/claude-haiku-4.5   System-Two comparator via the same gateway ("" to skip)
#   JEV_PROVIDER=typesafe                        call api.typesafe.ai directly (uses TYPESAFE_API_KEY)
#   JEV_PROVIDER=openrouter                      call Jev via OpenRouter (uses OPENROUTER_API_KEY)
#   JEV_MOCK=1                                   pipeline test with fake answers; writes to results/mock only
#
# Expect roughly 60-80 minutes (the turn-taking run is sequential on purpose, to measure clean latency)
# and a few dollars at list price. Every response is cached in data/cache/, so a re-run is free.
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${AI_GATEWAY_API_KEY:-}" ] && [ -z "${TYPESAFE_API_KEY:-}" ] && [ -z "${OPENROUTER_API_KEY:-}" ] && [ "${JEV_MOCK:-}" != "1" ]; then
  echo "error: set AI_GATEWAY_API_KEY (or run with JEV_MOCK=1 to test the pipeline)" >&2
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
PY=.venv/bin/python
LLM="${LLM_COMPARATOR-anthropic/claude-haiku-4.5}"

./scripts/fetch_data.sh
$PY cs1_grey_mirror/generate_threads.py
$PY cs1_grey_mirror/prepare_cga.py
$PY cs2_turn_taking/prepare_swb.py

echo "== 1A turning points"
$PY cs1_grey_mirror/run_turning_points.py
echo "== 1B Conversations Gone Awry"
$PY cs1_grey_mirror/run_cga.py ${LLM:+--llm "$LLM"}
echo "== 2 voice turn-taking"
$PY cs2_turn_taking/run_turns.py ${LLM:+--llm "$LLM" --llm-limit 600}

$PY report/build_reports.py
echo "done: open reports/jev-case-studies.html"
