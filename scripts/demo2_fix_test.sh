#!/usr/bin/env bash
# Demo 2: hippo fixes a failing test suite. Planner delegates, worker edits with --write,
# reviewer checks the worker's evidence (local__run_pytest exit code) before accepting.
# Usage (from the hippo/ directory):  bash scripts/demo2_fix_test.sh
set -euo pipefail
SANDBOX="examples/sandbox"

echo; echo "=== Reset sandbox and show the failure ==="
git checkout -- "$SANDBOX"
(cd "$SANDBOX" && python -m pytest -q -p no:cacheprovider 2>&1 | tail -n 4) || true

echo; echo "=== hippo run --write ==="
hippo run -w "$SANDBOX" --write "The pytest suite in this repo has failing tests. Reproduce the failure, find the single root cause in the library source, fix the source (do not edit the tests), and re-run the whole suite to confirm it is green. Report what was wrong and what you changed."

echo; echo "=== Diff produced by hippo ==="
git --no-pager diff --stat -- "$SANDBOX"
git --no-pager diff -- "$SANDBOX"

echo; echo "=== Tests now ==="
(cd "$SANDBOX" && python -m pytest -q -p no:cacheprovider 2>&1 | tail -n 2)
echo; echo "(Undo with: git checkout -- $SANDBOX)"
