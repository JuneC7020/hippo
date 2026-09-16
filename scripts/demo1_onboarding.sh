#!/usr/bin/env bash
# Demo 1: onboard hippo to a repo once, then ask about it in a fresh session with tools off.
# Session 2 can only answer from long-term memory (recall), which is the point.
# Usage (from the hippo/ directory):  bash scripts/demo1_onboarding.sh
set -euo pipefail
SANDBOX="examples/sandbox"

echo; echo "=== Session 1: onboarding (tools ON, memory written at the end) ==="
hippo run -w "$SANDBOX" "Onboard me to this repository. Read the README and every source module, and run the test suite. Then tell me: what the project does, what each module is for, how to run the tests and whether they currently pass, and the money and rounding conventions."

echo; echo "=== Session 2: new process, --oneshot = no tools, only recalled memory ==="
hippo run --oneshot "In the invoicely repo, which module handles discounts, what unit is money stored in, and how do I run the tests? Were any tests failing last time?"

echo; echo "=== What was recalled ==="
hippo memory "invoicely tests discounts"
