# Demo 1: onboard hippo to a repo once, then ask about it in a fresh session with tools off.
# Session 2 can only answer from long-term memory (recall), which is the point.
# Usage (from the hippo/ directory):  .\scripts\demo1_onboarding.ps1
$ErrorActionPreference = "Stop"
$Sandbox = "examples/sandbox"

Write-Host "`n=== Session 1: onboarding (tools ON, memory written at the end) ===" -ForegroundColor Cyan
hippo run -w $Sandbox "Onboard me to this repository. Read the README and every source module, and run the test suite. Then tell me: what the project does, what each module is for, how to run the tests and whether they currently pass, and the money and rounding conventions."

Write-Host "`n=== Session 2: new process, --oneshot = no tools, only recalled memory ===" -ForegroundColor Cyan
hippo run --oneshot "In the invoicely repo, which module handles discounts, what unit is money stored in, and how do I run the tests? Were any tests failing last time?"

Write-Host "`n=== What was recalled ===" -ForegroundColor Cyan
hippo memory "invoicely tests discounts"
