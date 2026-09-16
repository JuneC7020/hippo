# Demo 2: hippo fixes a failing test suite. Planner delegates, worker edits with --write,
# reviewer checks the worker's evidence (local__run_pytest exit code) before accepting.
# Usage (from the hippo/ directory):  .\scripts\demo2_fix_test.ps1
$ErrorActionPreference = "Stop"
$Sandbox = "examples/sandbox"

Write-Host "`n=== Reset sandbox and show the failure ===" -ForegroundColor Cyan
git checkout -- $Sandbox
Push-Location $Sandbox
python -m pytest -q -p no:cacheprovider 2>&1 | Select-Object -Last 4
Pop-Location

Write-Host "`n=== hippo run --write ===" -ForegroundColor Cyan
hippo run -w $Sandbox --write "The pytest suite in this repo has failing tests. Reproduce the failure, find the single root cause in the library source, fix the source (do not edit the tests), and re-run the whole suite to confirm it is green. Report what was wrong and what you changed."

Write-Host "`n=== Diff produced by hippo ===" -ForegroundColor Cyan
git --no-pager diff --stat -- $Sandbox
git --no-pager diff -- $Sandbox

Write-Host "`n=== Tests now ===" -ForegroundColor Cyan
Push-Location $Sandbox
python -m pytest -q -p no:cacheprovider 2>&1 | Select-Object -Last 2
Pop-Location
Write-Host "`n(Undo with: git checkout -- $Sandbox)"
