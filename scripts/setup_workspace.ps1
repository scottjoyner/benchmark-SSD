$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$artifactRoot = Join-Path $root "artifacts"
$benchmarkRoot = Join-Path $artifactRoot "lmstudio-benchmarks"
$reportRoot = Join-Path $root "reports"

New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
New-Item -ItemType Directory -Force -Path $benchmarkRoot | Out-Null
New-Item -ItemType Directory -Force -Path $reportRoot | Out-Null

Write-Host "Created workspace directories."
Write-Host "Snapshotting LM Studio model inventory..."

python (Join-Path $PSScriptRoot "discover_lmstudio_models.py")

Write-Host ""
Write-Host "Workspace setup complete."
Write-Host "If you want benchmark runs next, start LM Studio's local server and load one of the recommended models."
