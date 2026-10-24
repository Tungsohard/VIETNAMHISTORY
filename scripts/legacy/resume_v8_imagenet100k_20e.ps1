$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$run = Join-Path $root "runs\vietcolornet_v8_imagenet100k_vietnam_20e_resumable"
$resume = @(
    (Join-Path $run "resume_latest.pth"),
    (Join-Path $run "pending_validation.pth"),
    (Join-Path $run "latest.pth")
) | Where-Object { Test-Path $_ } | Select-Object -First 1

$arguments = @(
    "fine_tune_vietnam.py",
    "--config", "configs/vietcolornet_v8_imagenet100k_vietnam.json",
    "--protocol", "data/vietnam_historical_craw_v8/manifest.json",
    "--source-checkpoint", "runs/vietcolornet_v6_refined/best.pth",
    "--imagenet-manifest", "data/processed/protocol_manifest_v3.json",
    "--craw-root", "..\craw_dataset",
    "--output", "runs/vietcolornet_v8_imagenet100k_vietnam_20e_resumable",
    "--device", "cuda"
)
if ($resume) {
    $arguments += @("--resume", $resume)
    Write-Host "Resuming from $resume"
} else {
    Write-Host "No recovery checkpoint found; starting a new run"
}

Set-Location $root
& ".\.venv\Scripts\python.exe" @arguments
