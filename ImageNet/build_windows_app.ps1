param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ReleaseRoot = Join-Path $ProjectRoot "release\SacViet"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Không tìm thấy môi trường Python: $Python"
}

if ($Clean) {
    $BuildPath = Join-Path $ProjectRoot "build\SacViet"
    $DistPath = Join-Path $ProjectRoot "dist\SacViet"
    if (Test-Path -LiteralPath $BuildPath) { Remove-Item -LiteralPath $BuildPath -Recurse -Force }
    if (Test-Path -LiteralPath $DistPath) { Remove-Item -LiteralPath $DistPath -Recurse -Force }
}

$PyInstallerArgs = @("-m", "PyInstaller", "--noconfirm")
if ($Clean) { $PyInstallerArgs += "--clean" }
$PyInstallerArgs += (Join-Path $ProjectRoot "SacViet.spec")
& $Python @PyInstallerArgs
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build thất bại." }

New-Item -ItemType Directory -Path $ReleaseRoot -Force | Out-Null
Copy-Item -Path (Join-Path $ProjectRoot "dist\SacViet\*") -Destination $ReleaseRoot -Recurse -Force

$ReleaseFolders = @(
    "reports",
    "runs\ddcolor_historical_still_replay_v6",
    "runs\vietcolornet_master_v4",
    "examples"
)
foreach ($RelativeFolder in $ReleaseFolders) {
    $Source = Join-Path $ProjectRoot $RelativeFolder
    $Destination = Join-Path $ReleaseRoot $RelativeFolder
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    if ($RelativeFolder -eq "reports") {
        Copy-Item -LiteralPath (Join-Path $Source "production_selection.json") -Destination $Destination -Force
    } elseif ($RelativeFolder -like "runs\*") {
        Copy-Item -LiteralPath (Join-Path $Source "best.pth") -Destination $Destination -Force
    } else {
        Copy-Item -Path (Join-Path $Source "*") -Destination $Destination -Recurse -Force
    }
}

Copy-Item -LiteralPath (Join-Path $ProjectRoot "APP_UPDATE.md") -Destination $ReleaseRoot -Force
Write-Host "Bản ứng dụng đã tạo tại: $ReleaseRoot"
Write-Host "Chạy: $ReleaseRoot\SacViet.exe"
