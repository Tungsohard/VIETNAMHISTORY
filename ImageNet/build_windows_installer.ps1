param(
    [switch]$SkipAppBuild
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $SkipAppBuild) {
    & (Join-Path $ProjectRoot "build_windows_app.ps1")
    if ($LASTEXITCODE -ne 0) { throw "Build ứng dụng thất bại." }
}

$CompilerCandidates = @(
    (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 7\ISCC.exe"),
    "C:\Program Files\Inno Setup 7\ISCC.exe",
    "C:\Program Files (x86)\Inno Setup 7\ISCC.exe",
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
)
$Compiler = $CompilerCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $Compiler) {
    throw "Chưa cài Inno Setup 7. Hãy cài Inno Setup rồi chạy lại script."
}

& $Compiler (Join-Path $ProjectRoot "SacVietInstaller.iss")
if ($LASTEXITCODE -ne 0) { throw "Tạo bộ cài thất bại." }

Write-Host "Bộ cài đã tạo trong: $(Join-Path $ProjectRoot 'release\installer')"
