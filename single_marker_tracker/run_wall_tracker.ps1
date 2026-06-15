$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Tracker = Join-Path $ScriptDir "wall_tracker.py"
$Layout = Join-Path $ScriptDir "tag_layout_8x8x6in_36h11_12front_13left_14back_15right.json"

$StreamPort = $env:STREAM_PORT
if ([string]::IsNullOrWhiteSpace($StreamPort)) {
    $StreamPort = "8090"
}

$Python = $env:PYTHON
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = "python"
}

$IpArgs = @()
if (-not [string]::IsNullOrWhiteSpace($env:OAK_IP)) {
    $IpArgs = @("--ip", $env:OAK_IP)
}

Write-Host ""
Write-Host "Starting wall tracker (MJPEG :$StreamPort)"
if ($IpArgs.Count -gt 0) {
    Write-Host "  OAK device: $($env:OAK_IP)"
} else {
    Write-Host "  OAK device: USB auto-discover"
}
Write-Host "  Layout: $Layout"
Write-Host ""

& $Python $Tracker `
    @IpArgs `
    --layout $Layout `
    --stream-port $StreamPort `
    @args
