Set-StrictMode -Version Latest

$shimPath = Join-Path $PSScriptRoot "lark-cli-autoupdate.ps1"
if (-not (Test-Path -LiteralPath $shimPath)) {
    throw "Cannot find shim script: $shimPath"
}

$profilePath = $PROFILE.CurrentUserAllHosts
$profileDir = Split-Path -Parent $profilePath
if (-not (Test-Path -LiteralPath $profileDir)) {
    New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
}
if (-not (Test-Path -LiteralPath $profilePath)) {
    New-Item -ItemType File -Path $profilePath -Force | Out-Null
}

$beginMarker = "# >>> AI4S PDE lark-cli autoupdate >>>"
$endMarker = "# <<< AI4S PDE lark-cli autoupdate <<<"
$line = ". '$shimPath'"
$block = @(
    $beginMarker
    $line
    $endMarker
) -join [Environment]::NewLine

$content = Get-Content -LiteralPath $profilePath -Raw
if ($content -match [regex]::Escape($beginMarker)) {
    $pattern = "(?s)" + [regex]::Escape($beginMarker) + ".*?" + [regex]::Escape($endMarker)
    $content = [regex]::Replace($content, $pattern, [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $block })
} else {
    if (-not $content.EndsWith([Environment]::NewLine)) {
        $content += [Environment]::NewLine
    }
    $content += $block + [Environment]::NewLine
}

Set-Content -LiteralPath $profilePath -Value $content -Encoding UTF8
. $shimPath

Write-Host "Installed lark-cli auto-update wrapper into PowerShell profile:" -ForegroundColor Green
Write-Host "  $profilePath"
Write-Host "It is active in this shell after dot-sourcing and will load automatically in new PowerShell sessions."
