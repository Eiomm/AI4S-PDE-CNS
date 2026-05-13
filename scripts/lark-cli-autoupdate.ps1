Set-StrictMode -Version Latest

$script:LarkCliAutoUpdateDefaultIntervalHours = 24

function Get-LarkCliAutoUpdateStatePath {
    $root = $env:LOCALAPPDATA
    if ([string]::IsNullOrWhiteSpace($root)) {
        $root = Join-Path $env:USERPROFILE ".lark-cli-autoupdate"
    } else {
        $root = Join-Path $root "lark-cli-autoupdate"
    }
    if (-not (Test-Path -LiteralPath $root)) {
        New-Item -ItemType Directory -Path $root -Force | Out-Null
    }
    Join-Path $root "state.json"
}

function Get-RealLarkCliCommand {
    $commands = Get-Command lark-cli -CommandType ExternalScript,Application -ErrorAction Stop
    $command = @($commands)[0]
    if ($null -eq $command) {
        throw "Could not locate the real lark-cli executable or script."
    }
    $command
}

function Get-LarkCliAutoUpdateIntervalHours {
    if (-not [string]::IsNullOrWhiteSpace($env:LARK_CLI_AUTOUPDATE_INTERVAL_HOURS)) {
        $parsed = 0
        if ([int]::TryParse($env:LARK_CLI_AUTOUPDATE_INTERVAL_HOURS, [ref]$parsed) -and $parsed -ge 0) {
            return $parsed
        }
    }
    return $script:LarkCliAutoUpdateDefaultIntervalHours
}

function Test-LarkCliAutoUpdateShouldCheck {
    param(
        [Parameter(Mandatory = $true)]
        [string]$StatePath,
        [Parameter(Mandatory = $true)]
        [int]$IntervalHours
    )

    if ($IntervalHours -eq 0) {
        return $true
    }
    if (-not (Test-Path -LiteralPath $StatePath)) {
        return $true
    }

    try {
        $state = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
        if ($null -eq $state.last_checked_utc) {
            return $true
        }
        $lastChecked = [datetime]::Parse(
            [string]$state.last_checked_utc,
            [Globalization.CultureInfo]::InvariantCulture,
            [Globalization.DateTimeStyles]::AssumeUniversal
        ).ToUniversalTime()
        return (([datetime]::UtcNow - $lastChecked).TotalHours -ge $IntervalHours)
    } catch {
        return $true
    }
}

function Write-LarkCliAutoUpdateState {
    param(
        [Parameter(Mandatory = $true)]
        [string]$StatePath,
        [Parameter(Mandatory = $true)]
        [object]$Payload
    )

    $Payload | Add-Member -NotePropertyName last_checked_utc -NotePropertyValue ([datetime]::UtcNow.ToString("o")) -Force
    $Payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $StatePath -Encoding UTF8
}

function Invoke-LarkCliAutoUpdate {
    param(
        [switch]$Force,
        [switch]$Quiet
    )

    if ($env:LARK_CLI_AUTOUPDATE_DISABLE -eq "1") {
        return
    }

    $intervalHours = Get-LarkCliAutoUpdateIntervalHours
    $statePath = Get-LarkCliAutoUpdateStatePath
    if (-not $Force -and -not (Test-LarkCliAutoUpdateShouldCheck -StatePath $statePath -IntervalHours $intervalHours)) {
        return
    }

    try {
        $real = Get-RealLarkCliCommand
        $checkText = & $real.Source update --check --json 2>$null | Out-String
        $check = $checkText | ConvertFrom-Json

        $latest = [string]$check.latest_version
        $current = [string]$check.current_version
        $needsCliUpdate = -not [string]::IsNullOrWhiteSpace($latest) -and $current -ne $latest
        $needsSkillsSync = $false
        if ($null -ne $check.skills_status -and $null -ne $check.skills_status.in_sync) {
            $needsSkillsSync = -not [bool]$check.skills_status.in_sync
        }

        if ($needsCliUpdate -or $needsSkillsSync) {
            if (-not $Quiet) {
                Write-Host "[lark-cli-autoupdate] Updating lark-cli/skills: $current -> $latest" -ForegroundColor Cyan
            }
            $updateText = & $real.Source update --json | Out-String
            if (-not $Quiet) {
                Write-Host "[lark-cli-autoupdate] Update finished." -ForegroundColor Green
            }
            Write-LarkCliAutoUpdateState -StatePath $statePath -Payload ([pscustomobject]@{
                checked_current_version = $current
                checked_latest_version = $latest
                action = "updated"
                update_output = $updateText
            })
        } else {
            Write-LarkCliAutoUpdateState -StatePath $statePath -Payload ([pscustomobject]@{
                checked_current_version = $current
                checked_latest_version = $latest
                action = "already_up_to_date"
            })
        }
    } catch {
        if (-not $Quiet) {
            Write-Warning "[lark-cli-autoupdate] Update check failed; continuing with original lark-cli command. $($_.Exception.Message)"
        }
        try {
            Write-LarkCliAutoUpdateState -StatePath $statePath -Payload ([pscustomobject]@{
                action = "check_failed"
                error = $_.Exception.Message
            })
        } catch {
            # Do not block the original lark-cli command if state writing fails.
        }
    }
}

function global:lark-cli {
    [CmdletBinding()]
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    $quiet = $env:LARK_CLI_AUTOUPDATE_QUIET -eq "1"
    if ($Arguments.Count -eq 0 -or $Arguments[0] -ne "update") {
        Invoke-LarkCliAutoUpdate -Quiet:$quiet
    }

    $real = Get-RealLarkCliCommand
    & $real.Source @Arguments
}
