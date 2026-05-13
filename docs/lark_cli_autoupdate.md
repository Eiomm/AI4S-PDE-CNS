# lark-cli Auto Update

This repository includes a small PowerShell wrapper for `lark-cli`.

## What It Does

After installation, you can keep using the normal command:

```powershell
lark-cli docs +fetch --api-version v2 --as user --doc "<url>"
```

Before running the real `lark-cli`, the wrapper checks whether `lark-cli` or its paired skills need an update. If an update is needed, it runs:

```powershell
lark-cli update --json
```

Then it continues with the original command.

## Files

- `scripts/lark-cli-autoupdate.ps1`: wrapper function and update check logic.
- `scripts/install-lark-cli-autoupdate.ps1`: installer that adds the wrapper to the PowerShell profile.

## Install

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install-lark-cli-autoupdate.ps1
```

The installer writes a marked block into:

```text
C:\Users\Junao\Documents\WindowsPowerShell\profile.ps1
```

New PowerShell sessions will load the wrapper automatically.

## Configuration

Default check interval is 24 hours.

Check every time:

```powershell
$env:LARK_CLI_AUTOUPDATE_INTERVAL_HOURS = "0"
```

Disable auto update for the current shell:

```powershell
$env:LARK_CLI_AUTOUPDATE_DISABLE = "1"
```

Hide wrapper messages:

```powershell
$env:LARK_CLI_AUTOUPDATE_QUIET = "1"
```

## Verify

```powershell
(Get-Command lark-cli).CommandType
lark-cli --version
```

Expected command type after installation:

```text
Function
```
