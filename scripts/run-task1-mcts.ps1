param(
    [string]$Config = "configs\task1_mcts_full.yaml",
    [string]$ProjectRoot = "",
    [int]$MaxSteps = 0,
    [switch]$Reset,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"

# Runs: -m agent.run_task1_mcts_experiment
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
} else {
    $ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
}

if ([string]::IsNullOrWhiteSpace($Python)) {
    if (-not [string]::IsNullOrWhiteSpace($env:AI4S_PROJECT_PYTHON)) {
        $Python = $env:AI4S_PROJECT_PYTHON
    } else {
        $Python = "D:\Junao\ProgramData\anaconda3\envs\Hwpytorch\python.exe"
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python not found: $Python. Set AI4S_PROJECT_PYTHON or pass -Python."
}

$argsList = @(
    "-m", "agent.run_task1_mcts_experiment",
    "--config", $Config,
    "--project-root", $ProjectRoot
)

Push-Location -LiteralPath $ProjectRoot
try {
    if ($MaxSteps -gt 0) {
        $argsList += @("--max-steps", "$MaxSteps")
    }

    if ($Reset) {
        $argsList += "--reset"
    }

    & $Python @argsList
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

exit $exitCode
