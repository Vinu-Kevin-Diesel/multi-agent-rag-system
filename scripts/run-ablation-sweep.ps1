<#
.SYNOPSIS
    Collect one eval run per ablation configuration (day 17).

.DESCRIPTION
    The ablation flags are read by the app at startup from .env, so measuring a configuration
    means restarting the app with different flags -- there is no runtime switch. This script does
    that for each configuration in turn, waits for the app to come back, and collects a run.

    Collection only. Scoring is separate (eval/run_ragas.py) because it depends on a hosted judge
    with a finite free-tier budget, while collection is GPU-bound and costs nothing but time. A
    quota failure must never waste hours of GPU work.

    .env is backed up and restored in a finally block, so an interrupted sweep does not leave the
    developer's configuration rewritten.

.PARAMETER Configs
    Subset of configurations to collect. Defaults to all five, in ablation order.

.PARAMETER Limit
    Items per run. Use a small number for a dry run of the sweep mechanics.

.NOTES
    ROUTER_MODEL is held at qwen3-router for EVERY configuration, including full-clf. Decompose
    also runs on the router model, so letting it differ between full and full-clf would change two
    variables at once and the router comparison would be confounded. This costs model swapping on
    an 8 GB card, which is a latency penalty only -- and latency is not what the ablation measures.
#>

[CmdletBinding()]
param(
    [string[]]$Configs = @("baseline", "+router", "+decompose", "full", "full-clf"),
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$envPath = Join-Path $repo ".env"
$docker = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"

# One row per configuration in the day-17 table. Each step changes exactly one flag from the row
# above it, which is what makes the deltas attributable to a single component.
$matrix = [ordered]@{
    "baseline"   = @{ ROUTER_MODE = "off";        DECOMPOSE_ENABLED = "false"; CRITIC_MODE = "off" }
    "+router"    = @{ ROUTER_MODE = "llm";        DECOMPOSE_ENABLED = "false"; CRITIC_MODE = "off" }
    "+decompose" = @{ ROUTER_MODE = "llm";        DECOMPOSE_ENABLED = "true";  CRITIC_MODE = "off" }
    "full"       = @{ ROUTER_MODE = "llm";        DECOMPOSE_ENABLED = "true";  CRITIC_MODE = "cosine" }
    "full-clf"   = @{ ROUTER_MODE = "classifier"; DECOMPOSE_ENABLED = "true";  CRITIC_MODE = "cosine" }
}

function Invoke-Docker {
    <#
      Native executables write progress to stderr, and PowerShell 5.1 turns each such line into an
      ErrorRecord. Under $ErrorActionPreference='Stop' that aborts the script on output that is not
      an error at all -- `docker compose up` printing "Container ... Running" was enough to kill the
      sweep. Drop to Continue for the call and judge success by the exit code, which is the only
      reliable signal a native command gives.
    #>
    param([string[]]$DockerArgs, [switch]$Quiet)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        # Write-Host, not the pipeline: anything emitted to the pipeline would be returned by this
        # function alongside the exit code, and the caller's `$rc` would become the whole log.
        if ($Quiet) { & $docker @DockerArgs 2>&1 | Out-Null }
        else { & $docker @DockerArgs 2>&1 | ForEach-Object { Write-Host "$_" } }
    } finally {
        $ErrorActionPreference = $prev
    }
    return $LASTEXITCODE
}

function Set-EnvFlags {
    param([hashtable]$Flags)
    # Read and write as UTF-8 without BOM. Get-Content/Set-Content round-tripping through the
    # ANSI codepage mangles the box-drawing characters in .env's comments.
    $text = [System.IO.File]::ReadAllText($envPath, [System.Text.UTF8Encoding]::new($false))
    $all = $Flags.Clone()
    $all["ROUTER_MODEL"] = "qwen3-router"   # held constant; see .NOTES
    foreach ($k in $all.Keys) {
        $line = "$k=$($all[$k])"
        if ($text -match "(?m)^$k=") {
            $text = [regex]::Replace($text, "(?m)^$k=.*$", $line)
        } else {
            $text = $text.TrimEnd() + "`n$line"
        }
    }
    [System.IO.File]::WriteAllText($envPath, $text.TrimEnd() + "`n", [System.Text.UTF8Encoding]::new($false))
}

function Wait-Healthy {
    for ($i = 0; $i -lt 60; $i++) {
        try {
            $h = Invoke-RestMethod "http://localhost:8000/health" -TimeoutSec 5
            if ($h.status -eq "healthy") { return $h }
        } catch {}
        Start-Sleep -Seconds 5
    }
    throw "app did not become healthy within 5 minutes"
}

$backup = "$envPath.sweep-backup"
Copy-Item $envPath $backup -Force
Write-Host "backed up .env -> $backup" -ForegroundColor DarkGray

try {
    foreach ($name in $Configs) {
        if (-not $matrix.Contains($name)) { throw "unknown config '$name'" }
        $flags = $matrix[$name]

        Write-Host ""
        Write-Host "=== $name ===" -ForegroundColor Cyan
        Write-Host "  ROUTER_MODE=$($flags.ROUTER_MODE) DECOMPOSE_ENABLED=$($flags.DECOMPOSE_ENABLED) CRITIC_MODE=$($flags.CRITIC_MODE)"

        Set-EnvFlags -Flags $flags

        # Recreate rather than restart: compose only re-reads env_file when the container is
        # recreated, so a plain restart would silently keep the previous configuration -- and the
        # run would be labelled with flags it was not collected under.
        $rc = Invoke-Docker -DockerArgs @("compose","up","-d","--force-recreate","app") -Quiet
        if ($rc -ne 0) { throw "docker compose up failed for '$name' (exit $rc)" }
        $health = Wait-Healthy

        Write-Host ("  app reports: router_mode={0} router_model={1} decompose_enabled={2} critic_mode={3}" -f `
            $health.flags.router_mode, $health.flags.router_model, $health.flags.decompose_enabled, $health.flags.critic_mode)

        # run_eval.py re-checks these against the --config name and exits non-zero on a mismatch,
        # so a wrong row in $matrix fails loudly here instead of producing a mislabelled run.
        $evalArgs = @("compose", "exec", "-T", "app", "python", "-u", "eval/run_eval.py", "--config", $name)
        if ($Limit -gt 0) { $evalArgs += @("--limit", "$Limit") }

        $rc = Invoke-Docker -DockerArgs $evalArgs
        if ($rc -ne 0) { throw "collection failed for '$name' (exit $rc)" }
    }

    Write-Host ""
    Write-Host "sweep complete. Score the runs with eval/run_ragas.py as judge quota allows." -ForegroundColor Green
}
finally {
    Copy-Item $backup $envPath -Force
    Remove-Item $backup -ErrorAction SilentlyContinue
    Write-Host "restored .env" -ForegroundColor DarkGray
    Invoke-Docker -DockerArgs @("compose","up","-d","--force-recreate","app") -Quiet | Out-Null
}
