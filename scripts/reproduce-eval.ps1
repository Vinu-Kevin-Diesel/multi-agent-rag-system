<#
.SYNOPSIS
    Reproduce the evaluation reported in the README, end to end.

.DESCRIPTION
    Runs the whole chain in the order the results depend on:

      1. verify the frozen corpus against its committed hash  (gate -- stop on drift)
      2. validate the golden set structurally                 (gate)
      3. collect one run per ablation configuration           (GPU-bound, no API cost)
      4. summarise routing, latency and critic cost           (no judge required)
      5. score with RAGAS                                     (hosted judge, rate-limited)
      6. produce the per-type / confusion / correlation analysis

    Steps 1-4 need no API key and no network beyond the model already pulled: routing accuracy,
    the confusion matrices and the critic's latency cost all come from gold labels and
    predictions. Only steps 5-6 need the judge, and they are the ones that fail on a free tier.
    Splitting there is deliberate -- a quota failure must never cost hours of GPU work.

    Expect roughly 50 minutes per configuration to collect, plus 2-4 hours per configuration to
    score, on the hardware in the README.

.PARAMETER SkipCollect
    Reuse the runs already in eval/runs/ instead of re-collecting. Use when only re-scoring.

.PARAMETER SkipScore
    Stop after the judge-free analysis. Produces the ablation table and confusion matrices with
    no API calls at all.

.PARAMETER Configs
    Which configurations to collect. Defaults to all six.

.EXAMPLE
    ./scripts/reproduce-eval.ps1 -SkipScore
    Everything that does not need a judge -- the ablation table and router confusion matrices.

.EXAMPLE
    ./scripts/reproduce-eval.ps1 -SkipCollect
    Re-score the existing runs, e.g. after judge quota resets.
#>

[CmdletBinding()]
param(
    [switch]$SkipCollect,
    [switch]$SkipScore,
    [string[]]$Configs = @("baseline", "+router", "+decompose", "full", "full-clf", "full-nli")
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$docker = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
if (-not (Test-Path $docker)) { $docker = "docker" }

# Native executables write progress to stderr, which PowerShell 5.1 turns into ErrorRecords and,
# under ErrorActionPreference=Stop, into a fatal error on output that is not an error at all.
# Judge success by the exit code instead.
function Invoke-Docker {
    param([string[]]$DockerArgs, [switch]$Quiet)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if ($Quiet) { & $docker @DockerArgs 2>&1 | Out-Null }
        else { & $docker @DockerArgs 2>&1 | ForEach-Object { Write-Host "$_" } }
    } finally { $ErrorActionPreference = $prev }
    return $LASTEXITCODE
}

function Step($n, $text) { Write-Host "`n[$n] $text" -ForegroundColor Cyan }

Step 1 "Bringing up db + app"
if ((Invoke-Docker @("compose", "up", "-d", "db", "app") -Quiet) -ne 0) { throw "compose up failed" }
for ($i = 0; $i -lt 60; $i++) {
    try { if ((Invoke-RestMethod "http://localhost:8000/health" -TimeoutSec 5).status -eq "healthy") { break } } catch {}
    Start-Sleep -Seconds 5
}

Step 2 "Verifying the frozen corpus against its committed hash"
# Determinism is checked by SHA-256 of the chunk text, not chunk count: a mutated corpus was
# observed producing the same 22 chunks with a different hash. Drift here invalidates every
# comparison downstream, so it gates the run.
if ((Invoke-Docker @("compose", "run", "--rm", "app", "python", "eval/ingest_corpus.py", "--verify")) -ne 0) {
    throw "corpus drift -- results would not be comparable to the README's"
}

Step 3 "Validating the golden set"
if ((Invoke-Docker @("compose", "run", "--rm", "--no-deps", "app", "python", "eval/validate_golden_set.py")) -ne 0) {
    throw "golden set failed validation"
}

if (-not $SkipCollect) {
    Step 4 "Collecting one run per configuration (no API cost; ~50 min each)"
    & "$PSScriptRoot\run-ablation-sweep.ps1" -Configs $Configs
} else {
    Step 4 "Skipping collection -- reusing eval/runs/"
}

Step 5 "Ablation summary (routing, latency, critic cost -- no judge required)"
if ((Invoke-Docker @("compose", "run", "--rm", "--no-deps", "app",
        "python", "eval/summarize_runs.py", "--out", "eval/runs/ablation_summary.csv")) -ne 0) {
    throw "summarise failed"
}

if ($SkipScore) {
    Write-Host "`nStopping before scoring (-SkipScore). The table above needs no judge." -ForegroundColor Green
    exit 0
}

Step 6 "Scoring with RAGAS (hosted judge; free tiers rate-limit and exhaust)"
# answer_correctness is dropped: it is the most expensive metric and was the least reliable,
# at 28% coverage on a full run. See eval/README.md.
$metrics = "faithfulness,answer_relevancy,context_precision,context_recall"
foreach ($run in (Get-ChildItem "$repo\eval\runs\*.jsonl" | Sort-Object Name)) {
    Write-Host "`n  scoring $($run.Name)" -ForegroundColor DarkGray
    $rc = Invoke-Docker @("compose", "run", "--rm", "--no-deps", "app", "python", "-u",
        "eval/run_ragas.py", "--run", "eval/runs/$($run.Name)",
        "--metrics", $metrics, "--timeout", "900", "--workers", "3")
    # run_ragas exits 2 when nothing scored at all, which means the judge is unavailable rather
    # than that the answers were bad. Stop instead of burning the remaining runs against it.
    if ($rc -eq 2) {
        Write-Warning "judge unavailable -- stopping. Re-run with -SkipCollect when quota resets."
        break
    }
}

Step 7 "Analysis: per-type breakdown, router confusion, critic correlation"
Invoke-Docker @("compose", "run", "--rm", "--no-deps", "app",
    "python", "eval/analyze.py", "--out-dir", "eval/runs") | Out-Null

Write-Host "`nDone. Results in eval/runs/:" -ForegroundColor Green
Write-Host "  ablation_summary.csv              routing, latency, critic cost"
Write-Host "  analysis_per_type.csv             RAGAS metrics per query type"
Write-Host "  analysis_confusion.csv            router confusion, per configuration"
Write-Host "  analysis_critic_correlation.csv   confidence vs faithfulness"
