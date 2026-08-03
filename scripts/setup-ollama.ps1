<#
.SYNOPSIS
    Configure a host-native Ollama server to serve the model under test to the app container.

.DESCRIPTION
    Runs Ollama natively on the Windows host (not in Docker — GPU passthrough into Compose on
    Windows is a WSL2 detour, and Ollama ships its own CUDA runtime). The app container reaches
    it via host.docker.internal:11434 (wired in docker-compose.yml).

    Sets the environment variables that matter, restarts the server so they take effect, and
    verifies the model is resident on the GPU.

    The context length is sized from the detected VRAM rather than hardcoded, because the
    setting that fits a 12 GB desktop card spills on an 8 GB laptop card and everything then
    runs several times slower. See the sizing table in CLAUDE.md.

.PARAMETER ContextLength
    Override the VRAM-derived OLLAMA_CONTEXT_LENGTH. Lower it if `ollama ps` shows a CPU split.

.PARAMETER BuildRouterModel
    Force building the qwen3-router no-think variant. By default it is built only where a second
    resident 8B fits (>= 12 GB) — on 8 GB it would swap on every multi-hop query, which costs
    more than it saves. Use ROUTER_MODE=classifier there instead.

.PARAMETER SkipRouterModel
    Never build the variant, regardless of VRAM.

.NOTES
    OLLAMA_HOST=0.0.0.0        Bind all interfaces. Ollama defaults to 127.0.0.1, which a Docker
                              container CANNOT reach — the connection arrives via the Docker
                              gateway IP, not loopback, and is refused. This is the #1 gotcha.
    OLLAMA_CONTEXT_LENGTH     Sized from VRAM (see below). Large enough for the prompts the
                              multi-hop path builds; small enough that weights + KV cache stay
                              100% on the GPU.
    OLLAMA_KEEP_ALIVE=-1      Keep the model resident. Each query makes 3-6 LLM calls; reloading
                              between them would dominate latency.
    OLLAMA_NUM_PARALLEL=2     Modest concurrency without thrashing VRAM.

    MAX_CONTEXT_TOKENS in .env is COUPLED to OLLAMA_CONTEXT_LENGTH and is not set by this
    script. If the app's budget exceeds Ollama's window, Ollama silently truncates the front of
    the prompt — dropping the source chunks and leaving the model to invent an answer. The
    summary printed at the end states the value to use.
#>

[CmdletBinding()]
param(
    [int]$ContextLength,
    [switch]$BuildRouterModel,
    [switch]$SkipRouterModel
)

$ErrorActionPreference = "Stop"

$model = "qwen3:8b"

Write-Host "0. Detecting GPU..." -ForegroundColor Cyan
$vramMb = 0
try {
    $vramMb = [int](& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits |
        Select-Object -First 1)
    $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader | Select-Object -First 1)
    Write-Host "   $gpuName, $vramMb MiB VRAM"
} catch {
    # Non-ASCII must stay out of quoted strings: this file has no BOM, so PowerShell 5.1 reads
    # it as ANSI, and a UTF-8 em dash decodes to a byte PowerShell treats as a closing quote.
    Write-Warning "   nvidia-smi unavailable - assuming 12 GB. Pass -ContextLength to override."
    $vramMb = 12288
}

# Sizing table (CLAUDE.md). Measured for qwen3:8b Q4_K_M: weights ~5.2 GB, KV cache ~2.3 GB at
# 16384 ctx and ~0.4 GB at 4096. The 8 GB row leaves headroom for the desktop compositor, which
# takes its share of VRAM whether or not the model wants it.
if ($vramMb -ge 11000) {
    $derivedCtx = 16384; $maxContextTokens = 8000; $routerFits = $true
} elseif ($vramMb -ge 7500) {
    $derivedCtx = 8192;  $maxContextTokens = 4000; $routerFits = $false
} else {
    $derivedCtx = 8192;  $maxContextTokens = 4000; $routerFits = $false
    Write-Warning "   Under ~7.5 GB an 8B will not fit with usable context. Drop the model size, "
    Write-Warning "   not just the context: use qwen3:4b (set LLM_MODEL accordingly)."
}

$ctx = if ($PSBoundParameters.ContainsKey('ContextLength')) { $ContextLength } else { $derivedCtx }
Write-Host "   OLLAMA_CONTEXT_LENGTH = $ctx (MAX_CONTEXT_TOKENS should be $maxContextTokens)"

$vars = @{
    OLLAMA_HOST           = "0.0.0.0"
    OLLAMA_KEEP_ALIVE     = "-1"
    OLLAMA_CONTEXT_LENGTH = "$ctx"
    OLLAMA_NUM_PARALLEL   = "2"
}

Write-Host "1. Persisting environment variables (User scope)..." -ForegroundColor Cyan
foreach ($k in $vars.Keys) {
    [Environment]::SetEnvironmentVariable($k, $vars[$k], "User")
    # Also set in THIS process, so the server we launch below inherits them. A process
    # started from a shell that predates these vars will NOT see them otherwise — that is
    # how you end up bound to 127.0.0.1 despite OLLAMA_HOST=0.0.0.0 in the registry.
    Set-Item -Path "Env:$k" -Value $vars[$k]
    Write-Host "   $k = $($vars[$k])"
}

Write-Host "2. Restarting Ollama so it picks up the new environment..." -ForegroundColor Cyan
Get-Process -Name "ollama", "ollama app" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 3
Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden

$up = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        if ((Invoke-WebRequest "http://localhost:11434/api/tags" -TimeoutSec 3 -UseBasicParsing).StatusCode -eq 200) { $up = $true; break }
    } catch {}
    Start-Sleep -Seconds 2
}
if (-not $up) { throw "Ollama server did not come up on :11434" }
$bound = (Get-NetTCPConnection -LocalPort 11434 -State Listen -ErrorAction SilentlyContinue | Select-Object -Expand LocalAddress -Unique) -join ", "
Write-Host "   server up, listening on: $bound"

Write-Host "3. Ensuring model is present ($model)..." -ForegroundColor Cyan
if (-not (& ollama list | Select-String -SimpleMatch $model)) {
    Write-Host "   pulling $model (~5 GB, one time)..."
    & ollama pull $model
}

Write-Host "4. Warming the model and checking GPU placement..." -ForegroundColor Cyan
$body = @{ model = $model; prompt = "OK"; stream = $false; think = $false } | ConvertTo-Json
$null = Invoke-RestMethod "http://localhost:11434/api/generate" -Method Post -Body $body -TimeoutSec 180
& ollama ps

# Two 8B models do not co-reside in 12 GB (main 7.5 GB + variant 5.6 GB), so the variant is only
# worth building where the swap is cheaper than the reasoning penalty. On 8 GB it is not: the
# trained classifier removes the LLM router entirely, and decompose falling back to the main
# model is slower per multi-hop query but keeps a single model resident.
if ($SkipRouterModel) {
    $buildRouter = $false
} elseif ($BuildRouterModel) {
    $buildRouter = $true
} else {
    $buildRouter = $routerFits
}

if ($buildRouter) {
    Write-Host "5. Building the thinking-disabled router variant..." -ForegroundColor Cyan
    & "$PSScriptRoot\build-router-model.ps1"
    $routerLines = @("ROUTER_MODE=llm", "ROUTER_MODEL=qwen3-router")
} else {
    Write-Host "5. Skipping the router variant - a second 8B does not fit in $vramMb MiB." -ForegroundColor Cyan
    Write-Host "   Route with the trained classifier instead (no second model, ~1ms)."
    $routerLines = @("ROUTER_MODE=classifier", "ROUTER_MODEL=$model")
}

Write-Host ""
Write-Host "Done. Point the app at it in .env:" -ForegroundColor Green
Write-Host "   LLM_MODEL=$model"
Write-Host "   LLM_BASE_URL=http://host.docker.internal:11434/v1"
Write-Host "   LLM_API_KEY=ollama"
Write-Host "   MAX_CONTEXT_TOKENS=$maxContextTokens"
foreach ($line in $routerLines) { Write-Host "   $line" }
Write-Host ""
Write-Host "Want PROCESSOR to read '100% GPU'. If it shows a CPU split, re-run with"
Write-Host "  -ContextLength $([math]::Max(4096, [int]($ctx / 2)))  and lower MAX_CONTEXT_TOKENS to match."
