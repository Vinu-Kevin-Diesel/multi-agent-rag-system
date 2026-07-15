<#
.SYNOPSIS
    Configure a host-native Ollama server to serve the model under test to the app container.

.DESCRIPTION
    Runs Ollama natively on the Windows host (not in Docker — GPU passthrough into Compose on
    Windows is a WSL2 detour, and Ollama ships its own CUDA runtime). The app container reaches
    it via host.docker.internal:11434 (wired in docker-compose.yml).

    Sets four environment variables that matter, restarts the server so they take effect, and
    verifies the model is resident on the GPU.

.NOTES
    OLLAMA_HOST=0.0.0.0        Bind all interfaces. Ollama defaults to 127.0.0.1, which a Docker
                              container CANNOT reach — the connection arrives via the Docker
                              gateway IP, not loopback, and is refused. This is the #1 gotcha.
    OLLAMA_CONTEXT_LENGTH     16384. Large enough for the ~8-10k-token prompts the multi-hop
                              path builds; small enough that weights + KV cache fit fully in
                              12 GB VRAM (a bigger context spills to CPU and halves throughput).
    OLLAMA_KEEP_ALIVE=-1      Keep the model resident. Each query makes 3-6 LLM calls; reloading
                              between them would dominate latency.
    OLLAMA_NUM_PARALLEL=2     Modest concurrency without thrashing VRAM.
#>

$ErrorActionPreference = "Stop"

$model = "qwen3:8b"
$vars = @{
    OLLAMA_HOST           = "0.0.0.0"
    OLLAMA_KEEP_ALIVE     = "-1"
    OLLAMA_CONTEXT_LENGTH = "16384"
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

Write-Host ""
Write-Host "Done. Point the app at it in .env:" -ForegroundColor Green
Write-Host "   LLM_MODEL=$model"
Write-Host "   LLM_BASE_URL=http://host.docker.internal:11434/v1"
Write-Host "   LLM_API_KEY=ollama"
Write-Host "Want PROCESSOR to read '100% GPU'. If it shows a CPU split, lower OLLAMA_CONTEXT_LENGTH."
