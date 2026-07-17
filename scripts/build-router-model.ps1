<#
.SYNOPSIS
    Build a thinking-disabled qwen3 variant for the router/decompose classification calls.

.DESCRIPTION
    The router only needs to emit one label from a fixed set. qwen3's reasoning mode turns that
    into a 17-70s generation whose answer is also non-deterministic across runs — and it cannot
    be turned off through Ollama's OpenAI-compatible /v1 endpoint (the `think` parameter is only
    honoured by the native /api/chat). Proven by direct probing:

        /v1  + response_format + think:false  ->  still thinks, ~30-70s, sometimes wrong
        native /api/chat + think:false        ->  0.3s, no thinking

    Since the app talks OpenAI /v1 (provider-agnostic by design), the fix lives in the model, not
    the code: this bakes a variant whose chat template forces the no-think branch unconditionally.
    Derived from whatever qwen3:8b is installed rather than committing a frozen 268-line template,
    so it survives `ollama pull` bumping the base model.

    Result via /v1: ~0.5s, correct, and the answer is the bare category word.
#>

$ErrorActionPreference = "Stop"

$base = "qwen3:8b"
$variant = "qwen3-router"

Write-Host "Deriving $variant from $base ..." -ForegroundColor Cyan

$modelfile = (& ollama show $base --modelfile) -split "`n" |
    Where-Object { $_ -notmatch '^\s*#' }         # drop the informational comment header

# Force both thinking gates in the qwen3 template to the no-think branch:
#   $.IsThinkSet -> true   (the /think|/no_think block only fires when this is set; /v1 never sets it)
#   $.Think      -> false  (selects /no_think, and emits the empty <think></think> that ends reasoning)
# $.Think uses a word boundary so it does not touch the unrelated `.Thinking` variable.
$modelfile = $modelfile -join "`n"
$modelfile = $modelfile -replace '\$\.IsThinkSet', 'true'
$modelfile = $modelfile -replace '\$\.Think\b', 'false'

$tmp = Join-Path $env:TEMP "qwen3-router.modelfile"
Set-Content -Path $tmp -Value $modelfile -NoNewline

& ollama create $variant -f $tmp
Remove-Item $tmp -ErrorAction SilentlyContinue

Write-Host "Built '$variant'. Point the router at it in .env:" -ForegroundColor Green
Write-Host "   ROUTER_MODEL=$variant"
