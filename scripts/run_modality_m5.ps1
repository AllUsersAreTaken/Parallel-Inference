# Run the m5 confound-free within-run cross-substrate probe for one model.
#
# One 24-stream batch per problem:
#   streams 0-7   cot   S0..S7
#   streams 8-15  cot   S8..S15
#   streams 16-23 code  S8..S15
#
# Usage:
#   powershell -File scripts\run_modality_m5.ps1 -Tag m5_coder15b `
#       -Model models\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf -Limit 100
#   powershell -File scripts\run_modality_m5.ps1 -Tag m5_1b `
#       -Model models\Llama-3.2-1B-Instruct-Q4_K_M.gguf -Limit 100
#
# Writes results\<Tag>_within.tsv (+ .run.json) and results\<Tag>_provenance.txt.
param(
    [string]$Tag = "m5_coder15b",
    [string]$Model = "models\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf",
    [string]$Data = "data\gsm8k_dev200.tsv",
    [int]$Limit = 100,
    [int]$MaxTrace = 1024,
    [string]$BuildDir = "build\cuda-graphs"
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$exe = Join-Path $root "harness\engine\modality.exe"
$bin = Join-Path $root "$BuildDir\bin"
$env:PATH = "$bin;" + $env:PATH

# First 16 values are the normal series 1234 + 101*j. The code half reuses
# S8..S15 so C16 and M16 differ only in the second half's mode.
$S = @(1234, 1335, 1436, 1537, 1638, 1739, 1840, 1941,
       2042, 2143, 2244, 2345, 2446, 2547, 2648, 2749)
$modes = ((1..16 | ForEach-Object { "cot" }) + (1..8 | ForEach-Object { "code" })) -join ","
$seeds = ($S + $S[8..15]) -join ","

$out = Join-Path $root "results\${Tag}_within.tsv"
Write-Host "[run] $Tag -> $out"
Write-Host "[modes] $modes"
Write-Host "[seeds] $seeds"
& $exe --model (Join-Path $root $Model) --data (Join-Path $root $Data) --out $out `
    --limit $Limit --modes $modes --max-trace $MaxTrace --seeds $seeds
if ($LASTEXITCODE -ne 0) { throw "modality failed (exit $LASTEXITCODE)" }

function Get-Sha256([string]$p) {
    if (-not (Test-Path -LiteralPath $p)) { return "missing" }
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $p).Hash.ToLower()
}

$prov = @(
    "model_file: $Model",
    "model_sha256: $(Get-Sha256 (Join-Path $root $Model))",
    "source_file: harness\engine\modality.cpp",
    "source_sha256: $(Get-Sha256 (Join-Path $root 'harness\engine\modality.cpp'))",
    "exe_file: harness\engine\modality.exe",
    "exe_sha256: $(Get-Sha256 $exe)",
    "data_file: $Data",
    "data_sha256: $(Get-Sha256 (Join-Path $root $Data))",
    "max_trace: $MaxTrace",
    "seed: 1234",
    "seeds: $seeds",
    "modes: $modes",
    "limit: $Limit",
    "generated: $(Get-Date -Format o)"
)
$provPath = Join-Path $root "results\${Tag}_provenance.txt"
$prov | Set-Content -LiteralPath $provPath -Encoding ascii
Write-Host "[provenance] $provPath"
