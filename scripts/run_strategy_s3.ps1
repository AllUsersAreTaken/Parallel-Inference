# Run the strategy-diversity probe (HOM8 vs DIV8) and record provenance.
#
# Usage:
#   powershell -File scripts\run_strategy_s3.ps1 -Tag s3_1b       -Model models\Llama-3.2-1B-Instruct-Q4_K_M.gguf            -Limit 40
#   powershell -File scripts\run_strategy_s3.ps1 -Tag s3_coder15b -Model models\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf -Limit 40
#
# Writes results\<Tag>_hom8.tsv, results\<Tag>_div8.tsv and results\<Tag>_provenance.txt
param(
    [string]$Tag = "s3_1b",
    [string]$Model = "models\Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    [string]$Data = "data\gsm8k_dev200.tsv",
    [int]$Limit = 40,
    [int]$MaxTrace = 1024,
    [int]$Seed = 1234,
    [string]$BuildDir = "build\cuda-graphs"
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$exe = Join-Path $root "harness\engine\modality.exe"
$bin = Join-Path $root "$BuildDir\bin"
$env:PATH = "$bin;" + $env:PATH

$homS = (1..8 | ForEach-Object { "cot" }) -join ","
$divS = "cot,fwd,bwd,table,subq,restate,units,check"

$runs = @(
    @{ suffix = "hom8"; modes = $homS },
    @{ suffix = "div8"; modes = $divS }
)

foreach ($r in $runs) {
    $out = Join-Path $root "results\${Tag}_$($r.suffix).tsv"
    Write-Host "[run] $Tag $($r.suffix) -> $out"
    & $exe --model (Join-Path $root $Model) --data (Join-Path $root $Data) --out $out `
        --limit $Limit --modes $r.modes --max-trace $MaxTrace --seed $Seed
    if ($LASTEXITCODE -ne 0) { throw "modality $($r.suffix) failed (exit $LASTEXITCODE)" }
}

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
    "seed: $Seed",
    "limit: $Limit",
    "hom8_modes: $homS",
    "div8_modes: $divS",
    "generated: $(Get-Date -Format o)"
)
$provPath = Join-Path $root "results\${Tag}_provenance.txt"
$prov | Set-Content -LiteralPath $provPath -Encoding ascii
Write-Host "[provenance] $provPath"
