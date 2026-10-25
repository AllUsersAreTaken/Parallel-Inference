# Run the in-context exemplar probe (ZS8 / FS8 / FD8) and record provenance.
#
# Usage:
#   powershell -File scripts\run_exemplar_s4.ps1 -Tag s4_1b       -Model models\Llama-3.2-1B-Instruct-Q4_K_M.gguf            -Limit 40
#   powershell -File scripts\run_exemplar_s4.ps1 -Tag s4_coder15b -Model models\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf -Limit 40
#
# Writes results\<Tag>_zs8.tsv, results\<Tag>_fs8.tsv, results\<Tag>_fd8.tsv
# and results\<Tag>_provenance.txt.
param(
    [string]$Tag = "s4_1b",
    [string]$Model = "models\Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    [string]$Data = "data\gsm8k_dev200.tsv",
    [int]$Limit = 40,
    [int]$MaxTrace = 1024,
    [int]$Seed = 1234,
    [string]$BuildDir = "build\cuda-graphs",
    [string]$DivExemplars = "results\exemplars_div.txt",
    [string]$SameExemplars = "results\exemplars_same.txt"
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$exe = Join-Path $root "harness\engine\modality.exe"
$bin = Join-Path $root "$BuildDir\bin"
$env:PATH = "$bin;" + $env:PATH

$modes = (1..8 | ForEach-Object { "cot" }) -join ","
$modelPath = Join-Path $root $Model
$dataPath = Join-Path $root $Data
$divPath = Join-Path $root $DivExemplars
$samePath = Join-Path $root $SameExemplars

$runs = @(
    @{ suffix = "zs8"; extra = @("--wrapper") },
    @{ suffix = "fs8"; extra = @("--exemplars", $samePath) },
    @{ suffix = "fd8"; extra = @("--exemplars", $divPath) }
)

foreach ($r in $runs) {
    $out = Join-Path $root "results\${Tag}_$($r.suffix).tsv"
    Write-Host "[run] $Tag $($r.suffix) -> $out"
    & $exe --model $modelPath --data $dataPath --out $out `
        --limit $Limit --modes $modes --max-trace $MaxTrace --seed $Seed @($r.extra)
    if ($LASTEXITCODE -ne 0) { throw "modality $($r.suffix) failed (exit $LASTEXITCODE)" }
}

function Get-Sha256([string]$p) {
    if (-not (Test-Path -LiteralPath $p)) { return "missing" }
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $p).Hash.ToLower()
}

$prov = @(
    "model_file: $Model",
    "model_sha256: $(Get-Sha256 $modelPath)",
    "source_file: harness\engine\modality.cpp",
    "source_sha256: $(Get-Sha256 (Join-Path $root 'harness\engine\modality.cpp'))",
    "exe_file: harness\engine\modality.exe",
    "exe_sha256: $(Get-Sha256 $exe)",
    "data_file: $Data",
    "data_sha256: $(Get-Sha256 $dataPath)",
    "div_exemplars: $DivExemplars",
    "div_exemplars_sha256: $(Get-Sha256 $divPath)",
    "same_exemplars: $SameExemplars",
    "same_exemplars_sha256: $(Get-Sha256 $samePath)",
    "modes: $modes",
    "zs8_flags: --wrapper",
    "fs8_flags: --exemplars $SameExemplars",
    "fd8_flags: --exemplars $DivExemplars",
    "max_trace: $MaxTrace",
    "seed: $Seed",
    "limit: $Limit",
    "generated: $(Get-Date -Format o)"
)
$provPath = Join-Path $root "results\${Tag}_provenance.txt"
$prov | Set-Content -LiteralPath $provPath -Encoding ascii
Write-Host "[provenance] $provPath"
