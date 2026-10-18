# Run the Phase 1b harness against a chosen CUDA build so the matching
# ggml-cuda.dll (graphs on/off) is loaded first.
#
# Usage:
#   .\run_phase1b.ps1 -Config graphs  -Args "--model M.gguf --depth 512 --C 0,1,2 ..."
#   .\run_phase1b.ps1 -Config nographs -FusionOff -Args "..."
param(
    [ValidateSet('graphs','nographs')]
    [string]$Config = 'graphs',
    [switch]$FusionOff,
    [Parameter(Mandatory=$true)]
    [string]$Args
)
$ErrorActionPreference = 'Stop'
$root = 'S:\Agent\QSExplore\RecycleLM'
$bin  = Join-Path $root "build\cuda-$Config\bin"
$exe  = Join-Path $root 'harness\phase1b.exe'
if (-not (Test-Path $exe)) { throw "missing $exe (run build_harness.bat)" }
if (-not (Test-Path $bin)) { throw "missing $bin" }

$env:PATH = "$bin;$env:PATH"
if ($FusionOff) { $env:GGML_CUDA_DISABLE_FUSION = '1' } else { Remove-Item Env:\GGML_CUDA_DISABLE_FUSION -ErrorAction SilentlyContinue }
$env:GGML_CUDA_GRAPH_DEBUG = '0'

Write-Host "[run_phase1b] config=$Config fusion_off=$($FusionOff.IsPresent)"
& $exe @($Args -split '\s+') 
exit $LASTEXITCODE
