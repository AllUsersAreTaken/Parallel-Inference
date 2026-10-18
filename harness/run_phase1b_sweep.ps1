# Phase 1b driver: runs the knee sweep across arms x depths x build configs.
#
#   .\run_phase1b_sweep.ps1 -Model <gguf> [-Depths 512,4096,8192] [-Configs graphs,nographs]
#
# Writes per-step and per-run CSVs under results\. Run from an elevated shell
# with other GPU apps closed. Energy fields are nan on this card (no NVML power).
param(
    [Parameter(Mandatory=$true)] [string]$Model,
    [object]   $Depths  = '512,4096,8192',
    [object]   $Arms    = 'shared,separate',
    [object]   $Configs = 'graphs,nographs',
    [string]   $C       = '0,1,2,4,8,16,32,64,128',
    [int]      $Steps   = 100,
    [int]      $Warmup  = 10,
    [int]      $IdleSecs= 30,
    [string]   $OutDir  = 'S:\Agent\QSExplore\RecycleLM\results',
    [switch]   $SkipVary
)
$ErrorActionPreference = 'Stop'
$Depths  = @([string[]]$Depths  | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ } | ForEach-Object { [int]$_ })
$Arms    = @([string[]]$Arms    | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ })
$Configs = @([string[]]$Configs | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ })
$root = 'S:\Agent\QSExplore\RecycleLM'
$exe  = Join-Path $root 'harness\phase1b.exe'
if (-not (Test-Path $exe)) { throw "missing $exe (run build_harness.bat)" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$stepsCsv = Join-Path $OutDir 'phase1b_steps.csv'
$runsCsv  = Join-Path $OutDir 'phase1b_runs.csv'

foreach ($cfg in $Configs) {
    $bin = Join-Path $root "build\cuda-$cfg\bin"
    if (-not (Test-Path $bin)) { throw "missing $bin" }
    $env:PATH = "$bin;$env:PATH"
    Write-Host "=== config=$cfg ==="

    foreach ($arm in $Arms) {
        foreach ($d in $Depths) {
            $label = "${cfg}_${arm}_D${d}"
            Write-Host "[phase1b] $label fixed C=$C"
            & $exe --model $Model --arm $arm --mode fixed --depth $d --C $C `
                   --steps $Steps --warmup $Warmup --idle-secs $IdleSecs --label $label `
                   --steps-csv $stepsCsv --runs-csv $runsCsv
            if ($LASTEXITCODE -ne 0) { throw "run failed: $label" }

            if (-not $SkipVary) {
                $vlabel = "${cfg}_${arm}_D${d}_vary"
                Write-Host "[phase1b] $vlabel vary"
                & $exe --model $Model --arm $arm --mode vary --depth $d --C $C `
                       --steps $Steps --warmup $Warmup --idle-secs $IdleSecs --label $vlabel `
                       --steps-csv $stepsCsv --runs-csv $runsCsv
                if ($LASTEXITCODE -ne 0) { throw "run failed: $vlabel" }
            }
        }
    }
}
Write-Host "[phase1b] done -> $runsCsv"
