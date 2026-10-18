# Phase 1a driver: pure-batching sweep with an explicit context-budget guard.
#
# For each (prompt, n_parallel) cell it computes n_ctx_req first and refuses to
# run (rather than letting llama-batched-bench silently skip) when the cell does
# not fit. Deep prompts use -pps (shared prefix) to fit 8 GB.
#
#   .\run_phase1a.ps1 -Model <gguf> [-NCtx 32768] [-SharedFrom 4096]
param(
    [Parameter(Mandatory=$true)] [string]$Model,
    [int]      $Ntg        = 128,
    [int]      $NCtx       = 32768,
    [int]      $SharedFrom = 4096,   # prompts >= this use -pps (shared prefix)
    [object]   $Prompts    = '512,4096,8192',
    [object]   $Npls       = '1,2,4,8,16',
    [string]   $Config     = 'graphs',
    [string]   $OutCsv     = 'S:\Agent\QSExplore\RecycleLM\results\phase1a.csv'
)
$ErrorActionPreference = 'Stop'
$Prompts = @([string[]]$Prompts | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ } | ForEach-Object { [int]$_ })
$Npls    = @([string[]]$Npls    | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ } | ForEach-Object { [int]$_ })
$root = 'S:\Agent\QSExplore\RecycleLM'
$bin  = Join-Path $root "build\cuda-$Config\bin"
$exe  = Join-Path $bin 'llama-batched-bench.exe'
if (-not (Test-Path $exe)) { throw "missing $exe" }

"config,pp,tg,pl,shared,n_ctx_req,status,t_pp,s_pp,t_tg,s_tg_agg,s_tg_per_stream" |
    Set-Content -Path $OutCsv

foreach ($pp in $Prompts) {
    $shared = $pp -ge $SharedFrom
    foreach ($pl in $Npls) {
        # Default KV is non-unified: n_ctx_seq = NCtx/pl, so each sequence needs
        # pp+tg positions regardless of -pps (which only shares prefill compute).
        $n_ctx_req = $pl * ($pp + $Ntg)
        if ($n_ctx_req -gt $NCtx) {
            $msg = "SKIP pp=$pp pl=$pl shared=$shared n_ctx_req=$n_ctx_req > NCtx=$NCtx"
            Write-Host "[phase1a] $msg"
            "$Config,$pp,$Ntg,$pl,$shared,$n_ctx_req,skip,,,,," | Add-Content -Path $OutCsv
            continue
        }
        $flags = @('-m', $Model, '-c', $NCtx, '-b', '2048', '-ub', '512',
                   '-npp', $pp, '-ntg', $Ntg, '-npl', $pl, '-ngl', '99', '-fa', '1')
        if ($shared) { $flags += '-pps' }
        $out = & $exe @flags 2>&1
        $row = $out | Where-Object { $_ -match '^\|\s*\d' } | Select-Object -Last 1
        if (-not $row) {
            Write-Host "[phase1a] FAILED pp=$pp pl=$pl"
            "$Config,$pp,$Ntg,$pl,$shared,$n_ctx_req,fail,,,,," | Add-Content -Path $OutCsv
            continue
        }
        $f = ($row -split '\|') | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }
        # f = PP,TG,B,N_KV,T_PP,S_PP,T_TG,S_TG,T,S
        $s_tg_agg = [double]$f[7]
        $per = $s_tg_agg / $pl
        Write-Host ("[phase1a] pp={0} pl={1} shared={2} | S_TG_agg={3:N1} t/s | per-stream={4:N1} t/s" -f $pp,$pl,$shared,$s_tg_agg,$per)
        "$Config,$pp,$Ntg,$pl,$shared,$n_ctx_req,ok,$($f[4]),$($f[5]),$($f[6]),$s_tg_agg,$per" | Add-Content -Path $OutCsv
    }
}
Write-Host "[phase1a] done -> $OutCsv"
