# Manual only. Does not build. Every route is measured separately against this binary's baseline.
param(
    [Parameter(Mandatory=$true)][string]$Executable,
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$Data,
    [Parameter(Mandatory=$true)][string]$Out,
    [int]$Images = 9,
    [ValidateSet('standard','pbr')][string]$Mode = 'standard',
    [ValidateSet('fp16','bf16','fp32')][string]$Precision = 'fp16',
    [int]$Runs = 1,
    [switch]$FullMatrix
)
$ErrorActionPreference = 'Stop'
$binary = (Resolve-Path -LiteralPath $Executable).Path
$modelPath = (Resolve-Path -LiteralPath $Model).Path
$dataPath = (Resolve-Path -LiteralPath $Data).Path
if (Test-Path -LiteralPath $Out) { throw 'Choose a new output directory; existing benchmark results are never overwritten.' }
if ($Runs -lt 1) { throw 'Runs must be positive.' }
$root = [IO.Path]::GetFullPath($Out)
New-Item -ItemType Directory -Path $root | Out-Null
$cases = @(
    @{Name='baseline'; Flags=@('--legacy-fast-path')},
    @{Name='direct'; Flags=@()},
    @{Name='direct_separate_activation'; Flags=@('--separate-conv-activation')},
    @{Name='winograd'; Flags=@('--winograd-convolutions')},
    @{Name='direct_window1'; Flags=@('--conv-layer-window','1')},
    @{Name='direct_window4'; Flags=@('--conv-layer-window','4')}
)
if ($FullMatrix) {
    $cases += @(
        @{Name='direct32'; Flags=@('--conv-workspace-mib','32')},
        @{Name='direct8'; Flags=@('--conv-workspace-mib','8')},
        @{Name='winograd8'; Flags=@('--winograd-convolutions','--conv-workspace-mib','8')},
        @{Name='winograd32'; Flags=@('--winograd-convolutions','--conv-workspace-mib','32')},
        @{Name='wide_fusion'; Flags=@('--wide-wavelet-fusion')},
        @{Name='mlp8192'; Flags=@('--mlp-chunk-tokens','8192')},
        @{Name='query1024'; Flags=@('--attention-query-chunk','1024')},
        @{Name='retain_pool'; Flags=@('--pool-cache-mib','0')},
        @{Name='direct_no_flash'; Flags=@('--no-flash-attention')}
    )
}
$results = [Collections.Generic.List[object]]::new()
for ($run = 1; $run -le $Runs; $run++) {
    # Reverse experimental order on alternate rounds, while creating that round's baseline first.
    $ordered = @($cases[0])
    $rest = @($cases | Select-Object -Skip 1)
    if ($run % 2 -eq 0) { [array]::Reverse($rest) }
    $ordered += $rest
    $baseline = Join-Path $root "run${run}_baseline"
    foreach ($case in $ordered) {
        $destination = Join-Path $root "run${run}_$($case.Name)"
        $log = Join-Path $root "run${run}_$($case.Name).log"
        $arguments = @('infer','--model',$modelPath,'--data',$dataPath,'--out',$destination,
            '--mode',$Mode,'--images',"$Images",'--precision',$Precision,'--seed','42','--encoder-batch','4',
            '--pixel-batch','2048','--formats','exr,png','--no-progress') + $case.Flags
        $watch = [Diagnostics.Stopwatch]::StartNew()
        & $binary @arguments *> $log
        $exitCode = $LASTEXITCODE
        $watch.Stop()
        $metadataPath = Join-Path $destination 'metadata.json'
        $metadata = if (Test-Path -LiteralPath $metadataPath) { Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json } else { $null }
        $equal = $null
        if ($exitCode -eq 0 -and $case.Name -ne 'baseline') {
            & $binary compare --reference $baseline --candidate $destination *> (Join-Path $root "run${run}_$($case.Name)_equality.log")
            $equal = $LASTEXITCODE -eq 0
        }
        $results.Add([pscustomobject]@{
            Run=$run; Case=$case.Name; ExitCode=$exitCode; EqualToBaseline=$equal;
            ProcessSeconds=$watch.Elapsed.TotalSeconds; InferenceSeconds=$metadata.inference_seconds;
            EncoderSeconds=$metadata.operation_timings.encoder.inclusive_wall_seconds;
            PoolReservedSampleBytes=$metadata.memory.sampled_pool_reserved_peak_bytes;
            Flags=($case.Flags -join ' ')
        })
        $results | Export-Csv -LiteralPath (Join-Path $root 'summary.csv') -NoTypeInformation
        $results | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $root 'summary.json')
        Write-Host "$($case.Name): exit=$exitCode inference=$($metadata.inference_seconds)s identical=$equal"
    }
}
