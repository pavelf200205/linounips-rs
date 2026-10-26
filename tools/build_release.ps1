# Run manually: builds the standard Cargo artifact and preserves a labelled copy.
param([string]$TargetDir)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    $cargoArgs = @('build', '--release', '--bin', 'lino', '--message-format=json-render-diagnostics')
    if ($TargetDir) { $cargoArgs += @('--target-dir', $TargetDir) }
    $executable = $null
    & cargo @cargoArgs | ForEach-Object {
        $record = $_ | ConvertFrom-Json
        if ($record.reason -eq 'compiler-message' -and $record.message.rendered) {
            Write-Host $record.message.rendered -NoNewline
        }
        if ($record.reason -eq 'compiler-artifact' -and $record.target.name -eq 'lino' -and $record.executable) {
            $executable = $record.executable
        }
    }
    if ($LASTEXITCODE -ne 0 -or !$executable) { throw 'Cargo build failed or did not report an executable.' }
    # --version exits in argument parsing, before device/cache initialization.
    # Read the built identity instead of assuming HEAD stayed fixed during build.
    $version = (& $executable --version | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $version -notmatch '^lino ([0-9a-f]{12,40}(?:-dirty)?|unknown)$') {
        throw "Unexpected executable version: $version"
    }
    $revision = $Matches[1]
    $digest = (Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash
    $suffix = [IO.Path]::GetExtension($executable)
    $destination = Join-Path (Split-Path $executable -Parent) "lino_$revision$suffix"
    if (Test-Path -LiteralPath $destination) {
        if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash -ne $digest) {
            # Preserve distinct dirty builds or builds with different flags.
            $destination = Join-Path (Split-Path $executable -Parent) "lino_${revision}_$($digest.Substring(0,12).ToLowerInvariant())$suffix"
        }
    }
    if (!(Test-Path -LiteralPath $destination)) {
        Copy-Item -LiteralPath $executable -Destination $destination
    } elseif ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash -ne $digest) {
        throw "Refusing to overwrite a different executable: $destination"
    }
    Write-Host "Preserved build: $destination"
} finally {
    Pop-Location
}
