[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 8765,
    [string]$DataDir,
    [string]$Python,
    [string]$ResourceDir,
    [string]$BaseModelDir,
    [string]$PlannerModelDir
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot

if ([string]::IsNullOrWhiteSpace($Python)) {
    $runtimePython = Join-Path $projectRoot '.venv/psai_runtime/Scripts/python.exe'
    $projectPython = Join-Path $projectRoot '.venv/Scripts/python.exe'
    if (Test-Path -LiteralPath $runtimePython -PathType Leaf) {
        $Python = $runtimePython
    }
    elseif (Test-Path -LiteralPath $projectPython -PathType Leaf) {
        $Python = $projectPython
    }
    else {
        $Python = (Get-Command python -CommandType Application -ErrorAction Stop).Source
    }
}
else {
    $Python = (Get-Command $Python -CommandType Application -ErrorAction Stop).Source
}

$backendArguments = @('-u', '-m', 'psai_extension', '--port', [string]$Port, '--python', $Python)
$options = @{
    '--data-dir' = $DataDir
    '--resource-dir' = $ResourceDir
    '--base-model-dir' = $BaseModelDir
    '--planner-model-dir' = $PlannerModelDir
}
foreach ($optionName in $options.Keys) {
    if (-not [string]::IsNullOrWhiteSpace($options[$optionName])) {
        $resolvedOption = (Resolve-Path -LiteralPath $options[$optionName] -ErrorAction SilentlyContinue).Path
        if ([string]::IsNullOrWhiteSpace($resolvedOption)) {
            if ($optionName -ne '--data-dir') {
                throw "Local directory does not exist: $($options[$optionName])"
            }
            $resolvedOption = [System.IO.Path]::GetFullPath($options[$optionName])
        }
        $backendArguments += @($optionName, $resolvedOption)
    }
}

Write-Host "PSAI local backend - Python: $Python"
Write-Host 'Keep this console open. Copy the pairing token printed below into the extension.'
Push-Location -LiteralPath $projectRoot
try {
    & $Python @backendArguments
    if ($LASTEXITCODE -ne 0) {
        throw "PSAI backend exited with code $LASTEXITCODE. See the error above."
    }
}
finally {
    Pop-Location
}
