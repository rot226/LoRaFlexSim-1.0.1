param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'src'

python -m mobilesfrdth @Args
exit $LASTEXITCODE
