$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'src'

python -m mobilesfrdth @args
exit $LASTEXITCODE
