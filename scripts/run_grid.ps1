param(
    [string]$RunArgs = "--help",
    [string]$AggregateArgs = "--help",
    [string]$PlotsArgs = "--help"
)

$ErrorActionPreference = 'Stop'

Write-Host "[1/3] mobilesfrdth run $RunArgs"
Invoke-Expression "mobilesfrdth run $RunArgs"

Write-Host "[2/3] mobilesfrdth aggregate $AggregateArgs"
Invoke-Expression "mobilesfrdth aggregate $AggregateArgs"

Write-Host "[3/3] mobilesfrdth plots $PlotsArgs"
Invoke-Expression "mobilesfrdth plots $PlotsArgs"

Write-Host "Pipeline mobilesfrdth terminé."
