param(
    [ValidateSet('smoke', 'core_article', 'full_article')]
    [string]$Profile = 'core_article',
    [string]$Out = 'runs/campaign_profiles'
)

$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'src'

$baseArgs = @(
    'run',
    '--config', 'experiments/default.yaml',
    '--out', $Out,
    '--grid'
)

switch ($Profile) {
    'smoke' {
        $grid = 'N=50;speed=1;mode=SNIR_OFF,SNIR_ON;algo=ADR,UCB;reps=1;duration_s=300;seed_base=1234'
        $extra = @('--max-runs', '4', '--max-walltime', '1200')
    }
    'core_article' {
        $grid = 'N=50,100,160;speed=1,3;mode=SNIR_OFF,SNIR_ON;algo=ADR,MIXRA_H,MIXRA_OPT,UCB;reps=3;duration_s=1800;seed_base=1234'
        $extra = @('--resume', '--max-walltime', '21600')
    }
    'full_article' {
        $grid = 'N=50,100,160,320;speed=0,1,3,6;mode=SNIR_OFF,SNIR_ON;algo=ADR,MIXRA_H,MIXRA_OPT,UCB;reps=5;duration_s=3600;seed_base=1234'
        $extra = @('--resume', '--max-walltime', '172800')
    }
    default {
        throw "Profil non supporté: $Profile"
    }
}

$cmd = @($baseArgs + $grid + $extra)
Write-Host "[mobilesfrdth] Profil=$Profile"
Write-Host "[mobilesfrdth] Sortie=$Out"
Write-Host "[mobilesfrdth] Commande: python -m mobilesfrdth $($cmd -join ' ')"

python -m mobilesfrdth @cmd
exit $LASTEXITCODE
