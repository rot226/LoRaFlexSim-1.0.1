# IWCMC – Organisation des expériences

Ce dossier centralise les campagnes IWCMC (SNIR statique, RL statique, RL mobilité),
ainsi que le matériel complémentaire (MED) et les archives. La structure reprend
les conventions utilisées dans les runbooks et scénarios existants afin de garder
une exécution homogène et reproductible.

## Structure des dossiers

- `snir_static/` : campagnes SNIR en environnement statique.
- `rl_static/` : campagnes RL en environnement statique.
- `rl_mobile/` : campagnes RL avec mobilité.
- `MED/` : matériel spécifique MED.
- `archive/` : anciennes campagnes, exports intermédiaires, sauvegardes.

Chaque campagne suit la même sous‑arborescence :

```
modules/   # scripts utilitaires, helpers de plots, modules locaux
scenarios/ # scripts de lancement des scénarios
plots/     # scripts de génération des graphiques
figures/   # figures exportées (PNG/SVG/PDF)
data/      # CSV et données intermédiaires
```

## Installation & dépendances

Pré-requis (alignés sur les documents existants) :

- Python ≥ 3.10
- `pip` (ou `pipx`) et un environnement virtuel
- Dépendances principales : `pandas`, `matplotlib`

Installation en mode développement :

```bash
pip install -e .
```

Si vous exécutez des scripts directement, assurez-vous que la racine du dépôt est
bien dans `PYTHONPATH`.

## Exécution en CLI

### Bash (Linux/macOS/WSL)

```bash
# Exemple générique d’un scénario SNIR statique
python IWCMC/snir_static/scenarios/run_snir_static.py --seed 1 --replications 5

# Exemple de génération de figures
python IWCMC/snir_static/plots/plot_snir_static.py IWCMC/snir_static/data/der_density_snir.csv \
  --output-dir IWCMC/snir_static/figures
```

### PowerShell (Windows 11)

```powershell
# Exemple générique d’un scénario RL statique
python IWCMC/rl_static/scenarios/run_rl_static.py --seed 1 --replications 5

# Exemple de génération de figures
python IWCMC/rl_static/plots/plot_rl_static.py IWCMC/rl_static/data/rl_rewards.csv `
  --output-dir IWCMC/rl_static/figures
```

> Remarque : adaptez les noms de scripts et chemins de sortie aux fichiers
> placés dans `scenarios/` et `plots/`.

## Phases de travail (1–4)

Les phases reprennent les recommandations du runbook SNIR et des scénarios
d’utilisation :

1. **Préparer l’environnement**
   - Activer un virtualenv.
   - Installer les dépendances en mode développement (`pip install -e .`).
   - Vérifier le `PYTHONPATH` si nécessaire.
2. **Exécuter les scénarios**
   - Placer les scripts dans `scenarios/`.
   - Conserver des paramètres cohérents (graine, trafic, zone, etc.).
   - Limiter `--replications` à 5 pour maîtriser le temps d’exécution.
   - Exporter les résultats en CSV dans `data/`.
3. **Générer les graphiques**
   - Centraliser les fonctions communes dans `modules/`.
   - Produire les figures depuis les CSV `data/` vers `figures/`.
   - Inclure le scénario, les options et la graine dans les légendes.
4. **Vérifications & archivage**
   - Valider la cohérence des colonnes CSV.
   - Vérifier les suffixes de figures (baseline, snir, etc.).
   - Archiver les exports ou documenter la commande de génération.

## Tableau des figures IWCMC

Les figures sont organisées par campagne. Le tableau ci‑dessous sert de repère
pour stocker et référencer les fichiers générés.

| ID | Campagne | Emplacement attendu | Description (à compléter) |
| --- | --- | --- | --- |
| S1 | SNIR statique | `snir_static/figures/S1.*` | |
| S2 | SNIR statique | `snir_static/figures/S2.*` | |
| S3 | SNIR statique | `snir_static/figures/S3.*` | |
| S4 | SNIR statique | `snir_static/figures/S4.*` | |
| S5 | SNIR statique | `snir_static/figures/S5.*` | |
| S6 | SNIR statique | `snir_static/figures/S6.*` | |
| S7 | SNIR statique | `snir_static/figures/S7.*` | |
| S8 | SNIR statique | `snir_static/figures/S8.*` | |
| RLS1 | RL statique | `rl_static/figures/RLS1.*` | |
| RLS2 | RL statique | `rl_static/figures/RLS2.*` | |
| RLS3 | RL statique | `rl_static/figures/RLS3.*` | |
| RLS4 | RL statique | `rl_static/figures/RLS4.*` | |
| RLS5 | RL statique | `rl_static/figures/RLS5.*` | |
| RLS6 | RL statique | `rl_static/figures/RLS6.*` | |
| RLS7 | RL statique | `rl_static/figures/RLS7.*` | |
| RLS8 | RL statique | `rl_static/figures/RLS8.*` | |
| RLM1 | RL mobilité | `rl_mobile/figures/RLM1.*` | |
| RLM2 | RL mobilité | `rl_mobile/figures/RLM2.*` | |
| RLM3 | RL mobilité | `rl_mobile/figures/RLM3.*` | |
| RLM4 | RL mobilité | `rl_mobile/figures/RLM4.*` | |
| RLM5 | RL mobilité | `rl_mobile/figures/RLM5.*` | |
| RLM6 | RL mobilité | `rl_mobile/figures/RLM6.*` | |
| RLM7 | RL mobilité | `rl_mobile/figures/RLM7.*` | |
| RLM8 | RL mobilité | `rl_mobile/figures/RLM8.*` | |
