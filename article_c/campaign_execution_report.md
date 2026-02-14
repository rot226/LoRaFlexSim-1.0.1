# Rapport d'exécution campagne (SNIR OFF/ON + tailles 80..1280)

## 1) Test rapide

Commande lancée :

```bash
python -m article_c.run_all --allow-non-article-c --clean-hard --network-sizes 80 --replications 1 --seeds_base 1 --snir_modes snir_off,snir_on
```

Résultat : exécution terminée, sorties Step1+Step2 produites, puis validation globale réussie (`Aucune anomalie résultats détectée.`).

## 2) Campagne complète demandée

### Step1 (SNIR OFF+ON)

```bash
python -m article_c.step1.run_step1 --network-sizes 80 160 320 640 1280 --replications 5 --seeds_base 1 --snir_modes snir_off,snir_on --workers 1
```

### Step2 (algorithmes ADR, MixRA-H, MixRA-Opt, UCB1-SF)

```bash
python -m article_c.step2.run_step2 --network-sizes 80 160 320 640 1280 --replications 5 --seeds_base 1 --workers 1
```

### Agrégation finale

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
for step in ['step1','step2']:
    base=Path('article_c')/step/'results'
    out=base/'aggregates'
    out.mkdir(parents=True,exist_ok=True)
    files=sorted((base/'by_size').glob('size_*/aggregated_results.csv'))
    df=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
    df.to_csv(out/'aggregated_results.csv',index=False)
    print(step,'rows',len(df))
PY
```

Résultat agrégats :
- `step1 rows 600`
- `step2 rows 400`

## 3) Validation de sortie

Commande de validation principale :

```bash
python -m article_c.validate_results --step1-dir article_c/step1/results --step2-dir article_c/step2/results
```

Résultat : `Aucune anomalie résultats détectée.`

## 4) Contrôle manuel d'un échantillon (anti-mélange OFF/ON + cohérence numérique)

Vérification manuelle par script Python :

- Step1 : modes SNIR observés = `['snir_off', 'snir_on']`.
- Step2 : mode SNIR observé = `['snir_on']`.
- Comptages `network_size × algo × snir_mode` cohérents (20 lignes par combinaison affichée dans l'échantillon).
- Cohérence numérique :
  - `step1 pdr_mean out_of_range = 0`
  - `step2 success_rate_mean out_of_range = 0`
  - `step2 reward_mean out_of_range = 0`

Extraits observés :
- Step1 (`network_size=1280`, `algo=adr`) contient des lignes distinctes `snir_off` et `snir_on`.
- Step2 (`network_size=1280`) contient des valeurs `success_rate_mean` et `reward_mean` dans [0,1] pour ADR/MixRA-H.

## 5) Note sur `verify_all`

`python -m article_c.tools.verify_all --replications 5` retourne un échec non bloquant lié à un CSV externe manquant de données :

- `CSV sans ligne de données: article_c/common/data/author_curves.csv`

Ce point n'affecte pas la validation des résultats simulés Step1/Step2 ni la cohérence des agrégats produits dans cette campagne.
