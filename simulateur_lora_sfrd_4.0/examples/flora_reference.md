# Scénarios de référence FLoRa

Ce document présente les fichiers permettant de reproduire à l'identique les scénarios officiels de FLoRa afin d'obtenir des résultats chiffrés comparables.

## Positions et paramètres

Le fichier `flora_full.ini` fournit la configuration complète :

```ini
[gateways]
gw0 = 500,500

[nodes]
n0 = 450.45,490,12,14
n1 = 555,563,12,14
n2 = 460,467,12,14
n3 = 565,571,12,14
n4 = 702,578,12,14
```

- **Spreading factor** fixé à 12 pour tous les nœuds
- **Puissance d’émission** fixée à 14 dBm
- **Canal unique** de 125 kHz avec un seuil de détection à –110 dBm
- Fenêtre d'interférence de 5 s

Ces valeurs reprennent les paramètres du projet FLoRa.

## Lancer une simulation identique

Le script `run_flora_example.py` exécute ce scénario et peut comparer les métriques obtenues à celles de FLoRa :

```bash
python examples/run_flora_example.py --runs 5 --seed 123 \
    --flora-csv examples/flora_full.csv
```

Les différentes sorties FLoRa de référence sont disponibles dans le dossier `examples` :

- `flora_full.csv`
- `flora_collisions.csv`
- `flora_interference.csv`
- `flora_rx_stats.csv`

Elles peuvent servir de point de comparaison avec les scripts `compare_flora_report.py` ou `compare_flora_multi.py`.

## Comparaison automatique

Pour générer un rapport graphique comparant un résultat à la référence :

```bash
python tools/compare_flora_report.py results.csv examples/flora_full.csv
```

Vous obtiendrez ainsi une comparaison systématique (PDR, distribution SF,
énergie consommée, etc.) entre la simulation Python et FLoRa.
