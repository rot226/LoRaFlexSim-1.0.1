# Études de cas FLoRa

Ce document décrit plusieurs scénarios permettant de reproduire fidèlement des résultats publiés avec le simulateur **FLoRa**. Ils servent de base de comparaison scientifique entre l'implémentation Python et l'environnement OMNeT++.

## 1. Scénario de référence

Le fichier `flora_full.ini` et le script `run_flora_example.py` correspondent au cas d'étude présenté dans l'article d'origine sur FLoRa. Cinq nœuds sont placés autour d'une passerelle unique et transmettent 80 paquets chacun. Les résultats de FLoRa sont fournis sous forme de CSV (`flora_full.csv`). Pour exécuter ce scénario :

```bash
python examples/run_flora_example.py --runs 5 --seed 123 \
    --flora-csv examples/flora_full.csv
```

Les métriques obtenues sont alors directement comparables aux valeurs publiées.

## 2. Analyse de montée en charge

L'article « Scalability of LoRaWAN » étudie l'impact du nombre de nœuds sur les performances. Le fichier `flora_scalability.ini` reproduit ce second cas avec cent nœuds disposés sur une grille régulière. Le script `run_flora_scalability.py` lance la simulation et peut être confronté à un export FLoRa équivalent.

```bash
python examples/run_flora_scalability.py --runs 3 --seed 42 \
    --flora-csv chemin/vers/flora_scalability.csv
```

Ce scénario permet de valider la capacité du simulateur à suivre les tendances observées dans la littérature.

## 3. Interférences contrôlées

FLoRa propose également un cas où les transmissions se chevauchent volontairement. Les données `flora_interference.csv` et `flora_collisions.csv` illustrent cette situation. En lançant `run_flora_example.py --degrade` il est possible d'obtenir un comportement similaire et de vérifier le traitement des collisions.

Ces trois études de cas constituent une base solide pour comparer vos résultats aux publications utilisant FLoRa.
