# Expériences UCB1

Ce dossier rassemble les scripts permettant d'évaluer l'algorithme UCB1
(``LoRaSFSelectorUCB1``) intégré à LoRaFlexSim. Chaque nœud explore les facteurs
d'étalement disponibles via un bandit UCB1 et renforce les choix offrant un taux
de réussite élevé. Les clusters QoS sont configurés avec des objectifs de PDR
(cible de probabilité de livraison) distincts, ce qui permet d'observer comment
l'algorithme ajuste la distribution des SF.

## Scripts de simulation

Trois campagnes sont fournies. Les commandes peuvent être lancées depuis la
racine du dépôt ou depuis ``experiments/ucb1``.

- **Balayage de densité** :
  ```bash
  python experiments/ucb1/run_ucb1_density_sweep.py
  ```
  Les simulations font varier le nombre de nœuds (par défaut : 2000 à 15000) et
  exportent ``experiments/ucb1/ucb1_density_metrics.csv``.

- **Balayage de charge** :
  ```bash
  python experiments/ucb1/run_ucb1_load_sweep.py
  ```
  Trois intervalles de génération de paquets (300, 600, 900 s) sont testés pour
  un parc de nœuds fixe ; les résultats sont écrits dans
  ``experiments/ucb1/ucb1_load_metrics.csv``.

- **Comparaison UCB1 / ADR / MixRA** :
  ```bash
  python experiments/ucb1/run_baseline_comparison.py
  ```
  Cette campagne produit ``experiments/ucb1/ucb1_baseline_metrics.csv`` en
  évaluant les trois stratégies avec les mêmes paramètres réseaux.

Tous les CSV partagent les colonnes suivantes (``algorithm`` n'est présent que
pour la comparaison) :

- ``num_nodes`` : nombre total de nœuds simulés.
- ``cluster`` : identifiant du cluster QoS du nœud (1, 2 ou 3).
- ``sf`` : facteur d'étalement moyen observé sur le cluster.
- ``reward_mean`` : récompense moyenne stockée par le bandit UCB1. Elle
  correspond à ``succès - pénalité`` (1 pour une trame reçue, 0 sinon) pondéré
  par un bonus d'exploration (bornes supérieures de confiance) ; plus elle est
  élevée, plus le SF testé est jugé efficace.
- ``der`` : Data Extraction Rate, ratio de paquets décodés par rapport aux
  tentatives.
- ``pdr`` : Packet Delivery Ratio calculé par le gestionnaire QoS (cible
  spécifique à chaque cluster).
- ``snir_avg`` : SNIR (Signal to Noise plus Interference Ratio) moyen sur le
  dernier paquet entendu par les nœuds du cluster ; des valeurs plus élevées
  traduisent de meilleures conditions radio et expliquent souvent des DER/PDR
  plus élevés.
- ``success_rate`` : taux de réussite des transmissions (équivalent au DER dans
  ces campagnes).

## Génération des figures

Les scripts de traçage lisent directement les CSV générés et déposent les PNG
dans ``experiments/ucb1/plots`` (les fichiers sont nommés automatiquement en
fonction de la campagne). Exemple d’utilisation :

```bash
python experiments/ucb1/plots/plot_der_by_cluster.py
python experiments/ucb1/plots/plot_load_sensitivity.py
python experiments/ucb1/plots/plot_throughput.py
```

- ``plot_der_by_cluster.py`` reproduit la figure DER par cluster sur le balayage
  de densité et ajoute les lignes horizontales correspondant aux cibles QoS.
- ``plot_load_sensitivity.py`` trace le DER par cluster en fonction de l’intervalle
  de génération de paquets (les intervalles sont inférés si la colonne manque
  dans le CSV).
- ``plot_throughput.py`` présente un débit relatif (taux de succès issu du DER
  ou de la PDR) par cluster ainsi qu’une moyenne globale pondérée par les
  proportions de cluster.

Chaque script accepte des options pour changer le chemin du CSV, l’output ou les
paramètres (cibles DER, intervalles ou proportions) afin de reproduire les
figures sur vos propres campagnes.
