# Journal d'exécution

Cette branche documente les tentatives pour suivre la séquence demandée.

## Étapes réalisées
- Création d'un environnement virtuel local (`python -m venv .venv`) puis activation.
- Installation hors ligne de `setuptools` à partir de la roue système pour satisfaire le backend PEP 517 (`/usr/share/python-wheels/setuptools-68.1.2-py3-none-any.whl`).

## Points bloquants
- `pip install -e .` échoue car le réseau est filtré (erreurs 403 sur le proxy lors du téléchargement des dépendances) et la commande `bdist_wheel` est indisponible sans le paquet `wheel`.
- Les dépendances principales (ex. `numpy`) ne peuvent pas être installées pour la même raison, ce qui bloque l'exécution de `python -m loraflexsim.run` même en mode rapide.
- Les répertoires `data/` et `plots/` n'existent pas et aucune nouvelle sortie CSV ou EPS n'a pu être générée tant que les prérequis ne sont pas installés.

## Actions requises pour terminer la séquence
- Fournir des roues hors ligne pour `wheel`, `numpy`, `pandas`, `scipy`, `matplotlib` et les autres dépendances listées dans `pyproject.toml`, ou lever les restrictions réseau le temps de l'installation.
- Relancer `pip install -e .` une fois ces paquets disponibles, puis exécuter successivement le runner en mode `--fast` puis complet, `prepare_ieee_figures.py` pour produire les `_ieee.csv`, et enfin les scripts de génération de figures EPS avec le style `ieee_plot_style.yaml`.
