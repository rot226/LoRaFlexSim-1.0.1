# Runbook – Comparaison SNIR (Étape 1)

Ce runbook décrit la marche à suivre pour préparer les comparaisons baseline vs SNIR en utilisant les scripts de `experiments/snir_stage1_compare/`.

## 1. Préparer l'environnement
1. Activer un virtualenv ou utiliser celui du dépôt.
2. Installer les dépendances en mode développement :
   ```bash
   pip install -e .
   ```
3. Vérifier que la variable `PYTHONPATH` inclut la racine du dépôt si vous lancez les scripts directement.

## 2. Exécuter les scénarios
1. Placer ou écrire les scripts dans `scenarios/` en exposant les paramètres suivants :
   - baseline : `flora_mode=True`, `snir_model=False`, `interference_model=False`.
   - SNIR uniquement : `flora_mode=True`, `snir_model=True`, `interference_model=False`.
   - SNIR + interférences : `flora_mode=True`, `snir_model=True`, `interference_model=True`.
2. Pour chaque script, utiliser les mêmes paramètres de trafic (graine, nombre de paquets, intervalle, zone) afin de permettre la comparaison.
3. La DER baseline reflète strictement `packets_delivered / packets_sent` (pas de biais artificiel). On s'attend donc à une DER baseline légèrement plus basse (jusqu'à ~0,05 auparavant) et à un écart SNIR/DER qui reflète mieux la performance réelle.
4. Exporter les résultats en CSV dans `data/` avec un nom explicite, par exemple `der_density_baseline.csv`, `der_density_snir.csv`, etc.
5. Le nombre de répétitions (`--replications`) est plafonné à 5 afin de limiter le temps de calcul.

## 3. Générer les graphiques
1. Centraliser les fonctions communes (palette, mise en forme des légendes, labels) dans `plots_helpers/`.
2. Écrire des scripts de tracé qui consomment les CSV de `data/` et produisent des PNG/PDF dans `plots/`.
3. Inclure dans chaque figure le scénario, les options SNIR activées et la graine utilisée.

## 4. Vérifications finales
1. S'assurer que chaque CSV contient les colonnes utilisées par les scripts de tracé (p. ex. `density`, `der`, `snir_db`).
2. Confirmer que les figures générées portent un suffixe cohérent (`baseline`, `snir`, `snir_interference`).
3. Documenter dans `README.md` toute hypothèse supplémentaire ou écart par rapport aux scripts d'origine.

## 5. Archivage
- Pousser les CSV et figures dans `data/` et `plots/` si une revue est nécessaire, sinon documenter la commande de génération.
- Conserver les commandes exécutées (avec options) pour garantir la reproductibilité.
