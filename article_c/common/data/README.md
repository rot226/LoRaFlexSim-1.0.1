# Courbes auteurs

Ce dossier est réservé aux données fournies par les auteurs (courbes de référence)
à superposer aux résultats LoRaFlexSim.

## Format CSV attendu

Le script `article_c/reproduce_author_results.py` recherche par défaut un fichier
`author_curves.csv` dans ce dossier. Il doit contenir les colonnes suivantes :

- `figure` : numéro de figure (`4`, `5`, `7`, `8`).
- `profile` : profil QoS (`mixra`, `apra`, `aimi`).
- `cluster` : nom du cluster (`gold`, `silver`, `bronze`, `all`, etc.).
- `x` : abscisse numérique (taille de réseau, niveau de charge, etc.).
- `y` : ordonnée numérique (DER, throughput, etc.).
- `label` : texte optionnel pour la légende.

## Exemple minimal

```
figure,profile,cluster,x,y,label
4,mixra,gold,50,0.12,Authors MixRA
```

Les lignes additionnelles sont optionnelles. Si le fichier est absent ou vide, le
script trace uniquement les résultats simulés.
