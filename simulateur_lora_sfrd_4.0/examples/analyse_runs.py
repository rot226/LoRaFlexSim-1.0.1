#!/usr/bin/env python3
"""Analyse les résultats de plusieurs runs enregistrés dans un seul CSV."""
import sys

try:
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - optional deps
    missing = "pandas" if "pandas" in str(exc) else "matplotlib"
    print(
        f"Le module '{missing}' est requis pour exécuter ce script. Installez les dépendances via 'pip install -r VERSION_4/requirements.txt'."
    )
    raise SystemExit(1)

if len(sys.argv) != 2:
    print("Usage: python analyse_runs.py resultats.csv")
    sys.exit(1)

csv_path = sys.argv[1]

df = pd.read_csv(csv_path)

# Conversion des colonnes numériques courantes
for col in ["PDR(%)", "collisions", "avg_delay", "avg_delay_s", "throughput_bps", "energy", "energy_J"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

energy_col = "energy_J" if "energy_J" in df.columns else "energy" if "energy" in df.columns else None

# Agrégation par numéro de run
if "run" not in df.columns:
    print("La colonne 'run' est absente du CSV.")
    sys.exit(1)

metrics = ["PDR(%)", "collisions", "avg_delay", "avg_delay_s", "throughput_bps"]
if energy_col:
    metrics.append(energy_col)

avg = df.groupby("run")[metrics].mean()
print(avg)

if "PDR(%)" in avg.columns:
    plt.figure()
    avg["PDR(%)"].plot(kind="bar")
    plt.ylabel("PDR (%)")
    plt.title("PDR moyen par run")
    plt.tight_layout()
    plt.savefig("pdr_par_run.png")
    print("Graphique enregistré dans pdr_par_run.png")
