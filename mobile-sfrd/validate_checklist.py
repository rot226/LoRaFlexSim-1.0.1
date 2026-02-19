"""Validation automatique de la checklist qualité mobile-sfrd.

Ce script lit les CSV générés dans ``mobile-sfrd/outputs/csv`` et vérifie:
- noms de colonnes exacts;
- monotonie PDR/DER vs vitesse + contrainte RWP < SM;
- ordre de la courbe d'apprentissage (v=1 > v=5 > v=10) en convergence et plateau;
- somme des barres Figure 3 = total de nœuds (par fenêtre/panneau);
- Figure 5: rupture nette à t=150 puis récupération partielle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "outputs" / "csv"


class ValidationError(RuntimeError):
    """Erreur levée lorsqu'une règle de validation échoue."""


def _require_columns(df: pd.DataFrame, expected: list[str], name: str) -> None:
    got = list(df.columns)
    if got != expected:
        raise ValidationError(f"{name}: colonnes inattendues. attendu={expected}, obtenu={got}")


def _assert_monotone_non_increasing(values: np.ndarray, label: str) -> None:
    if not np.all(np.diff(values) <= 1e-12):
        raise ValidationError(f"{label}: la série n'est pas monotone décroissante.")


def _convergence_episode(values: np.ndarray) -> int:
    plateau = float(np.mean(values[-30:]))
    threshold = 0.95 * plateau
    indices = np.where(values >= threshold)[0]
    if len(indices) == 0:
        raise ValidationError("courbe sans convergence détectable")
    return int(indices[0] + 1)


def validate() -> list[str]:
    messages: list[str] = []

    fig1 = pd.read_csv(CSV_DIR / "fig1.csv")
    fig2 = pd.read_csv(CSV_DIR / "fig2.csv")
    fig3 = pd.read_csv(CSV_DIR / "fig3.csv")
    fig4 = pd.read_csv(CSV_DIR / "fig4.csv")
    fig5 = pd.read_csv(CSV_DIR / "fig5.csv")

    _require_columns(fig1, ["speed", "pdr_sm", "pdr_rwp"], "fig1.csv")
    _require_columns(fig2, ["episode", "reward_v1", "reward_v5", "reward_v10"], "fig2.csv")
    _require_columns(fig3, ["mobility", "speed", "window", "sf", "nodes_count"], "fig3.csv")
    _require_columns(fig4, ["speed", "der_sm", "der_rwp"], "fig4.csv")
    _require_columns(fig5, ["t", "pdr", "changepoint_t"], "fig5.csv")
    messages.append("Colonnes CSV: OK")

    _assert_monotone_non_increasing(fig1["pdr_sm"].to_numpy(dtype=float), "fig1.pdr_sm")
    _assert_monotone_non_increasing(fig1["pdr_rwp"].to_numpy(dtype=float), "fig1.pdr_rwp")
    if not np.all(fig1["pdr_rwp"].to_numpy(dtype=float) < fig1["pdr_sm"].to_numpy(dtype=float)):
        raise ValidationError("fig1: contrainte RWP < SM violée")

    _assert_monotone_non_increasing(fig4["der_sm"].to_numpy(dtype=float), "fig4.der_sm")
    _assert_monotone_non_increasing(fig4["der_rwp"].to_numpy(dtype=float), "fig4.der_rwp")
    if not np.all(fig4["der_rwp"].to_numpy(dtype=float) < fig4["der_sm"].to_numpy(dtype=float)):
        raise ValidationError("fig4: contrainte RWP < SM violée")
    messages.append("Monotonicité PDR/DER + RWP<SM: OK")

    c1 = _convergence_episode(fig2["reward_v1"].to_numpy(dtype=float))
    c5 = _convergence_episode(fig2["reward_v5"].to_numpy(dtype=float))
    c10 = _convergence_episode(fig2["reward_v10"].to_numpy(dtype=float))
    p1 = float(fig2["reward_v1"].tail(30).mean())
    p5 = float(fig2["reward_v5"].tail(30).mean())
    p10 = float(fig2["reward_v10"].tail(30).mean())

    if not (c1 < c5 < c10):
        raise ValidationError(
            f"fig2: ordre de convergence invalide (v1={c1}, v5={c5}, v10={c10})"
        )
    if not (p1 > p5 > p10):
        raise ValidationError(
            f"fig2: ordre des plateaux invalide (v1={p1:.4f}, v5={p5:.4f}, v10={p10:.4f})"
        )
    messages.append("Courbe apprentissage (ordre convergence+plateau): OK")

    for (mobility, speed, window), grp in fig3.groupby(["mobility", "speed", "window"]):
        total = int(grp["nodes_count"].sum())
        if total != 200:
            raise ValidationError(
                f"fig3: somme des barres != 200 pour ({mobility}, v={speed}, {window}) -> {total}"
            )
    messages.append("Figure 3 (sommes des barres=200): OK")

    cp = int(fig5["changepoint_t"].iloc[0])
    if cp != 150:
        raise ValidationError(f"fig5: changepoint_t attendu=150, obtenu={cp}")

    pre = float(fig5.loc[fig5["t"] < cp, "pdr"].tail(20).mean())
    post = float(fig5.loc[(fig5["t"] > cp) & (fig5["t"] <= cp + 10), "pdr"].mean())
    end = float(fig5.tail(40)["pdr"].mean())
    if (pre - post) < 0.15:
        raise ValidationError(f"fig5: rupture insuffisante (drop={pre - post:.4f})")
    if not (post < end < pre):
        raise ValidationError(
            f"fig5: récupération partielle invalide (pre={pre:.4f}, post={post:.4f}, end={end:.4f})"
        )
    messages.append("Figure 5 (rupture franche puis récupération partielle): OK")

    return messages


def main() -> int:
    messages = validate()
    print("Checklist validée:")
    for msg in messages:
        print(f"- {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
