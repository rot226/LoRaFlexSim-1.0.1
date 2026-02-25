"""Génération des figures à partir de ``aggregates/*.csv`` uniquement."""

from __future__ import annotations

import argparse
import csv
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_FILES = {
    "metric_by_factor": "metric_by_factor.csv",
    "distribution_sf": "distribution_sf.csv",
    "convergence_tc": "convergence_tc.csv",
    "sinr_cdf": "sinr_cdf.csv",
    "fairness_airtime_switching": "fairness_airtime_switching.csv",
}

FIGURE_SPECS = [
    ("fig01_pdr_vs_n_snir_off.png", "metric_by_factor", "pdr_mean", {"mode": {"snir_off"}}),
    ("fig02_pdr_vs_n_snir_on.png", "metric_by_factor", "pdr_mean", {"mode": {"snir_on"}}),
    ("fig03_der_vs_n_snir_off.png", "metric_by_factor", "der_mean", {"mode": {"snir_off"}}),
    ("fig04_der_vs_n_snir_on.png", "metric_by_factor", "der_mean", {"mode": {"snir_on"}}),
    ("fig05_throughput_vs_n_snir_off.png", "metric_by_factor", "throughput_bps_mean", {"mode": {"snir_off"}}),
    ("fig06_throughput_vs_n_snir_on.png", "metric_by_factor", "throughput_bps_mean", {"mode": {"snir_on"}}),
]

BONUS_SPECS = [
    ("fig11_airtime_vs_n.png", "fairness_airtime_switching", "airtime_total_s", {}),
    ("fig12_switch_count_vs_n.png", "fairness_airtime_switching", "switch_count", {}),
]


@dataclass(frozen=True)
class ScenarioFilters:
    by_column: dict[str, set[str]]

    @classmethod
    def from_tokens(cls, tokens: list[str] | None) -> "ScenarioFilters":
        mapping: dict[str, set[str]] = defaultdict(set)
        for token in tokens or []:
            if "=" not in token:
                warnings.warn(f"Filtre ignoré (format attendu clé=valeur1,valeur2): {token}", stacklevel=2)
                continue
            key, values = token.split("=", 1)
            key = key.strip()
            if not key:
                warnings.warn(f"Filtre ignoré (clé vide): {token}", stacklevel=2)
                continue
            parsed = [item.strip() for item in values.split(",") if item.strip()]
            if not parsed:
                warnings.warn(f"Filtre ignoré (aucune valeur): {token}", stacklevel=2)
                continue
            mapping[key].update(parsed)
        return cls(by_column=dict(mapping))

    def merge(self, extra: dict[str, set[str]]) -> "ScenarioFilters":
        merged = {key: set(values) for key, values in self.by_column.items()}
        for key, values in extra.items():
            if key in merged:
                merged[key] = merged[key].intersection(values)
            else:
                merged[key] = set(values)
        return ScenarioFilters(merged)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        warnings.warn(f"Fichier agrégé manquant: {path}.", stacklevel=2)
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    token = value.strip()
    if not token:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _apply_filters(rows: list[dict[str, str]], filters: ScenarioFilters) -> list[dict[str, str]]:
    if not filters.by_column:
        return rows
    filtered: list[dict[str, str]] = []
    for row in rows:
        keep = True
        for key, allowed in filters.by_column.items():
            if key not in row or row.get(key, "") not in allowed:
                keep = False
                break
        if keep:
            filtered.append(row)
    return filtered


def _warn_skip(fig_name: str, reason: str) -> None:
    warnings.warn(f"{fig_name} ignorée: {reason}", stacklevel=2)


def _plot_xy_by_algo(rows: list[dict[str, str]], *, fig_name: str, y_col: str, out_path: Path) -> bool:
    needed = {"N", "algo", y_col}
    if not rows:
        _warn_skip(fig_name, "aucune ligne disponible")
        return False
    missing = [column for column in needed if column not in rows[0]]
    if missing:
        _warn_skip(fig_name, f"colonnes manquantes {missing}")
        return False

    by_algo: dict[str, list[tuple[float, float]]] = defaultdict(list)
    dropped = 0
    for row in rows:
        x = _to_float(row.get("N"))
        y = _to_float(row.get(y_col))
        if x is None or y is None:
            dropped += 1
            continue
        by_algo[row.get("algo", "unknown")].append((x, y))

    if dropped:
        warnings.warn(f"{fig_name}: {dropped} lignes ignorées (valeurs non numériques).", stacklevel=2)
    if not by_algo:
        _warn_skip(fig_name, "aucune donnée traçable après nettoyage")
        return False

    plt.figure(figsize=(8, 5))
    for algo in sorted(by_algo):
        points = sorted(by_algo[algo], key=lambda item: item[0])
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        plt.plot(xs, ys, marker="o", label=algo)
    plt.grid(alpha=0.3)
    plt.xlabel("N")
    plt.ylabel(y_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def _plot_tc_vs_speed(rows: list[dict[str, str]], out_path: Path) -> bool:
    fig_name = out_path.name
    if not rows:
        _warn_skip(fig_name, "fichier convergence_tc.csv vide ou absent")
        return False
    needed = {"speed", "algo", "Tc_s"}
    missing = [column for column in needed if column not in rows[0]]
    if missing:
        _warn_skip(fig_name, f"colonnes manquantes {missing}")
        return False

    grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        speed = _to_float(row.get("speed"))
        tc_s = _to_float(row.get("Tc_s"))
        if speed is None or tc_s is None:
            continue
        grouped[row.get("algo", "unknown")][speed].append(tc_s)

    if not grouped:
        _warn_skip(fig_name, "pas de couples speed/Tc_s exploitables")
        return False

    plt.figure(figsize=(8, 5))
    for algo in sorted(grouped):
        speeds = sorted(grouped[algo])
        means = [sum(grouped[algo][speed]) / len(grouped[algo][speed]) for speed in speeds]
        plt.plot(speeds, means, marker="o", label=algo)
    plt.grid(alpha=0.3)
    plt.xlabel("speed")
    plt.ylabel("Tc_s moyen")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def _plot_sinr_cdf(rows: list[dict[str, str]], out_path: Path) -> bool:
    fig_name = out_path.name
    if not rows:
        _warn_skip(fig_name, "fichier sinr_cdf.csv vide ou absent")
        return False
    needed = {"quantile", "sinr_db", "algo"}
    missing = [column for column in needed if column not in rows[0]]
    if missing:
        _warn_skip(fig_name, f"colonnes manquantes {missing}")
        return False

    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        q = _to_float(row.get("quantile"))
        sinr = _to_float(row.get("sinr_db"))
        if q is None or sinr is None:
            continue
        grouped[row.get("algo", "unknown")].append((q, sinr))

    if not grouped:
        _warn_skip(fig_name, "pas de points quantile/sinr exploitables")
        return False

    plt.figure(figsize=(8, 5))
    for algo in sorted(grouped):
        points = sorted(grouped[algo], key=lambda item: item[0])
        xs = [point[1] for point in points]
        ys = [point[0] for point in points]
        plt.plot(xs, ys, label=algo)
    plt.grid(alpha=0.3)
    plt.xlabel("SINR (dB)")
    plt.ylabel("Probabilité cumulée")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def _plot_sf_distribution(rows: list[dict[str, str]], out_path: Path) -> bool:
    fig_name = out_path.name
    if not rows:
        _warn_skip(fig_name, "fichier distribution_sf.csv vide ou absent")
        return False
    needed = {"sf", "ratio", "algo"}
    missing = [column for column in needed if column not in rows[0]]
    if missing:
        _warn_skip(fig_name, f"colonnes manquantes {missing}")
        return False

    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        sf = _to_float(row.get("sf"))
        ratio = _to_float(row.get("ratio"))
        if sf is None or ratio is None:
            continue
        grouped[row.get("algo", "unknown")][int(sf)].append(ratio)

    if not grouped:
        _warn_skip(fig_name, "pas de points SF/ratio exploitables")
        return False

    plt.figure(figsize=(8, 5))
    for algo in sorted(grouped):
        sfs = sorted(grouped[algo])
        means = [sum(grouped[algo][sf]) / len(grouped[algo][sf]) for sf in sfs]
        plt.plot(sfs, means, marker="o", label=algo)
    plt.grid(alpha=0.3)
    plt.xlabel("Spreading Factor")
    plt.ylabel("Ratio moyen")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def generate_minimal_figures(*, aggregates_dir: Path, out_dir: Path, filters: ScenarioFilters, include_bonus: bool = True) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads = {name: _read_csv_rows(aggregates_dir / filename) for name, filename in REQUIRED_FILES.items()}

    generated: list[Path] = []

    for fig_name, source, metric, local_filter in FIGURE_SPECS:
        selected = _apply_filters(payloads[source], filters.merge(local_filter))
        out_path = out_dir / fig_name
        if _plot_xy_by_algo(selected, fig_name=fig_name, y_col=metric, out_path=out_path):
            generated.append(out_path)

    fig07 = out_dir / "fig07_tc_vs_speed.png"
    if _plot_tc_vs_speed(_apply_filters(payloads["convergence_tc"], filters), fig07):
        generated.append(fig07)

    fig08 = out_dir / "fig08_fairness_vs_n.png"
    if _plot_xy_by_algo(_apply_filters(payloads["fairness_airtime_switching"], filters), fig_name=fig08.name, y_col="jain_fairness", out_path=fig08):
        generated.append(fig08)

    fig09 = out_dir / "fig09_sf_distribution.png"
    if _plot_sf_distribution(_apply_filters(payloads["distribution_sf"], filters), fig09):
        generated.append(fig09)

    fig10 = out_dir / "fig10_sinr_cdf.png"
    if _plot_sinr_cdf(_apply_filters(payloads["sinr_cdf"], filters), fig10):
        generated.append(fig10)

    if include_bonus:
        for fig_name, source, metric, local_filter in BONUS_SPECS:
            selected = _apply_filters(payloads[source], filters.merge(local_filter))
            out_path = out_dir / fig_name
            if _plot_xy_by_algo(selected, fig_name=fig_name, y_col=metric, out_path=out_path):
                generated.append(out_path)

    return generated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Génère les figures fig01..fig10 (et bonus fig11..fig12) depuis aggregates/*.csv")
    parser.add_argument("--aggregates-dir", required=True, type=Path, help="Répertoire contenant les CSV agrégés.")
    parser.add_argument("--out", required=True, type=Path, help="Répertoire cible pour les PNG.")
    parser.add_argument(
        "--scenario-filter",
        action="append",
        default=[],
        help="Filtre clé=val1,val2 (répétable), ex: --scenario-filter mode=snir_on --scenario-filter algo=ucb,legacy",
    )
    parser.add_argument("--no-bonus", action="store_true", help="N'écrit pas les figures bonus fig11/fig12.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generated = generate_minimal_figures(
        aggregates_dir=args.aggregates_dir,
        out_dir=args.out,
        filters=ScenarioFilters.from_tokens(args.scenario_filter),
        include_bonus=not args.no_bonus,
    )
    print(f"{len(generated)} figure(s) générée(s).")
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
