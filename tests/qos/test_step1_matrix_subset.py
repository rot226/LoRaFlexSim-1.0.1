from __future__ import annotations

import csv
import json
import sys
from collections.abc import Iterable
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_step1_matrix  # noqa: E402  pylint: disable=wrong-import-position


def _select_histogram_json(row: dict[str, str]) -> str:
    use_snir = row.get("use_snir") == "True"
    if use_snir:
        return row["snir_histogram_json"]
    return row["snir_histogram_json"]


def _mean_snir(row: dict[str, str]) -> float:
    histogram = json.loads(_select_histogram_json(row))
    total = sum(histogram.values()) or 1
    return sum(float(bin_key) * count for bin_key, count in histogram.items()) / total


def _snir_quantiles(row: dict[str, str], quantiles: Iterable[float]) -> dict[float, float]:
    histogram = json.loads(_select_histogram_json(row))
    total = sum(histogram.values()) or 1
    cumulative = 0
    targets = sorted(set(quantiles))
    quantile_values: dict[float, float] = {}

    for bin_key, count in sorted(histogram.items(), key=lambda item: float(item[0])):
        cumulative += count
        ratio = cumulative / total
        while targets and ratio >= targets[0]:
            quantile_values[targets.pop(0)] = float(bin_key)
        if not targets:
            break

    # Au cas où des quantiles manqueraient (histogramme vide), on duplique le dernier bin connu.
    if histogram and targets:
        last_bin = float(sorted(histogram, key=lambda key: float(key))[-1])
        for target in targets:
            quantile_values[target] = last_bin

    return quantile_values


def _moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []

    if window <= 0:
        raise ValueError("La taille de fenêtre doit être positive")

    if len(values) < window:
        return [sum(values) / len(values)]

    return [sum(values[idx : idx + window]) / window for idx in range(len(values) - window + 1)]


def _collect_metrics(results_dir: Path) -> dict[bool, list[dict[str, float]]]:
    grouped: dict[bool, list[dict[str, float]]] = {True: [], False: []}

    for csv_path in sorted(results_dir.glob("**/*.csv")):
        with csv_path.open(newline="", encoding="utf8") as handle:
            row = next(csv.DictReader(handle))

        use_snir = row["use_snir"] == "True"
        if use_snir:
            assert row.get("snir_mean") not in (None, ""), (
                f"snir_mean manquant pour un scénario SNIR activé ({csv_path})"
            )
        grouped[use_snir].append(
            {
                "pdr": float(row["PDR"]),
                "der": float(row["DER"]),
                "collisions": float(row["collisions"]),
                "mean_snir": _mean_snir(row),
                "snir_quantiles": _snir_quantiles(row, quantiles=(0.1, 0.5, 0.9)),
            }
        )

    return grouped


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_step1_subset_metrics_stay_within_bounds(tmp_path: Path) -> None:
    campaigns = {
        "subset": [
            "--algos",
            "adr",
            "--with-snir",
            "true",
            "false",
            "--seeds",
            "1",
            "--nodes",
            "40",
            "80",
            "--packet-intervals",
            "1.0",
            "--duration",
            "45",
        ],
        "subset_dense": [
            "--algos",
            "adr",
            "--with-snir",
            "true",
            "false",
            "--seeds",
            "1",
            "--nodes",
            "60",
            "120",
            "--packet-intervals",
            "0.6",
            "--duration",
            "60",
        ],
    }

    metrics_by_campaign: dict[str, dict[bool, list[dict[str, float]]]] = {}
    for name, args in campaigns.items():
        results_dir = tmp_path / name
        run_step1_matrix.main([
            *args,
            "--results-dir",
            str(results_dir),
        ])
        metrics_by_campaign[name] = _collect_metrics(results_dir)

    for metrics in metrics_by_campaign.values():
        assert metrics[True] and metrics[False], "Les états SNIR doivent être couverts"

    snir_spread_threshold = 8.0
    der_threshold = 0.25
    pdr_threshold = 0.25
    snir_gap_threshold = 6.0
    # Seuils non nuls pour éviter les écarts arbitraires : avec 1 graine, des durées
    # de 45–60 s et des charges modérées (32/64 ou 48/96 noeuds), on s'attend à des
    # écarts SNIR de quelques dB, des variations PDR/DER d'au moins 5 points, et une
    # différence de collisions d'au moins ~5 (variance suffisante sans allonger le test).
    min_snir_gap_between_states = 5.0
    min_rate_gap_between_states = 0.05
    min_collision_gap_between_states = 5.0
    min_collision_mean_gap_between_states = 1.0
    min_collision_median_gap_between_states = 1.0

    for campaign_name, metrics in metrics_by_campaign.items():
        state_means: dict[bool, dict[str, float]] = {}
        for use_snir, rows in metrics.items():
            pdr_values = [item["pdr"] for item in rows]
            der_values = [item["der"] for item in rows]
            collisions_values = sorted(item["collisions"] for item in rows)
            mean_snir_values = [item["mean_snir"] for item in rows]
            snir_quantiles = [item["snir_quantiles"] for item in rows]

            pdr_mean = sum(pdr_values) / len(pdr_values)
            collisions_mean = sum(collisions_values) / len(collisions_values)
            snir_mean = sum(mean_snir_values) / len(mean_snir_values)
            collisions_median = collisions_values[len(collisions_values) // 2]

            assert 0.75 <= pdr_mean <= 1.01, (
                f"PDR moyen inattendu pour SNIR={use_snir} ({campaign_name}): {pdr_mean:.3f}"
            )
            assert 0.0 <= collisions_mean <= 10.0, (
                f"Taux de collisions moyen hors bornes pour SNIR={use_snir} ({campaign_name}): {collisions_mean:.3f}"
            )
            assert -10.0 <= snir_mean <= 40.0, (
                f"SNIR moyen irréaliste pour SNIR={use_snir} ({campaign_name}): {snir_mean:.2f} dB"
            )
            assert 0.0 <= collisions_median <= 12.0, (
                f"Médiane des collisions incohérente pour SNIR={use_snir} ({campaign_name}): {collisions_median:.3f}"
            )

            q10_avg = sum(item[0.1] for item in snir_quantiles) / len(snir_quantiles)
            q90_avg = sum(item[0.9] for item in snir_quantiles) / len(snir_quantiles)
            assert -30.0 <= q10_avg <= 25.0, (
                f"Quantile 10% SNIR suspect pour SNIR={use_snir} ({campaign_name}): {q10_avg:.2f} dB"
            )
            assert -5.0 <= q90_avg <= 55.0, (
                f"Quantile 90% SNIR suspect pour SNIR={use_snir} ({campaign_name}): {q90_avg:.2f} dB"
            )
            assert (q90_avg - q10_avg) >= snir_spread_threshold, (
                f"Dispersion SNIR trop faible pour SNIR={use_snir} ({campaign_name}): {(q90_avg - q10_avg):.2f} dB"
            )

            for metric_values, threshold, label in [
                (pdr_values, pdr_threshold, "PDR"),
                (der_values, der_threshold, "DER"),
                (mean_snir_values, snir_gap_threshold, "SNIR"),
            ]:
                ma_5 = _moving_average(metric_values, window=5)[-1]
                ma_20 = _moving_average(metric_values, window=20)[-1]
                assert abs(ma_5 - ma_20) <= threshold, (
                    f"Moyenne glissante incohérente pour {label} SNIR={use_snir} ({campaign_name})"
                    f" (Δ={abs(ma_5 - ma_20):.3f})"
                )

            state_means[use_snir] = {
                "pdr": pdr_mean,
                "der": sum(der_values) / len(der_values),
                "mean_snir": snir_mean,
                "collisions_mean": collisions_mean,
                "collisions_median": collisions_median,
            }

        mean_gap = abs(state_means[True]["mean_snir"] - state_means[False]["mean_snir"])
        assert mean_gap >= min_snir_gap_between_states, (
            f"Écart moyen de SNIR insuffisant entre états ({campaign_name}): {mean_gap:.2f} dB"
        )

        for metric in ("pdr", "der"):
            gap = abs(state_means[True][metric] - state_means[False][metric])
            assert gap >= min_rate_gap_between_states, (
                f"Écart moyen de {metric.upper()} trop faible entre états ({campaign_name}): {gap:.3f}"
            )

        collisions_mean_gap = abs(state_means[True]["collisions_mean"] - state_means[False]["collisions_mean"])
        collisions_median_gap = abs(
            state_means[True]["collisions_median"] - state_means[False]["collisions_median"]
        )
        assert collisions_mean_gap >= min_collision_mean_gap_between_states, (
            f"Écart moyen de collisions trop faible entre états ({campaign_name}): {collisions_mean_gap:.3f}"
        )
        assert collisions_median_gap >= min_collision_median_gap_between_states, (
            f"Écart médian de collisions trop faible entre états ({campaign_name}): {collisions_median_gap:.3f}"
        )
        assert max(collisions_mean_gap, collisions_median_gap) >= min_collision_gap_between_states, (
            f"Dispersion des collisions trop faible entre états ({campaign_name}): "
            f"Δmoyenne={collisions_mean_gap:.3f}, Δmédiane={collisions_median_gap:.3f}"
        )

    reference_campaign, comparison_campaign = (metrics_by_campaign[name] for name in campaigns)
    for use_snir in (True, False):
        ref_rows = reference_campaign[use_snir]
        cmp_rows = comparison_campaign[use_snir]

        ref_pdr_ma = _moving_average([item["pdr"] for item in ref_rows], 20)[-1]
        cmp_pdr_ma = _moving_average([item["pdr"] for item in cmp_rows], 20)[-1]
        assert abs(ref_pdr_ma - cmp_pdr_ma) <= 0.3, (
            f"Écart PDR entre campagnes trop marqué pour SNIR={use_snir}: {abs(ref_pdr_ma - cmp_pdr_ma):.3f}"
        )

        ref_der_ma = _moving_average([item["der"] for item in ref_rows], 20)[-1]
        cmp_der_ma = _moving_average([item["der"] for item in cmp_rows], 20)[-1]
        assert abs(ref_der_ma - cmp_der_ma) <= 0.3, (
            f"Écart DER entre campagnes trop marqué pour SNIR={use_snir}: {abs(ref_der_ma - cmp_der_ma):.3f}"
        )

        ref_snir_ma = _moving_average([item["mean_snir"] for item in ref_rows], 20)[-1]
        cmp_snir_ma = _moving_average([item["mean_snir"] for item in cmp_rows], 20)[-1]
        assert abs(ref_snir_ma - cmp_snir_ma) <= 10.0, (
            f"Écart SNIR entre campagnes trop marqué pour SNIR={use_snir}: {abs(ref_snir_ma - cmp_snir_ma):.3f} dB"
        )
