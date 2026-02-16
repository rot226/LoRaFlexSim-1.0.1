"""Point d'entrée module: agrégation des résultats."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .reward_ucb import aggregate_learning_curves


LOGGER = logging.getLogger(__name__)


_ALGORITHM_ALIASES = {
    "adr": "adrpure",
    "adrpure": "adrpure",
    "apra": "apralike",
    "apralike": "apralike",
    "aimi": "aimilike",
    "aimilike": "aimilike",
    "mixraopt": "mixraopt",
    "mixrah": "mixrah",
    "ucb": "ucb1",
    "ucb1": "ucb1",
}


def _to_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(float(value.strip()))
    raise ValueError(f"Valeur entière invalide: {value!r}")


def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value.strip())
    raise ValueError(f"Valeur numérique invalide: {value!r}")


def _normalize_snir(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"on", "snir_on", "true", "1"}:
        return "ON"
    if normalized in {"off", "snir_off", "false", "0"}:
        return "OFF"
    return str(value or "").strip().upper()


def _normalize_algorithm_key(value: Any) -> str:
    raw = str(value or "").strip().lower()
    compact = "".join(ch for ch in raw if ch.isalnum())
    return _ALGORITHM_ALIASES.get(compact, compact)


def _run_match_key(run: tuple[str, int, str, int]) -> tuple[str, int, str, int]:
    snir, network_size, algorithm, seed = run
    return (_normalize_snir(snir), int(network_size), _normalize_algorithm_key(algorithm), int(seed))


def _extract_run_from_path(summary_path: Path) -> tuple[str, int, str, int] | None:
    try:
        seed_part = summary_path.parent.name
        algo_part = summary_path.parent.parent.name
        ns_part = summary_path.parent.parent.parent.name
        snir_part = summary_path.parent.parent.parent.parent.name
        if not seed_part.startswith("seed_"):
            return None
        if not algo_part.startswith("algo_"):
            return None
        if not ns_part.startswith("ns_"):
            return None
        if not snir_part.startswith("SNIR_"):
            return None
        snir = _normalize_snir(snir_part.replace("SNIR_", "", 1))
        network_size = _to_int(ns_part.replace("ns_", "", 1))
        algorithm = algo_part.replace("algo_", "", 1).strip()
        seed = _to_int(seed_part.replace("seed_", "", 1))
    except (ValueError, TypeError):
        return None

    if snir not in {"ON", "OFF"} or network_size <= 0 or not algorithm or seed < 0:
        return None
    return snir, network_size, algorithm, seed


def _compute_missing_expected_runs(
    expected_runs: set[tuple[str, int, str, int]],
    available_runs: set[tuple[str, int, str, int]],
) -> list[tuple[str, int, str, int]]:
    available_keys = {_run_match_key(run) for run in available_runs}
    return sorted(run for run in expected_runs if _run_match_key(run) not in available_keys)


def _resolve_expected_run_dir(
    logs_root: Path,
    run: tuple[str, int, str, int],
) -> tuple[Path, bool]:
    snir, network_size, algorithm, seed = run
    primary = logs_root / f"SNIR_{snir}" / f"ns_{network_size}" / f"algo_{algorithm}" / f"seed_{seed}"
    if primary.exists():
        return primary, True

    base = logs_root / f"SNIR_{snir}" / f"ns_{network_size}"
    if base.exists():
        for candidate in base.glob("algo_*/seed_*"):
            if not candidate.is_dir():
                continue
            extracted = _extract_run_from_path(candidate / "campaign_summary.json")
            if extracted is None:
                try:
                    extracted = (
                        _normalize_snir(snir),
                        _to_int(network_size),
                        candidate.parent.name.replace("algo_", "", 1),
                        _to_int(candidate.name.replace("seed_", "", 1)),
                    )
                except (ValueError, TypeError):
                    continue
            if _run_match_key(extracted) == _run_match_key(run):
                return candidate, False

    return primary, False


def _log_missing_runs_debug(
    logs_root: Path,
    missing_runs: list[tuple[str, int, str, int]],
) -> None:
    for run in missing_runs:
        run_dir, is_exact = _resolve_expected_run_dir(logs_root, run)
        summary = run_dir / "campaign_summary.json"
        raw_packets = run_dir / "raw_packets.csv"
        raw_energy = run_dir / "raw_energy.csv"
        LOGGER.debug(
            "Missing run debug | combo=%s | expected_path=%s | exact_match=%s | dir_exists=%s | campaign_summary=%s | raw_packets=%s | raw_energy=%s",
            run,
            run_dir,
            "yes" if is_exact else "no",
            "yes" if run_dir.exists() else "no",
            "yes" if summary.is_file() else "no",
            "yes" if raw_packets.is_file() else "no",
            "yes" if raw_energy.is_file() else "no",
        )


def _extract_metric(metrics: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in metrics and metrics[key] not in (None, ""):
            return _to_float(metrics[key])
    return 0.0


def _extract_sf_counts(metrics: dict[str, Any], network_size: int) -> dict[int, float]:
    raw = (
        metrics.get("sf_distribution_counts")
        or metrics.get("sf_distribution")
        or metrics.get("sf_counts")
        or {}
    )
    if not isinstance(raw, dict):
        return {}

    def _parse_sf(value: Any) -> int | None:
        try:
            numeric = _to_float(value)
        except (ValueError, TypeError):
            return None
        sf = int(numeric)
        if sf not in {7, 8, 9, 10, 11, 12}:
            return None
        if abs(numeric - float(sf)) > 1e-9:
            return None
        return sf

    parsed: dict[int, float] = {}
    for sf_key, raw_value in raw.items():
        try:
            value = _to_float(raw_value)
        except (ValueError, TypeError):
            continue
        sf = _parse_sf(sf_key)
        if sf is None:
            continue
        parsed[sf] = value

    total = sum(parsed.values())
    if total <= 0:
        return parsed

    if total <= 1.000001 and network_size > 0:
        return {sf: value * network_size for sf, value in parsed.items()}

    return parsed


def _extract_rewards(summary: dict[str, Any]) -> Iterable[tuple[int, float]]:
    metrics = summary.get("metrics", {})
    candidates = [
        summary.get("learning_curve"),
        summary.get("rewards"),
        metrics.get("learning_curve"),
        metrics.get("ucb_learning_curve"),
        metrics.get("rewards"),
    ]

    for candidate in candidates:
        if isinstance(candidate, list):
            for idx, item in enumerate(candidate, start=1):
                if isinstance(item, dict):
                    ep = item.get("episode", idx)
                    reward = item.get("reward", 0.0)
                else:
                    ep = idx
                    reward = item
                try:
                    yield _to_int(ep), _to_float(reward)
                except (ValueError, TypeError):
                    continue
            return

        if isinstance(candidate, dict):
            for ep, reward in candidate.items():
                try:
                    yield _to_int(ep), _to_float(reward)
                except (ValueError, TypeError):
                    continue
            return


def _write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _load_expected_runs(
    logs_root: Path,
    *,
    manifest: dict[str, Any] | None = None,
) -> set[tuple[str, int, str, int]]:
    expected: set[tuple[str, int, str, int]] = set()

    manifest_runs = manifest.get("expected_runs") if isinstance(manifest, dict) else None
    if isinstance(manifest_runs, list):
        for row in manifest_runs:
            if not isinstance(row, dict):
                continue
            try:
                snir = _normalize_snir(row.get("snir", ""))
                network_size = _to_int(row.get("network_size", 0))
                algorithm = str(row.get("algorithm", "")).strip()
                seed = _to_int(row.get("seed", -1))
            except (ValueError, TypeError):
                continue
            if snir in {"ON", "OFF"} and network_size > 0 and algorithm and seed >= 0:
                expected.add((snir, network_size, algorithm, seed))
        return expected

    state_path = logs_root / "campaign_state.json"
    if state_path.exists():
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValueError):
            payload = None

        runs = payload.get("runs") if isinstance(payload, dict) else None
        if isinstance(runs, dict):
            for run_payload in runs.values():
                if not isinstance(run_payload, dict):
                    continue
                try:
                    snir = _normalize_snir(run_payload.get("snir", ""))
                    network_size = _to_int(run_payload.get("network_size", 0))
                    algorithm = str(run_payload.get("algo", "")).strip()
                    seed = _to_int(run_payload.get("seed", -1))
                except (ValueError, TypeError):
                    continue
                if snir not in {"ON", "OFF"} or network_size <= 0 or not algorithm or seed < 0:
                    continue
                expected.add((snir, network_size, algorithm, seed))

    missing_report_path = logs_root / "campaign_missing_combinations.csv"
    if missing_report_path.exists():
        try:
            with missing_report_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    try:
                        snir = _normalize_snir(row.get("snir", ""))
                        network_size = _to_int(row.get("network_size", 0))
                        algorithm = str(row.get("algorithm", "")).strip()
                        seed = _to_int(row.get("seed", -1))
                    except (ValueError, TypeError):
                        continue
                    if snir in {"ON", "OFF"} and network_size > 0 and algorithm and seed >= 0:
                        expected.add((snir, network_size, algorithm, seed))
        except OSError:
            pass

    return expected


def _compute_completeness_rows(
    expected_runs: set[tuple[str, int, str, int]],
    available_runs: set[tuple[str, int, str, int]],
) -> list[dict[str, Any]]:
    expected_by_combo: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    available_by_combo: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    display_by_combo: dict[tuple[str, int, str], tuple[str, int, str]] = {}

    for snir, network_size, algorithm, seed in expected_runs:
        combo_key = (_normalize_snir(snir), network_size, _normalize_algorithm_key(algorithm))
        expected_by_combo[combo_key].add(seed)
        display_by_combo.setdefault(combo_key, (_normalize_snir(snir), network_size, algorithm))
    for snir, network_size, algorithm, seed in available_runs:
        combo_key = (_normalize_snir(snir), network_size, _normalize_algorithm_key(algorithm))
        available_by_combo[combo_key].add(seed)
        display_by_combo.setdefault(combo_key, (_normalize_snir(snir), network_size, algorithm))

    rows: list[dict[str, Any]] = []
    for combo_key in sorted(expected_by_combo):
        expected_seeds = expected_by_combo[combo_key]
        available_seeds = available_by_combo.get(combo_key, set())
        missing = sorted(expected_seeds - available_seeds)
        snir, network_size, algorithm = display_by_combo[combo_key]
        rows.append(
            {
                "snir": snir,
                "network_size": network_size,
                "algorithm": algorithm,
                "expected_runs": len(expected_seeds),
                "available_runs": len(available_seeds),
                "is_complete": "yes" if not missing else "no",
                "missing_seeds": ",".join(str(seed) for seed in missing),
            }
        )
    return rows


def _compute_missing_combinations_rows(
    expected_runs: set[tuple[str, int, str, int]],
    available_runs: set[tuple[str, int, str, int]],
    run_statuses: dict[tuple[str, int, str, int], str] | None = None,
) -> list[dict[str, Any]]:
    available_keys = {_run_match_key(run) for run in available_runs}
    statuses_by_key = {
        _run_match_key(run): status
        for run, status in (run_statuses or {}).items()
    }

    missing_by_combo: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    display_by_combo: dict[tuple[str, int, str], tuple[str, int, str]] = {}
    for snir, network_size, algorithm, seed in expected_runs:
        match_key = _run_match_key((snir, network_size, algorithm, seed))
        if match_key in available_keys:
            continue
        combo_key = match_key[:3]
        missing_by_combo[combo_key].add(seed)
        display_by_combo.setdefault(combo_key, (_normalize_snir(snir), network_size, algorithm))

    rows: list[dict[str, Any]] = []
    for combo_key in sorted(missing_by_combo):
        missing_seeds = sorted(missing_by_combo[combo_key])
        snir, network_size, algorithm = display_by_combo[combo_key]
        rows.append(
            {
                "snir": snir,
                "network_size": network_size,
                "algorithm": algorithm,
                "missing_runs": len(missing_seeds),
                "missing_seeds": ",".join(str(seed) for seed in missing_seeds),
                "statuses": ",".join(
                    statuses_by_key.get((combo_key[0], combo_key[1], combo_key[2], seed), "missing")
                    for seed in missing_seeds
                ),
            }
        )
    return rows


def _load_run_statuses(logs_root: Path) -> dict[tuple[str, int, str, int], str]:
    statuses: dict[tuple[str, int, str, int], str] = {}
    state_path = logs_root / "campaign_state.json"
    if not state_path.exists():
        return statuses

    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        return statuses

    runs = payload.get("runs") if isinstance(payload, dict) else None
    if not isinstance(runs, dict):
        return statuses

    for run_payload in runs.values():
        if not isinstance(run_payload, dict):
            continue
        try:
            snir = _normalize_snir(run_payload.get("snir", ""))
            network_size = _to_int(run_payload.get("network_size", 0))
            algorithm = str(run_payload.get("algo", "")).strip()
            seed = _to_int(run_payload.get("seed", -1))
        except (ValueError, TypeError):
            continue
        if snir not in {"ON", "OFF"} or network_size <= 0 or not algorithm or seed < 0:
            continue
        statuses[(snir, network_size, algorithm, seed)] = (
            str(run_payload.get("status", "")).strip().lower() or "missing"
        )

    return statuses




def _load_campaign_manifest(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest de campagne invalide: {manifest_path}")
    return payload


def _resolve_campaign_scope(
    logs_root: str | Path,
    *,
    campaign_id: str | None = None,
    manifest_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any] | None]:
    root = Path(logs_root)

    explicit_manifest = Path(manifest_path) if manifest_path is not None else None
    if explicit_manifest is not None:
        manifest = _load_campaign_manifest(explicit_manifest)
        manifest_logs_root = manifest.get("logs_root")
        scoped_root = Path(manifest_logs_root) if isinstance(manifest_logs_root, str) and manifest_logs_root else explicit_manifest.parent
        return scoped_root, manifest

    if campaign_id:
        candidates = [
            root / "campaign_manifests" / f"{campaign_id}.json",
            root / campaign_id / "campaign_manifest.json",
        ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            manifest = _load_campaign_manifest(candidate)
            manifest_logs_root = manifest.get("logs_root")
            scoped_root = Path(manifest_logs_root) if isinstance(manifest_logs_root, str) and manifest_logs_root else candidate.parent
            return scoped_root, manifest
        raise FileNotFoundError(
            f"Impossible de trouver un manifest pour campaign_id={campaign_id!r} sous {root.resolve()}"
        )

    default_manifest = root / "campaign_manifest.json"
    if default_manifest.exists():
        manifest = _load_campaign_manifest(default_manifest)
        manifest_logs_root = manifest.get("logs_root")
        scoped_root = Path(manifest_logs_root) if isinstance(manifest_logs_root, str) and manifest_logs_root else root
        return scoped_root, manifest

    return root, None


def aggregate_logs(
    logs_root: str | Path,
    *,
    allow_partial: bool = False,
    campaign_id: str | None = None,
    manifest_path: str | Path | None = None,
    debug_missing: bool = False,
) -> Path:
    """Agrège les fichiers ``campaign_summary.json`` en sorties CSV dédiées."""

    root, manifest = _resolve_campaign_scope(
        logs_root,
        campaign_id=campaign_id,
        manifest_path=manifest_path,
    )
    summaries = sorted(root.glob("SNIR_*/ns_*/algo_*/seed_*/campaign_summary.json"))
    expected_runs = _load_expected_runs(root, manifest=manifest)
    run_statuses = _load_run_statuses(root)
    available_runs: set[tuple[str, int, str, int]] = set()

    metric_sums: dict[tuple[int, str, str], dict[str, float]] = defaultdict(
        lambda: {"pdr": 0.0, "throughput": 0.0, "energy": 0.0, "count": 0.0}
    )
    sf_sums: dict[tuple[int, str, str, int], dict[str, float]] = defaultdict(
        lambda: {"count_sum": 0.0, "replications": 0.0}
    )
    run_learning_curves: list[list[dict[str, float | int]]] = []

    for summary_path in summaries:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        contract = data.get("contract", {})
        metrics = data.get("metrics", {})

        run_from_path = _extract_run_from_path(summary_path)
        if run_from_path is not None:
            snir, network_size, algorithm, seed = run_from_path
        else:
            network_size = _to_int(contract.get("network_size", 0))
            algorithm = str(contract.get("algorithm", "")).strip()
            snir = _normalize_snir(contract.get("snir_mode", ""))
            if not algorithm or snir not in {"ON", "OFF"}:
                continue
            seed = _to_int(contract.get("seed", summary_path.parent.name.replace("seed_", "")))
        available_runs.add((snir, network_size, algorithm, seed))

        key = (network_size, algorithm, snir)
        metric_sums[key]["pdr"] += _extract_metric(metrics, "pdr", "PDR")
        metric_sums[key]["throughput"] += _extract_metric(
            metrics,
            "throughput_packets_per_s",
            "throughput_pps",
            "throughput_bps",
        )
        metric_sums[key]["energy"] += _extract_metric(
            metrics,
            "energy_joule_per_packet",
            "energy_per_packet",
            "energy",
        )
        metric_sums[key]["count"] += 1.0

        sf_counts = _extract_sf_counts(metrics, network_size)
        for sf in range(7, 13):
            sf_key = (network_size, algorithm, snir, sf)
            sf_sums[sf_key]["count_sum"] += sf_counts.get(sf, 0.0)
            sf_sums[sf_key]["replications"] += 1.0

        algorithm_key = algorithm.strip().upper().replace("-", "")
        if algorithm_key not in {"UCB", "UCB1"}:
            continue
        run_curve: list[dict[str, float | int]] = []
        for episode, reward in _extract_rewards(data):
            run_curve.append({"episode": max(1, episode), "reward": reward})
        if run_curve:
            run_learning_curves.append(run_curve)

    pdr_rows: list[dict[str, Any]] = []
    throughput_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []

    for network_size, algorithm, snir in sorted(metric_sums):
        values = metric_sums[(network_size, algorithm, snir)]
        replications = values["count"] or 1.0
        pdr_rows.append(
            {
                "network_size": network_size,
                "algorithm": algorithm,
                "snir": snir,
                "pdr": values["pdr"] / replications,
            }
        )
        throughput_rows.append(
            {
                "network_size": network_size,
                "algorithm": algorithm,
                "snir": snir,
                "throughput_packets_per_s": values["throughput"] / replications,
            }
        )
        energy_rows.append(
            {
                "network_size": network_size,
                "algorithm": algorithm,
                "snir": snir,
                "energy_joule_per_packet": values["energy"] / replications,
            }
        )

    sf_rows: list[dict[str, Any]] = []
    for network_size, algorithm, snir in sorted(metric_sums):
        for sf in range(7, 13):
            values = sf_sums.get((network_size, algorithm, snir, sf))
            count = 0.0
            if values is not None:
                replications = values["replications"] or 1.0
                count = values["count_sum"] / replications
            sf_rows.append(
                {
                    "network_size": network_size,
                    "algorithm": algorithm,
                    "snir": snir,
                    "sf": sf,
                    "count": count,
                }
            )

    learning_rows: list[dict[str, Any]] = aggregate_learning_curves(run_learning_curves)

    output_root = root / "output"
    off_root = output_root / "SNIR_OFF"
    on_root = output_root / "SNIR_ON"

    _write_csv(
        off_root / "pdr_results.csv",
        ["network_size", "algorithm", "snir", "pdr"],
        [row for row in pdr_rows if row["snir"] == "OFF"],
    )
    _write_csv(
        off_root / "throughput_results.csv",
        ["network_size", "algorithm", "snir", "throughput_packets_per_s"],
        [row for row in throughput_rows if row["snir"] == "OFF"],
    )
    _write_csv(
        off_root / "energy_results.csv",
        ["network_size", "algorithm", "snir", "energy_joule_per_packet"],
        [row for row in energy_rows if row["snir"] == "OFF"],
    )
    _write_csv(
        off_root / "sf_distribution.csv",
        ["network_size", "algorithm", "snir", "sf", "count"],
        [row for row in sf_rows if row["snir"] == "OFF"],
    )

    _write_csv(
        on_root / "pdr_results.csv",
        ["network_size", "algorithm", "snir", "pdr"],
        [row for row in pdr_rows if row["snir"] == "ON"],
    )
    _write_csv(
        on_root / "throughput_results.csv",
        ["network_size", "algorithm", "snir", "throughput_packets_per_s"],
        [row for row in throughput_rows if row["snir"] == "ON"],
    )
    _write_csv(
        on_root / "energy_results.csv",
        ["network_size", "algorithm", "snir", "energy_joule_per_packet"],
        [row for row in energy_rows if row["snir"] == "ON"],
    )
    _write_csv(
        on_root / "sf_distribution.csv",
        ["network_size", "algorithm", "snir", "sf", "count"],
        [row for row in sf_rows if row["snir"] == "ON"],
    )

    _write_csv(
        output_root / "learning_curve_ucb.csv",
        ["episode", "reward"],
        learning_rows,
    )

    completeness_rows = _compute_completeness_rows(expected_runs, available_runs)
    _write_csv(
        output_root / "campaign_completeness.csv",
        [
            "snir",
            "network_size",
            "algorithm",
            "expected_runs",
            "available_runs",
            "is_complete",
            "missing_seeds",
        ],
        completeness_rows,
    )

    missing_combinations_rows = _compute_missing_combinations_rows(
        expected_runs,
        available_runs,
        run_statuses,
    )
    missing_expected_runs = _compute_missing_expected_runs(expected_runs, available_runs)
    if debug_missing and missing_expected_runs:
        _log_missing_runs_debug(root, missing_expected_runs)
    _write_csv(
        output_root / "campaign_missing_combinations.csv",
        ["snir", "network_size", "algorithm", "missing_runs", "missing_seeds", "statuses"],
        missing_combinations_rows,
    )

    if expected_runs and not allow_partial:
        missing = _compute_missing_expected_runs(expected_runs, available_runs)
        if missing:
            preview = ", ".join(
                f"{snir}/ns_{size}/algo_{algo}/seed_{seed}"
                for snir, size, algo, seed in missing[:8]
            )
            suffix = " ..." if len(missing) > 8 else ""
            raise RuntimeError(
                "Agrégation incomplète: "
                f"{len(missing)} run(s) manquant(s). Exemples: {preview}{suffix}. "
                "Relancer avec --allow-partial pour agréger uniquement le disponible. "
                f"Détail: {(output_root / 'campaign_missing_combinations.csv').resolve()}"
            )

    return output_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agrège les résumés de campagne SFRD.")
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("sfrd/logs"),
        help="Racine des logs (contient SNIR_OFF/SNIR_ON)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Agrège uniquement les runs disponibles même si la campagne est incomplète.",
    )
    parser.add_argument(
        "--campaign-id",
        type=str,
        default=None,
        help="Identifiant de campagne à agréger (résolu via logs_root/campaign_manifests).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Chemin explicite vers un campaign_manifest.json.",
    )
    parser.add_argument(
        "--debug-missing",
        action="store_true",
        help="Journalise le diagnostic détaillé pour chaque combinaison manquante.",
    )
    return parser.parse_args()


def main() -> None:
    """Exécution principale."""

    args = _parse_args()
    path = aggregate_logs(
        args.logs_root,
        allow_partial=args.allow_partial,
        campaign_id=args.campaign_id,
        manifest_path=args.manifest,
        debug_missing=args.debug_missing,
    )
    print(f"Agrégation écrite: {path}")


if __name__ == "__main__":
    main()
