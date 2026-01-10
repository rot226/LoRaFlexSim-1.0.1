"""Exécute une grille Step 1 et agrège les métriques clés.

Le script lance une grille (n_nodes, charge, canaux) avec un nombre de
réplications configurable, puis écrit les CSV bruts dans
``results/step1/raw`` et un résumé agrégé (moyenne, écart-type, CI95)
 dans ``results/step1/agg``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loraflexsim.launcher import Channel, MultiChannel, Simulator
from loraflexsim.launcher.non_orth_delta import DEFAULT_NON_ORTH_DELTA
from loraflexsim.launcher.qos import QoSManager
from loraflexsim.launcher.simulator import InterferenceTracker
from loraflexsim.scenarios.qos_cluster_bench import (
    AREA_SIZE_M,
    FREQUENCIES_HZ,
    PAYLOAD_BYTES,
    SF_ORDER,
    _apply_adr_pure,
    _apply_apra_like,
    _apply_mixra_h,
    _apply_mixra_opt,
    _compute_additional_metrics,
    _configure_clusters,
    _flatten_metrics,
)

DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "step1"
DEFAULT_RAW_DIR = DEFAULT_RESULTS_DIR / "raw"
DEFAULT_AGG_DIR = DEFAULT_RESULTS_DIR / "agg"
DEFAULT_NODE_COUNTS: Sequence[int] = (1000, 5000, 10000)
DEFAULT_CHARGES: Sequence[float] = (600.0, 300.0, 150.0)
DEFAULT_CHANNELS: Sequence[int] = (1, 3, 8)
DEFAULT_SNIR_STATES: Sequence[bool] = (False, True)
DEFAULT_REPLICATIONS = 5
MAX_REPLICATIONS = 5
DEFAULT_DURATION = 6 * 3600.0
STATE_LABELS = {True: "snir_on", False: "snir_off"}

ALGORITHMS = {
    "adr": _apply_adr_pure,
    "apra": _apply_apra_like,
    "mixra_h": _apply_mixra_h,
    "mixra_opt": _apply_mixra_opt,
}


def _parse_snir_window(value: str) -> str | float:
    text = str(value).strip().lower()
    if text in {"packet", "preamble", "symbol"}:
        return text
    try:
        return float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "snir_window doit être 'packet', 'preamble', 'symbol' ou une durée en secondes."
        ) from exc


def _snir_window_label(value: str | float | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return f"{value:g}s"


def _parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "Valeur booléenne attendue (true/false, 1/0, yes/no, on/off)"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nodes",
        nargs="+",
        type=int,
        default=list(DEFAULT_NODE_COUNTS),
        help="Valeurs de n_nodes à simuler",
    )
    parser.add_argument(
        "--charges",
        nargs="+",
        type=float,
        default=list(DEFAULT_CHARGES),
        help="Charges (intervalle moyen d'émission en secondes)",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        type=int,
        default=list(DEFAULT_CHANNELS),
        help="Nombre de canaux LoRa à activer",
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        choices=sorted(ALGORITHMS),
        default=list(ALGORITHMS.keys()),
        help="Algorithmes QoS à exécuter",
    )
    parser.add_argument(
        "--snir-modes",
        nargs="+",
        type=_parse_bool,
        default=list(DEFAULT_SNIR_STATES),
        metavar="BOOL",
        help="Modes SNIR à exécuter (true/false)",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=DEFAULT_REPLICATIONS,
        help=f"Nombre de répétitions par scénario (max {MAX_REPLICATIONS})",
    )
    parser.add_argument("--seed", type=int, default=1, help="Graine de base")
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help="Durée max de simulation (secondes)",
    )
    parser.add_argument(
        "--mixra-solver",
        choices=["auto", "greedy"],
        default="auto",
        help="Solveur MixRA-Opt à utiliser",
    )
    parser.add_argument(
        "--snir-window",
        type=_parse_snir_window,
        default=None,
        help="Fenêtre SNIR (packet, preamble, symbol ou durée en secondes)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Répertoire de sortie pour les CSV bruts",
    )
    parser.add_argument(
        "--agg-dir",
        type=Path,
        default=DEFAULT_AGG_DIR,
        help="Répertoire de sortie pour les CSV agrégés",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="N'exécute pas les simulations si le CSV brut existe déjà",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Réduit les impressions de progression",
    )
    return parser


def _build_multichannel(
    channel_count: int,
    *,
    snir_window: str | float | None = None,
) -> MultiChannel:
    if channel_count < 1:
        raise ValueError("Le nombre de canaux doit être >= 1")
    if channel_count > len(FREQUENCIES_HZ):
        raise ValueError(
            f"Nombre de canaux demandé ({channel_count}) supérieur aux {len(FREQUENCIES_HZ)} fréquences disponibles."
        )
    channels = []
    for idx, freq in enumerate(FREQUENCIES_HZ[:channel_count]):
        channel = Channel(
            frequency_hz=freq,
            bandwidth=125_000.0,
            capture_threshold_dB=1.0,
            capture_window_symbols=6,
            channel_index=idx,
            advanced_capture=True,
            multipath_taps=4,
            fast_fading_std=1.0,
            snir_fading_std=1.5,
            variable_noise_std=0.5,
            snir_window=snir_window,
        )
        channel.orthogonal_sf = False
        channels.append(channel)
    multichannel = MultiChannel(channels)
    multichannel.force_non_orthogonal(DEFAULT_NON_ORTH_DELTA)
    return multichannel


def _sync_snir_state(simulator: Simulator, requested: bool) -> bool:
    channel = getattr(simulator, "channel", None)
    if channel is not None:
        setattr(channel, "use_snir", requested)
    multichannel = getattr(simulator, "multichannel", None)
    channels = list(getattr(multichannel, "channels", []) or [])
    for sub_channel in channels:
        setattr(sub_channel, "use_snir", requested)

    observed_states: list[bool] = []
    if channel is not None:
        observed_states.append(bool(getattr(channel, "use_snir", requested)))
    for sub_channel in channels:
        observed_states.append(bool(getattr(sub_channel, "use_snir", requested)))

    if not observed_states:
        return bool(getattr(simulator, "use_snir", requested))

    effective_state = observed_states[0]
    if any(state != effective_state for state in observed_states):
        raise ValueError("Les canaux SNIR ne sont pas synchronisés (états divergents détectés).")
    if effective_state != requested:
        raise ValueError(
            "L'état SNIR effectif ne correspond pas à l'état demandé après synchronisation."
        )
    return effective_state


def _instantiate_simulator(
    *,
    nodes: int,
    charge_s: float,
    seed: int,
    channels: int,
    use_snir: bool,
    snir_window: str | float | None = None,
) -> Simulator:
    simulator = Simulator(
        num_nodes=nodes,
        num_gateways=1,
        area_size=AREA_SIZE_M,
        transmission_mode="Random",
        packet_interval=charge_s,
        first_packet_interval=charge_s,
        packets_to_send=0,
        adr_node=False,
        adr_server=False,
        duty_cycle=0.01,
        mobility=False,
        channels=_build_multichannel(channels, snir_window=snir_window),
        channel_distribution="round-robin",
        payload_size_bytes=PAYLOAD_BYTES,
        seed=seed,
        capture_mode="advanced",
        phy_model="omnet",
        pure_poisson_mode=False,
    )
    simulator.use_snir = bool(use_snir)
    simulator.snir_window = snir_window
    setattr(simulator, "capture_delta_db", 1.0)
    simulator._interference_tracker = InterferenceTracker()
    _sync_snir_state(simulator, use_snir)
    return simulator


def _apply_algorithm(name: str, simulator: Simulator, manager: QoSManager, solver: str) -> None:
    handler = ALGORITHMS.get(name)
    if handler is None:
        raise ValueError(f"Algorithme inconnu : {name}")
    if name == "adr":
        handler(simulator)
        return
    if name == "mixra_opt":
        handler(simulator, manager, solver)
        simulator.qos_mixra_solver = solver
        return
    handler(simulator, manager)


def _sf_percentages(metrics: Mapping[str, Any], total_nodes: int) -> Dict[str, float]:
    distribution = metrics.get("sf_distribution", {}) or {}
    total = total_nodes if total_nodes else sum(distribution.values())
    if total == 0:
        return {f"sf_{sf}_pct": 0.0 for sf in SF_ORDER}
    return {
        f"sf_{sf}_pct": 100.0 * float(distribution.get(sf, 0)) / float(total)
        for sf in SF_ORDER
    }


def _build_cluster_columns(metrics: Mapping[str, Any]) -> Dict[str, float]:
    clusters = metrics.get("qos_cluster_pdr", {}) or {}
    cluster_values: Dict[str, float] = {}
    for key, value in clusters.items():
        try:
            cluster_id = int(key)
        except (TypeError, ValueError):
            continue
        cluster_values[f"cluster_pdr_{cluster_id}"] = float(value)
    return cluster_values


def _run_simulation(
    *,
    nodes: int,
    charge_s: float,
    channels: int,
    algo: str,
    use_snir: bool,
    seed: int,
    duration: float,
    mixra_solver: str,
    snir_window: str | float | None,
) -> Dict[str, Any]:
    simulator = _instantiate_simulator(
        nodes=nodes,
        charge_s=charge_s,
        seed=seed,
        channels=channels,
        use_snir=use_snir,
        snir_window=snir_window,
    )
    manager = QoSManager()
    _configure_clusters(manager, charge_s)
    _apply_algorithm(algo, simulator, manager, mixra_solver)

    effective_snir = _sync_snir_state(simulator, use_snir)
    simulator.run(max_time=duration)

    metrics = simulator.get_metrics()
    metrics.update(
        {
            "num_nodes": nodes,
            "packet_interval_s": charge_s,
            "random_seed": seed,
            "simulation_duration_s": getattr(simulator, "current_time", duration),
            "payload_bytes": PAYLOAD_BYTES,
            "use_snir": use_snir,
            "with_snir": use_snir,
            "snir_state": STATE_LABELS.get(use_snir, "snir_unknown"),
            "snir_state_effective": STATE_LABELS.get(effective_snir, "snir_unknown"),
            "channels": channels,
            "snir_window": _snir_window_label(snir_window),
        }
    )
    enriched = _compute_additional_metrics(simulator, dict(metrics), algo, mixra_solver)
    return enriched


def _collect_rows(
    *,
    nodes: Sequence[int],
    charges: Sequence[float],
    channels: Sequence[int],
    algos: Sequence[str],
    snir_modes: Sequence[bool],
    replications: int,
    seed: int,
    duration: float,
    mixra_solver: str,
    snir_window: str | float | None,
    quiet: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    scenario_index = 0
    snir_window_label = _snir_window_label(snir_window)
    for snir_mode in snir_modes:
        snir_label = STATE_LABELS.get(snir_mode, "snir_unknown")
        for channel_count in channels:
            for charge_s in charges:
                for node_count in nodes:
                    for algo in algos:
                        for rep in range(replications):
                            scenario_index += 1
                            run_seed = seed + scenario_index
                            if not quiet:
                                print(
                                    "[RUN] "
                                    f"algo={algo} snir={snir_label} "
                                    f"N={node_count} charge={charge_s:g}s "
                                    f"C={channel_count} rep={rep + 1}/{replications} seed={run_seed}"
                                )
                            metrics = _run_simulation(
                                nodes=node_count,
                                charge_s=charge_s,
                                channels=channel_count,
                                algo=algo,
                                use_snir=snir_mode,
                                seed=run_seed,
                                duration=duration,
                                mixra_solver=mixra_solver,
                                snir_window=snir_window,
                            )
                            flat = _flatten_metrics(metrics)
                            total_nodes = int(metrics.get("num_nodes", node_count))
                            sf_pct = _sf_percentages(metrics, total_nodes)
                            cluster_cols = _build_cluster_columns(metrics)
                            cluster_payload = metrics.get("qos_cluster_pdr", {}) or {}
                            row: Dict[str, Any] = {
                                "scenario_id": f"N{node_count}_L{charge_s:g}_C{channel_count}",
                                "replication": rep + 1,
                                "seed": run_seed,
                                "n_nodes": node_count,
                                "charge": charge_s,
                                "channels": channel_count,
                                "algos": algo,
                                "snir_mode": snir_label,
                                "snir_window": snir_window_label,
                                "collisions": flat.get("collisions", 0),
                                "snir_mean": flat.get("snir_mean", 0.0),
                                "snr_mean": flat.get("snr_mean", 0.0),
                                "clusters": json.dumps(cluster_payload, ensure_ascii=False, sort_keys=True),
                                **cluster_cols,
                                **sf_pct,
                            }
                            rows.append(row)
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _coerce_row_types(row: Mapping[str, Any]) -> Dict[str, Any]:
    int_fields = {"replication", "seed", "n_nodes", "channels", "collisions"}
    float_fields = {"charge", "snir_mean", "snr_mean"}
    coerced: Dict[str, Any] = {}
    for key, value in row.items():
        if value in ("", None):
            coerced[key] = value
            continue
        if key in int_fields:
            try:
                coerced[key] = int(float(value))
            except (TypeError, ValueError):
                coerced[key] = value
            continue
        if key in float_fields or key.startswith("cluster_pdr_") or (
            key.startswith("sf_") and key.endswith("_pct")
        ):
            try:
                coerced[key] = float(value)
            except (TypeError, ValueError):
                coerced[key] = value
            continue
        coerced[key] = value
    return coerced


def _numeric_columns(rows: List[Dict[str, Any]], exclude: Iterable[str]) -> List[str]:
    columns: List[str] = []
    excluded = set(exclude)
    if not rows:
        return columns
    for key in rows[0].keys():
        if key in excluded:
            continue
        try:
            float(rows[0].get(key))
        except (TypeError, ValueError):
            continue
        columns.append(key)
    return columns


def _aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["n_nodes"],
            row["charge"],
            row["channels"],
            row["algos"],
            row["snir_mode"],
            row.get("snir_window"),
        )
        grouped[key].append(row)

    exclude = {
        "scenario_id",
        "replication",
        "seed",
        "clusters",
        "n_nodes",
        "charge",
        "channels",
        "algos",
        "snir_mode",
        "snir_window",
    }
    numeric_cols = _numeric_columns(rows, exclude)
    aggregated: List[Dict[str, Any]] = []
    for (n_nodes, charge, channels, algo, snir_mode, snir_window), items in grouped.items():
        entry: Dict[str, Any] = {
            "n_nodes": n_nodes,
            "charge": charge,
            "channels": channels,
            "algos": algo,
            "snir_mode": snir_mode,
            "snir_window": snir_window,
            "replications": len(items),
            "clusters": items[0].get("clusters", "{}"),
        }
        for col in numeric_cols:
            values = [float(item.get(col, 0.0)) for item in items]
            mean_value = fmean(values) if values else 0.0
            if len(values) > 1:
                std_value = stdev(values)
            else:
                std_value = 0.0
            ci95 = 0.0
            if len(values) > 1:
                ci95 = 1.96 * std_value / math.sqrt(len(values))
            entry[f"{col}_mean"] = mean_value
            entry[f"{col}_std"] = std_value
            entry[f"{col}_ci95"] = ci95
        aggregated.append(entry)
    return aggregated


def main(argv: List[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.replications > MAX_REPLICATIONS:
        parser.error(f"replications ne doit pas dépasser {MAX_REPLICATIONS}.")
    if args.replications < 1:
        parser.error("replications doit être >= 1.")

    raw_dir: Path = args.raw_dir
    agg_dir: Path = args.agg_dir
    raw_path = raw_dir / "step1_raw.csv"
    agg_path = agg_dir / "step1_agg.csv"

    if args.skip_existing and raw_path.exists():
        if not args.quiet:
            print(f"[SKIP] CSV brut déjà présent : {raw_path}")
        rows: List[Dict[str, Any]] = []
        with raw_path.open("r", encoding="utf8") as handle:
            reader = csv.DictReader(handle)
            rows.extend(_coerce_row_types(row) for row in reader)
    else:
        rows = _collect_rows(
            nodes=args.nodes,
            charges=args.charges,
            channels=args.channels,
            algos=args.algos,
            snir_modes=args.snir_modes,
            replications=args.replications,
            seed=args.seed,
            duration=args.duration,
            mixra_solver=args.mixra_solver,
            snir_window=args.snir_window,
            quiet=args.quiet,
        )
        _write_csv(raw_path, rows)
        if not args.quiet:
            print(f"[OK] CSV brut écrit : {raw_path}")

    if rows:
        aggregated = _aggregate(rows)
        _write_csv(agg_path, aggregated)
        if not args.quiet:
            print(f"[OK] CSV agrégé écrit : {agg_path}")
    elif not args.quiet:
        print("[WARN] Aucun résultat brut à agréger.")


if __name__ == "__main__":
    main()
