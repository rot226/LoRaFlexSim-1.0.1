"""Automatise l'exécution des expériences de l'étape 1.

Le script permet de choisir l'algorithme QoS (ADR, APRA, MixRA-H ou
MixRA-Opt), de contrôler l'utilisation du calcul SNIR et d'exporter les
métriques clés au format CSV via l'infrastructure de journalisation
existante.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Mapping

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loraflexsim.launcher import Simulator
from loraflexsim.launcher.qos import QoSManager
from loraflexsim.launcher.simulator import InterferenceTracker
from loraflexsim.scenarios.qos_cluster_bench import (
    PAYLOAD_BYTES,
    _apply_adr_pure,
    _apply_apra_like,
    _apply_mixra_h,
    _apply_mixra_opt,
    _compute_additional_metrics,
    _create_simulator,
    _flatten_metrics,
    _write_csv,
    _configure_clusters,
)


def _apply_ucb1(simulator: Simulator, manager: QoSManager) -> None:
    """Active la sélection SF par bandit UCB1 sur tous les nœuds."""

    simulator.adr_node = False
    simulator.adr_server = False
    simulator.qos_active = False
    simulator.qos_algorithm = "UCB1"
    simulator.qos_mixra_solver = None
    for node in getattr(simulator, "nodes", []) or []:
        node.learning_method = "ucb1"

DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "step1"
ALGORITHMS: Mapping[str, Callable[..., None]] = {
    "adr": _apply_adr_pure,
    "apra": _apply_apra_like,
    "mixra_h": _apply_mixra_h,
    "mixra_opt": _apply_mixra_opt,
    "ucb1": _apply_ucb1,
}
STATE_LABELS = {True: "snir_on", False: "snir_off"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm",
        choices=sorted(ALGORITHMS),
        default="adr",
        help="Algorithme testé (adr, apra, mixra_h, mixra_opt, ucb1)",
    )
    parser.add_argument("--nodes", type=int, default=5000, help="Nombre de nœuds simulés")
    parser.add_argument(
        "--packet-interval",
        type=float,
        default=300.0,
        help="Intervalle moyen d'émission en secondes",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=6 * 3600.0,
        help="Durée maximale de simulation en secondes",
    )
    parser.add_argument("--seed", type=int, default=1, help="Graine de simulation")
    snir_group = parser.add_mutually_exclusive_group(required=True)
    snir_group.add_argument(
        "--use-snir",
        action="store_true",
        dest="use_snir",
        help="Active explicitement le calcul SNIR sur les canaux",
    )
    snir_group.add_argument(
        "--no-snir",
        action="store_false",
        dest="use_snir",
        help="Désactive explicitement le calcul SNIR sur les canaux",
    )
    parser.add_argument(
        "--channel-config",
        type=Path,
        default=ROOT_DIR / "config.ini",
        help="Fichier INI pour configurer le bruit et le fading du canal",
    )
    parser.add_argument(
        "--fading-std-db",
        type=float,
        default=None,
        help="Écart-type (dB) du fading aléatoire appliqué au calcul SNIR",
    )
    parser.add_argument(
        "--noise-floor-std-db",
        type=float,
        default=None,
        help="Écart-type (dB) du bruit de fond du canal",
    )
    parser.add_argument(
        "--capture-threshold-db",
        type=float,
        default=None,
        help="Seuil de capture (dB) utilisé dans le modèle de collision",
    )
    parser.add_argument(
        "--marginal-snir-margin-db",
        type=float,
        default=None,
        help="Marge sous laquelle une capture peut échouer aléatoirement",
    )
    parser.add_argument(
        "--marginal-snir-drop-prob",
        type=float,
        default=None,
        help="Probabilité max d'échec lorsque le SNIR est marginal",
    )
    parser.add_argument(
        "--mixra-solver",
        choices=["auto", "greedy"],
        default="auto",
        help="Solveur utilisé pour MixRA-Opt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Répertoire de sortie pour les fichiers CSV",
    )
    parser.add_argument("--quiet", action="store_true", help="Réduit les impressions de progression")
    return parser


def _instantiate_simulator(
    nodes: int,
    packet_interval: float,
    seed: int,
    use_snir: bool,
    *,
    channel_config: Path | None = None,
    fading_std_db: float | None = None,
    noise_floor_std_db: float | None = None,
    capture_threshold_db: float | None = None,
    marginal_snir_margin_db: float | None = None,
    marginal_snir_drop_prob: float | None = None,
) -> Simulator:
    simulator = _create_simulator(
        nodes,
        packet_interval,
        seed,
        channel_config=channel_config,
        channel_overrides={
            "snir_fading_std": fading_std_db,
            "noise_floor_std": noise_floor_std_db,
            "capture_threshold_dB": capture_threshold_db,
            "marginal_snir_margin_db": marginal_snir_margin_db,
            "marginal_snir_drop_prob": marginal_snir_drop_prob,
        },
    )
    simulator._interference_tracker = InterferenceTracker()
    _sync_snir_state(simulator, use_snir)
    _ensure_multichannel_snir_consistency(simulator, use_snir)
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


def _snir_suffix(use_snir: bool) -> str:
    return "_snir-on" if use_snir else "_snir-off"


def _ensure_collisions_snir(csv_row: Mapping[str, object]) -> None:
    if "collisions_snir" not in csv_row:
        raise ValueError("Le CSV exporté doit contenir le champ collisions_snir.")


def _ensure_snir_state_effective(csv_row: Mapping[str, object]) -> None:
    if "snir_state_effective" not in csv_row:
        raise ValueError("Le CSV exporté doit contenir le champ snir_state_effective.")


def _ensure_multichannel_snir_consistency(
    simulator: Simulator,
    requested: bool | None = None,
) -> None:
    multichannel = getattr(simulator, "multichannel", None)
    channels = list(getattr(multichannel, "channels", []) or [])
    if not channels:
        return

    states = [bool(getattr(channel, "use_snir", False)) for channel in channels]
    baseline = states[0]
    if any(state != baseline for state in states[1:]):
        raise ValueError("Les canaux multichannel ne partagent pas le même état use_snir.")
    if requested is not None and baseline != requested:
        raise ValueError(
            "L'état SNIR effectif du multichannel ne correspond pas à l'état demandé."
        )


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


def main(argv: list[str] | None = None) -> Mapping[str, object]:
    parser = _build_parser()
    args = parser.parse_args(argv)

    print(
        "[RUN] "
        f"algo={args.algorithm} use_snir={args.use_snir} seed={args.seed} "
        f"nodes={args.nodes} interval={args.packet_interval:g}s "
        f"fading={args.fading_std_db or 'config'}dB noise_std={args.noise_floor_std_db or 'config'}dB"
    )

    simulator = _instantiate_simulator(
        args.nodes,
        args.packet_interval,
        args.seed,
        args.use_snir,
        channel_config=args.channel_config,
        fading_std_db=args.fading_std_db,
        noise_floor_std_db=args.noise_floor_std_db,
        capture_threshold_db=args.capture_threshold_db,
        marginal_snir_margin_db=args.marginal_snir_margin_db,
        marginal_snir_drop_prob=args.marginal_snir_drop_prob,
    )
    manager = QoSManager()
    _configure_clusters(manager, args.packet_interval)
    _apply_algorithm(args.algorithm, simulator, manager, args.mixra_solver)

    effective_use_snir = _sync_snir_state(simulator, args.use_snir)

    simulator.run(max_time=args.duration)

    metrics = simulator.get_metrics()
    metrics.update(
        {
            "num_nodes": args.nodes,
            "packet_interval_s": args.packet_interval,
            "random_seed": args.seed,
            "simulation_duration_s": getattr(simulator, "current_time", args.duration),
            "payload_bytes": PAYLOAD_BYTES,
            "use_snir": args.use_snir,
            "with_snir": args.use_snir,
            "snir_state": STATE_LABELS.get(args.use_snir, "snir_unknown"),
            "snir_state_effective": STATE_LABELS.get(effective_use_snir, "snir_unknown"),
            "channel_config": str(args.channel_config) if args.channel_config else None,
            "snir_fading_std": getattr(simulator, "snir_fading_std", None),
            "noise_floor_std": getattr(simulator, "noise_floor_std", None),
            "capture_threshold_dB": getattr(simulator, "capture_delta_db", None),
            "marginal_snir_margin_db": getattr(simulator, "marginal_snir_margin_db", None),
            "marginal_snir_drop_prob": getattr(simulator, "marginal_snir_drop_prob", None),
        }
    )
    enriched = _compute_additional_metrics(simulator, dict(metrics), args.algorithm, args.mixra_solver)
    csv_row = _flatten_metrics(enriched)
    _ensure_collisions_snir(csv_row)
    _ensure_snir_state_effective(csv_row)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    interval_label = int(args.packet_interval) if float(args.packet_interval).is_integer() else args.packet_interval
    csv_path = output_dir / f"{args.algorithm}_N{args.nodes}_T{interval_label}{_snir_suffix(effective_use_snir)}.csv"
    _write_csv(csv_path, csv_row)

    if not args.quiet:
        cluster_pdr = enriched.get("qos_cluster_pdr", {}) or {}
        print(f"Résultats enregistrés dans {csv_path}")
        if cluster_pdr:
            print("PDR par cluster : " + ", ".join(f"{k}={v:.3f}" for k, v in cluster_pdr.items()))
        print(
            f"DER={enriched.get('DER', 0.0):.3f} | Collisions={int(enriched.get('collisions', 0))} "
            f"| Jain={enriched.get('jain_index', 0.0):.3f} | Capacité={enriched.get('throughput_bps', 0.0):.1f} bps"
        )

    return {"metrics": enriched, "csv_path": csv_path}


if __name__ == "__main__":
    main()
