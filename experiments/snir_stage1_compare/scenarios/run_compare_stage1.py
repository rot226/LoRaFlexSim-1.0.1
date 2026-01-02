"""Campagne de comparaison SNIR (étape 1).

Ce script explore plusieurs combinaisons d'algorithmes, de profils PHY,
de tailles de réseau et d'intervalles d'émission afin de générer des CSV
suffixés ``_compare`` consolidant les métriques principales (DER, PDR,
SNIR, collisions, débit). La CLI permet de personnaliser les listes
ciblées via des options séparées par des virgules.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loraflexsim.launcher import MultiChannel, Simulator
from loraflexsim.launcher.non_orth_delta import DEFAULT_NON_ORTH_DELTA

FREQUENCIES_HZ: tuple[float, ...] = (
    868_100_000.0,
    868_300_000.0,
    868_500_000.0,
    867_100_000.0,
    867_300_000.0,
    867_500_000.0,
    867_700_000.0,
    867_900_000.0,
)
PAYLOAD_BYTES = 20
PACKETS_PER_NODE = 10
DEFAULT_ALGORITHMS = ("baseline", "snir", "snir_interference")
DEFAULT_PROFILES = ("flora_full", "omnet_full")
DEFAULT_NODES = (2000, 4000, 6000)
DEFAULT_INTERVALS = (300.0, 600.0)
DEFAULT_SEED = 1
DEFAULT_REPS = 1
DEFAULT_JOBS = 1


@dataclass(frozen=True)
class AlgorithmConfig:
    """Configuration liée à un algorithme de comparaison.

    Les drapeaux ``snir_model`` et ``interference_model`` sont conservés pour
    piloter les scénarios, même si l'API ``Simulator`` ne prend plus ces
    arguments en paramètre. Le routage se fait désormais via les attributs des
    canaux instanciés (``use_snir``) et le suivi d'interférence intégré au
    simulateur.
    """

    flora_mode: bool
    snir_model: bool
    interference_model: bool


ALGO_PRESETS: dict[str, AlgorithmConfig] = {
    "baseline": AlgorithmConfig(True, False, False),
    "snir": AlgorithmConfig(True, True, False),
    "snir_interference": AlgorithmConfig(True, True, True),
    "interference_only": AlgorithmConfig(True, False, True),
}


@dataclass(frozen=True)
class SimulationTask:
    """Paramètres minimaux pour une exécution."""

    algorithm: str
    phy_profile: str
    num_nodes: int
    packet_interval: float
    seed: int
    rep: int


@dataclass
class SimulationResult:
    """Résultat agrégé d'un run."""

    algorithm: str
    phy_profile: str
    num_nodes: int
    packet_interval_s: float
    rep: int
    seed: int
    interference_enabled: bool
    packets_sent: int
    packets_attempted: int
    packets_delivered: int
    collisions: int
    der: float
    pdr: float
    throughput_bps: float
    snir_mean: float
    snir_median: float
    sim_time_s: float
    baseline_loss_rate: float

    def as_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "phy_profile": self.phy_profile,
            "num_nodes": self.num_nodes,
            "packet_interval_s": self.packet_interval_s,
            "rep": self.rep,
            "seed": self.seed,
            "interference_enabled": self.interference_enabled,
            "packets_sent": self.packets_sent,
            "packets_attempted": self.packets_attempted,
            "packets_delivered": self.packets_delivered,
            "collisions": self.collisions,
            "der": self.der,
            "pdr": self.pdr,
            "throughput_bps": self.throughput_bps,
            "snir_mean": self.snir_mean,
            "snir_median": self.snir_median,
            "sim_time_s": self.sim_time_s,
            "baseline_loss_rate": self.baseline_loss_rate,
        }


def _parse_list(values: str | Sequence[str], cast_func) -> list:
    if isinstance(values, str):
        split_values = [v.strip() for v in values.split(",") if v.strip()]
    else:
        split_values = list(values)
    return [cast_func(v) for v in split_values]


def _snir_stats(values: Iterable[float | None]) -> tuple[float, float]:
    filtered = [v for v in values if v is not None and math.isfinite(v)]
    if not filtered:
        return 0.0, 0.0
    return float(mean(filtered)), float(median(filtered))


def _build_multichannel(profile: str, force_snir: bool) -> MultiChannel:
    multichannel = MultiChannel(FREQUENCIES_HZ)
    multichannel.force_non_orthogonal(DEFAULT_NON_ORTH_DELTA)
    for channel in multichannel.channels:
        channel.phy_model = profile
        channel.use_snir = bool(force_snir)
        channel.use_flora_curves = profile.startswith("flora")
    return multichannel


def _collect_metrics(
    simulator: Simulator,
    *,
    include_snir: bool,
    interference_model: bool,
) -> tuple[int, int, int, int, float, float, float, float, float, float]:
    sim_time = float(getattr(simulator, "current_time", 0.0))
    payload_bits = PAYLOAD_BYTES * 8.0
    events = list(getattr(simulator, "events_log", []) or [])

    sent = sum(node.packets_sent for node in simulator.nodes)
    attempts = sum(node.tx_attempted for node in simulator.nodes)
    delivered = sum(node.rx_delivered for node in simulator.nodes)
    collisions = sum(node.packets_collision for node in simulator.nodes)
    if collisions == 0 and interference_model and include_snir:
        collisions = max(attempts - delivered, 0)

    der = delivered / sent if sent else 0.0
    pdr = delivered / attempts if attempts else 0.0
    throughput = (delivered * payload_bits / sim_time) if sim_time > 0 else 0.0

    snir_values = [] if not include_snir else [entry.get("snir_dB") for entry in events if "snir_dB" in entry]
    snir_mean, snir_median = _snir_stats(snir_values)
    return (
        sent,
        attempts,
        delivered,
        collisions,
        der,
        pdr,
        throughput,
        snir_mean,
        snir_median,
        sim_time,
    )


def _run_single(task: SimulationTask) -> SimulationResult:
    preset = ALGO_PRESETS.get(task.algorithm, AlgorithmConfig(True, False, False))
    multichannel = _build_multichannel(task.phy_profile, preset.snir_model)

    simulator = Simulator(
        num_nodes=task.num_nodes,
        num_gateways=1,
        area_size=5000.0,
        transmission_mode="Random",
        packet_interval=task.packet_interval,
        first_packet_interval=task.packet_interval,
        packets_to_send=PACKETS_PER_NODE,
        duty_cycle=0.01,
        mobility=False,
        channels=multichannel,
        channel_distribution="round-robin",
        payload_size_bytes=PAYLOAD_BYTES,
        flora_mode=preset.flora_mode,
        seed=task.seed,
        phy_model=task.phy_profile,
        capture_mode="advanced",
    )

    # L'API ``Simulator`` ne propose plus de paramètre ``snir_model`` : on
    # force donc l'état du calcul SNIR au niveau des canaux. Cela permet de
    # comparer un mode purement RSSI (SNIR désactivé) et un mode SNIR complet
    # avec suivi d'interférence.
    for channel in getattr(simulator.multichannel, "channels", []) or []:
        channel.use_snir = bool(preset.snir_model)

    # Le suivi d'interférence intégré (``InterferenceTracker``) est toujours
    # actif ; si un scénario demande explicitement de modéliser
    # l'interférence, on laisse le tracker en place. Sinon, on remplace le
    # tracker par une implémentation minimale qui retourne systématiquement
    # zéro afin de désactiver l'interférence sans modifier le reste du
    # pipeline de simulation.
    interference_enabled = bool(preset.interference_model)
    if not interference_enabled:
        class _ConfigurableNullTracker:
            def __init__(
                self,
                *,
                log_events: list[dict] | None = None,
                log_message: str = "interference_off",
            ) -> None:
                if log_events is not None:
                    log_events.append({"event": log_message})

            def add(self, *_, **__):
                return None

            def remove(self, *_, **__):
                return None

            def total_interference(self, *_, **__):
                return 0.0

        simulator._interference_tracker = _ConfigurableNullTracker(
            log_events=simulator.events_log
        )
    simulator.run()
    (
        sent,
        attempts,
        delivered,
        collisions,
        der,
        pdr,
        throughput,
        snir_mean,
        snir_median,
        sim_time,
    ) = _collect_metrics(
        simulator,
        include_snir=preset.snir_model,
        interference_model=preset.interference_model,
    )
    baseline_loss_rate = float(
        getattr(getattr(simulator, "channel", None), "baseline_loss_rate", 0.0) or 0.0
    )

    return SimulationResult(
        algorithm=task.algorithm,
        phy_profile=task.phy_profile,
        num_nodes=task.num_nodes,
        packet_interval_s=task.packet_interval,
        rep=task.rep,
        seed=task.seed,
        interference_enabled=interference_enabled,
        packets_sent=sent,
        packets_attempted=attempts,
        packets_delivered=delivered,
        collisions=collisions,
        der=der,
        pdr=pdr,
        throughput_bps=throughput,
        snir_mean=snir_mean,
        snir_median=snir_median,
        sim_time_s=sim_time,
        baseline_loss_rate=baseline_loss_rate,
    )


def _write_outputs(results: Iterable[SimulationResult], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    by_algo: dict[str, list[SimulationResult]] = defaultdict(list)
    for res in results:
        by_algo[res.algorithm].append(res)

    fieldnames = [
        "algorithm",
        "phy_profile",
        "num_nodes",
        "packet_interval_s",
        "rep",
        "seed",
        "interference_enabled",
        "packets_sent",
        "packets_attempted",
        "packets_delivered",
        "collisions",
        "der",
        "pdr",
        "throughput_bps",
        "snir_mean",
        "snir_median",
        "sim_time_s",
        "baseline_loss_rate",
    ]

    for algorithm, rows in by_algo.items():
        path = outdir / f"{algorithm}_compare.csv"
        with path.open("w", newline="", encoding="utf8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row.as_dict())


def _build_tasks(
    algorithms: Sequence[str],
    profiles: Sequence[str],
    nodes: Sequence[int],
    intervals: Sequence[float],
    *,
    seed: int,
    reps: int,
) -> list[SimulationTask]:
    tasks: list[SimulationTask] = []
    for rep in range(reps):
        for algorithm in algorithms:
            for profile in profiles:
                for num_nodes in nodes:
                    for interval in intervals:
                        tasks.append(
                            SimulationTask(
                                algorithm=algorithm,
                                phy_profile=profile,
                                num_nodes=num_nodes,
                                packet_interval=float(interval),
                                seed=seed + rep,
                                rep=rep + 1,
                            )
                        )
    return tasks


def run_campaign(
    algorithms: Sequence[str],
    profiles: Sequence[str],
    nodes: Sequence[int],
    intervals: Sequence[float],
    *,
    seed: int,
    reps: int,
    jobs: int,
) -> list[SimulationResult]:
    tasks = _build_tasks(algorithms, profiles, nodes, intervals, seed=seed, reps=reps)
    runner = _run_single
    if jobs <= 1:
        return [runner(task) for task in tasks]

    results: list[SimulationResult] = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        for res in executor.map(runner, tasks):
            results.append(res)
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Campagne de comparaison SNIR (étape 1)")
    parser.add_argument(
        "--algorithms",
        type=str,
        default=",".join(DEFAULT_ALGORITHMS),
        help="Liste des algorithmes à tester (séparés par des virgules)",
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default=",".join(DEFAULT_PROFILES),
        help="Liste des profils PHY à tester (séparés par des virgules)",
    )
    parser.add_argument(
        "--nodes",
        type=str,
        default=",".join(str(n) for n in DEFAULT_NODES),
        help="Tailles de réseau à parcourir (séparées par des virgules)",
    )
    parser.add_argument(
        "--intervals",
        type=str,
        default=",".join(str(v) for v in DEFAULT_INTERVALS),
        help="Intervalles moyens d'émission en secondes (séparés par des virgules)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Graine de base")
    parser.add_argument(
        "--reps",
        type=int,
        default=DEFAULT_REPS,
        help="Nombre de répétitions pour chaque combinaison",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help="Nombre de processus en parallèle",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT_DIR / "experiments" / "snir_stage1_compare" / "data"),
        help="Répertoire de sortie pour les CSV _compare",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    algorithms = _parse_list(args.algorithms, str)
    profiles = _parse_list(args.profiles, str)
    nodes = _parse_list(args.nodes, int)
    intervals = _parse_list(args.intervals, float)

    results = run_campaign(
        algorithms,
        profiles,
        nodes,
        intervals,
        seed=args.seed,
        reps=args.reps,
        jobs=args.jobs,
    )

    outdir = Path(args.outdir)
    _write_outputs(results, outdir)
    print(f"CSV générés dans {outdir} (suffixe _compare)")


if __name__ == "__main__":
    main()
