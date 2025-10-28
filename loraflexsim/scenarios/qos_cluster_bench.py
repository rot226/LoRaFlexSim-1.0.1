"""Scénario de benchmark QoS avec trois clusters et multiples algorithmes."""

from __future__ import annotations

import csv
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from loraflexsim.launcher import Channel, MultiChannel, Simulator
from loraflexsim.launcher.non_orth_delta import DEFAULT_NON_ORTH_DELTA
from loraflexsim.launcher.qos import QoSManager

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "qos_clusters"
DEFAULT_REPORT_PATH = ROOT_DIR / "docs" / "qos_cluster_bench_report.md"

AREA_RADIUS_M = 2500.0
AREA_SIZE_M = AREA_RADIUS_M * 2.0
PAYLOAD_BYTES = 20
DEFAULT_NODE_COUNTS: Sequence[int] = (1000, 5000, 10000, 13000, 15000)
DEFAULT_TX_PERIODS: Sequence[float] = (600.0, 300.0, 150.0)
DEFAULT_SIMULATION_DURATION_S = 24.0 * 3600.0
SF_ORDER = [7, 8, 9, 10, 11, 12]
FREQUENCIES_HZ = [
    868_100_000.0,
    868_300_000.0,
    868_500_000.0,
    867_100_000.0,
    867_300_000.0,
    867_500_000.0,
    867_700_000.0,
    867_900_000.0,
]


@dataclass(frozen=True)
class AlgorithmSpec:
    """Description d'un algorithme testé dans le banc."""

    key: str
    label: str
    requires_qos: bool
    apply: Callable[[Simulator, QoSManager, str], None]


@dataclass
class RunRecord:
    """Résultat d'une exécution élémentaire."""

    num_nodes: int
    packet_interval_s: float
    algorithm: str
    csv_path: Path
    metrics: Dict[str, Any]
    targets_met: bool


ALGORITHMS: Sequence[AlgorithmSpec] = (
    AlgorithmSpec("adr", "ADR pur", False, lambda sim, manager, solver: _apply_adr_pure(sim)),
    AlgorithmSpec("apra", "APRA-like", True, lambda sim, manager, solver: _apply_apra_like(sim, manager)),
    AlgorithmSpec("aimi", "Aimi-like", True, lambda sim, manager, solver: _apply_aimi_like(sim, manager)),
    AlgorithmSpec("mixrah", "MixRA-H", True, lambda sim, manager, solver: _apply_mixra_h(sim, manager)),
    AlgorithmSpec("mixraopt", "MixRA-Opt", True, lambda sim, manager, solver: _apply_mixra_opt(sim, manager, solver)),
)


def _apply_adr_pure(simulator: Simulator) -> None:
    """Active l'ADR côté serveur sans gestion QoS."""

    setattr(simulator, "adr_server", True)
    setattr(simulator, "adr_node", True)
    setattr(simulator, "qos_active", False)
    setattr(simulator, "qos_algorithm", "ADR pur")
    setattr(simulator, "qos_clusters_config", {})
    setattr(simulator, "qos_node_clusters", {})
    setattr(simulator, "qos_mixra_solver", None)


def _apply_apra_like(simulator: Simulator, manager: QoSManager) -> None:
    """Heuristique inspirée d'APRA : SF minimal pour clusters prioritaires."""

    manager.active_algorithm = "APRA-like"
    manager._update_qos_context(simulator)
    if not getattr(manager, "clusters", None):
        return
    gateways = list(getattr(simulator, "gateways", []))

    def _distance(node) -> float:
        if not gateways:
            return 0.0
        return min(math.hypot(node.x - gw.x, node.y - gw.y) for gw in gateways)

    ordered_nodes = sorted(getattr(simulator, "nodes", []), key=_distance)
    for node in ordered_nodes:
        accessible = list(getattr(node, "qos_accessible_sf", []) or [])
        if not accessible:
            accessible = list(SF_ORDER)
        cluster_id = getattr(node, "qos_cluster_id", None)
        if cluster_id == manager.clusters[0].cluster_id:
            chosen_sf = accessible[0]
        elif cluster_id == manager.clusters[1].cluster_id:
            idx = min(1, len(accessible) - 1)
            chosen_sf = accessible[idx]
        else:
            chosen_sf = accessible[-1]
        node.sf = chosen_sf
        sf_index = SF_ORDER.index(chosen_sf) if chosen_sf in SF_ORDER else len(SF_ORDER) - 1
        node.tx_power = QoSManager._assign_tx_power(sf_index)
    setattr(simulator, "qos_active", True)
    setattr(simulator, "qos_algorithm", "APRA-like")
    setattr(simulator, "qos_mixra_solver", None)


def _apply_aimi_like(simulator: Simulator, manager: QoSManager) -> None:
    """Heuristique inspirée d'Aimi : compromis SF médian et équilibrage canaux."""

    manager.active_algorithm = "Aimi-like"
    manager._update_qos_context(simulator)
    if not getattr(manager, "clusters", None):
        return
    multichannel = getattr(simulator, "multichannel", None)
    channels = list(getattr(multichannel, "channels", []) or [])
    if not channels:
        base_channel = getattr(simulator, "channel", None)
        if base_channel is not None:
            channels = [base_channel]
    channel_count = len(channels)
    channel_index = 0

    for node in getattr(simulator, "nodes", []):
        accessible = list(getattr(node, "qos_accessible_sf", []) or [])
        if not accessible:
            accessible = list(SF_ORDER)
        cluster_id = getattr(node, "qos_cluster_id", None)
        if cluster_id == manager.clusters[0].cluster_id:
            idx = min(len(accessible) // 2, len(accessible) - 1)
            chosen_sf = accessible[idx]
        elif cluster_id == manager.clusters[1].cluster_id:
            idx = min(max(len(accessible) // 2, 1), len(accessible) - 1)
            chosen_sf = accessible[idx]
        else:
            chosen_sf = accessible[-1]
        node.sf = chosen_sf
        sf_index = SF_ORDER.index(chosen_sf) if chosen_sf in SF_ORDER else len(SF_ORDER) - 1
        node.tx_power = QoSManager._assign_tx_power(sf_index)
        if channels:
            channel = channels[channel_index % channel_count]
            node.channel = channel
            channel_index += 1
    setattr(simulator, "qos_active", True)
    setattr(simulator, "qos_algorithm", "Aimi-like")
    setattr(simulator, "qos_mixra_solver", None)


def _apply_mixra_h(simulator: Simulator, manager: QoSManager) -> None:
    manager.apply(simulator, "MixRA-H")
    setattr(simulator, "qos_mixra_solver", "heuristic")


@contextmanager
def _mixra_solver_context(mode: str):
    from loraflexsim.launcher import qos as qos_module

    if mode == "greedy":
        original = qos_module.minimize
        qos_module.minimize = None
        try:
            yield "greedy"
        finally:
            qos_module.minimize = original
    else:
        solver = "scipy" if qos_module.minimize is not None else "greedy"
        yield solver


def _apply_mixra_opt(simulator: Simulator, manager: QoSManager, solver_mode: str) -> None:
    with _mixra_solver_context(solver_mode) as solver_used:
        manager.apply(simulator, "MixRA-Opt")
        setattr(simulator, "qos_mixra_solver", solver_used)


def _configure_clusters(manager: QoSManager, packet_interval: float) -> None:
    rate = 1.0 / packet_interval if packet_interval > 0 else 0.0
    manager.configure_clusters(
        3,
        proportions=[0.1, 0.3, 0.6],
        arrival_rates=[rate, rate, rate],
        pdr_targets=[0.90, 0.80, 0.70],
    )


def _build_multichannel() -> MultiChannel:
    channels = []
    for idx, freq in enumerate(FREQUENCIES_HZ):
        channel = Channel(
            frequency_hz=freq,
            bandwidth=125_000.0,
            capture_threshold_dB=1.0,
            capture_window_symbols=6,
            channel_index=idx,
            advanced_capture=True,
            multipath_taps=4,
            fast_fading_std=0.0,
            variable_noise_std=0.5,
        )
        channel.orthogonal_sf = False
        channels.append(channel)
    multichannel = MultiChannel(channels)
    multichannel.force_non_orthogonal(DEFAULT_NON_ORTH_DELTA)
    return multichannel


def _create_simulator(num_nodes: int, packet_interval: float, seed: int) -> Simulator:
    multichannel = _build_multichannel()
    simulator = Simulator(
        num_nodes=num_nodes,
        num_gateways=1,
        area_size=AREA_SIZE_M,
        transmission_mode="Random",
        packet_interval=packet_interval,
        first_packet_interval=packet_interval,
        packets_to_send=0,
        adr_node=False,
        adr_server=False,
        duty_cycle=0.01,
        mobility=False,
        channels=multichannel,
        channel_distribution="round-robin",
        payload_size_bytes=PAYLOAD_BYTES,
        seed=seed,
        capture_mode="advanced",
        phy_model="omnet",
        pure_poisson_mode=True,
    )
    setattr(simulator, "capture_delta_db", 1.0)
    return simulator


def _round_frequency(freq: float) -> int:
    return int(round(freq))


def _resolve_channel_index(mapping: Mapping[int, int], freq: float | None) -> int:
    if not mapping:
        return 0
    if freq is None:
        return next(iter(mapping.values()))
    key = _round_frequency(freq)
    if key in mapping:
        return mapping[key]
    closest_key = min(mapping, key=lambda k: abs(k - key))
    return mapping[closest_key]


def _frequency_mapping(simulator: Simulator) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    multichannel = getattr(simulator, "multichannel", None)
    channels = list(getattr(multichannel, "channels", []) or [])
    for idx, channel in enumerate(channels):
        freq = getattr(channel, "frequency_hz", None)
        if freq is None:
            continue
        mapping[_round_frequency(freq)] = getattr(channel, "channel_index", idx)
    base_channel = getattr(simulator, "channel", None)
    if base_channel is not None:
        freq = getattr(base_channel, "frequency_hz", None)
        if freq is not None:
            mapping.setdefault(_round_frequency(freq), getattr(base_channel, "channel_index", 0))
    return mapping


def _compute_additional_metrics(
    simulator: Simulator,
    metrics: MutableMapping[str, Any],
    algorithm_label: str,
    mixra_solver: str,
) -> Dict[str, Any]:
    payload_bits = PAYLOAD_BYTES * 8.0
    duration = float(getattr(simulator, "current_time", 0.0) or 0.0)
    if duration <= 0.0:
        duration = 1.0
    total_sent = float(metrics.get("tx_attempted", 0.0) or 0.0)
    delivered = float(metrics.get("delivered", 0.0) or 0.0)
    metrics["DER"] = delivered / total_sent if total_sent > 0 else 0.0

    nodes = list(getattr(simulator, "nodes", []) or [])
    energy_nodes = float(metrics.get("energy_nodes_J", 0.0) or 0.0)
    metrics["avg_energy_per_node_J"] = energy_nodes / len(nodes) if nodes else 0.0

    per_node_throughput = [
        getattr(node, "rx_delivered", 0) * payload_bits / duration for node in nodes
    ]
    if per_node_throughput and sum(value ** 2 for value in per_node_throughput) > 0:
        numerator = sum(per_node_throughput) ** 2
        denominator = len(per_node_throughput) * sum(value ** 2 for value in per_node_throughput)
        metrics["jain_index"] = numerator / denominator if denominator > 0 else 0.0
    else:
        metrics["jain_index"] = 0.0

    freq_map = _frequency_mapping(simulator)
    throughput_map: Dict[int, Dict[int, float]] = {}
    collisions_by_sf: Dict[int, int] = {}
    collisions_by_channel: Dict[int, int] = {}
    snr_values: List[float] = []
    histogram: Dict[str, int] = {}

    for event in getattr(simulator, "events_log", []):
        result = event.get("result")
        sf = int(event.get("sf", 0) or 0)
        channel_idx = _resolve_channel_index(freq_map, event.get("frequency_hz"))
        if result == "Success":
            throughput_sf = throughput_map.setdefault(sf, {})
            throughput_sf[channel_idx] = throughput_sf.get(channel_idx, 0.0) + 1.0
            snr = event.get("snr_dB")
            if snr is not None:
                snr_values.append(float(snr))
        elif result == "Collision":
            collisions_by_sf[sf] = collisions_by_sf.get(sf, 0) + 1
            collisions_by_channel[channel_idx] = collisions_by_channel.get(channel_idx, 0) + 1

    for sf, channel_counts in throughput_map.items():
        for channel_idx, count in channel_counts.items():
            channel_counts[channel_idx] = count * payload_bits / duration

    if snr_values:
        min_bin = math.floor(min(snr_values))
        max_bin = math.ceil(max(snr_values))
        for value in snr_values:
            bin_key = str(int(math.floor(value)))
            histogram[bin_key] = histogram.get(bin_key, 0) + 1
        bins = list(range(min_bin, max_bin + 1))
    else:
        bins = list(range(-30, 31))
        histogram = {str(b): 0 for b in bins}

    total_samples = sum(histogram.values())
    cdf: List[List[float]] = []
    cumulative = 0
    for bin_key in sorted(histogram, key=lambda x: float(x)):
        cumulative += histogram[bin_key]
        probability = cumulative / total_samples if total_samples > 0 else 0.0
        cdf.append([float(bin_key), probability])

    metrics["throughput_sf_channel"] = throughput_map
    metrics["collision_breakdown"] = {
        "total": int(metrics.get("collisions", 0) or 0),
        "by_sf": collisions_by_sf,
        "by_channel": collisions_by_channel,
    }
    metrics["snr_histogram"] = histogram
    metrics["snr_cdf"] = cdf
    metrics["snr_samples"] = total_samples
    metrics["algorithm"] = algorithm_label
    metrics.setdefault("mixra_solver", getattr(simulator, "qos_mixra_solver", mixra_solver))
    metrics["throughput_sf_channel_json"] = json.dumps(throughput_map, ensure_ascii=False, sort_keys=True)
    metrics["collision_breakdown_json"] = json.dumps(metrics["collision_breakdown"], ensure_ascii=False, sort_keys=True)
    metrics["snr_histogram_json"] = json.dumps(histogram, ensure_ascii=False, sort_keys=True)
    metrics["snr_cdf_json"] = json.dumps(cdf, ensure_ascii=False)
    metrics["sf_distribution_json"] = json.dumps(metrics.get("sf_distribution", {}), ensure_ascii=False, sort_keys=True)
    return dict(metrics)


def _flatten_metrics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}

    def _flatten(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, val in value.items():
                next_prefix = f"{prefix}__{key}" if prefix else str(key)
                _flatten(next_prefix, val)
        elif isinstance(value, list):
            flat[prefix] = json.dumps(value, ensure_ascii=False)
        else:
            flat[prefix] = value

    for key, val in payload.items():
        _flatten(str(key), val)
    return flat


def _write_csv(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(row.keys())
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def _targets_met(metrics: Mapping[str, Any]) -> bool:
    targets = metrics.get("qos_cluster_targets") or {}
    pdrs = metrics.get("qos_cluster_pdr") or {}
    if not isinstance(targets, Mapping) or not targets:
        return False
    tolerance = 1e-6
    for key, target in targets.items():
        cluster_value = pdrs.get(key)
        if cluster_value is None and isinstance(key, str):
            try:
                cluster_value = pdrs.get(int(key))
            except Exception:
                cluster_value = None
        if cluster_value is None:
            return False
        if float(cluster_value) + tolerance < float(target):
            return False
    return True


def _mean(values: Iterable[float]) -> float:
    data = list(values)
    if not data:
        return 0.0
    return float(sum(data)) / len(data)


def _compute_breakpoint(runs: Sequence[RunRecord]) -> Dict[str, Any] | None:
    ordered = sorted(runs, key=lambda r: (r.num_nodes, r.packet_interval_s))
    for run in ordered:
        if not run.targets_met:
            return {
                "num_nodes": run.num_nodes,
                "packet_interval_s": run.packet_interval_s,
            }
    return None


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    if isinstance(value, Path):
        return str(value)
    return value


def _summarize_algorithm(runs: Sequence[RunRecord]) -> Dict[str, Any]:
    averages = {
        "PDR": _mean(run.metrics.get("PDR", 0.0) for run in runs),
        "DER": _mean(run.metrics.get("DER", 0.0) for run in runs),
        "throughput_bps": _mean(run.metrics.get("throughput_bps", 0.0) for run in runs),
        "avg_energy_per_node_J": _mean(run.metrics.get("avg_energy_per_node_J", 0.0) for run in runs),
        "jain_index": _mean(run.metrics.get("jain_index", 0.0) for run in runs),
    }
    return {
        "averages": averages,
        "all_targets_met": all(run.targets_met for run in runs if run.metrics.get("qos_cluster_targets")),
        "breakpoint": _compute_breakpoint(runs),
    }


def _generate_report(summary: Mapping[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    settings = summary.get("settings", {})
    node_counts = settings.get("node_counts", [])
    tx_periods = settings.get("tx_periods", [])
    mixra_solver = settings.get("mixra_solver", "auto")
    lines: List[str] = []
    lines.append("# Rapport du banc QoS par clusters")
    lines.append("")
    lines.append("## Paramètres de simulation")
    lines.append(f"- Rayon simulé : {AREA_RADIUS_M / 1000:.1f} km (aire carrée {AREA_SIZE_M / 1000:.1f} km)")
    lines.append(f"- Charges évaluées (nœuds) : {', '.join(str(n) for n in node_counts)}")
    lines.append(
        "- Périodes d'émission (s) : "
        + ", ".join(f"{int(p) if p.is_integer() else p:g}" for p in tx_periods)
    )
    lines.append(f"- Taille de payload : {PAYLOAD_BYTES} octets")
    lines.append(f"- Capture delta configuré : 1 dB")
    lines.append(f"- Solveur MixRA-Opt : {mixra_solver}")
    lines.append("")
    lines.append("## Synthèse par algorithme")
    lines.append("| Algorithme | Point de rupture | Respect des cibles | PDR moyen | DER moyen | Débit moyen (bps) | Indice de Jain |")
    lines.append("|---|---|---|---|---|---|---|")
    algorithms = summary.get("algorithms", {})
    for label, data in algorithms.items():
        averages = data.get("averages", {})
        breakpoint = data.get("breakpoint")
        if breakpoint:
            bp_text = f"N={breakpoint['num_nodes']} (TX={breakpoint['packet_interval_s']:.0f}s)"
        else:
            bp_text = "Aucun"
        respect = "✅" if data.get("all_targets_met") else "❌"
        lines.append(
            "| {label} | {bp} | {respect} | {pdr:.3f} | {der:.3f} | {thr:.2f} | {jain:.3f} |".format(
                label=label,
                bp=bp_text,
                respect=respect,
                pdr=averages.get("PDR", 0.0),
                der=averages.get("DER", 0.0),
                thr=averages.get("throughput_bps", 0.0),
                jain=averages.get("jain_index", 0.0),
            )
        )
    lines.append("")
    for label, data in algorithms.items():
        runs: Sequence[Mapping[str, Any]] = data.get("runs", [])
        if not runs:
            continue
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| Nœuds | Période (s) | PDR | DER | Débit (bps) | Énergie moyenne (J) | Jain | Cibles OK | CSV |")
        lines.append("|---|---|---|---|---|---|---|---|")
        sorted_runs = sorted(runs, key=lambda r: (r["num_nodes"], r["packet_interval_s"]))
        for run in sorted_runs:
            metrics = run.get("metrics", {})
            period = run["packet_interval_s"]
            period_text = f"{period:.0f}" if float(period).is_integer() else f"{period:g}"
            respect = "✅" if run.get("targets_met") else "❌"
            csv_rel = _relative_path(Path(run["csv_path"]))
            lines.append(
                "| {nodes} | {period} | {pdr:.3f} | {der:.3f} | {thr:.2f} | {energy:.4f} | {jain:.3f} | {ok} | {csv} |".format(
                    nodes=run["num_nodes"],
                    period=period_text,
                    pdr=metrics.get("PDR", 0.0),
                    der=metrics.get("DER", 0.0),
                    thr=metrics.get("throughput_bps", 0.0),
                    energy=metrics.get("avg_energy_per_node_J", 0.0),
                    jain=metrics.get("jain_index", 0.0),
                    ok=respect,
                    csv=csv_rel,
                )
            )
        lines.append("")
    lines.append("## Checklist PASS/FAIL – implémentation QoS conforme")
    lines.append("")
    lines.append("- [x] Capture delta fixé à 1 dB sur les huit canaux 125 kHz")
    lines.append("- [x] Fading Rayleigh activé via multipath_taps=4")
    for label, data in algorithms.items():
        runs = data.get("runs", [])
        if not runs:
            continue
        has_qos = any(run.get("metrics", {}).get("qos_cluster_targets") for run in runs)
        if not has_qos:
            status = "[ ]"
        else:
            status = "[x]" if data.get("all_targets_met") else "[ ]"
        lines.append(f"- {status} {label} : cibles PDR respectées sur toutes les charges testées")
    report_path.write_text("\n".join(lines), encoding="utf8")


def run_bench(
    *,
    node_counts: Sequence[int] = DEFAULT_NODE_COUNTS,
    tx_periods: Sequence[float] = DEFAULT_TX_PERIODS,
    seed: int = 1,
    output_dir: Path | None = None,
    simulation_duration_s: float | None = DEFAULT_SIMULATION_DURATION_S,
    mixra_solver: str = "auto",
    quiet: bool = False,
    progress_callback: Callable[[int, int, Dict[str, Any]], None] | None = None,
) -> Dict[str, Any]:
    """Exécute le banc QoS pour toutes les combinaisons et exporte les résultats."""

    if simulation_duration_s is None:
        simulation_duration_s = DEFAULT_SIMULATION_DURATION_S
    if output_dir is None:
        output_root = DEFAULT_RESULTS_DIR
    else:
        output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = DEFAULT_REPORT_PATH

    combos = [(n, p) for n in node_counts for p in tx_periods]
    total_runs = len(combos) * len(ALGORITHMS)
    records_by_algorithm: Dict[str, List[RunRecord]] = {spec.label: [] for spec in ALGORITHMS}
    run_index = 0

    for combo_index, (num_nodes, packet_interval) in enumerate(combos):
        combo_seed = seed + combo_index
        for spec in ALGORITHMS:
            run_index += 1
            context = {
                "num_nodes": num_nodes,
                "packet_interval_s": packet_interval,
                "algorithm": spec.label,
                "run_index": run_index,
                "total_runs": total_runs,
            }
            if progress_callback is not None:
                progress_callback(run_index, total_runs, context)
            elif not quiet:
                print(
                    f"[{run_index}/{total_runs}] {spec.label} – N={num_nodes} TX={packet_interval:.0f}s"
                )
            simulator = _create_simulator(num_nodes, packet_interval, combo_seed)
            manager = QoSManager()
            if spec.requires_qos:
                _configure_clusters(manager, packet_interval)
            try:
                spec.apply(simulator, manager, mixra_solver)
            except Exception:
                if not quiet:
                    print(f"Échec de l'initialisation pour {spec.label}, la simulation est ignorée.")
                raise
            simulator.run(max_time=simulation_duration_s)
            base_metrics = simulator.get_metrics()
            base_metrics["num_nodes"] = num_nodes
            base_metrics["packet_interval_s"] = packet_interval
            base_metrics["random_seed"] = combo_seed
            base_metrics["simulation_duration_s"] = getattr(simulator, "current_time", simulation_duration_s)
            enriched = _compute_additional_metrics(simulator, dict(base_metrics), spec.label, mixra_solver)
            csv_row = _flatten_metrics(enriched)
            csv_filename = f"{num_nodes}_{int(packet_interval) if float(packet_interval).is_integer() else packet_interval:g}.csv"
            csv_path = output_root / spec.key / csv_filename
            _write_csv(csv_path, csv_row)
            run_record = RunRecord(
                num_nodes=num_nodes,
                packet_interval_s=packet_interval,
                algorithm=spec.label,
                csv_path=csv_path,
                metrics=enriched,
                targets_met=_targets_met(enriched),
            )
            records_by_algorithm[spec.label].append(run_record)

    algorithms_summary: Dict[str, Any] = {}
    for spec in ALGORITHMS:
        runs = records_by_algorithm.get(spec.label, [])
        summary = _summarize_algorithm(runs)
        summary["runs"] = [
            {
                "num_nodes": run.num_nodes,
                "packet_interval_s": run.packet_interval_s,
                "targets_met": run.targets_met,
                "csv_path": _relative_path(run.csv_path),
                "metrics": {
                    "PDR": run.metrics.get("PDR", 0.0),
                    "DER": run.metrics.get("DER", 0.0),
                    "throughput_bps": run.metrics.get("throughput_bps", 0.0),
                    "avg_energy_per_node_J": run.metrics.get("avg_energy_per_node_J", 0.0),
                    "jain_index": run.metrics.get("jain_index", 0.0),
                    "qos_cluster_pdr": run.metrics.get("qos_cluster_pdr", {}),
                    "qos_cluster_targets": run.metrics.get("qos_cluster_targets", {}),
                    "mixra_solver": run.metrics.get("mixra_solver"),
                },
            }
            for run in runs
        ]
        algorithms_summary[spec.label] = summary

    summary_payload = {
        "settings": {
            "node_counts": list(node_counts),
            "tx_periods": list(tx_periods),
            "seed": seed,
            "simulation_duration_s": simulation_duration_s,
            "mixra_solver": mixra_solver,
            "capture_delta_db": 1.0,
            "output_dir": _relative_path(output_root),
        },
        "algorithms": algorithms_summary,
        "total_runs": total_runs,
        "report_path": _relative_path(report_path),
    }

    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(_sanitize_for_json(summary_payload), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf8",
    )

    _generate_report(summary_payload, report_path)
    summary_payload["summary_path"] = _relative_path(summary_path)
    return summary_payload


__all__ = ["run_bench"]
