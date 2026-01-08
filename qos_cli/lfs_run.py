"""CLI pour exécuter un scénario QoS LoRaFlexSim et exporter les traces."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import yaml

from loraflexsim.launcher import Channel, MultiChannel, Simulator
from loraflexsim.launcher.non_orth_delta import DEFAULT_NON_ORTH_DELTA
from loraflexsim.launcher.qos import QoSManager


DEFAULT_CONFIG = Path(__file__).with_name("scenarios.yaml")
DEFAULT_DURATION_FACTOR = 100.0
AREA_RADIUS_M = 2500.0
AREA_SIZE_M = AREA_RADIUS_M * 2.0


MethodName = str


@dataclass(frozen=True)
class PhyProfile:
    """Enveloppe des paramètres radio appliqués aux canaux."""

    name: str
    use_snir: bool
    flora_capture: bool
    interference_model: bool
    snir_model: bool


class ScenarioQoSManager(QoSManager):
    """Spécialisation de :class:`QoSManager` appliquant des limites externes."""

    def __init__(self) -> None:
        super().__init__()
        self._cluster_limits: dict[int, dict[int, dict[int, float]]] = {}
        self._cluster_names: dict[int, str] = {}
        self._channel_map: dict[str, int] = {}

    def set_cluster_names(self, mapping: Mapping[int, str]) -> None:
        self._cluster_names = {int(k): str(v) for k, v in mapping.items()}

    def set_channel_mapping(self, mapping: Mapping[str, int]) -> None:
        self._channel_map = {str(k): int(v) for k, v in mapping.items()}

    def set_capacity_overrides(
        self, overrides: Mapping[int, Mapping[int, Mapping[int, float]]]
    ) -> None:
        mapped: dict[int, dict[int, dict[int, float]]] = {}
        for cluster_id, sf_map in overrides.items():
            cluster_key = int(cluster_id)
            mapped_sf: dict[int, dict[int, float]] = {}
            for sf, channel_map in sf_map.items():
                sf_key = int(sf)
                mapped_channels: dict[int, float] = {}
                for channel_name, value in channel_map.items():
                    channel_key = self._channel_map.get(str(channel_name))
                    if channel_key is None:
                        continue
                    mapped_channels[channel_key] = float(value)
                if mapped_channels:
                    mapped_sf[sf_key] = mapped_channels
            if mapped_sf:
                mapped[cluster_key] = mapped_sf
        self._cluster_limits = mapped

    def _update_qos_context(self, simulator) -> None:  # type: ignore[override]
        super()._update_qos_context(simulator)
        if not self._cluster_limits:
            return
        overrides: dict[int, dict[int, dict[int, float]]] = {}
        for cluster in self.clusters:
            cluster_limits = self._cluster_limits.get(cluster.cluster_id)
            if not cluster_limits:
                continue
            overrides[cluster.cluster_id] = {}
            for sf, channel_map in cluster_limits.items():
                overrides[cluster.cluster_id][sf] = dict(channel_map)
        if overrides:
            self.cluster_sf_channel_capacity = overrides
            setattr(simulator, "qos_sf_channel_capacity", overrides)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exécute un scénario QoS LoRaFlexSim et exporte les traces en CSV."
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help="Identifiant du scénario (clé dans le YAML).",
    )
    parser.add_argument(
        "--method",
        required=True,
        help="Méthode à exécuter (ADR, APRA_like, MixRA_H, MixRA_Opt, Greedy).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Fichier YAML décrivant les scénarios (défaut: %(default)s).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Répertoire de sortie pour écrire packets.csv et nodes.csv.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Durée de la simulation en secondes (défaut: période × 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Graine aléatoire pour l'initialisation du simulateur.",
    )
    snir_group = parser.add_mutually_exclusive_group()
    snir_group.add_argument(
        "--snir",
        choices=("on", "off", "auto"),
        default="auto",
        help="Force l'activation SNIR (on/off) ou laisse la config YAML (auto).",
    )
    snir_group.add_argument(
        "--enable-snir",
        action="store_const",
        const="on",
        dest="snir",
        help="Alias pour --snir on.",
    )
    snir_group.add_argument(
        "--disable-snir",
        action="store_const",
        const="off",
        dest="snir",
        help="Alias pour --snir off.",
    )
    return parser


def load_yaml(path: Path) -> Mapping[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable: {path}")
    with path.open("r", encoding="utf8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError("Le fichier YAML doit contenir un dictionnaire racine.")
    return data


def _ensure_mapping(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"La section '{name}' doit être un dictionnaire.")
    return value


def _parse_optional_float(
    mapping: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> Optional[float]:
    if key not in mapping:
        return None
    value = mapping.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Valeur invalide pour '{context}.{key}': {value!r}") from exc


def _parse_common(common: Mapping[str, object]) -> Tuple[dict, dict, List[int]]:
    gateway = _ensure_mapping("common.gateway", common.get("gateway", {}))
    channels_cfg = _ensure_mapping("common.channels", common.get("channels", {}))
    sf_list = list(map(int, common.get("spreading_factors", [7, 8, 9, 10, 11, 12])))
    return gateway, channels_cfg, sf_list


def _parse_clusters(
    clusters_cfg: Mapping[str, object],
) -> Tuple[List[str], List[float], List[float], List[dict]]:
    names: List[str] = []
    shares: List[float] = []
    targets: List[float] = []
    limits: List[dict] = []
    for name, payload in clusters_cfg.items():
        if not isinstance(payload, Mapping):
            raise ValueError(f"Cluster '{name}' doit être un dictionnaire.")
        share = float(payload.get("share", 0.0))
        target = float(payload.get("target_pdr", 0.0))
        if share <= 0.0:
            raise ValueError(f"Cluster '{name}': 'share' doit être > 0.")
        if not (0.0 < target <= 1.0):
            raise ValueError(f"Cluster '{name}': 'target_pdr' doit être dans ]0,1].")
        names.append(str(name))
        shares.append(share)
        targets.append(target)
        limits.append(dict(payload.get("nu_max", {})))
    total_share = sum(shares)
    if total_share <= 0.0:
        raise ValueError("La somme des parts de clusters doit être > 0.")
    shares = [share / total_share for share in shares]
    return names, shares, targets, limits


def _build_channels(
    channels_cfg: Mapping[str, object],
    propagation_cfg: Mapping[str, object],
    snir_override: Optional[bool] = None,
) -> Tuple[MultiChannel, dict[str, int], PhyProfile]:
    raw_profile = propagation_cfg.get("phy_profile", "qos_original")
    profile_name = str(raw_profile).strip().lower() if raw_profile is not None else ""
    if snir_override is True:
        profile_name = "snir_enhanced"
    elif snir_override is False:
        profile_name = "qos_original"
    enhanced = profile_name == "snir_enhanced"
    profile = PhyProfile(
        name="snir_enhanced" if enhanced else "qos_original",
        use_snir=enhanced,
        flora_capture=enhanced,
        interference_model=enhanced,
        snir_model=enhanced,
    )

    override_keys = (
        "baseline_loss_rate",
        "baseline_collision_rate",
        "residual_collision_prob",
        "snir_off_noise_prob",
        "snir_fading_std",
        "marginal_snir_margin_db",
        "marginal_snir_drop_prob",
    )
    global_overrides = {
        key: _parse_optional_float(propagation_cfg, key, context="propagation")
        for key in override_keys
    }

    channels: List[Channel] = []
    channel_map: dict[str, int] = {}
    capture_threshold = float(propagation_cfg.get("capture_threshold_db", 1.0))
    variable_noise = float(propagation_cfg.get("variable_noise_std", 0.5))
    for index, (name, payload) in enumerate(channels_cfg.items()):
        if not isinstance(payload, Mapping):
            raise ValueError(f"Le canal '{name}' doit être un dictionnaire.")
        frequency = float(payload.get("frequency_hz", 868_100_000.0))
        bandwidth = float(payload.get("bandwidth_hz", 125_000.0))
        channel_overrides: dict[str, float] = {}
        for key in override_keys:
            value = _parse_optional_float(payload, key, context=f"canal '{name}'")
            if value is None:
                value = global_overrides.get(key)
            if value is not None:
                channel_overrides[key] = value
        channel = Channel(
            frequency_hz=frequency,
            bandwidth=bandwidth,
            capture_threshold_dB=capture_threshold,
            capture_window_symbols=6,
            advanced_capture=True,
            multipath_taps=4,
            fast_fading_std=0.0,
            variable_noise_std=variable_noise,
            **channel_overrides,
        )
        channel.channel_index = index
        channel.orthogonal_sf = False
        channel.use_snir = profile.use_snir
        channel.flora_capture = profile.flora_capture
        channel.interference_model = profile.interference_model
        channel.snir_model = profile.snir_model
        channels.append(channel)
        channel_map[str(name)] = index
    multichannel = MultiChannel(channels)
    multichannel.force_non_orthogonal(DEFAULT_NON_ORTH_DELTA)
    return multichannel, channel_map, profile


def _apply_method(
    simulator: Simulator,
    manager: ScenarioQoSManager,
    method: MethodName,
) -> None:
    method_normalized = method.strip().lower()
    if method_normalized == "adr":
        _apply_adr(simulator, manager)
        return
    if method_normalized == "apra_like":
        _apply_apra_like(simulator, manager)
        return
    if method_normalized == "mixra_h":
        manager.apply(simulator, "MixRA-H")
        return
    if method_normalized in {"mixra_opt", "greedy"}:
        solver_mode = "greedy" if method_normalized == "greedy" else "auto"
        _apply_mixra_opt(simulator, manager, solver_mode)
        return
    raise ValueError(f"Méthode inconnue: {method}")


def _apply_adr(simulator: Simulator, manager: ScenarioQoSManager) -> None:
    manager._update_qos_context(simulator)
    setattr(simulator, "adr_server", True)
    setattr(simulator, "adr_node", True)
    setattr(simulator, "qos_active", False)
    setattr(simulator, "qos_algorithm", "ADR-Pure")
    setattr(simulator, "qos_mixra_solver", None)


def _apply_apra_like(simulator: Simulator, manager: ScenarioQoSManager) -> None:
    manager.active_algorithm = "APRA-like"
    manager._update_qos_context(simulator)
    clusters = list(getattr(manager, "clusters", []) or [])
    if not clusters:
        return
    gateways = list(getattr(simulator, "gateways", []))

    def _distance(node) -> float:
        if not gateways:
            return 0.0
        return min(((node.x - gw.x) ** 2 + (node.y - gw.y) ** 2) ** 0.5 for gw in gateways)

    ordered_nodes = sorted(getattr(simulator, "nodes", []), key=_distance)
    sf_order = [7, 8, 9, 10, 11, 12]
    for node in ordered_nodes:
        accessible = list(getattr(node, "qos_accessible_sf", []) or [])
        if not accessible:
            accessible = list(sf_order)
        cluster_id = getattr(node, "qos_cluster_id", None)
        if cluster_id == clusters[0].cluster_id:
            chosen_sf = accessible[0]
        elif len(clusters) > 1 and cluster_id == clusters[1].cluster_id:
            idx = min(1, len(accessible) - 1)
            chosen_sf = accessible[idx]
        else:
            chosen_sf = accessible[-1]
        node.sf = chosen_sf
        sf_index = sf_order.index(chosen_sf) if chosen_sf in sf_order else len(sf_order) - 1
        node.tx_power = QoSManager._assign_tx_power(sf_index)
    setattr(simulator, "qos_active", True)
    setattr(simulator, "qos_algorithm", "APRA-like")
    setattr(simulator, "qos_mixra_solver", None)


def _apply_mixra_opt(
    simulator: Simulator,
    manager: ScenarioQoSManager,
    mode: str,
) -> None:
    from loraflexsim.launcher import qos as qos_module

    def _solver_context():
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

    from contextlib import contextmanager

    @contextmanager
    def _context():
        with _solver_context() as value:
            yield value

    with _context() as solver_used:
        manager.apply(simulator, "MixRA-Opt")
        setattr(simulator, "qos_mixra_solver", solver_used)


def _override_limits(
    manager: ScenarioQoSManager,
    cluster_names: Sequence[str],
    limits_payload: Sequence[Mapping[str, object]],
    channel_map: Mapping[str, int],
) -> None:
    overrides: dict[int, dict[int, dict[int, float]]] = {}
    for index, limit_cfg in enumerate(limits_payload):
        cluster_id = manager.clusters[index].cluster_id
        per_sf: dict[int, dict[int, float]] = {}
        for sf_key, channels in limit_cfg.items():
            if isinstance(sf_key, str) and sf_key.upper().startswith("SF"):
                sf_value = int(sf_key[2:])
            else:
                try:
                    sf_value = int(sf_key)
                except Exception:
                    continue
            if not isinstance(channels, Mapping):
                continue
            per_channel: dict[int, float] = {}
            for channel_name, raw_value in channels.items():
                channel_index = channel_map.get(str(channel_name))
                if channel_index is None:
                    continue
                try:
                    per_channel[channel_index] = float(raw_value)
                except (TypeError, ValueError):
                    continue
            if per_channel:
                per_sf[sf_value] = per_channel
        if per_sf:
            overrides[cluster_id] = per_sf
    manager.set_capacity_overrides(overrides)
    manager.set_cluster_names({manager.clusters[i].cluster_id: name for i, name in enumerate(cluster_names)})


def _scenario_duration(period: float, explicit: Optional[float]) -> float:
    if explicit is not None and explicit > 0.0:
        return float(explicit)
    return float(period * DEFAULT_DURATION_FACTOR)


def _parse_snir_override(raw_value: object, *, source: str) -> Optional[bool]:
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"on", "true", "1", "yes"}:
            return True
        if normalized in {"off", "false", "0", "no"}:
            return False
        if normalized in {"auto", ""}:
            return None
    raise ValueError(f"Valeur SNIR invalide ({source}): {raw_value!r}")


def _prepare_gateway(
    simulator: Simulator, gateway_cfg: Mapping[str, object]
) -> Tuple[float, float, float]:
    position = gateway_cfg.get("position_m", [simulator.area_size / 2.0, simulator.area_size / 2.0, 15.0])
    if isinstance(position, Sequence) and len(position) >= 3:
        x, y, z = (float(position[0]), float(position[1]), float(position[2]))
    else:
        x = y = simulator.area_size / 2.0
        z = 15.0
    if simulator.gateways:
        gateway = simulator.gateways[0]
        gateway.x = x
        gateway.y = y
        gateway.altitude = z
        rx_gain = gateway_cfg.get("rx_gain_db")
        if rx_gain is not None:
            gateway.rx_gain_dB = float(rx_gain)
    return x, y, z


def _collect_packets(
    simulator: Simulator,
    cluster_lookup: Mapping[int, str],
    phy_profile: PhyProfile,
) -> "Optional['pd.DataFrame']":
    df = simulator.get_events_dataframe()
    if df is None or df.empty:
        return df
    snir_state = "snir_on" if phy_profile.use_snir else "snir_off"
    df["use_snir"] = bool(phy_profile.use_snir)
    df["snir_state"] = snir_state
    if phy_profile.name == "qos_original":
        for column in (
            "snir_dB",
            "snir_db",
            "snir",
        ):
            if column in df.columns:
                df[column] = float("nan")
    node_clusters = {getattr(node, "id", None): getattr(node, "qos_cluster_id", None) for node in simulator.nodes}
    df["cluster_id"] = df["node_id"].map(node_clusters)
    df["cluster"] = df["cluster_id"].map(lambda cid: cluster_lookup.get(cid) if cid is not None else None)
    result_series = df["result"].astype(str).str.strip().str.lower()
    df["delivered"] = result_series.eq("success")
    df["collision"] = result_series.isin({"collisionloss", "collision"})
    df["no_coverage"] = result_series.eq("nocoverage")
    loss_reason = None
    df["loss_reason"] = loss_reason
    df.loc[df["collision"], "loss_reason"] = "collision"
    df.loc[df["no_coverage"], "loss_reason"] = "no_coverage"
    df.loc[~df["delivered"] & df["loss_reason"].isna(), "loss_reason"] = "other"
    if not phy_profile.use_snir and "snr_dB" in df.columns and "snir_dB" not in df.columns:
        # En mode SNIR désactivé, le simulateur ne calcule que le SNR ; on
        # réplique donc la colonne pour conserver des courbes SNIR/SNR cohérentes.
        df["snir_dB"] = df["snr_dB"]
    return df


def _collect_nodes(
    simulator: Simulator,
    cluster_lookup: Mapping[int, str],
) -> "Optional['pd.DataFrame']":
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover - pandas obligatoire ici
        raise RuntimeError("pandas est requis pour exporter nodes.csv") from exc

    rows: List[dict] = []
    for node in simulator.nodes:
        data = node.to_dict()
        cluster_id = getattr(node, "qos_cluster_id", None)
        data["node_id"] = getattr(node, "id", data.get("node_id"))
        data["cluster_id"] = cluster_id
        data["cluster"] = cluster_lookup.get(cluster_id)
        data["sf"] = getattr(node, "sf", data.get("final_sf"))
        energy = getattr(node, "energy_consumed", None)
        if energy is not None:
            data["energy_J"] = float(energy)
        accessible = getattr(node, "qos_accessible_sf", None)
        if accessible is not None:
            data["accessible_sf"] = json.dumps(list(accessible))
        rows.append(data)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    yaml_data = load_yaml(args.config)
    common_cfg = _ensure_mapping("common", yaml_data.get("common", {}))
    scenarios_cfg = _ensure_mapping("scenarios", yaml_data.get("scenarios", {}))
    scenario_cfg = scenarios_cfg.get(args.scenario)
    if scenario_cfg is None:
        raise ValueError(f"Scénario '{args.scenario}' introuvable dans {args.config}.")
    if not isinstance(scenario_cfg, Mapping):
        raise ValueError(f"Le scénario '{args.scenario}' doit être un dictionnaire.")

    gateway_cfg, channels_cfg, _ = _parse_common(common_cfg)
    propagation_cfg = _ensure_mapping("common.propagation", common_cfg.get("propagation", {}))
    scenario_propagation_raw = scenario_cfg.get("propagation")
    if scenario_propagation_raw is not None:
        scenario_propagation = _ensure_mapping(
            f"scenario '{args.scenario}'.propagation", scenario_propagation_raw
        )
        merged_propagation = dict(propagation_cfg)
        merged_propagation.update(scenario_propagation)
        propagation_cfg = merged_propagation
    clusters_cfg = _ensure_mapping("clusters", scenario_cfg.get("clusters", {}))

    names, shares, targets, limits_payload = _parse_clusters(clusters_cfg)

    period = float(scenario_cfg.get("period", 600.0))
    num_nodes = int(scenario_cfg.get("N", 120))
    duration = _scenario_duration(period, args.duration)

    snir_override = _parse_snir_override(scenario_cfg.get("snir_enabled"), source="scénario")
    snir_override_cli = _parse_snir_override(args.snir, source="CLI")
    if snir_override_cli is not None:
        snir_override = snir_override_cli

    multichannel, channel_map, phy_profile = _build_channels(
        channels_cfg, propagation_cfg, snir_override
    )

    simulator = Simulator(
        num_nodes=num_nodes,
        num_gateways=1,
        area_size=AREA_SIZE_M,
        transmission_mode="Random",
        packet_interval=period,
        first_packet_interval=period,
        packets_to_send=0,
        adr_node=False,
        adr_server=False,
        duty_cycle=float(common_cfg.get("duty_cycle", 0.01)),
        mobility=False,
        channels=multichannel,
        channel_distribution="round-robin",
        payload_size_bytes=int(common_cfg.get("payload_size_bytes", 20)),
        seed=int(args.seed),
        capture_mode="advanced",
        phy_model="omnet",
        pure_poisson_mode=True,
    )
    setattr(simulator, "capture_delta_db", float(propagation_cfg.get("capture_threshold_db", 1.0)))
    simulator.use_snir = phy_profile.use_snir
    simulator.phy_profile = phy_profile.name
    simulator.snir_model = phy_profile.snir_model
    simulator.interference_model = phy_profile.interference_model
    simulator.flora_capture = phy_profile.flora_capture
    gw_x, gw_y, gw_z = _prepare_gateway(simulator, gateway_cfg)

    manager = ScenarioQoSManager()
    arrival_rate = 1.0 / period if period > 0.0 else 0.0
    manager.configure_clusters(
        len(names),
        proportions=shares,
        arrival_rates=[arrival_rate] * len(names),
        pdr_targets=targets,
    )
    manager.set_channel_mapping(channel_map)
    _override_limits(manager, names, limits_payload, channel_map)

    _apply_method(simulator, manager, args.method)

    simulator.run(max_time=duration)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_payload = {
        "gateway": {"x": gw_x, "y": gw_y, "z": gw_z},
        "phy_profile": phy_profile.name,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    cluster_lookup = {cluster.cluster_id: names[index] for index, cluster in enumerate(manager.clusters)}

    packets_df = _collect_packets(simulator, cluster_lookup, phy_profile)
    if packets_df is not None:
        packets_df.to_csv(out_dir / "packets.csv", index=False)

    nodes_df = _collect_nodes(simulator, cluster_lookup)
    if nodes_df is not None:
        nodes_df.to_csv(out_dir / "nodes.csv", index=False)

    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
