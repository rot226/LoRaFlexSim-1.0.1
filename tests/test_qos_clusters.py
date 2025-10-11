"""Tests unitaires pour la configuration des clusters QoS."""

import math

import pytest

from loraflexsim.launcher import qos as qos_module
from loraflexsim.launcher.qos import Cluster, QoSManager, build_clusters
from loraflexsim.launcher.simulator import Simulator


def test_build_clusters_returns_normalised_instances():
    clusters = build_clusters(
        2,
        proportions=[0.6, 0.4],
        arrival_rates=[0.1, 0.3],
        pdr_targets=[0.95, 0.85],
    )
    assert clusters == [
        Cluster(cluster_id=1, device_share=0.6, arrival_rate=0.1, pdr_target=0.95),
        Cluster(cluster_id=2, device_share=0.4, arrival_rate=0.3, pdr_target=0.85),
    ]


def test_build_clusters_validates_lengths_and_values():
    with pytest.raises(ValueError):
        build_clusters(
            2,
            proportions=[0.5, 0.5],
            arrival_rates=[0.1],
            pdr_targets=[0.9, 0.8],
        )
    with pytest.raises(ValueError):
        build_clusters(
            2,
            proportions=[0.3, 0.3],
            arrival_rates=[0.1, 0.2],
            pdr_targets=[0.9, 0.8],
        )
    with pytest.raises(ValueError):
        build_clusters(
            1,
            proportions=[1.0],
            arrival_rates=[0.0],
            pdr_targets=[0.9],
        )
    with pytest.raises(ValueError):
        build_clusters(
            1,
            proportions=[1.0],
            arrival_rates=[0.1],
            pdr_targets=[0.0],
        )


def test_qos_manager_configure_clusters():
    manager = QoSManager()
    configured = manager.configure_clusters(
        3,
        proportions=[0.5, 0.3, 0.2],
        arrival_rates=[0.2, 0.4, 0.8],
        pdr_targets=[0.9, 0.92, 0.85],
    )
    assert configured == manager.clusters
    assert [cluster.cluster_id for cluster in configured] == [1, 2, 3]


class DummyChannel:
    def __init__(self, channel_index=0):
        self.frequency_hz = 868e6
        self.path_loss_exp = 2.0
        self.bandwidth = 125000.0
        self.frontend_filter_bw = 125000.0
        self.receiver_noise_floor_dBm = -174.0
        self.noise_figure_dB = 6.0
        self.channel_index = channel_index
        self.low_data_rate_threshold = 11
        self.coding_rate = 1
        self.preamble_symbols = 8

    def airtime(self, sf: int, payload_size: int = 20) -> float:
        rs = self.bandwidth / (2 ** sf)
        ts = 1.0 / rs
        de = 1 if sf >= self.low_data_rate_threshold else 0
        cr_denom = self.coding_rate + 4
        numerator = 8 * payload_size - 4 * sf + 28 + 16
        denominator = 4 * (sf - 2 * de)
        n_payload = max(math.ceil(numerator / denominator), 0) * cr_denom + 8
        t_preamble = (self.preamble_symbols + 4.25) * ts
        t_payload = n_payload * ts
        return t_preamble + t_payload


class DummyNode:
    def __init__(self, node_id, x, y, tx_power, channel):
        self.id = node_id
        self.x = x
        self.y = y
        self.tx_power = tx_power
        self.sf = 7
        self.channel = channel
        self.tx_attempted = 0
        self.rx_delivered = 0
        self.history: list[dict] = []

    @property
    def pdr(self) -> float:
        return self.rx_delivered / self.tx_attempted if self.tx_attempted > 0 else 0.0

    @property
    def recent_pdr(self) -> float:
        if not self.history:
            return 0.0
        success = sum(1 for entry in self.history if entry.get("delivered"))
        return success / len(self.history)


class DummyGateway:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class DummySimulator:
    REQUIRED_SNR = Simulator.REQUIRED_SNR

    def __init__(self, nodes, gateways, channel, *, extra_channels=None, duty_cycle=None):
        self.nodes = nodes
        self.gateways = gateways
        channels = [channel]
        if extra_channels:
            channels.extend(extra_channels)
        self.channel = channels[0]
        self.multichannel = type("MC", (), {"channels": channels})()
        self.fixed_tx_power = None
        self.payload_size_bytes = 20
        if duty_cycle is not None:
            self.duty_cycle_manager = type("Duty", (), {"duty_cycle": float(duty_cycle)})()
        else:
            self.duty_cycle_manager = None


def test_qos_manager_computes_sf_limits_and_accessible_sets():
    channel = DummyChannel()
    nodes = [
        DummyNode(1, 100.0, 0.0, 14.0, channel),
        DummyNode(2, 200.0, 0.0, 14.0, channel),
        DummyNode(3, 250.0, 0.0, 14.0, channel),
        DummyNode(4, 950.0, 0.0, 14.0, channel),
    ]
    gateways = [DummyGateway(0.0, 0.0)]
    simulator = DummySimulator(nodes, gateways, channel)

    manager = QoSManager()
    manager.configure_clusters(
        2,
        proportions=[0.5, 0.5],
        arrival_rates=[0.1, 0.1],
        pdr_targets=[0.95, 0.999999],
    )

    original_powers = {node.id: node.tx_power for node in nodes}
    manager.apply(simulator, "MixRA-Opt")

    assert manager.sf_limits
    assert set(manager.sf_limits) == {1, 2}

    noise_dbm = channel.receiver_noise_floor_dBm + 10 * math.log10(channel.bandwidth) + channel.noise_figure_dB
    noise_w = qos_module.QoSManager._dbm_to_w(noise_dbm)
    alpha = channel.path_loss_exp
    wavelength = qos_module.QoSManager.SPEED_OF_LIGHT / channel.frequency_hz
    factor = wavelength / (4.0 * math.pi)
    snr_table = {sf: 10 ** (snr / 10.0) for sf, snr in simulator.REQUIRED_SNR.items()}
    ref_tx_w = qos_module.QoSManager._dbm_to_w(14.0)

    cluster1 = manager.clusters[0]
    base1 = -math.log(cluster1.pdr_target)
    expected_sf7_cluster1 = qos_module.QoSManager._compute_limit(
        base1,
        ref_tx_w,
        noise_w,
        snr_table[7],
        alpha,
        factor,
    )
    assert manager.sf_limits[cluster1.cluster_id][7] == pytest.approx(expected_sf7_cluster1, rel=1e-6)

    cluster2 = manager.clusters[1]
    base2 = -math.log(cluster2.pdr_target)
    expected_sf12_cluster2 = qos_module.QoSManager._compute_limit(
        base2,
        ref_tx_w,
        noise_w,
        snr_table[12],
        alpha,
        factor,
    )
    assert manager.sf_limits[cluster2.cluster_id][12] == pytest.approx(expected_sf12_cluster2, rel=1e-6)

    distances = [math.hypot(node.x, node.y) for node in nodes]
    expected_access = {}
    for node, distance in zip(nodes, distances):
        cluster = manager.node_clusters[node.id]
        base = -math.log(manager.clusters[cluster - 1].pdr_target)
        node_tx_w = qos_module.QoSManager._dbm_to_w(original_powers[node.id])
        accessible = []
        for sf in sorted(snr_table):
            limit = qos_module.QoSManager._compute_limit(
                base,
                node_tx_w,
                noise_w,
                snr_table[sf],
                alpha,
                factor,
            )
            if limit > 0 and distance <= limit:
                accessible.append(sf)
        expected_access[node.id] = accessible

    for node in nodes:
        assert manager.node_sf_access[node.id] == expected_access[node.id]
        assert node.qos_accessible_sf == expected_access[node.id]
        assert node.qos_cluster_id == manager.node_clusters[node.id]
        if expected_access[node.id]:
            assert node.qos_min_sf == expected_access[node.id][0]
        else:
            assert node.qos_min_sf is None

    assert simulator.qos_sf_limits == manager.sf_limits
    assert simulator.qos_node_sf_access == manager.node_sf_access
    assert simulator.qos_node_clusters == manager.node_clusters
    expected_d = {
        cluster.cluster_id: {sf: 0 for sf in sorted(simulator.REQUIRED_SNR)}
        for cluster in manager.clusters
    }
    for node in nodes:
        cluster_id = manager.node_clusters[node.id]
        access = expected_access[node.id]
        if not access:
            continue
        expected_d[cluster_id][access[0]] += 1
    assert manager.cluster_d_matrix == expected_d
    assert simulator.qos_d_matrix == expected_d

    # --- Vérification du trafic offert et des capacités -------------------
    airtimes = {
        sf: channel.airtime(sf, simulator.payload_size_bytes)
        for sf in sorted(simulator.REQUIRED_SNR)
    }
    expected_offered: dict[int, dict[int, dict[int, float]]] = {
        cluster.cluster_id: {sf: {} for sf in airtimes}
        for cluster in manager.clusters
    }
    for node in nodes:
        cluster_id = manager.node_clusters[node.id]
        sf = node.qos_min_sf
        if sf is None:
            continue
        lam = manager.clusters[cluster_id - 1].arrival_rate
        expected_offered[cluster_id].setdefault(sf, {})
        expected_offered[cluster_id][sf].setdefault(node.channel.channel_index, 0.0)
        expected_offered[cluster_id][sf][node.channel.channel_index] += lam * airtimes[sf]

    for cluster in manager.clusters:
        cluster_id = cluster.cluster_id
        for sf, channels in expected_offered[cluster_id].items():
            for chan_idx, value in channels.items():
                assert manager.cluster_offered_traffic[cluster_id][sf][chan_idx] == pytest.approx(value)

    totals = {
        cluster_id: sum(sum(channels.values()) for channels in sf_map.values())
        for cluster_id, sf_map in expected_offered.items()
    }
    assert manager.cluster_offered_totals == pytest.approx(totals)
    total_load = sum(totals.values())
    for cluster in manager.clusters:
        cluster_id = cluster.cluster_id
        interference = max(total_load - totals.get(cluster_id, 0.0), 0.0)
        expected_capacity = qos_module.QoSManager._capacity_from_pdr(
            cluster.pdr_target, interference
        )
        margin_factor = 1.0 - manager.capacity_margin
        if expected_capacity > 0.0:
            expected_capacity *= margin_factor
        assert manager.cluster_capacity_limits[cluster_id] == pytest.approx(expected_capacity)
        assert simulator.qos_capacity_limits[cluster_id] == pytest.approx(expected_capacity)

    assert simulator.qos_offered_traffic == manager.cluster_offered_traffic
    assert simulator.qos_offered_totals == manager.cluster_offered_totals
    assert simulator.qos_interference == manager.cluster_interference


def test_mixra_opt_respects_duty_cycle_and_capacity():
    channel0 = DummyChannel(channel_index=0)
    channel1 = DummyChannel(channel_index=1)
    nodes = [
        DummyNode(1, 80.0, 0.0, 14.0, channel0),
        DummyNode(2, 120.0, 0.0, 14.0, channel0),
        DummyNode(3, 180.0, 0.0, 14.0, channel0),
        DummyNode(4, 220.0, 0.0, 14.0, channel0),
        DummyNode(5, 260.0, 0.0, 14.0, channel0),
        DummyNode(6, 300.0, 0.0, 14.0, channel0),
    ]
    gateways = [DummyGateway(0.0, 0.0)]
    simulator = DummySimulator(
        nodes,
        gateways,
        channel0,
        extra_channels=[channel1],
        duty_cycle=0.01,
    )

    manager = QoSManager()
    manager.configure_clusters(
        1,
        proportions=[1.0],
        arrival_rates=[0.05],
        pdr_targets=[0.9],
    )

    manager.apply(simulator, "MixRA-Opt")

    # Vérifie que les SF attribués sont compatibles avec les ensembles accessibles
    for node in nodes:
        access = manager.node_sf_access.get(node.id, [])
        if access:
            assert node.sf in access

    cluster = manager.clusters[0]
    cluster_id = cluster.cluster_id
    airtimes = manager.sf_airtimes
    duty_cycle = simulator.duty_cycle_manager.duty_cycle
    channel_loads: dict[int, float] = {}
    for node in nodes:
        if manager.node_clusters.get(node.id) != cluster_id:
            continue
        sf = node.sf
        tau = airtimes.get(sf, 0.0)
        channel_index = getattr(getattr(node, "channel", None), "channel_index", 0)
        channel_loads[channel_index] = channel_loads.get(channel_index, 0.0) + cluster.arrival_rate * tau

    # Le trafic est réparti sur plusieurs canaux et respecte le duty-cycle
    assert len(channel_loads) >= 2
    for load in channel_loads.values():
        assert load <= duty_cycle + 1e-6

    capacity = manager.cluster_capacity_limits.get(cluster_id)
    if capacity:
        total_load = sum(channel_loads.values())
        assert total_load <= capacity + 1e-6


def test_qos_manager_reconfiguration_policy(monkeypatch):
    channel = DummyChannel()
    nodes = [
        DummyNode(1, 80.0, 0.0, 14.0, channel),
        DummyNode(2, 120.0, 0.0, 14.0, channel),
    ]
    simulator = DummySimulator(nodes, [DummyGateway(0.0, 0.0)], channel)

    manager = QoSManager()
    manager.configure_clusters(
        1,
        proportions=[1.0],
        arrival_rates=[0.05],
        pdr_targets=[0.9],
    )
    manager.min_reconfig_interval = 8.0

    time_sequence = iter([5.0, 15.0, 25.0, 35.0, 45.0])
    monkeypatch.setattr(qos_module.time, "monotonic", lambda: next(time_sequence))

    manager.apply(simulator, "MixRA-Opt")
    first_time = manager._last_reconfig_time
    assert first_time == pytest.approx(5.0)

    for node in simulator.nodes:
        node.tx_attempted = 10
        node.rx_delivered = 9
        node.history = [{"delivered": True}] * 9 + [{"delivered": False}]

    manager.apply(simulator, "MixRA-Opt")
    assert manager._last_reconfig_time == pytest.approx(first_time)

    new_node = DummyNode(3, 160.0, 0.0, 14.0, channel)
    new_node.tx_attempted = 10
    new_node.rx_delivered = 9
    new_node.history = [{"delivered": True}] * 9 + [{"delivered": False}]
    simulator.nodes.append(new_node)

    manager.apply(simulator, "MixRA-Opt")
    second_time = manager._last_reconfig_time
    assert second_time == pytest.approx(15.0)

    for node in simulator.nodes:
        node.tx_attempted = 10
        node.rx_delivered = 2
        node.history = [{"delivered": True}] * 2 + [{"delivered": False}] * 8

    manager.apply(simulator, "MixRA-Opt")
    third_time = manager._last_reconfig_time
    assert third_time == pytest.approx(35.0)
