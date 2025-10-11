"""Tests unitaires pour la configuration des clusters QoS."""

import math

import pytest

from types import SimpleNamespace

from loraflexsim.launcher import qos as qos_module
from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.node import Node
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
        self._recent_pdr = 0.0
        self.arrival_interval_sum = 0.0
        self.arrival_interval_count = 0

    @property
    def recent_pdr(self) -> float:
        return self._recent_pdr

    @recent_pdr.setter
    def recent_pdr(self, value: float) -> None:
        self._recent_pdr = float(value)


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
        self.current_time = 0.0
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
        expected_capacity *= max(0.0, 1.0 - manager.capacity_margin)
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

def test_qos_reconfig_triggers_on_node_change_and_pdr_drift():
    channel = DummyChannel()
    nodes = [
        DummyNode(1, 100.0, 0.0, 14.0, channel),
        DummyNode(2, 150.0, 0.0, 14.0, channel),
    ]
    gateways = [DummyGateway(0.0, 0.0)]
    simulator = DummySimulator(nodes, gateways, channel)

    manager = QoSManager()
    manager.configure_clusters(
        1,
        proportions=[1.0],
        arrival_rates=[0.1],
        pdr_targets=[0.9],
    )
    manager.reconfig_interval_s = 100.0
    manager.pdr_drift_threshold = 0.05

    simulator.current_time = 0.0
    manager.apply(simulator, "MixRA-Opt")
    initial_time = manager._last_reconfig_time
    assert initial_time is not None

    simulator.current_time = initial_time + 10.0
    manager.apply(simulator, "MixRA-Opt")
    assert manager._last_reconfig_time == initial_time

    simulator.nodes.append(DummyNode(3, 200.0, 0.0, 14.0, channel))
    simulator.current_time = initial_time + 20.0
    manager.apply(simulator, "MixRA-Opt")
    node_change_time = manager._last_reconfig_time
    assert node_change_time is not None and node_change_time > initial_time

    manager.reconfig_interval_s = 1000.0
    simulator.nodes[0].recent_pdr = manager.pdr_drift_threshold + 0.1
    simulator.current_time = node_change_time + 1.0
    manager.apply(simulator, "MixRA-Opt")
    drift_time = manager._last_reconfig_time
    assert drift_time is not None and drift_time > node_change_time


def test_qos_reconfig_triggers_on_traffic_variation():
    channel = DummyChannel()
    nodes = [
        DummyNode(1, 100.0, 0.0, 14.0, channel),
        DummyNode(2, 150.0, 0.0, 14.0, channel),
    ]
    for node in nodes:
        node.arrival_interval_sum = 100.0
        node.arrival_interval_count = 10
    gateways = [DummyGateway(0.0, 0.0)]
    simulator = DummySimulator(nodes, gateways, channel)

    manager = QoSManager()
    manager.configure_clusters(
        1,
        proportions=[1.0],
        arrival_rates=[0.1],
        pdr_targets=[0.9],
    )
    manager.reconfig_interval_s = 1000.0
    manager.traffic_drift_threshold = 0.1

    simulator.current_time = 0.0
    manager.apply(simulator, "MixRA-Opt")
    initial_time = manager._last_reconfig_time
    assert initial_time is not None

    for node in simulator.nodes:
        node.arrival_interval_sum = 200.0
        node.arrival_interval_count = 20
    simulator.current_time = initial_time + 5.0
    manager.apply(simulator, "MixRA-Opt")
    assert manager._last_reconfig_time == initial_time

    for node in simulator.nodes:
        node.arrival_interval_sum = 120.0
        node.arrival_interval_count = 30
    simulator.current_time = initial_time + 10.0
    manager.apply(simulator, "MixRA-Opt")
    traffic_time = manager._last_reconfig_time
    assert traffic_time is not None and traffic_time > initial_time


def test_update_qos_context_records_recent_metrics():
    channel = DummyChannel()
    nodes = [
        DummyNode(1, 100.0, 0.0, 14.0, channel),
        DummyNode(2, 200.0, 0.0, 14.0, channel),
    ]
    for index, node in enumerate(nodes, start=1):
        node.recent_pdr = 0.2 * index
        node.arrival_interval_sum = 50.0 * index
        node.arrival_interval_count = 5.0 * index
    gateways = [DummyGateway(0.0, 0.0)]
    simulator = DummySimulator(nodes, gateways, channel)

    manager = QoSManager()
    manager.configure_clusters(
        1,
        proportions=[1.0],
        arrival_rates=[0.2],
        pdr_targets=[0.95],
    )

    simulator.current_time = 42.0
    manager.apply(simulator, "MixRA-Opt")

    assert manager._last_reconfig_time == pytest.approx(42.0)
    for node in nodes:
        node_id = node.id
        assert manager._last_recent_pdr[node_id] == pytest.approx(node.recent_pdr)
        expected_rate = node.arrival_interval_count / node.arrival_interval_sum
        assert manager._last_arrival_rates[node_id] == pytest.approx(expected_rate)

def test_qos_manager_compute_sf_airtimes_follows_channel_airtime():
    channel = DummyChannel()
    simulator = DummySimulator([], [DummyGateway(0.0, 0.0)], channel)
    manager = QoSManager()

    sfs = [7, 9, 12]
    airtimes = manager._compute_sf_airtimes(simulator, sfs)

    assert set(airtimes) == set(sfs)
    for sf in sfs:
        expected = channel.airtime(sf, simulator.payload_size_bytes)
        assert airtimes[sf] == pytest.approx(expected)


def test_qos_manager_compute_sf_airtimes_handles_invalid_channel():
    class FaultyChannel(DummyChannel):
        def airtime(self, sf: int, payload_size: int = 20) -> float:  # pragma: no cover - simple override
            raise RuntimeError("airtime failure")

    simulator = SimpleNamespace(
        channel=FaultyChannel(), payload_size_bytes=20, multichannel=None
    )
    manager = QoSManager()

    result = manager._compute_sf_airtimes(simulator, [7, 8])

    assert result == {7: 0.0, 8: 0.0}


def test_qos_manager_compute_offered_traffic_aggregates_by_channel():
    manager = QoSManager()
    cluster_a = Cluster(cluster_id=1, device_share=0.5, arrival_rate=0.2, pdr_target=0.95)
    cluster_b = Cluster(cluster_id=2, device_share=0.5, arrival_rate=0.4, pdr_target=0.9)
    manager.clusters = [cluster_a, cluster_b]

    channel0 = DummyChannel(channel_index=0)
    channel1 = DummyChannel(channel_index=1)
    node_a = DummyNode(1, 0.0, 0.0, 14.0, channel0)
    node_b = DummyNode(2, 0.0, 0.0, 14.0, channel1)
    node_c = DummyNode(3, 0.0, 0.0, 14.0, channel0)

    assignments = {node_a: cluster_a, node_b: cluster_a, node_c: cluster_b}
    node_sf_access = {1: [7, 8], 2: [7], 3: []}
    airtimes = {7: 0.5, 8: 0.25}

    offered = manager._compute_offered_traffic(assignments, node_sf_access, airtimes)

    assert offered[1][7][0] == pytest.approx(cluster_a.arrival_rate * airtimes[7])
    assert offered[1][7][1] == pytest.approx(cluster_a.arrival_rate * airtimes[7])
    assert offered[1][8] == {}
    assert offered[2][7] == {}
    assert offered[2][8] == {}


@pytest.mark.parametrize(
    "pdr, delta",
    [
        (0.9, 0.0),
        (0.99, 0.2),
        (0.5, 1.5),
        (0.999999, 0.0),
    ],
)
def test_capacity_from_pdr_stays_non_negative(pdr, delta):
    capacity = QoSManager._capacity_from_pdr(pdr, delta)
    assert capacity >= 0.0


def test_lambertw_neg1_properties():
    x = -0.2
    w = QoSManager._lambertw_neg1(x)
    assert w <= -1.0
    assert math.isclose(w * math.exp(w), x, rel_tol=1e-12, abs_tol=1e-12)

    limit = -1.0 / math.e
    assert QoSManager._lambertw_neg1(limit) == pytest.approx(-1.0)

    with pytest.raises(ValueError):
        QoSManager._lambertw_neg1(0.0)


def test_simulator_metrics_expose_qos_statistics():
    sim = Simulator.__new__(Simulator)
    sim.tx_attempted = 0
    sim.rx_delivered = 0
    sim.total_delay = 0.0
    sim.delivered_count = 0
    sim.current_time = 100.0
    sim.packets_delivered = 0
    sim.payload_size_bytes = 20
    sim.nodes = []
    sim.gateways = []
    sim.network_server = SimpleNamespace(duplicate_packets=0, event_gateway={})
    sim.packets_lost_collision = 0
    sim.total_energy_J = 0.0
    sim.energy_nodes_J = 0.0
    sim.energy_gateways_J = 0.0
    sim.retransmissions = 0
    sim.events_log = []
    sim._events_log_map = {}
    sim.warm_up_intervals = 0
    sim.dump_intervals = False

    channel0 = Channel()
    channel0.channel_index = 0
    channel1 = Channel()
    channel1.channel_index = 1

    node1 = Node(1, 0.0, 0.0, 7, 14.0, channel=channel0)
    node1.tx_attempted = 10
    node1.rx_delivered = 9
    node1.qos_cluster_id = 1

    node2 = Node(2, 0.0, 0.0, 8, 14.0, channel=channel1)
    node2.tx_attempted = 5
    node2.rx_delivered = 4
    node2.qos_cluster_id = 2

    node3 = Node(3, 0.0, 0.0, 9, 14.0, channel=channel0)
    node3.tx_attempted = 8
    node3.rx_delivered = 6
    node3.qos_cluster_id = 1

    sim.nodes = [node1, node2, node3]
    for node in sim.nodes:
        node.interval_log = []
        node.arrival_interval_sum = 0.0
        node.arrival_interval_count = 0
    sim.tx_attempted = sum(node.tx_attempted for node in sim.nodes)
    sim.rx_delivered = sum(node.rx_delivered for node in sim.nodes)
    sim.packets_delivered = sim.rx_delivered
    sim.delivered_count = sim.rx_delivered

    sim.qos_clusters_config = {
        1: {"arrival_rate": 0.2, "pdr_target": 0.95, "device_share": 0.6},
        2: {"arrival_rate": 0.1, "pdr_target": 0.85, "device_share": 0.4},
    }
    sim.qos_node_clusters = {1: 1, 2: 2, 3: 1}

    metrics = sim.get_metrics()

    cluster_pdr = metrics["qos_cluster_pdr"]
    assert cluster_pdr[1] == pytest.approx(15 / 18)
    assert cluster_pdr[2] == pytest.approx(4 / 5)

    targets = metrics["qos_cluster_targets"]
    assert targets == {1: 0.95, 2: 0.85}

    gaps = metrics["qos_cluster_pdr_gap"]
    assert gaps[1] == pytest.approx((15 / 18) - 0.95)
    assert gaps[2] == pytest.approx((4 / 5) - 0.85)

    throughputs = metrics["qos_cluster_throughput_bps"]
    payload_bits = sim.payload_size_bytes * 8
    expected_throughput1 = 15 * payload_bits / sim.current_time
    expected_throughput2 = 4 * payload_bits / sim.current_time
    assert throughputs[1] == pytest.approx(expected_throughput1)
    assert throughputs[2] == pytest.approx(expected_throughput2)

    gini_expected = (
        sum(
            abs(a - b)
            for a in (expected_throughput1, expected_throughput2)
            for b in (expected_throughput1, expected_throughput2)
        )
        / (2 * 2 * (expected_throughput1 + expected_throughput2))
    )
    assert metrics["qos_throughput_gini"] == pytest.approx(gini_expected)

    counts = metrics["qos_cluster_node_counts"]
    assert counts == {1: 2, 2: 1}

    sf_channel = metrics["qos_cluster_sf_channel"]
    assert sf_channel[1][7][sim.channel_index(channel0)] == 1
    assert sf_channel[1][9][sim.channel_index(channel0)] == 1
    assert sf_channel[2][8][sim.channel_index(channel1)] == 1


def test_simulator_metrics_empty_qos_when_disabled():
    sim = Simulator.__new__(Simulator)
    sim.tx_attempted = 0
    sim.rx_delivered = 0
    sim.total_delay = 0.0
    sim.delivered_count = 0
    sim.current_time = 0.0
    sim.packets_delivered = 0
    sim.payload_size_bytes = 20
    sim.nodes = []
    sim.gateways = []
    sim.network_server = SimpleNamespace(duplicate_packets=0, event_gateway={})
    sim.packets_lost_collision = 0
    sim.total_energy_J = 0.0
    sim.energy_nodes_J = 0.0
    sim.energy_gateways_J = 0.0
    sim.retransmissions = 0
    sim.events_log = []
    sim._events_log_map = {}
    sim.warm_up_intervals = 0
    sim.dump_intervals = False
    sim.qos_clusters_config = {}
    sim.qos_node_clusters = {}

    metrics = sim.get_metrics()

    assert metrics["qos_cluster_throughput_bps"] == {}
    assert metrics["qos_cluster_pdr"] == {}
    assert metrics["qos_cluster_targets"] == {}
    assert metrics["qos_cluster_node_counts"] == {}
    assert metrics["qos_cluster_pdr_gap"] == {}
    assert metrics["qos_cluster_sf_channel"] == {}
    assert metrics["qos_throughput_gini"] == 0.0


def test_compute_limit_guard_clamps_invalid_inputs():
    compute = qos_module.QoSManager._compute_limit

    assert compute(-1.0, 1.0, 1.0, 1.0, 2.0, 1.0) == 0.0
    assert compute(1.0, 0.0, 1.0, 1.0, 2.0, 1.0) == 0.0
    assert compute(1.0, 1.0, 0.0, 1.0, 2.0, 1.0) == 0.0
    assert compute(1.0, 1.0, 1.0, 0.0, 2.0, 1.0) == 0.0
    assert compute(1.0, 1.0, 1.0, 1.0, -5.0, 1.0) == 0.0
    assert compute(1.0, 1.0, 1.0, 1.0, 2.0, -1.0) == 0.0


def test_build_d_matrix_counts_minimum_sf_only():
    manager = QoSManager()
    manager.clusters = [
        Cluster(cluster_id=1, device_share=0.5, arrival_rate=0.1, pdr_target=0.95),
        Cluster(cluster_id=2, device_share=0.5, arrival_rate=0.1, pdr_target=0.9),
    ]

    class Dummy:
        pass

    node_a = Dummy()
    node_a.id = 10
    node_b = Dummy()
    node_b.id = 11
    node_c = Dummy()
    node_c.id = 12

    assignments = {node_a: manager.clusters[0], node_b: manager.clusters[1], node_c: manager.clusters[1]}
    node_sf_access = {10: [7, 8, 9], 11: [9], 12: []}
    sfs = [7, 8, 9]

    matrix = manager._build_d_matrix(assignments, node_sf_access, sfs)

    assert matrix == {
        1: {7: 1, 8: 0, 9: 0},
        2: {7: 0, 8: 0, 9: 1},
    }
