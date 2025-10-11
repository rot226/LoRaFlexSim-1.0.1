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
    def __init__(self):
        self.frequency_hz = 868e6
        self.path_loss_exp = 2.0
        self.bandwidth = 125000.0
        self.frontend_filter_bw = 125000.0
        self.receiver_noise_floor_dBm = -174.0
        self.noise_figure_dB = 6.0


class DummyNode:
    def __init__(self, node_id, x, y, tx_power):
        self.id = node_id
        self.x = x
        self.y = y
        self.tx_power = tx_power
        self.sf = 7


class DummyGateway:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class DummySimulator:
    REQUIRED_SNR = Simulator.REQUIRED_SNR

    def __init__(self, nodes, gateways, channel):
        self.nodes = nodes
        self.gateways = gateways
        self.channel = channel
        self.multichannel = type("MC", (), {"channels": [channel]})()
        self.fixed_tx_power = None


def test_qos_manager_computes_sf_limits_and_accessible_sets():
    channel = DummyChannel()
    nodes = [
        DummyNode(1, 100.0, 0.0, 14.0),
        DummyNode(2, 200.0, 0.0, 14.0),
        DummyNode(3, 250.0, 0.0, 14.0),
        DummyNode(4, 950.0, 0.0, 14.0),
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

    assert simulator.qos_sf_limits == manager.sf_limits
    assert simulator.qos_node_sf_access == manager.node_sf_access
    assert simulator.qos_node_clusters == manager.node_clusters
