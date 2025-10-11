"""Tests unitaires pour la configuration des clusters QoS."""

import pytest

from loraflexsim.launcher.qos import Cluster, QoSManager, build_clusters


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
