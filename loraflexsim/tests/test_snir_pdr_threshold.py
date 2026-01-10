"""Vérifie l'impact du SNIR proche de REQUIRED_SNR sur le PDR.

Le seuil REQUIRED_SNR (même table que Channel.SNR_THRESHOLDS) sert de limite
de décodage : un SNIR juste au-dessus du seuil reste soumis à la perte
"marginale", tandis qu'une marge confortable (~+5 dB) maintient un PDR élevé.
"""

from __future__ import annotations

import pytest

from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.config_loader import write_flora_ini
from loraflexsim.launcher.server import REQUIRED_SNR
from loraflexsim.launcher.simulator import Simulator


def _snr_at_distance(channel: Channel, distance_m: float, tx_power_dbm: float, sf: int) -> float:
    _, snr = channel.compute_rssi(tx_power_dbm, distance_m, sf=sf)
    return snr


def _distance_for_target_snr(
    channel: Channel,
    target_snr_db: float,
    tx_power_dbm: float,
    sf: int,
    *,
    max_distance_m: float = 20000.0,
) -> float:
    low = 1.0
    high = 1.0
    while _snr_at_distance(channel, high, tx_power_dbm, sf) > target_snr_db:
        high *= 2.0
        if high >= max_distance_m:
            break
    high = min(high, max_distance_m)
    for _ in range(40):
        mid = 0.5 * (low + high)
        if _snr_at_distance(channel, mid, tx_power_dbm, sf) > target_snr_db:
            low = mid
        else:
            high = mid
    return high


def test_pdr_drop_near_required_snr(tmp_path):
    tx_power_dbm = 14.0
    sf = 7
    required = REQUIRED_SNR[sf]

    channel = Channel(
        shadowing_std=0.0,
        fast_fading_std=0.0,
        time_variation_std=0.0,
        snir_fading_std=0.0,
        noise_floor_std=0.0,
        interference_dB=0.0,
        baseline_loss_rate=0.0,
        baseline_collision_rate=0.0,
        phy_model="omnet",
    )

    limit_target = required + 0.2
    medium_target = required + 5.0

    limit_distance = _distance_for_target_snr(
        channel, limit_target, tx_power_dbm, sf
    )
    medium_distance = _distance_for_target_snr(
        channel, medium_target, tx_power_dbm, sf
    )

    assert medium_distance < limit_distance
    assert _snr_at_distance(channel, limit_distance, tx_power_dbm, sf) == pytest.approx(
        limit_target, abs=0.25
    )
    assert _snr_at_distance(channel, medium_distance, tx_power_dbm, sf) == pytest.approx(
        medium_target, abs=0.25
    )

    cfg_path = tmp_path / "snir_threshold.ini"
    write_flora_ini(
        [
            {"x": limit_distance, "y": 0.0, "sf": sf, "tx_power": tx_power_dbm},
            {"x": medium_distance, "y": 0.0, "sf": sf, "tx_power": tx_power_dbm},
        ],
        [{"x": 0.0, "y": 0.0}],
        cfg_path,
    )

    sim = Simulator(
        num_nodes=2,
        num_gateways=1,
        transmission_mode="Periodic",
        packet_interval=20.0,
        packets_to_send=200,
        duty_cycle=None,
        mobility=False,
        config_file=cfg_path,
        channels=[channel],
        seed=2024,
    )
    sim.run()

    limit_node, medium_node = sim.nodes
    limit_pdr = limit_node.pdr
    medium_pdr = medium_node.pdr

    assert limit_pdr <= medium_pdr * 0.8
