import pytest

import math

from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.obstacle_loss import ObstacleLoss


def test_obstacle_loss_reduces_rssi():
    raster = [
        [0, 0, 0],
        [0, 10, 0],
        [0, 0, 0],
    ]
    loss = ObstacleLoss.from_raster(raster, cell_size=1.0, material="concrete")
    ch_clear = Channel(shadowing_std=0.0, fast_fading_std=0.0, time_variation_std=0.0)
    ch_obst = Channel(
        obstacle_loss=loss,
        shadowing_std=0.0,
        fast_fading_std=0.0,
        time_variation_std=0.0,
    )
    tx = (0.0, 1.5)
    rx = (2.0, 1.5)
    dist = math.dist(tx, rx)
    rssi_clear, _ = ch_clear.compute_rssi(14.0, dist, tx_pos=tx, rx_pos=rx)
    rssi_obs, _ = ch_obst.compute_rssi(14.0, dist, tx_pos=tx, rx_pos=rx)
    # Concrete base loss 15 dB + 0.5 * height (10) = 20 dB
    assert rssi_clear - rssi_obs == pytest.approx(20.0, abs=1e-6)
