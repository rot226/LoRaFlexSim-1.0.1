import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.advanced_channel import AdvancedChannel  # noqa: E402


def test_cost231_path_loss_vs_log_distance():
    adv = AdvancedChannel()
    pl = adv.path_loss(1000)
    assert pl > 100


def test_okumura_hata_and_weather():
    adv = AdvancedChannel(propagation_model="okumura_hata", weather_loss_dB_per_km=1.0)
    base = AdvancedChannel(propagation_model="okumura_hata")
    pl_weather = adv.path_loss(2000)
    pl_base = base.path_loss(2000)
    assert pl_weather > pl_base


def test_3d_path_loss_includes_height():
    adv = AdvancedChannel(propagation_model="3d", base_station_height=50.0, mobile_height=1.0)
    base_loss = adv.base.path_loss(1000)
    adv_loss = adv.path_loss(1000)
    assert adv_loss > base_loss


def test_rayleigh_fading_variability():
    adv = AdvancedChannel(fading="rayleigh")
    r1, _ = adv.compute_rssi(14.0, 100.0)
    r2, _ = adv.compute_rssi(14.0, 100.0)
    assert r1 != r2


def test_rician_fading_variability():
    adv = AdvancedChannel(fading="rician", rician_k=2.0)
    r1, _ = adv.compute_rssi(14.0, 100.0)
    r2, _ = adv.compute_rssi(14.0, 100.0)
    assert r1 != r2
