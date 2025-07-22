import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.advanced_channel import AdvancedChannel  # noqa: E402
import random  # noqa: E402


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


def test_variable_noise_changes_snr():
    random.seed(0)
    adv = AdvancedChannel(fading="", shadowing_std=0, variable_noise_std=5.0)
    _, snr1 = adv.compute_rssi(14.0, 100.0)
    _, snr2 = adv.compute_rssi(14.0, 100.0)
    assert snr1 != snr2


def test_time_varying_offsets():
    random.seed(0)
    adv = AdvancedChannel(
        fading="",
        shadowing_std=0,
        frequency_offset_hz=1000.0,
        freq_offset_std_hz=50.0,
        sync_offset_s=0.001,
        sync_offset_std_s=0.0001,
    )
    _, snr1 = adv.compute_rssi(14.0, 100.0)
    _, snr2 = adv.compute_rssi(14.0, 100.0)
    assert snr1 != snr2


def test_obstacle_map_extra_loss():
    obstacle = [
        [0.0, 0.0],
        [0.0, 10.0],
    ]
    adv = AdvancedChannel(obstacle_map=obstacle, map_area_size=100.0, fading="")
    r_clear, _ = adv.compute_rssi(
        14.0,
        80.0,
        tx_pos=(10.0, 90.0),
        rx_pos=(10.0, 10.0),
    )
    r_obst, _ = adv.compute_rssi(
        14.0,
        80.0,
        tx_pos=(10.0, 90.0),
        rx_pos=(90.0, 90.0),
    )
    assert r_obst < r_clear


def test_obstacle_height_blocks_link():
    obstacle_h = [
        [0.0, 0.0],
        [5.0, 0.0],
    ]
    obstacle = [
        [0.0, 0.0],
        [-1.0, 0.0],
    ]
    adv = AdvancedChannel(
        obstacle_height_map=obstacle_h,
        obstacle_map=obstacle,
        map_area_size=100.0,
        fading="",
    )
    r, s = adv.compute_rssi(
        14.0,
        80.0,
        tx_pos=(10.0, 90.0, 0.0),
        rx_pos=(90.0, 90.0, 0.0),
    )
    assert r == -float("inf")


def test_3d_compute_rssi_uses_altitude():
    adv = AdvancedChannel(propagation_model="3d", fading="", shadowing_std=0)
    r1, _ = adv.compute_rssi(14.0, 100.0, tx_pos=(0.0, 0.0, 0.0), rx_pos=(0.0, 100.0, 0.0))
    r2, _ = adv.compute_rssi(14.0, 100.0, tx_pos=(0.0, 0.0, 10.0), rx_pos=(0.0, 100.0, 0.0))
    assert r2 < r1


def test_device_specific_offset():
    random.seed(0)
    adv = AdvancedChannel(
        fading="",
        shadowing_std=0,
        dev_frequency_offset_hz=500.0,
        dev_freq_offset_std_hz=50.0,
    )
    _, snr1 = adv.compute_rssi(14.0, 100.0)
    _, snr2 = adv.compute_rssi(14.0, 100.0)
    assert snr1 != snr2


def test_temperature_dependent_noise():
    random.seed(0)
    adv = AdvancedChannel(fading="", shadowing_std=0, temperature_std_K=30.0)
    _, snr1 = adv.compute_rssi(14.0, 100.0)
    _, snr2 = adv.compute_rssi(14.0, 100.0)
    assert snr1 != snr2


def test_humidity_dependent_noise():
    random.seed(0)
    adv = AdvancedChannel(
        fading="",
        shadowing_std=0,
        humidity_std_percent=10.0,
        humidity_noise_coeff_dB=1.0,
    )
    _, snr1 = adv.compute_rssi(14.0, 100.0)
    _, snr2 = adv.compute_rssi(14.0, 100.0)
    assert snr1 != snr2


def test_pa_non_linearity():
    random.seed(0)
    adv = AdvancedChannel(
        fading="",
        shadowing_std=0,
        pa_non_linearity_std_dB=1.0,
    )
    r1, _ = adv.compute_rssi(14.0, 100.0)
    r2, _ = adv.compute_rssi(14.0, 100.0)
    assert r1 != r2


def test_phase_noise_penalizes_snr():
    random.seed(0)
    adv = AdvancedChannel(fading="", shadowing_std=0, phase_noise_std_dB=2.0)
    _, snr1 = adv.compute_rssi(14.0, 100.0)
    _, snr2 = adv.compute_rssi(14.0, 100.0)
    assert snr1 != snr2


def test_dynamic_weather_variation():
    random.seed(0)
    adv = AdvancedChannel(
        fading="",
        shadowing_std=0,
        weather_loss_dB_per_km=1.0,
        weather_loss_std_dB_per_km=0.5,
    )
    r1, _ = adv.compute_rssi(14.0, 1000.0)
    r2, _ = adv.compute_rssi(14.0, 1000.0)
    assert r1 != r2


def test_pa_nonlinearity_curve():
    adv1 = AdvancedChannel(fading="", shadowing_std=0)
    adv2 = AdvancedChannel(
        fading="",
        shadowing_std=0,
        pa_non_linearity_curve=(-0.01, 0.1, 0.0),
    )
    r1, _ = adv1.compute_rssi(20.0, 50.0)
    r2, _ = adv2.compute_rssi(20.0, 50.0)
    assert r1 != r2

def test_interference_penalty_inf():
    adv = AdvancedChannel(fading="", shadowing_std=0)
    freq_offset = adv.base.bandwidth / 2
    symbol_time = (2 ** 7) / adv.base.bandwidth
    penalty = adv._interference_penalty_db(freq_offset, symbol_time, 7)
    assert penalty == float("inf")


def test_clock_jitter_variation():
    random.seed(0)
    adv = AdvancedChannel(
        fading="",
        shadowing_std=0,
        clock_jitter_std_s=0.0005,
    )
    _, snr1 = adv.compute_rssi(14.0, 100.0)
    _, snr2 = adv.compute_rssi(14.0, 100.0)
    assert snr1 != snr2


def test_pa_distortion_affects_rssi():
    random.seed(0)
    adv = AdvancedChannel(
        fading="",
        shadowing_std=0,
        pa_distortion_std_dB=1.0,
    )
    r1, _ = adv.compute_rssi(14.0, 100.0)
    r2, _ = adv.compute_rssi(14.0, 100.0)
    assert r1 != r2
