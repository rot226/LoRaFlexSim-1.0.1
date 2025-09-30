import math

import pytest

from loraflexsim.launcher.channel import Channel


def flora_snrs(rssi_list, start_list, end_list, noise_dBm):
    """Reference SNIR calculation mirroring FLoRa's analog model."""

    noise_lin = 10 ** (noise_dBm / 10.0)
    powers = [10 ** (r / 10.0) for r in rssi_list]
    snrs = []

    n = len(rssi_list)
    for i in range(n):
        start_i = start_list[i]
        end_i = end_list[i]
        dur_i = max(end_i - start_i, 1e-9)
        events = {start_i: noise_lin, end_i: -noise_lin}
        for j in range(n):
            if j == i:
                continue
            o_start = max(start_i, start_list[j])
            o_end = min(end_i, end_list[j])
            if o_end <= o_start:
                continue
            p = powers[j]
            events[o_start] = events.get(o_start, 0.0) + p
            events[o_end] = events.get(o_end, 0.0) - p
        level = 0.0
        last = start_i
        energy = 0.0
        for t in sorted(events):
            energy += level * (t - last)
            level += events[t]
            last = t
        avg_noise = energy / dur_i
        snrs.append(10 * math.log10(powers[i] / avg_noise))
    return snrs


def test_overlap_snir_matches_flora():
    # Three partially overlapping transmissions
    rssi = [-90.0, -95.0, -100.0]
    start = [0.0, 0.25, 0.5]
    end = [1.0, 1.25, 0.75]

    ch = Channel(
        phy_model="omnet",
        receiver_noise_floor_dBm=-120.0,
        noise_figure_dB=0.0,
        variable_noise_std=0.0,
        noise_floor_std=0.0,
        fine_fading_std=0.0,
    )
    phy = ch.omnet_phy

    # Use a common deterministic noise value for both models
    noise = phy.noise_floor()

    snrs_flex = phy.compute_snrs(rssi, start, end, noise)
    snrs_flora = flora_snrs(rssi, start, end, noise)

    for a, b in zip(snrs_flex, snrs_flora):
        assert a == pytest.approx(b, abs=1e-9)

