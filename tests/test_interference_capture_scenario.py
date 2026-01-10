from __future__ import annotations

import numpy as np

from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.gateway import Gateway
from traffic.numpy_compat import create_generator


class _StubServer:
    def __init__(self) -> None:
        self.received: list[int] = []
        self.collision_reasons: dict[int, str] = {}

    def schedule_receive(self, event_id: int, *_args, **_kwargs) -> None:
        self.received.append(event_id)

    def register_collision_reason(self, event_id: int, reason: str) -> None:
        self.collision_reasons[event_id] = reason


def _run_trial(
    *,
    interference_db: float,
    capture_threshold_db: float,
    marginal_drop_prob: float,
    seed: int,
) -> tuple[bool, dict[int, str]]:
    rng = create_generator(seed)
    channel = Channel(
        phy_model="omnet",
        shadowing_std=0.0,
        fast_fading_std=0.0,
        noise_floor_std=0.0,
        fine_fading_std=0.0,
        variable_noise_std=0.0,
        interference_dB=interference_db,
        marginal_snir_drop_prob=marginal_drop_prob,
        rng=rng,
    )

    # Deux transmissions de SF7 se chevauchent brièvement (200 ms sur 1 s)
    rssi = [-100.0, -101.0]
    start = [0.0, 0.4]
    end = [1.0, 0.6]
    noise = channel.omnet_phy.noise_floor()
    snirs = channel.omnet_phy.compute_snrs(rssi, start, end, noise)

    gw = Gateway(1, 0.0, 0.0, rng=rng)
    server = _StubServer()

    for event_id, (rssi_dbm, snir_db) in enumerate(zip(rssi, snirs), start=1):
        gw.start_reception(
            event_id=event_id,
            node_id=event_id,
            sf=7,
            rssi=rssi_dbm,
            end_time=end[event_id - 1],
            capture_threshold=capture_threshold_db,
            required_snr_db_by_sf={7: capture_threshold_db},
            current_time=start[event_id - 1],
            frequency=868e6,
            noise_floor=noise,
            snir=snir_db,
            capture_mode="basic",
            orthogonal_sf=True,
            capture_window_symbols=5,
            non_orth_delta=None,
            snir_fading_std=0.0,
            marginal_snir_db=channel.marginal_snir_margin_db,
            marginal_drop_prob=marginal_drop_prob,
        )

    gw.end_reception(1, server, 1)
    gw.end_reception(2, server, 2)

    return bool(server.received), server.collision_reasons


def _aggregate(
    *, interference_db: float, capture_threshold_db: float, marginal_drop_prob: float
) -> tuple[float, float, int]:
    successes = 0
    collision_reasons: dict[int, int] = {}
    trials = 40
    for seed in range(trials):
        captured, reasons = _run_trial(
            interference_db=interference_db,
            capture_threshold_db=capture_threshold_db,
            marginal_drop_prob=marginal_drop_prob,
            seed=seed,
        )
        successes += int(captured)
        for reason in reasons.values():
            collision_reasons[reason] = collision_reasons.get(reason, 0) + 1

    capture_rate = successes / trials
    der = successes / (2 * trials)
    marginal_losses = collision_reasons.get("snir_marginal", 0)
    return capture_rate, der, marginal_losses


def _run_three_node_scenario(
    *,
    rssi: list[float],
    start: list[float],
    end: list[float],
    noise_floor: float,
    snirs: list[float],
    use_snir: bool,
) -> tuple[list[int], dict[int, str]]:
    rng = create_generator(42)
    gw = Gateway(1, 0.0, 0.0, rng=rng)
    server = _StubServer()

    for event_id, (rssi_dbm, snir_db) in enumerate(zip(rssi, snirs), start=1):
        gw.start_reception(
            event_id=event_id,
            node_id=event_id,
            sf=7,
            rssi=rssi_dbm,
            end_time=end[event_id - 1],
            capture_threshold=0.0,
            required_snr_db_by_sf={7: 7.0},
            current_time=start[event_id - 1],
            frequency=868e6,
            noise_floor=noise_floor,
            snir=snir_db,
            capture_mode="basic",
            orthogonal_sf=True,
            capture_window_symbols=5,
            non_orth_delta=None,
            snir_fading_std=0.0,
            marginal_snir_db=0.0,
            marginal_drop_prob=0.0,
            use_snir=use_snir,
        )

    for event_id in range(1, len(rssi) + 1):
        gw.end_reception(event_id, server, event_id)

    return server.received, server.collision_reasons


def test_three_node_snir_off_vs_on() -> None:
    channel = Channel(
        phy_model="omnet",
        shadowing_std=0.0,
        fast_fading_std=0.0,
        noise_floor_std=0.0,
        fine_fading_std=0.0,
        variable_noise_std=0.0,
        rng=create_generator(123),
    )
    # Deux nœuds proches et un nœud lointain avec chevauchement partiel.
    rssi = [-96.0, -110.0, -95.0]
    start = [0.0, 0.2, 0.4]
    end = [1.0, 0.8, 1.2]
    noise_floor = channel.omnet_phy.noise_floor()
    snirs = channel.omnet_phy.compute_snrs(rssi, start, end, noise_floor)

    strongest_snr = rssi[2] - noise_floor
    assert strongest_snr > 7.0
    assert snirs[2] < 7.0

    # SNIR_OFF = SNR seul (pas d'interférences prises en compte).
    received_off, reasons_off = _run_three_node_scenario(
        rssi=rssi,
        start=start,
        end=end,
        noise_floor=noise_floor,
        snirs=snirs,
        use_snir=False,
    )
    assert received_off == [3]
    assert reasons_off[1] == "capture"
    assert reasons_off[2] == "capture"

    # SNIR_ON = SNR + interférences combinées.
    received_on, reasons_on = _run_three_node_scenario(
        rssi=rssi,
        start=start,
        end=end,
        noise_floor=noise_floor,
        snirs=snirs,
        use_snir=True,
    )
    assert received_on == []
    assert reasons_on[1] == "snir_below_threshold"
    assert reasons_on[2] == "snir_below_threshold"
    assert reasons_on[3] == "snir_below_threshold"


def test_capture_rate_declines_with_interference_and_threshold() -> None:
    base_rate, base_der, _ = _aggregate(
        interference_db=0.0, capture_threshold_db=7.0, marginal_drop_prob=0.0
    )
    noisy_rate, noisy_der, _ = _aggregate(
        interference_db=6.0, capture_threshold_db=7.0, marginal_drop_prob=0.0
    )
    strict_rate, strict_der, _ = _aggregate(
        interference_db=0.0, capture_threshold_db=9.0, marginal_drop_prob=0.0
    )

    assert base_rate > noisy_rate
    assert base_rate > strict_rate
    assert base_der > noisy_der
    assert base_der > strict_der


def test_partial_overlaps_trigger_marginal_snir_losses() -> None:
    # Caler le seuil légèrement sous le SNIR moyen pour rester dans la zone marginale
    capture_threshold = 7.0
    baseline_rate, baseline_der, _ = _aggregate(
        interference_db=0.0,
        capture_threshold_db=capture_threshold,
        marginal_drop_prob=0.0,
    )
    drop_rate, drop_der, marginal_losses = _aggregate(
        interference_db=0.0,
        capture_threshold_db=capture_threshold,
        marginal_drop_prob=1.0,
    )

    assert drop_rate < baseline_rate
    assert drop_der < baseline_der
    assert marginal_losses > 0
