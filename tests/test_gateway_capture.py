import json
from pathlib import Path

import pytest

from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.gateway import Gateway, FLORA_NON_ORTH_DELTA
from loraflexsim.launcher.server import NetworkServer


def _start_tx(
    gateway: Gateway,
    event_id: int,
    node_id: int,
    sf: int,
    rssi: float,
    start_time: float,
    end_time: float,
    *,
    frequency: float = 868e6,
    capture_threshold: float = 6.0,
    capture_window_symbols: int = 5,
    orthogonal_sf: bool = True,
    non_orth_delta=None,
    capture_mode: str | None = None,
) -> dict:
    gateway.start_reception(
        event_id,
        node_id,
        sf,
        rssi,
        end_time,
        capture_threshold,
        start_time,
        frequency,
        orthogonal_sf=orthogonal_sf,
        non_orth_delta=non_orth_delta,
        capture_window_symbols=capture_window_symbols,
        capture_mode="basic" if capture_mode is None else capture_mode,
    )
    _, tx = gateway.active_by_event[event_id]
    return tx


def test_orthogonal_sf_no_collision():
    gw = Gateway(0, 0, 0)
    server = NetworkServer()
    server.gateways = [gw]

    gw.start_reception(
        1, 1, 7, -60, 1.0, 6.0, 0.0, 868e6, orthogonal_sf=True
    )
    gw.start_reception(
        2, 2, 9, -60, 1.0, 6.0, 0.0, 868e6, orthogonal_sf=True
    )
    gw.end_reception(1, server, 1)
    gw.end_reception(2, server, 2)

    assert server.packets_received == 2


def test_capture_requires_five_symbols():
    gw = Gateway(0, 0, 0)
    server = NetworkServer()
    server.gateways = [gw]

    # Strong signal starts first
    gw.start_reception(1, 1, 7, -50, 1.0, 6.0, 0.0, 868e6)
    # Weaker packet starts after more than 5 symbols
    gw.start_reception(2, 2, 7, -60, 1.0, 6.0, 0.006, 868e6)
    gw.end_reception(1, server, 1)
    gw.end_reception(2, server, 2)

    assert server.packets_received == 1


def test_no_capture_before_five_symbols():
    gw = Gateway(0, 0, 0)
    server = NetworkServer()
    server.gateways = [gw]

    # Strong signal starts first but second arrives too soon
    gw.start_reception(1, 1, 7, -50, 1.0, 6.0, 0.0, 868e6)
    gw.start_reception(2, 2, 7, -60, 1.0, 6.0, 0.003, 868e6)
    gw.end_reception(1, server, 1)
    gw.end_reception(2, server, 2)

    assert server.packets_received == 0


def test_interference_ends_before_cs_begin():
    gw = Gateway(0, 0, 0)
    server = NetworkServer()
    server.gateways = [gw]

    sf = 7
    bw = 125e3
    symbol_time = (2 ** sf) / bw
    interferer_start = -0.002
    interferer_end = interferer_start + 3 * symbol_time  # ends before csBegin (~5 symbols)

    gw.start_reception(1, 1, sf, -60, interferer_end, 6.0, interferer_start, 868e6)
    gw.start_reception(2, 2, sf, -50, 0.1, 6.0, 0.0, 868e6)

    gw.end_reception(1, server, 1)
    gw.end_reception(2, server, 2)

    assert server.packets_received == 1


def test_strong_signal_arrives_late():
    gw = Gateway(0, 0, 0)
    server = NetworkServer()
    server.gateways = [gw]

    # Weak packet first
    gw.start_reception(1, 1, 7, -60, 1.0, 6.0, 0.0, 868e6)
    # Strong packet after 5 symbols should not capture
    gw.start_reception(2, 2, 7, -50, 1.0, 6.0, 0.006, 868e6)
    gw.end_reception(1, server, 1)
    gw.end_reception(2, server, 2)

    assert server.packets_received == 0


def test_cross_sf_collision():
    gw = Gateway(0, 0, 0)
    server = NetworkServer()
    server.gateways = [gw]

    # Two packets on the same frequency with different SF collide
    gw.start_reception(
        1, 1, 7, -60, 1.0, 6.0, 0.0, 868e6,
        orthogonal_sf=False,
        non_orth_delta=FLORA_NON_ORTH_DELTA,
    )
    gw.start_reception(
        2, 2, 9, -60, 1.0, 6.0, 0.0, 868e6,
        orthogonal_sf=False,
        non_orth_delta=FLORA_NON_ORTH_DELTA,
    )
    gw.end_reception(1, server, 1)
    gw.end_reception(2, server, 2)

    assert server.packets_received == 0


def test_cross_sf_capture_after_delay():
    gw = Gateway(0, 0, 0)
    server = NetworkServer()
    server.gateways = [gw]

    # Strong signal starts first and should capture the weaker one
    gw.start_reception(
        1, 1, 7, -50, 1.0, 6.0, 0.0, 868e6,
        orthogonal_sf=False,
        non_orth_delta=FLORA_NON_ORTH_DELTA,
    )
    # Weaker packet with higher SF starts after more than 5 of its symbols
    gw.start_reception(
        2, 2, 9, -60, 1.0, 6.0, 0.03, 868e6,
        orthogonal_sf=False,
        non_orth_delta=FLORA_NON_ORTH_DELTA,
    )
    gw.end_reception(1, server, 1)
    gw.end_reception(2, server, 2)

    assert server.packets_received == 1


def test_non_orth_same_sf_uses_capture_threshold():
    gw = Gateway(0, 0, 0)
    server = NetworkServer()
    server.gateways = [gw]

    # Two packets with the same SF start simultaneously with a 5 dB RSSI gap.
    # With ``orthogonal_sf`` disabled, the capture matrix should NOT apply to
    # same-SF collisions, so the regular capture threshold (6 dB) must be used.
    gw.start_reception(
        1,
        1,
        7,
        -50,
        1.0,
        6.0,
        0.0,
        868e6,
        orthogonal_sf=False,
        non_orth_delta=FLORA_NON_ORTH_DELTA,
        capture_window_symbols=0,
    )
    gw.start_reception(
        2,
        2,
        7,
        -55,
        1.0,
        6.0,
        0.0,
        868e6,
        orthogonal_sf=False,
        non_orth_delta=FLORA_NON_ORTH_DELTA,
        capture_window_symbols=0,
    )
    gw.end_reception(1, server, 1)
    gw.end_reception(2, server, 2)

    # RSSI difference (5 dB) is below capture threshold (6 dB) so none captured
    assert server.packets_received == 0


def test_non_orth_multi_sf_capture_respects_matrix_and_preamble():
    cases = [
        (7, 8, 8),
        (8, 10, 10),
        (9, 12, 12),
    ]
    for idx, (sf_main, sf_interferer, preamble) in enumerate(cases, start=1):
        gw = Gateway(idx, 0, 0)
        server = NetworkServer()
        server.gateways = [gw]

        symbol_duration = (2 ** sf_main) / 125e3
        strong_duration = symbol_duration * 14
        delay_symbols = max(5.5, (preamble - 6) + 0.5)
        interferer_start = symbol_duration * delay_symbols
        interferer_sym = (2 ** sf_interferer) / 125e3
        interferer_end = interferer_start + interferer_sym * 6

        strong = _start_tx(
            gw,
            1,
            1,
            sf_main,
            -48.0,
            0.0,
            strong_duration,
            orthogonal_sf=False,
            non_orth_delta=FLORA_NON_ORTH_DELTA,
            capture_window_symbols=5,
        )
        strong["preamble_symbols"] = preamble

        _start_tx(
            gw,
            2,
            2,
            sf_interferer,
            -55.0,
            interferer_start,
            interferer_end,
            orthogonal_sf=False,
            non_orth_delta=FLORA_NON_ORTH_DELTA,
            capture_window_symbols=5,
        )

        assert not gw.active_by_event[1][1]["lost_flag"]
        assert gw.active_by_event[2][1]["lost_flag"]

        gw.end_reception(1, server, 1)
        gw.end_reception(2, server, 2)

        assert server.packets_received == 1


def test_non_orth_capture_blocked_by_contaminated_preamble():
    gw = Gateway(0, 0, 0)
    server = NetworkServer()
    server.gateways = [gw]

    sf_main = 8
    symbol_duration = (2 ** sf_main) / 125e3
    strong = _start_tx(
        gw,
        1,
        1,
        sf_main,
        -50.0,
        0.0,
        symbol_duration * 14,
        orthogonal_sf=False,
        non_orth_delta=FLORA_NON_ORTH_DELTA,
        capture_window_symbols=6,
    )
    strong["preamble_symbols"] = 12

    interferer_start = symbol_duration * 5.0  # begins before csBegin (6 symbols)
    interferer_end = interferer_start + symbol_duration * 6

    _start_tx(
        gw,
        2,
        2,
        10,
        -58.0,
        interferer_start,
        interferer_end,
        orthogonal_sf=False,
        non_orth_delta=FLORA_NON_ORTH_DELTA,
        capture_window_symbols=6,
    )

    assert gw.active_by_event[1][1]["lost_flag"]
    assert gw.active_by_event[2][1]["lost_flag"]

    gw.end_reception(1, server, 1)
    gw.end_reception(2, server, 2)

    assert server.packets_received == 0


def test_flora_capture_requires_six_symbols():
    gw = Gateway(0, 0, 0)
    server = NetworkServer()
    server.gateways = [gw]

    channel = Channel(phy_model="flora", shadowing_std=0.0, use_flora_curves=True)
    flora_phy = channel.flora_phy
    assert flora_phy is not None
    assert channel.capture_window_symbols == 6

    sf = 7
    symbol_time = (2 ** sf) / channel.bandwidth
    strong_start = 0.0
    strong_end = strong_start + symbol_time * 10
    weak_start = strong_start + 1.5 * symbol_time
    weak_end = weak_start + symbol_time  # ends between the 5th and 6th symbols

    gw.start_reception(
        1,
        1,
        sf,
        -50,
        strong_end,
        channel.capture_threshold_dB,
        strong_start,
        868e6,
        capture_mode="flora",
        flora_phy=flora_phy,
        capture_window_symbols=channel.capture_window_symbols,
    )
    gw.start_reception(
        2,
        2,
        sf,
        -50,
        weak_end,
        channel.capture_threshold_dB,
        weak_start,
        868e6,
        capture_mode="flora",
        flora_phy=flora_phy,
        capture_window_symbols=channel.capture_window_symbols,
    )

    gw.end_reception(1, server, 1)
    gw.end_reception(2, server, 2)

    # The weaker packet still overlaps when only five symbols were clean, so
    # FLoRa's receiver (which requires six symbols) reports a collision.
    assert server.packets_received == 0


def test_aloha_mode_disables_capture_effect():
    gw = Gateway(0, 0, 0)
    server = NetworkServer()
    server.gateways = [gw]

    # First packet is strong and would normally win a capture.
    gw.start_reception(1, 1, 7, -50, 1.0, 6.0, 0.0, 868e6, capture_mode="aloha")
    # Second packet overlaps after a short delay with a much weaker power.
    gw.start_reception(2, 2, 7, -70, 1.0, 6.0, 0.1, 868e6, capture_mode="aloha")

    gw.end_reception(1, server, 1)
    gw.end_reception(2, server, 2)

    # Pure ALOHA mode destroys both packets regardless of the RSSI delta.
    assert server.packets_received == 0


def test_gateway_omnet_capture_matches_reference():
    data_path = Path(__file__).parent / "data" / "flora_gateway_reference.json"
    if not data_path.exists():
        pytest.skip("référence OMNeT++ manquante")

    with data_path.open("r", encoding="utf8") as fh:
        reference = json.load(fh)

    channel = Channel(
        phy_model="omnet",
        flora_capture=True,
        shadowing_std=0.0,
        fast_fading_std=0.0,
    )
    capture_window = channel.capture_window_symbols
    expected_cases = reference.get("cases", [])

    for case in expected_cases:
        gw = Gateway(0, 0, 0)
        gw.omnet_phy = channel.omnet_phy
        server = NetworkServer()
        server.gateways = [gw]

        preambles = case.get("preamble")
        frequency = case.get("frequency", 868e6)
        count = len(case.get("rssi", []))
        for idx, (rssi, start, end, sf) in enumerate(
            zip(
                case.get("rssi", []),
                case.get("start", []),
                case.get("end", []),
                case.get("sf", []),
            ),
            start=1,
        ):
            tx = _start_tx(
                gw,
                idx,
                idx,
                int(sf),
                float(rssi),
                float(start),
                float(end),
                frequency=frequency,
                orthogonal_sf=False,
                non_orth_delta=FLORA_NON_ORTH_DELTA,
                capture_window_symbols=capture_window,
                capture_mode="omnet",
                capture_threshold=channel.capture_threshold_dB,
            )
            if preambles:
                tx["preamble_symbols"] = preambles[idx - 1]

        winners = [
            not gw.active_by_event[event_id][1]["lost_flag"]
            for event_id in range(1, count + 1)
        ]
        expected = [bool(value) for value in case.get("expected", [])]
        assert winners == expected

        for event_id in range(1, count + 1):
            gw.end_reception(event_id, server, event_id)
