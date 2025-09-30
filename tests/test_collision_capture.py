from loraflexsim.launcher.gateway import Gateway, FLORA_NON_ORTH_DELTA
from loraflexsim.launcher.server import NetworkServer


def _setup():
    gw = Gateway(0, 0, 0)
    server = NetworkServer()
    server.gateways = [gw]
    return gw, server


def _start_tx(
    gw: Gateway,
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
):
    gw.start_reception(
        event_id,
        node_id,
        sf,
        rssi,
        end_time,
        capture_threshold,
        start_time,
        frequency,
        orthogonal_sf=orthogonal_sf,
        non_orth_delta=FLORA_NON_ORTH_DELTA if not orthogonal_sf else None,
        capture_window_symbols=capture_window_symbols,
    )
    return gw.active_by_event[event_id][1]


def test_same_sf_collision():
    gw, server = _setup()
    # two transmissions start at the same time with same SF/frequency
    gw.start_reception(1, 1, 7, -50, 0.1, 6.0, 0.0, 868e6)
    gw.start_reception(2, 2, 7, -58, 0.1, 6.0, 0.0, 868e6)
    gw.end_reception(1, server, 1)
    gw.end_reception(2, server, 2)
    assert server.packets_received == 0


def test_capture_after_five_preamble():
    gw, server = _setup()
    sf = 7
    symbol_time = (2 ** sf) / 125e3
    start2 = 5.1 * symbol_time
    gw.start_reception(1, 1, sf, -50, 0.1, 6.0, 0.0, 868e6)
    gw.start_reception(2, 2, sf, -60, 0.1, 6.0, start2, 868e6)
    gw.end_reception(1, server, 1)
    gw.end_reception(2, server, 2)
    assert server.packets_received == 1


def test_residual_interference_blocks_followup_packet():
    gw, server = _setup()
    # Deux paquets simultanés entrent en collision totale
    gw.start_reception(1, 1, 7, -50, 0.1, 6.0, 0.0, 868e6)
    gw.start_reception(2, 2, 7, -58, 0.1, 6.0, 0.0, 868e6)
    # Tant que les transmissions brouillées ne sont pas terminées,
    # un troisième paquet reste bloqué par l'interférence résiduelle
    gw.start_reception(3, 3, 7, -48, 0.15, 6.0, 0.05, 868e6)

    key = (7, 868e6)
    active_entries = gw.active_map.get(key, [])
    assert len(active_entries) == 3
    assert all(t["lost_flag"] for t in active_entries)

    gw.end_reception(1, server, 1)
    gw.end_reception(2, server, 2)
    gw.end_reception(3, server, 3)
    assert server.packets_received == 0


def test_multi_sf_non_orthogonal_collision_marks_winners():
    gw, server = _setup()

    sf_main = 7
    sym_time_main = (2 ** sf_main) / 125e3
    strong_duration = sym_time_main * 14
    strong = _start_tx(
        gw,
        1,
        1,
        sf_main,
        -48.0,
        0.0,
        strong_duration,
        orthogonal_sf=False,
        capture_window_symbols=5,
    )
    strong["preamble_symbols"] = 9

    interferer1_start = sym_time_main * 6.5
    interferer1_sym = (2 ** 9) / 125e3
    interferer1_end = interferer1_start + interferer1_sym * 6
    _start_tx(
        gw,
        2,
        2,
        9,
        -55.0,
        interferer1_start,
        interferer1_end,
        orthogonal_sf=False,
        capture_window_symbols=5,
    )

    interferer2_start = interferer1_start + sym_time_main
    interferer2_sym = (2 ** 10) / 125e3
    interferer2_end = interferer2_start + interferer2_sym * 5
    _start_tx(
        gw,
        3,
        3,
        10,
        -60.0,
        interferer2_start,
        interferer2_end,
        orthogonal_sf=False,
        capture_window_symbols=5,
    )

    assert not gw.active_by_event[1][1]["lost_flag"]
    assert gw.active_by_event[2][1]["lost_flag"]
    assert gw.active_by_event[3][1]["lost_flag"]

    gw.end_reception(2, server, 2)
    gw.end_reception(3, server, 3)
    gw.end_reception(1, server, 1)

    assert server.packets_received == 1


def test_multi_sf_capture_fails_with_short_preamble():
    gw, server = _setup()

    sf_main = 7
    sym_time = (2 ** sf_main) / 125e3
    strong = _start_tx(
        gw,
        1,
        1,
        sf_main,
        -49.0,
        0.0,
        sym_time * 12,
        orthogonal_sf=False,
        capture_window_symbols=6,
    )
    strong["preamble_symbols"] = 7  # csBegin après un seul symbole

    interferer_start = sym_time * 0.5
    interferer_end = interferer_start + sym_time * 8
    _start_tx(
        gw,
        2,
        2,
        9,
        -56.0,
        interferer_start,
        interferer_end,
        orthogonal_sf=False,
        capture_window_symbols=6,
    )

    assert gw.active_by_event[1][1]["lost_flag"]
    assert gw.active_by_event[2][1]["lost_flag"]

    gw.end_reception(1, server, 1)
    gw.end_reception(2, server, 2)

    assert server.packets_received == 0

