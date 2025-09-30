"""Tests dédiés à l'accélération des fenêtres RX en classe C."""

from loraflexsim.launcher.simulator import Simulator

from tests.test_no_random_drop import make_clean_channel


def test_class_c_fast_polling_stops_after_quota():
    """Vérifie qu'un polling accéléré ne maintient pas indéfiniment la file d'événements."""

    channel = make_clean_channel()
    sim = Simulator(
        num_nodes=1,
        num_gateways=1,
        area_size=1.0,
        transmission_mode="Periodic",
        packet_interval=1.0,
        packets_to_send=1,
        mobility=False,
        channels=[channel],
        node_class="C",
        class_c_rx_interval=0.01,
        fixed_sf=7,
        fixed_tx_power=14,
        seed=2024,
    )

    node = sim.nodes[0]
    post_quota_lengths: list[int] = []

    while sim.event_queue:
        sim.step()
        if node.packets_sent >= sim.packets_to_send:
            # Lors des optimisations de fast-forward, un polling trop agressif
            # réactivait sans fin des RX_WINDOW. On s'assure qu'aucun événement
            # résiduel ne persiste durablement une fois le quota atteint afin
            # d'éviter toute régression sur l'accélération.
            remaining = len(sim.event_queue)
            post_quota_lengths.append(remaining)
            if len(post_quota_lengths) > 1:
                assert remaining <= post_quota_lengths[-2]
            assert len(post_quota_lengths) <= 10, "La file met trop longtemps à se vider"

    assert post_quota_lengths, "Le quota n'a jamais été atteint pendant la simulation"
    assert post_quota_lengths[-1] == 0
    assert node.packets_sent == sim.packets_to_send
    assert not sim.event_queue
