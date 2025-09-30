"""Tests dédiés à l'accélération des fenêtres RX en classe C."""

from loraflexsim.launcher.gateway import Gateway
from loraflexsim.launcher.simulator import EventType, Simulator

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


class _NullScheduler:
    """Planificateur factice qui n'injecte jamais de downlink."""

    def schedule_class_c(self, *args, **kwargs):
        return None

    def next_time(self, node_id):  # noqa: D401 - signature imposée
        return None

    def pop_ready(self, node_id, current_time):  # noqa: D401 - signature imposée
        return None


def test_class_c_polling_stops_without_downlink():
    """Les sondes de classe C s'arrêtent lorsqu'aucun downlink n'est planifié."""

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
        seed=1337,
    )
    sim.network_server.scheduler = _NullScheduler()

    sim.run(max_time=20.0)

    node = sim.nodes[0]
    assert node.downlink_pending == 0
    assert node.id not in sim._class_c_polling_nodes
    assert all(event.type != EventType.RX_WINDOW for event in sim.event_queue)


def test_class_c_polling_clears_after_dropped_downlink(monkeypatch):
    """Un downlink planifié mais perdu ne doit pas bloquer l'accélération."""

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
        seed=4242,
    )

    original_pop = Gateway.pop_downlink

    def _drop_downlink(self, node_id):
        original_pop(self, node_id)
        return None

    monkeypatch.setattr(Gateway, "pop_downlink", _drop_downlink)

    node = sim.nodes[0]
    sim.network_server.send_downlink(node, payload=b"X")

    sim.run(max_time=20.0)

    assert node.downlink_pending == 0
    assert node.id not in sim._class_c_polling_nodes
    assert all(event.type != EventType.RX_WINDOW for event in sim.event_queue)
