"""Tests dédiés à l'accélération des fenêtres RX en classe C."""

import heapq

from loraflexsim.launcher.downlink_scheduler import ScheduledDownlink
from loraflexsim.launcher.gateway import Gateway
from loraflexsim.launcher.simulator import Event, EventType, Simulator

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


def test_class_c_purge_multiple_fake_windows():
    """La purge doit retirer toutes les fenêtres RX résiduelles sans downlink."""

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
        seed=9001,
    )

    node = sim.nodes[0]

    # Injecter artificiellement plusieurs fenêtres RX_WINDOW en file pour
    # vérifier que la logique de purge supprime bien chaque entrée lorsque
    # le quota est atteint et qu'aucun downlink n'est en attente.
    for offset in (1.5, 2.5, 3.5):
        scheduled_id = sim.event_id_counter
        sim.schedule_event(node, sim.current_time + offset)
        for idx, event in enumerate(sim.event_queue):
            if event.id == scheduled_id:
                sim.event_queue[idx] = Event(
                    event.time, EventType.RX_WINDOW, event.id, event.node_id
                )
                break
        else:  # pragma: no cover - protection contre l'absence d'insertion
            raise AssertionError("Événement factice introuvable dans la file")
    heapq.heapify(sim.event_queue)

    sim.run(max_time=20.0)

    assert node.downlink_pending == 0
    assert node.id not in sim._class_c_polling_nodes
    assert all(event.type != EventType.RX_WINDOW for event in sim.event_queue)


class _FutureScheduler:
    """Planificateur conservant un downlink jusqu'à une échéance future."""

    def __init__(self, ready_time: float):
        self._ready_time = ready_time
        self._scheduled: dict[int, tuple[float, ScheduledDownlink]] = {}

    def schedule_class_c(self, node, time, frame, gateway, *, priority=0, data_rate=None, tx_power=None):
        scheduled_time = max(time, self._ready_time)
        self._scheduled[node.id] = (
            scheduled_time,
            ScheduledDownlink(frame, gateway, data_rate, tx_power),
        )
        return scheduled_time

    def next_time(self, node_id):  # noqa: D401 - signature imposée
        entry = self._scheduled.get(node_id)
        if entry is None:
            return None
        return entry[0]

    def pop_ready(self, node_id, current_time):  # noqa: D401 - signature imposée
        entry = self._scheduled.get(node_id)
        if not entry:
            return None
        scheduled_time, downlink = entry
        if current_time >= scheduled_time:
            self._scheduled.pop(node_id, None)
            return downlink
        return None


def test_class_c_preserves_window_when_downlink_future():
    """Tant qu'un downlink est planifié dans le futur, une fenêtre doit rester."""

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
        seed=2025,
    )

    future_time = 5.0
    scheduler = _FutureScheduler(future_time)
    sim.network_server.scheduler = scheduler

    node = sim.nodes[0]
    sim.network_server.send_downlink(node, payload=b"F")

    # Exécuter la simulation uniquement jusqu'à avant l'échéance du downlink.
    sim.run(max_time=future_time - 0.1)

    assert scheduler.next_time(node.id) == future_time
    assert node.downlink_pending > 0
    assert node.id in sim._class_c_polling_nodes
    assert any(
        event.type == EventType.RX_WINDOW and event.node_id == node.id
        for event in sim.event_queue
    )
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
