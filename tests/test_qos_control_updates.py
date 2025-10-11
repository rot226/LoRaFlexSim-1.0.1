"""Tests ciblés pour la diffusion des mises à jour de contrôle QoS."""

from __future__ import annotations

from types import SimpleNamespace

from loraflexsim.launcher.lorawan import ControlUpdate
from loraflexsim.launcher.qos import QoSManager
from loraflexsim.launcher.simulator import Simulator


class _DummyServer:
    def __init__(self) -> None:
        self.sent: list[tuple[object, bytes, float | None, object | None]] = []

    def send_downlink(self, node, *, payload, at_time, channel) -> None:
        self.sent.append((node, payload, at_time, channel))


class _DummySimulator:
    """Simulateur minimal utilisant :meth:`Simulator.channel_index`."""

    def __init__(self) -> None:
        self.control_channel_index = 99
        self.control_channel = object()
        self._channel_lookup: dict[int, object] = {}
        self._channel_reverse: dict[int, int] = {}
        self.network_server = _DummyServer()
        self.nodes: list[object] = []
        self.current_time: float | None = 0.0

    def channel_index(self, channel) -> int:
        return Simulator.channel_index(self, channel)


def _new_channel() -> object:
    return object()


def test_channel_index_dynamic_and_control_support() -> None:
    simulator = _DummySimulator()

    # Aucun canal enregistré : un premier appel doit créer un index 0.
    channel_a = _new_channel()
    index_a = simulator.channel_index(channel_a)
    assert index_a == 0
    assert simulator._channel_lookup[index_a] is channel_a
    assert simulator._channel_reverse[id(channel_a)] == index_a

    # Nouvel objet canal : un index supplémentaire est alloué et réutilisé.
    channel_b = _new_channel()
    index_b = simulator.channel_index(channel_b)
    assert index_b == 1
    assert simulator.channel_index(channel_b) == index_b

    # Le canal de contrôle (ou l'absence de canal) renvoie l'index dédié.
    assert (
        simulator.channel_index(simulator.control_channel)
        == simulator.control_channel_index
    )
    assert simulator.channel_index(None) == simulator.control_channel_index


def test_broadcast_control_updates_only_on_assignment_changes() -> None:
    manager = QoSManager()
    simulator = _DummySimulator()

    # Canal commun aux deux nœuds au départ.
    shared_channel = _new_channel()
    simulator.channel_index(shared_channel)

    node_1 = SimpleNamespace(id=1, sf=7, channel=shared_channel)
    node_2 = SimpleNamespace(id=2, sf=9, channel=shared_channel)
    simulator.nodes = [node_1, node_2]

    # Première diffusion : chaque nœud doit recevoir une mise à jour.
    manager._broadcast_control_updates(simulator)
    assert len(simulator.network_server.sent) == 2
    payloads = [entry[1] for entry in simulator.network_server.sent]
    assert ControlUpdate(7, 0).to_bytes() in payloads
    assert ControlUpdate(9, 0).to_bytes() in payloads
    assert node_1.assigned_channel_index == 0
    assert node_2.assigned_channel_index == 0

    # Aucune modification : aucune diffusion supplémentaire ne doit avoir lieu.
    simulator.network_server.sent.clear()
    manager._broadcast_control_updates(simulator)
    assert simulator.network_server.sent == []

    # Modifier le SF d'un nœud doit déclencher une seule diffusion.
    node_1.sf = 8
    manager._broadcast_control_updates(simulator)
    assert len(simulator.network_server.sent) == 1
    node, payload, *_ = simulator.network_server.sent[0]
    assert node is node_1
    assert payload == ControlUpdate(8, 0).to_bytes()

    # Changer le canal attribué provoque également une diffusion unique.
    simulator.network_server.sent.clear()
    new_channel = _new_channel()
    node_2.channel = new_channel
    manager._broadcast_control_updates(simulator)
    assert len(simulator.network_server.sent) == 1
    node, payload, *_ = simulator.network_server.sent[0]
    new_index = simulator.channel_index(new_channel)
    assert node is node_2
    assert payload == ControlUpdate(node_2.sf, new_index).to_bytes()
