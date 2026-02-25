"""Moteur de simulation event-driven pour uplinks périodiques."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import random
from typing import Any


@dataclass
class Node:
    node_id: int
    period_s: float
    next_uplink_s: float = 0.0
    payload_size: int = 12
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(order=True)
class Event:
    time_s: float
    kind: str
    node_id: int


@dataclass
class SimulationResult:
    uplink_count: int = 0
    events: list[Event] = field(default_factory=list)


class EventDrivenEngine:
    """Boucle event-driven basée sur une file de priorité.

    Chaque nœud planifie un événement ``uplink`` périodique.
    """

    def __init__(self, *, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def _schedule_initial_events(self, nodes: list[Node]) -> list[Event]:
        queue: list[Event] = []
        for node in nodes:
            jitter = self.rng.uniform(0.0, min(node.period_s, 1.0))
            node.next_uplink_s = max(0.0, jitter)
            heapq.heappush(queue, Event(time_s=node.next_uplink_s, kind="uplink", node_id=node.node_id))
        return queue

    def run(self, *, nodes: list[Node], until_s: float) -> SimulationResult:
        if until_s <= 0:
            return SimulationResult()

        node_by_id = {n.node_id: n for n in nodes}
        queue = self._schedule_initial_events(nodes)
        result = SimulationResult()

        while queue:
            event = heapq.heappop(queue)
            if event.time_s > until_s:
                break
            result.events.append(event)

            if event.kind == "uplink":
                result.uplink_count += 1
                node = node_by_id[event.node_id]
                next_time = event.time_s + max(node.period_s, 1e-6)
                node.next_uplink_s = next_time
                heapq.heappush(queue, Event(time_s=next_time, kind="uplink", node_id=node.node_id))

        return result
