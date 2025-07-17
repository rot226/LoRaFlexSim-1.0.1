import heapq


class DownlinkScheduler:
    """Simple scheduler for downlink frames for class B/C nodes."""

    def __init__(self):
        self.queue: dict[int, list[tuple[float, int, object, object]]] = {}
        self._counter = 0

    def schedule(self, node_id: int, time: float, frame, gateway):
        """Schedule a frame for a given node at ``time`` via ``gateway``."""
        heapq.heappush(
            self.queue.setdefault(node_id, []),
            (time, self._counter, frame, gateway),
        )
        self._counter += 1

    def schedule_class_b(
        self,
        node,
        after_time: float,
        frame,
        gateway,
        beacon_interval: float,
        ping_slot_interval: float,
        ping_slot_offset: float,
    ) -> float:
        """Schedule ``frame`` for ``node`` at its next ping slot."""
        t = node.next_ping_slot_time(
            after_time, beacon_interval, ping_slot_interval, ping_slot_offset
        )
        self.schedule(node.id, t, frame, gateway)
        return t

    def pop_ready(self, node_id: int, current_time: float):
        """Return the next ready frame for ``node_id`` if any."""
        q = self.queue.get(node_id)
        if not q or q[0][0] > current_time:
            return None, None
        _, _, frame, gw = heapq.heappop(q)
        if not q:
            self.queue.pop(node_id, None)
        return frame, gw

    def next_time(self, node_id: int):
        q = self.queue.get(node_id)
        if not q:
            return None
        return q[0][0]
