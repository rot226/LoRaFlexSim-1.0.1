import heapq


class DownlinkScheduler:
    """Simple scheduler for downlink frames for class B/C nodes."""

    def __init__(self):
        self.queue: dict[int, list[tuple[float, int, object, object]]] = {}
        self._counter = 0
        # Track when each gateway becomes free to transmit
        self._gateway_busy: dict[int, float] = {}

    @staticmethod
    def _payload_length(frame) -> int:
        """Return the byte length of ``frame`` payload."""
        if hasattr(frame, "payload"):
            try:
                return len(frame.payload)
            except Exception:
                pass
        if hasattr(frame, "to_bytes"):
            try:
                return len(frame.to_bytes())
            except Exception:
                pass
        return 0

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
        *,
        last_beacon_time: float | None = None,
    ) -> float:
        """Schedule ``frame`` for ``node`` at its next ping slot."""
        duration = node.channel.airtime(node.sf, self._payload_length(frame))
        t = node.next_ping_slot_time(
            after_time,
            beacon_interval,
            ping_slot_interval,
            ping_slot_offset,
            last_beacon_time=last_beacon_time,
        )
        busy = self._gateway_busy.get(gateway.id, 0.0)
        if t < busy:
            t = busy
        self.schedule(node.id, t, frame, gateway)
        self._gateway_busy[gateway.id] = t + duration
        return t

    def schedule_class_c(self, node, time: float, frame, gateway):
        """Schedule a frame for a Class C node at ``time`` and return it."""
        duration = node.channel.airtime(node.sf, self._payload_length(frame))
        busy = self._gateway_busy.get(gateway.id, 0.0)
        if time < busy:
            time = busy
        self.schedule(node.id, time, frame, gateway)
        self._gateway_busy[gateway.id] = time + duration
        return time

    def schedule_beacon(self, after_time: float, frame, gateway, beacon_interval: float) -> float:
        """Schedule a beacon frame at the next beacon time after ``after_time``."""
        from .lorawan import next_beacon_time

        t = next_beacon_time(after_time, beacon_interval)
        self.schedule(0, t, frame, gateway)
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
