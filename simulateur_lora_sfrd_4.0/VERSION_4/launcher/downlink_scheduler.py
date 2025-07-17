class DownlinkScheduler:
    """Simple scheduler for downlink frames for class B/C nodes."""

    def __init__(self):
        self.queue: dict[int, list[tuple[float, object, object]]] = {}

    def schedule(self, node_id: int, time: float, frame, gateway):
        """Schedule a frame for a given node at ``time`` via ``gateway``."""
        self.queue.setdefault(node_id, []).append((time, frame, gateway))

    def pop_ready(self, node_id: int, current_time: float):
        """Return the next ready frame for ``node_id`` if any."""
        q = self.queue.get(node_id)
        if not q:
            return None, None
        q.sort(key=lambda x: x[0])
        if q[0][0] <= current_time:
            _, frame, gw = q.pop(0)
            return frame, gw
        return None, None

    def next_time(self, node_id: int):
        q = self.queue.get(node_id)
        if not q:
            return None
        return min(t for t, _, _ in q)
