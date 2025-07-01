class DutyCycleManager:
    """Simple per-node duty cycle manager."""
    def __init__(self, duty_cycle: float = 0.01):
        if duty_cycle <= 0 or duty_cycle > 1:
            raise ValueError("duty_cycle must be in (0,1]")
        self.duty_cycle = duty_cycle
        # next allowed transmission time per node id
        self.next_tx_time: dict[int, float] = {}

    def can_transmit(self, node_id: int, time: float) -> bool:
        """Return True if node can transmit at given time."""
        return time >= self.next_tx_time.get(node_id, 0.0)

    def update_after_tx(self, node_id: int, start_time: float, duration: float):
        """Update duty cycle info after a transmission."""
        wait_time = duration * (1.0 / self.duty_cycle - 1.0)
        next_time = start_time + duration + wait_time
        cur = self.next_tx_time.get(node_id, 0.0)
        if next_time > cur:
            self.next_tx_time[node_id] = next_time

    def enforce(self, node_id: int, requested_time: float) -> float:
        """Return the earliest time the node can transmit given the duty cycle."""
        return max(requested_time, self.next_tx_time.get(node_id, 0.0))
