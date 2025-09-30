import heapq
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(slots=True)
class ScheduledDownlink:
    """Container describing a scheduled downlink frame."""

    frame: Any
    gateway: Any
    data_rate: Optional[int] = None
    tx_power: Optional[float] = None


class DownlinkScheduler:
    """Simple scheduler for downlink frames for class B/C nodes."""

    def __init__(self, link_delay: float = 0.0):
        self.queue: dict[int, list[tuple[float, int, int, ScheduledDownlink]]] = {}
        self._counter = 0
        # Track when each gateway becomes free to transmit
        self._gateway_busy: dict[int, float] = {}
        # Track the last scheduled downlink per gateway to allow re-planning
        self._last_gateway_entry: dict[int, dict[str, Any]] = {}
        self.link_delay = link_delay

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

    def schedule(
        self,
        node_id: int,
        time: float,
        frame,
        gateway,
        *,
        priority: int = 0,
        data_rate: int | None = None,
        tx_power: float | None = None,
    ) -> None:
        """Schedule a frame for a given node at ``time`` via ``gateway`` with optional ``priority``."""
        item = ScheduledDownlink(frame, gateway, data_rate, tx_power)
        heapq.heappush(
            self.queue.setdefault(node_id, []),
            (time, priority, self._counter, item),
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
        priority: int = 0,
        data_rate: int | None = None,
        tx_power: float | None = None,
    ) -> float:
        """Schedule ``frame`` for ``node`` at its next ping slot."""
        sf = node.sf
        dr = data_rate
        if dr is None:
            dr = getattr(node, "ping_slot_dr", None)
        if dr is not None:
            from .lorawan import DR_TO_SF

            sf = DR_TO_SF.get(dr, sf)
        duration = node.channel.airtime(sf, self._payload_length(frame))
        t = node.next_ping_slot_time(
            after_time,
            beacon_interval,
            ping_slot_interval,
            ping_slot_offset,
            last_beacon_time=last_beacon_time,
        )
        slot_time = t
        start_time = slot_time + self.link_delay
        busy = self._gateway_busy.get(gateway.id, 0.0)
        tolerance = 1e-9

        if start_time < busy - tolerance:
            last = self._last_gateway_entry.get(gateway.id)
            if (
                last
                and priority < last["priority"]
                and slot_time <= last["slot_time"] + tolerance
            ):
                prev_node = last["node"]
                new_slot = prev_node.next_ping_slot_time(
                    last["slot_time"] + 1e-6,
                    last["beacon_interval"],
                    last["ping_slot_interval"],
                    last["ping_slot_offset"],
                    last_beacon_time=last["last_beacon_time"],
                )
                new_start = new_slot + self.link_delay
                self._retime_entry(last["node_id"], last["counter"], new_start)
                last["slot_time"] = new_slot
                last["start_time"] = new_start
                last["end_time"] = new_start + last["duration"]
                busy = start_time
                # Gateway is now free at the requested slot for the priority frame
            else:
                after = max(busy - self.link_delay, slot_time)
                slot_time = node.next_ping_slot_time(
                    after + 1e-6,
                    beacon_interval,
                    ping_slot_interval,
                    ping_slot_offset,
                    last_beacon_time=last_beacon_time,
                )
                start_time = slot_time + self.link_delay
                busy = self._gateway_busy.get(gateway.id, 0.0)

        while start_time < busy - tolerance:
            slot_time = node.next_ping_slot_time(
                busy - self.link_delay + 1e-6,
                beacon_interval,
                ping_slot_interval,
                ping_slot_offset,
                last_beacon_time=last_beacon_time,
            )
            start_time = slot_time + self.link_delay
            busy = self._gateway_busy.get(gateway.id, 0.0)

        counter = self._counter
        self.schedule(
            node.id,
            start_time,
            frame,
            gateway,
            priority=priority,
            data_rate=dr,
            tx_power=tx_power,
        )

        entry_end = start_time + duration
        last_entry = self._last_gateway_entry.get(gateway.id)
        if (
            last_entry is None
            or entry_end >= last_entry["end_time"] - tolerance
        ):
            self._last_gateway_entry[gateway.id] = {
                "slot_time": slot_time,
                "start_time": start_time,
                "end_time": entry_end,
                "duration": duration,
                "priority": priority,
                "node": node,
                "node_id": node.id,
                "counter": counter,
                "beacon_interval": beacon_interval,
                "ping_slot_interval": ping_slot_interval,
                "ping_slot_offset": ping_slot_offset,
                "last_beacon_time": last_beacon_time,
            }
        self._gateway_busy[gateway.id] = max(
            entry_end,
            self._last_gateway_entry.get(gateway.id, {}).get("end_time", entry_end),
        )
        return start_time

    def _retime_entry(self, node_id: int, counter: int, new_time: float) -> None:
        queue = self.queue.get(node_id)
        if not queue:
            return
        for index, (time, priority, cnt, item) in enumerate(queue):
            if cnt == counter:
                queue[index] = (new_time, priority, cnt, item)
                heapq.heapify(queue)
                break

    def schedule_class_c(
        self,
        node,
        time: float,
        frame,
        gateway,
        *,
        priority: int = 0,
        data_rate: int | None = None,
        tx_power: float | None = None,
    ):
        """Schedule a frame for a Class C node at ``time`` with optional ``priority`` and return the scheduled time."""
        sf = node.sf
        if data_rate is not None:
            from .lorawan import DR_TO_SF

            sf = DR_TO_SF.get(data_rate, sf)
        duration = node.channel.airtime(sf, self._payload_length(frame))
        busy = self._gateway_busy.get(gateway.id, 0.0)
        if time < busy:
            time = busy
        time += self.link_delay
        self.schedule(
            node.id,
            time,
            frame,
            gateway,
            priority=priority,
            data_rate=data_rate,
            tx_power=tx_power,
        )
        self._gateway_busy[gateway.id] = time + duration
        return time

    def schedule_class_a(
        self,
        node,
        after_time: float,
        rx1: float,
        rx2: float,
        frame,
        gateway,
        *,
        priority: int = 0,
    ) -> float:
        """Schedule ``frame`` for a Class A node in the next available window."""
        duration = node.channel.airtime(node.sf, self._payload_length(frame))
        busy = self._gateway_busy.get(gateway.id, 0.0)
        candidate = max(after_time, busy)
        if candidate <= rx1:
            t = rx1
        elif candidate <= rx2:
            t = rx2
        else:
            t = candidate
        t += self.link_delay
        self.schedule(node.id, t, frame, gateway, priority=priority)
        self._gateway_busy[gateway.id] = t + duration
        return t

    def schedule_beacon(self, after_time: float, frame, gateway, beacon_interval: float, *, priority: int = 0) -> float:
        """Schedule a beacon frame at the next beacon time after ``after_time``."""
        from .lorawan import next_beacon_time

        t = next_beacon_time(after_time, beacon_interval)
        t += self.link_delay
        self.schedule(0, t, frame, gateway, priority=priority)
        return t

    def pop_ready(self, node_id: int, current_time: float):
        """Return the next ready :class:`ScheduledDownlink` for ``node_id`` if any."""
        q = self.queue.get(node_id)
        if not q or q[0][0] > current_time:
            return None
        _, _, _, item = heapq.heappop(q)
        if not q:
            self.queue.pop(node_id, None)
        return item

    def next_time(self, node_id: int):
        q = self.queue.get(node_id)
        if not q:
            return None
        return q[0][0]
