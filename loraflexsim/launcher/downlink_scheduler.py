import heapq
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ScheduledDownlink:
    """Container describing a scheduled downlink frame."""

    frame: Any
    gateway: Any
    data_rate: Optional[int] = None
    tx_power: Optional[float] = None


def _identity(t: float) -> float:
    return t


class DownlinkScheduler:
    """Simple scheduler for downlink frames for class B/C nodes."""

    def __init__(
        self,
        link_delay: float = 0.0,
        *,
        quantize: Callable[[float], float] | None = None,
    ):
        self.queue: dict[int, list[tuple[float, int, int, ScheduledDownlink]]] = {}
        self._counter = 0
        # Track when each gateway becomes free to transmit
        self._gateway_busy: dict[int, float] = {}
        # Track the last scheduled downlink per gateway to allow re-planning
        self._last_gateway_entry: dict[int, dict[str, Any]] = {}
        self.link_delay = link_delay
        self._identity_quantize: Callable[[float], float] = _identity
        self.quantize: Callable[[float], float] = quantize or self._identity_quantize

    def _apply_quantize(self, value: float) -> float:
        try:
            return self.quantize(value)
        except TypeError:
            return value

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
        q_time = self._apply_quantize(time)
        heapq.heappush(
            self.queue.setdefault(node_id, []),
            (q_time, priority, self._counter, item),
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
        slot_time = self._apply_quantize(t)
        start_time = self._apply_quantize(slot_time + self.link_delay)
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
                new_slot = self._apply_quantize(new_slot)
                new_start = self._apply_quantize(new_slot + self.link_delay)
                self._retime_entry(last["node_id"], last["counter"], new_start)
                last["slot_time"] = new_slot
                last["start_time"] = new_start
                new_end = self._apply_quantize(new_start + last["duration"])
                if new_end < new_start:
                    new_end = new_start
                last["end_time"] = new_end
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
                slot_time = self._apply_quantize(slot_time)
                start_time = self._apply_quantize(slot_time + self.link_delay)
                busy = self._gateway_busy.get(gateway.id, 0.0)

        while start_time < busy - tolerance:
            slot_time = node.next_ping_slot_time(
                busy - self.link_delay + 1e-6,
                beacon_interval,
                ping_slot_interval,
                ping_slot_offset,
                last_beacon_time=last_beacon_time,
            )
            slot_time = self._apply_quantize(slot_time)
            start_time = self._apply_quantize(slot_time + self.link_delay)
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
        quantized_end = self._apply_quantize(entry_end)
        if quantized_end < start_time:
            quantized_end = start_time
        last_entry = self._last_gateway_entry.get(gateway.id)
        if (
            last_entry is None
            or quantized_end >= last_entry["end_time"] - tolerance
        ):
            self._last_gateway_entry[gateway.id] = {
                "slot_time": slot_time,
                "start_time": start_time,
                "end_time": quantized_end,
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
        else:
            quantized_end = last_entry["end_time"]
        self._gateway_busy[gateway.id] = quantized_end
        return start_time

    def _retime_entry(self, node_id: int, counter: int, new_time: float) -> None:
        queue = self.queue.get(node_id)
        if not queue:
            return
        adjusted_time = self._apply_quantize(new_time)
        for index, (time, priority, cnt, item) in enumerate(queue):
            if cnt == counter:
                queue[index] = (adjusted_time, priority, cnt, item)
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
        start_time = time
        if start_time < busy:
            start_time = busy
        start_time = self._apply_quantize(start_time + self.link_delay)
        self.schedule(
            node.id,
            start_time,
            frame,
            gateway,
            priority=priority,
            data_rate=data_rate,
            tx_power=tx_power,
        )
        end_time = self._apply_quantize(start_time + duration)
        if end_time < start_time:
            end_time = start_time
        self._gateway_busy[gateway.id] = end_time
        return start_time

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
    ) -> float | None:
        """Schedule ``frame`` for a Class A node in the next available window.

        Returns the scheduled time if the frame could be planned, otherwise
        :data:`None` when both RX windows are unavailable.
        """
        duration = node.channel.airtime(node.sf, self._payload_length(frame))
        busy = self._gateway_busy.get(gateway.id, 0.0)
        candidate = max(after_time, busy)
        if candidate <= rx1:
            t = rx1
        elif candidate <= rx2:
            t = rx2
        else:
            logger.warning(
                "Rejet du downlink classe A pour le nœud %s via la passerelle %s : "
                "fenêtre RX2 dépassée (libre à %.3fs, RX2 à %.3fs).",
                getattr(node, "id", "?"),
                getattr(gateway, "id", "?"),
                candidate,
                rx2,
            )
            return None
        t = self._apply_quantize(t + self.link_delay)
        self.schedule(node.id, t, frame, gateway, priority=priority)
        end_time = self._apply_quantize(t + duration)
        if end_time < t:
            end_time = t
        self._gateway_busy[gateway.id] = end_time
        return t

    def schedule_beacon(self, after_time: float, frame, gateway, beacon_interval: float, *, priority: int = 0) -> float:
        """Schedule a beacon frame at the next beacon time after ``after_time``."""
        from .lorawan import next_beacon_time

        t = next_beacon_time(after_time, beacon_interval)
        t = self._apply_quantize(t + self.link_delay)
        self.schedule(0, t, frame, gateway, priority=priority)
        return t

    def pop_ready(self, node_id: int, current_time: float):
        """Return the next ready :class:`ScheduledDownlink` for ``node_id`` if any."""
        q = self.queue.get(node_id)
        ready_time = self._apply_quantize(current_time)
        if not q or q[0][0] > ready_time:
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

