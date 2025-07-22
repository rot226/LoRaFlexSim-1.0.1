import csv
from pathlib import Path
from typing import Iterable, Sequence


class GPSTraceMobility:
    """Mobility based on time-stamped GPS traces."""

    def __init__(self, trace: str | Iterable[Sequence[float]], loop: bool = True) -> None:
        if isinstance(trace, str) or isinstance(trace, Path):
            rows = []
            with open(trace, "r", newline="") as f:
                for row in csv.reader(f):
                    if not row:
                        continue
                    values = [float(v) for v in row]
                    if len(values) == 3:
                        t, x, y = values
                        z = 0.0
                    else:
                        t, x, y, z = (values + [0.0])[:4]
                    rows.append((t, x, y, z))
        else:
            rows = [tuple(map(float, r + (0.0,) * (4 - len(r)))) for r in trace]
        rows.sort(key=lambda r: r[0])
        if len(rows) < 2:
            raise ValueError("Trace must contain at least two points")
        self.trace = rows
        self.loop = loop

    # ------------------------------------------------------------------
    def assign(self, node) -> None:
        node.trace_index = 0
        node.x = self.trace[0][1]
        node.y = self.trace[0][2]
        node.altitude = self.trace[0][3]
        node.last_move_time = self.trace[0][0]

    # ------------------------------------------------------------------
    def move(self, node, current_time: float) -> None:
        if current_time <= node.last_move_time:
            return
        while (
            node.trace_index < len(self.trace) - 1
            and current_time >= self.trace[node.trace_index + 1][0]
        ):
            node.trace_index += 1
            if node.trace_index >= len(self.trace) - 1:
                if self.loop:
                    node.trace_index = 0
                    current_time = current_time % self.trace[-1][0]
                else:
                    node.last_move_time = current_time
                    return
        t0, x0, y0, z0 = self.trace[node.trace_index]
        t1, x1, y1, z1 = self.trace[node.trace_index + 1]
        ratio = (current_time - t0) / (t1 - t0)
        node.x = x0 + (x1 - x0) * ratio
        node.y = y0 + (y1 - y0) * ratio
        node.altitude = z0 + (z1 - z0) * ratio
        node.last_move_time = current_time
