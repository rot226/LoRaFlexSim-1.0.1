import configparser
from pathlib import Path
from typing import List, Tuple, Dict

def load_config(path: str | Path) -> tuple[list[dict], list[dict]]:
    """Load node and gateway positions from an INI-style configuration file.

    The file must define optional ``[gateways]`` and ``[nodes]`` sections. Each
    value is a comma-separated list ``x,y[,sf,tx_power]``. Example::

        [gateways]
        gw0 = 500,500

        [nodes]
        n0 = 450,490,12,14
        n1 = 555,563

    SF defaults to 7 and TX power to 14 dBm when omitted.
    Returns two lists of dictionaries for nodes and gateways respectively.
    """
    cp = configparser.ConfigParser()
    cp.read(path)
    nodes: list[dict] = []
    gateways: list[dict] = []

    if cp.has_section("gateways"):
        for _, value in cp.items("gateways"):
            parts = [p.strip() for p in value.split(",")]
            if len(parts) < 2:
                continue
            gateways.append({
                "x": float(parts[0]),
                "y": float(parts[1]),
            })

    if cp.has_section("nodes"):
        for _, value in cp.items("nodes"):
            parts = [p.strip() for p in value.split(",")]
            if len(parts) < 2:
                continue
            node = {
                "x": float(parts[0]),
                "y": float(parts[1]),
                "sf": int(parts[2]) if len(parts) > 2 else 7,
                "tx_power": float(parts[3]) if len(parts) > 3 else 14.0,
            }
            nodes.append(node)

    return nodes, gateways
