import configparser
import json
from pathlib import Path

def load_config(path: str | Path) -> tuple[list[dict], list[dict]]:
    """Load node and gateway positions from an INI or JSON configuration file.

    ``path`` may point to a classic FLoRa-style INI file or a JSON document
    describing the scenario.  INI files must define optional ``[gateways]`` and
    ``[nodes]`` sections where each value is ``x,y[,sf,tx_power]``. JSON files
    should contain two lists ``nodes`` and ``gateways`` with dictionaries.  SF
    defaults to 7 and TX power to 14 dBm when omitted.

    Returns two lists of dictionaries for nodes and gateways respectively.
    """

    path = Path(path)
    nodes: list[dict] = []
    gateways: list[dict] = []

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        for gw in data.get("gateways", []):
            gateways.append({
                "x": float(gw.get("x", 0)),
                "y": float(gw.get("y", 0)),
            })
        for nd in data.get("nodes", []):
            nodes.append({
                "x": float(nd.get("x", 0)),
                "y": float(nd.get("y", 0)),
                "sf": int(nd.get("sf", 7)),
                "tx_power": float(nd.get("tx_power", 14.0)),
            })
        return nodes, gateways

    cp = configparser.ConfigParser()
    cp.read(path)

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


def write_flora_ini(
    nodes: list[dict], gateways: list[dict], path: str | Path
) -> None:
    """Write ``nodes`` and ``gateways`` positions to a FLoRa compatible INI.

    Parameters
    ----------
    nodes : list of dict
        Each dictionary must contain ``x`` and ``y`` coordinates and may
        optionally define ``sf`` and ``tx_power``.
    gateways : list of dict
        Each dictionary must contain ``x`` and ``y`` coordinates.
    path : str or Path
        Destination file.
    """

    cp = configparser.ConfigParser()
    cp.optionxform = str  # keep case for section keys
    cp["gateways"] = {
        f"gw{i}": f"{gw['x']},{gw['y']}" for i, gw in enumerate(gateways)
    }
    cp["nodes"] = {
        f"n{i}": f"{nd['x']},{nd['y']},{nd.get('sf', 7)},{nd.get('tx_power', 14.0)}"
        for i, nd in enumerate(nodes)
    }
    with open(path, "w") as f:
        cp.write(f)
