import configparser
import json
import re
from pathlib import Path

YamlData = dict[str, object]

def load_config(
    path: str | Path,
) -> tuple[list[dict], list[dict], float | None, float | None]:
    """Load node and gateway positions from an INI or JSON configuration file.

    ``path`` may point to a classic FLoRa-style INI file or a JSON document
    describing the scenario.  INI files must define optional ``[gateways]`` and
    ``[nodes]`` sections where each value is ``x,y[,sf,tx_power]``. JSON files
    should contain two lists ``nodes`` and ``gateways`` with dictionaries.  SF
    defaults to 7 and TX power to 14 dBm when omitted.

    Returns two lists of dictionaries for nodes and gateways respectively.
    When ``path`` points to a FLoRa compatible INI file, the mean values for
    ``timeToNextPacket`` and ``timeToFirstPacket`` (when present) are also
    returned.  ``None`` is returned for each value if the parameter cannot be
    found or when loading a JSON file.
    """

    path = Path(path)
    nodes: list[dict] = []
    gateways: list[dict] = []

    next_interval = None
    first_interval = None

    suffix = path.suffix.lower()
    if suffix in {".json", ".yaml", ".yml"}:
        text = path.read_text()
        data: YamlData | None = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            if suffix == ".json":
                raise
        if data is None:
            try:  # pragma: no cover - optional dependency
                import yaml  # type: ignore
            except Exception as exc:  # pragma: no cover - handled via message
                raise RuntimeError(
                    "PyYAML is required to load YAML scenario files"
                ) from exc
            data = yaml.safe_load(text)
            if data is None:
                data = {}
        if not isinstance(data, dict):
            raise ValueError("YAML/JSON configuration must be a mapping at the top level")
        for gw in data.get("gateways", []):
            gateways.append({
                "x": float(gw.get("x", 0)),
                "y": float(gw.get("y", 0)),
                "tx_power": float(gw["tx_power"]) if "tx_power" in gw else None,
            })
        for nd in data.get("nodes", []):
            nodes.append({
                "x": float(nd.get("x", 0)),
                "y": float(nd.get("y", 0)),
                "sf": int(nd.get("sf", 7)),
                "tx_power": float(nd.get("tx_power", 14.0)),
            })
        return nodes, gateways, next_interval, first_interval

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
                "tx_power": float(parts[2]) if len(parts) > 2 else None,
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

    next_interval = parse_flora_interval(path)
    first_interval = parse_flora_first_interval(path)

    return nodes, gateways, next_interval, first_interval


def write_flora_ini(
    nodes: list[dict],
    gateways: list[dict],
    path: str | Path,
    *,
    next_interval: float | None = None,
    first_interval: float | None = None,
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
    next_interval : float, optional
        Mean delay between packets. When set, ``timeToNextPacket`` is appended to
        the INI file.
    first_interval : float, optional
        Mean delay before the first packet. When set,
        ``timeToFirstPacket`` is appended to the INI file.
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
        if first_interval is not None:
            f.write(f"timeToFirstPacket = exponential({first_interval}s)\n")
        if next_interval is not None:
            f.write(f"timeToNextPacket = exponential({next_interval}s)\n")


def parse_flora_interval(path: str | Path) -> float | None:
    """Return the mean packet interval defined in a FLoRa INI file.

    The function searches for a parameter ``timeToNextPacket`` expressed as
    ``exponential(<value>s)`` and returns ``<value>`` as a float. ``None`` is
    returned when the parameter cannot be found.
    """

    text = Path(path).read_text()
    match = re.search(r"timeToNextPacket\s*=\s*exponential\((\d+(?:\.\d+)?)s\)", text)
    if match:
        return float(match.group(1))
    return None


def parse_flora_first_interval(path: str | Path) -> float | None:
    """Return the mean delay before the first packet defined in a FLoRa INI.

    ``timeToFirstPacket`` is ignored when ``timeToNextPacket`` is present.  This
    helper therefore returns the same value as :func:`parse_flora_interval`
    whenever possible so that the first packet uses the same mean interval as
    the subsequent ones.
    """

    next_val = parse_flora_interval(path)
    if next_val is not None:
        return next_val

    text = Path(path).read_text()
    match = re.search(r"timeToFirstPacket\s*=\s*exponential\((\d+(?:\.\d+)?)s\)", text)
    if match:
        return float(match.group(1))
    return None
