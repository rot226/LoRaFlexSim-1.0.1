#!/usr/bin/env python3
"""Convert LoRa scenario files between JSON and FLoRa INI formats."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from VERSION_4.launcher.config_loader import load_config, write_flora_ini


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert scenario between JSON and FLoRa INI formats"
    )
    parser.add_argument("input", help="Input scenario (.ini or .json)")
    parser.add_argument(
        "output", help="Output file (.ini or .json)")
    args = parser.parse_args()

    nodes, gws = load_config(args.input)
    out = Path(args.output)
    if out.suffix.lower() == ".json":
        data = {"nodes": nodes, "gateways": gws}
        out.write_text(json.dumps(data, indent=2))
    else:
        write_flora_ini(nodes, gws, out)


if __name__ == "__main__":
    main()
