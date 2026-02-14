#!/usr/bin/env python3
"""Compare two run_config.json files and print exact configuration differences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _compare(left: Any, right: Any, prefix: str = "") -> list[str]:
    diffs: list[str] = []
    if isinstance(left, dict) and isinstance(right, dict):
        keys = sorted(set(left) | set(right))
        for key in keys:
            path = f"{prefix}.{key}" if prefix else key
            if key not in left:
                diffs.append(f"+ {path} = {right[key]!r}")
            elif key not in right:
                diffs.append(f"- {path} = {left[key]!r}")
            else:
                diffs.extend(_compare(left[key], right[key], path))
        return diffs

    if isinstance(left, list) and isinstance(right, list):
        max_len = max(len(left), len(right))
        for idx in range(max_len):
            path = f"{prefix}[{idx}]"
            if idx >= len(left):
                diffs.append(f"+ {path} = {right[idx]!r}")
            elif idx >= len(right):
                diffs.append(f"- {path} = {left[idx]!r}")
            else:
                diffs.extend(_compare(left[idx], right[idx], path))
        return diffs

    if left != right:
        diffs.append(f"~ {prefix}: {left!r} -> {right!r}")
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare un ancien run_config (PDR non nul) et un run actuel (PDR nul) "
            "pour identifier la différence exacte de configuration."
        )
    )
    parser.add_argument("old_config", type=Path, help="run_config.json de référence")
    parser.add_argument("new_config", type=Path, help="run_config.json à diagnostiquer")
    args = parser.parse_args()

    old_payload = json.loads(args.old_config.read_text(encoding="utf-8"))
    new_payload = json.loads(args.new_config.read_text(encoding="utf-8"))

    diffs = _compare(old_payload, new_payload)
    if not diffs:
        print("Aucune différence de configuration détectée.")
        return 0

    print("Différences détectées :")
    for entry in diffs:
        print(entry)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
