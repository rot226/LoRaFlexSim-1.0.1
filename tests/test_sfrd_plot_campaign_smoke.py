from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_plot_campaign_smoke_generates_pngs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    logs_root = tmp_path / "logs"

    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(repo_root) if not pythonpath else f"{repo_root}{os.pathsep}{pythonpath}"
    )

    run_cmd = [
        sys.executable,
        "-m",
        "sfrd.cli.run_campaign",
        "--network-sizes",
        "8",
        "--replications",
        "1",
        "--seeds-base",
        "7",
        "--snir",
        "OFF,ON",
        "--algos",
        "ADR",
        "UCB",
        "--warmup-s",
        "0",
        "--logs-root",
        str(logs_root),
        "--precheck",
        "off",
        "--max-run-seconds",
        "120",
    ]
    subprocess.run(
        run_cmd,
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    manifests_dir = logs_root / "campaign_manifests"
    manifest_files = sorted(manifests_dir.glob("*.json"))
    assert manifest_files, "Aucun manifest de campagne généré."

    campaign_payload = json.loads(manifest_files[0].read_text(encoding="utf-8"))
    campaign_id = campaign_payload["campaign_id"]

    plot_cmd = [
        sys.executable,
        "-m",
        "sfrd.cli.plot_campaign",
        "--campaign-id",
        campaign_id,
        "--logs-root",
        str(logs_root),
        "--format",
        "png",
    ]
    subprocess.run(
        plot_cmd,
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    figures_dir = tmp_path / "figures" / campaign_id
    png_files = sorted(figures_dir.glob("*.png"))
    assert len(png_files) >= 3

    figures_manifest = figures_dir / "figures_manifest.json"
    assert figures_manifest.is_file()
