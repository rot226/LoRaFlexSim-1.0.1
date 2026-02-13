"""Smoke test pipeline article C.

Exécute un pipeline minimal :
1) Step1 + Step2 sur N=80, replications=1
2) agrégation Step1/Step2
3) make_all_plots
4) validations (agrégats, plots, légendes RL, absence de FAIL bloquant)

Retourne un code non-zéro en cas d'échec.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
ARTICLE_DIR = ROOT_DIR / "article_c"
STEP1_AGG = ARTICLE_DIR / "step1" / "results" / "aggregates" / "aggregated_results.csv"
STEP2_AGG = ARTICLE_DIR / "step2" / "results" / "aggregates" / "aggregated_results.csv"
LEGEND_REPORT = ARTICLE_DIR / "legend_check_report.csv"
STEP1_PLOTS_DIR = ARTICLE_DIR / "step1" / "plots" / "output"
STEP2_PLOTS_DIR = ARTICLE_DIR / "step2" / "plots" / "output"

REQUIRED_LEGEND_MODULES = {
    "article_c.step2.plots.plot_RL1",
    "article_c.step2.plots.plot_RL2",
    "article_c.step2.plots.plot_RL3",
    "article_c.step2.plots.plot_RL4",
    "article_c.step2.plots.plot_RL6_cluster_outage_vs_density",
    "article_c.step2.plots.plot_RL7_reward_vs_density",
}


def _run(cmd: list[str]) -> tuple[int, str]:
    print(f"$ {' '.join(cmd)}")
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        text=True,
        capture_output=True,
        check=False,
    )
    output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    if output.strip():
        print(output.strip())
    return completed.returncode, output


def _csv_has_rows(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return any(True for _ in reader)


def _count_recent_plot_files(directory: Path, *, since_epoch: float) -> int:
    if not directory.exists():
        return 0
    count = 0
    for path in directory.iterdir():
        if path.suffix.lower() not in {".png", ".pdf", ".eps", ".svg"}:
            continue
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            continue
        if mtime >= since_epoch:
            count += 1
    return count


def _check_required_legends() -> list[str]:
    errors: list[str] = []
    if not LEGEND_REPORT.exists():
        return [f"Rapport des légendes absent: {LEGEND_REPORT}"]

    by_module: dict[str, dict[str, str]] = {}
    with LEGEND_REPORT.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            module = str(row.get("module", "")).strip()
            if module:
                by_module[module] = row

    for module in sorted(REQUIRED_LEGEND_MODULES):
        row = by_module.get(module)
        if row is None:
            errors.append(f"Légende non vérifiée (module absent du rapport): {module}")
            continue
        status = str(row.get("status", "")).strip().upper()
        legend_entries = str(row.get("legend_entries", "")).strip()
        try:
            legend_entries_value = int(float(legend_entries or "0"))
        except ValueError:
            legend_entries_value = 0
        if status != "PASS" or legend_entries_value <= 0:
            errors.append(
                f"Légende invalide pour {module}: status={status}, legend_entries={legend_entries_value}"
            )
    return errors


def main() -> int:
    failures: list[str] = []

    make_all_plots_started_at = 0.0
    commands = [
        [
            sys.executable,
            "-m",
            "article_c.step1.run_step1",
            "--network-sizes",
            "80",
            "--replications",
            "1",
            "--workers",
            "1",
        ],
        [
            sys.executable,
            "-m",
            "article_c.step2.run_step2",
            "--network-sizes",
            "80",
            "--replications",
            "1",
            "--workers",
            "1",
        ],
        [sys.executable, "-m", "article_c.tools.aggregate_step1"],
        [sys.executable, "-m", "article_c.tools.aggregate_step2"],
        [
            sys.executable,
            "article_c/make_all_plots.py",
            "--network-sizes",
            "80",
        ],
    ]

    make_all_plots_output = ""
    for cmd in commands:
        if "article_c/make_all_plots.py" in " ".join(cmd):
            make_all_plots_started_at = time.time()
        rc, output = _run(cmd)
        if rc != 0:
            failures.append(f"Commande en échec (code {rc}): {' '.join(cmd)}")
        if "article_c/make_all_plots.py" in " ".join(cmd):
            make_all_plots_output = output

    if not _csv_has_rows(STEP1_AGG):
        failures.append(f"Agrégat Step1 absent ou vide: {STEP1_AGG}")
    if not _csv_has_rows(STEP2_AGG):
        failures.append(f"Agrégat Step2 absent ou vide: {STEP2_AGG}")

    if _count_recent_plot_files(STEP1_PLOTS_DIR, since_epoch=make_all_plots_started_at) < 1:
        failures.append("Aucun plot Step1 généré pendant make_all_plots")
    if _count_recent_plot_files(STEP2_PLOTS_DIR, since_epoch=make_all_plots_started_at) < 1:
        failures.append("Aucun plot Step2 généré pendant make_all_plots")

    failures.extend(_check_required_legends())

    blocking_fail_markers = (
        "sortie finale bloquée",
        "ERREUR: sortie finale bloquée",
    )
    lowered_output = make_all_plots_output.lower()
    if any(marker.lower() in lowered_output for marker in blocking_fail_markers):
        failures.append("Détection d'un FAIL bloquant dans la sortie de make_all_plots")

    if failures:
        print("\nSMOKE PIPELINE: FAIL")
        for item in failures:
            print(f"- {item}")
        return 1

    print("\nSMOKE PIPELINE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
