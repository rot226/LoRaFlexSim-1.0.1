"""Run all MNE3SD article scenarios and plots in a single command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from scripts.mne3sd.common import add_execution_profile_argument, resolve_execution_profile

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


@dataclass(frozen=True)
class Task:
    """A runnable CLI module and the artefacts it generates."""

    module: str
    description: str
    outputs: tuple[Path, ...]


ARTICLE_SCENARIOS: dict[str, tuple[Task, ...]] = {
    "a": (
        Task(
            module="scripts.mne3sd.article_a.scenarios.run_class_density_sweep",
            description="Class density sweep",
            outputs=(Path("results/mne3sd/article_a/class_density_metrics.csv"),),
        ),
        Task(
            module="scripts.mne3sd.article_a.scenarios.run_class_downlink_energy_profile",
            description="Class downlink energy profile",
            outputs=(Path("results/mne3sd/article_a/class_downlink_energy.csv"),),
        ),
        Task(
            module="scripts.mne3sd.article_a.scenarios.simulate_energy_classes",
            description="Class energy consumption simulation",
            outputs=(
                Path("results/mne3sd/article_a/energy_consumption.csv"),
                Path("results/mne3sd/article_a/energy_consumption_summary.csv"),
            ),
        ),
        Task(
            module="scripts.mne3sd.article_a.scenarios.run_class_load_sweep",
            description="Class load sweep",
            outputs=(Path("results/mne3sd/article_a/class_load_metrics.csv"),),
        ),
    ),
    "b": (
        Task(
            module="scripts.mne3sd.article_b.scenarios.run_mobility_range_sweep",
            description="Mobility range sweep",
            outputs=(Path("results/mne3sd/article_b/mobility_range_metrics.csv"),),
        ),
        Task(
            module="scripts.mne3sd.article_b.scenarios.run_mobility_speed_sweep",
            description="Mobility speed sweep",
            outputs=(Path("results/mne3sd/article_b/mobility_speed_metrics.csv"),),
        ),
        Task(
            module="scripts.mne3sd.article_b.scenarios.run_mobility_gateway_sweep",
            description="Mobility gateway sweep",
            outputs=(Path("results/mne3sd/article_b/mobility_gateway_metrics.csv"),),
        ),
    ),
}


ARTICLE_PLOTS: dict[str, tuple[Task, ...]] = {
    "a": (
        Task(
            module="scripts.mne3sd.article_a.plots.plot_class_load_results",
            description="Class load plots",
            outputs=(
                Path(
                    "figures/mne3sd/article_a/class_load/energy_vs_interval/"
                    "class_energy_vs_interval.png"
                ),
                Path(
                    "figures/mne3sd/article_a/class_load/energy_vs_interval/"
                    "class_energy_vs_interval.eps"
                ),
                Path(
                    "figures/mne3sd/article_a/class_load/pdr_vs_interval/"
                    "class_pdr_vs_interval.png"
                ),
                Path(
                    "figures/mne3sd/article_a/class_load/pdr_vs_interval/"
                    "class_pdr_vs_interval.eps"
                ),
            ),
        ),
        Task(
            module="scripts.mne3sd.article_a.plots.plot_energy_duty_cycle",
            description="Energy consumption versus duty cycle plots",
            outputs=(
                Path(
                    "figures/mne3sd/article_a/energy_duty_cycle/"
                    "energy_per_node_vs_duty_cycle/"
                    "energy_per_node_vs_duty_cycle.png"
                ),
                Path(
                    "figures/mne3sd/article_a/energy_duty_cycle/"
                    "energy_per_node_vs_duty_cycle/"
                    "energy_per_node_vs_duty_cycle.eps"
                ),
                Path(
                    "figures/mne3sd/article_a/energy_duty_cycle/pdr_vs_duty_cycle/"
                    "pdr_vs_duty_cycle.png"
                ),
                Path(
                    "figures/mne3sd/article_a/energy_duty_cycle/pdr_vs_duty_cycle/"
                    "pdr_vs_duty_cycle.eps"
                ),
                Path(
                    "figures/mne3sd/article_a/energy_duty_cycle/"
                    "energy_breakdown_vs_duty_cycle/"
                    "energy_breakdown_vs_duty_cycle.png"
                ),
                Path(
                    "figures/mne3sd/article_a/energy_duty_cycle/"
                    "energy_breakdown_vs_duty_cycle/"
                    "energy_breakdown_vs_duty_cycle.eps"
                ),
            ),
        ),
        Task(
            module="scripts.mne3sd.article_a.plots.plot_class_downlink_energy",
            description="Class downlink energy plots",
            outputs=(
                Path(
                    "figures/mne3sd/article_a/class_downlink_energy/energy_breakdown/"
                    "energy_breakdown.png"
                ),
                Path(
                    "figures/mne3sd/article_a/class_downlink_energy/energy_breakdown/"
                    "energy_breakdown.eps"
                ),
                Path(
                    "figures/mne3sd/article_a/class_downlink_energy/pdr_comparison/"
                    "pdr_comparison.png"
                ),
                Path(
                    "figures/mne3sd/article_a/class_downlink_energy/pdr_comparison/"
                    "pdr_comparison.eps"
                ),
            ),
        ),
    ),
    "b": (
        Task(
            module="scripts.mne3sd.article_b.plots.plot_mobility_range_metrics",
            description="Mobility range plots",
            outputs=(
                Path(
                    "figures/mne3sd/article_b/mobility_range/pdr_vs_range/"
                    "pdr_vs_communication_range.png"
                ),
                Path(
                    "figures/mne3sd/article_b/mobility_range/pdr_vs_range/"
                    "pdr_vs_communication_range.eps"
                ),
                Path(
                    "figures/mne3sd/article_b/mobility_range/average_delay_vs_range/"
                    "average_delay_vs_communication_range.png"
                ),
                Path(
                    "figures/mne3sd/article_b/mobility_range/average_delay_vs_range/"
                    "average_delay_vs_communication_range.eps"
                ),
            ),
        ),
        Task(
            module="scripts.mne3sd.article_b.plots.plot_mobility_speed_metrics",
            description="Mobility speed plots",
            outputs=(
                Path(
                    "figures/mne3sd/article_b/mobility_speed/pdr_by_speed_profile/"
                    "pdr_by_speed_profile.png"
                ),
                Path(
                    "figures/mne3sd/article_b/mobility_speed/pdr_by_speed_profile/"
                    "pdr_by_speed_profile.eps"
                ),
                Path(
                    "figures/mne3sd/article_b/mobility_speed/"
                    "average_delay_by_speed_profile/"
                    "average_delay_by_speed_profile.png"
                ),
                Path(
                    "figures/mne3sd/article_b/mobility_speed/"
                    "average_delay_by_speed_profile/"
                    "average_delay_by_speed_profile.eps"
                ),
                Path(
                    "figures/mne3sd/article_b/mobility_speed/"
                    "pdr_heatmap_speed_profile_range/"
                    "pdr_heatmap_speed_profile_range.png"
                ),
                Path(
                    "figures/mne3sd/article_b/mobility_speed/"
                    "pdr_heatmap_speed_profile_range/"
                    "pdr_heatmap_speed_profile_range.eps"
                ),
            ),
        ),
        Task(
            module="scripts.mne3sd.article_b.plots.plot_mobility_gateway_metrics",
            description="Mobility gateway plots",
            outputs=(
                Path(
                    "figures/mne3sd/article_b/mobility_gateway/"
                    "pdr_distribution_by_gateway/"
                    "pdr_distribution_by_gateway.png"
                ),
                Path(
                    "figures/mne3sd/article_b/mobility_gateway/"
                    "pdr_distribution_by_gateway/"
                    "pdr_distribution_by_gateway.eps"
                ),
                Path(
                    "figures/mne3sd/article_b/mobility_gateway/"
                    "downlink_delay_vs_gateways/"
                    "average_downlink_delay_vs_gateways.png"
                ),
                Path(
                    "figures/mne3sd/article_b/mobility_gateway/"
                    "downlink_delay_vs_gateways/"
                    "average_downlink_delay_vs_gateways.eps"
                ),
                Path(
                    "figures/mne3sd/article_b/mobility_gateway/"
                    "model_comparison/"
                    "pdr_vs_delay_model_comparison.png"
                ),
                Path(
                    "figures/mne3sd/article_b/mobility_gateway/"
                    "model_comparison/"
                    "pdr_vs_delay_model_comparison.eps"
                ),
            ),
        ),
    ),
}


FIGURE_SUFFIXES = {".png", ".pdf", ".eps", ".svg"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Return the parsed command line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Execute all MNE3SD article scenarios and/or plotting scripts. "
            "Use '--profile fast' for quicker local iterations (recommended on "
            "Windows 11)."
        ),
    )
    add_execution_profile_argument(parser)
    parser.add_argument(
        "--article",
        choices=("a", "b", "both"),
        default="both",
        help="Select which article pipeline to run (defaults to both).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip the plotting stage for the selected articles.",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help=(
            "Skip tasks whose outputs exist and are newer than the corresponding"
            " script."
        ),
    )
    parser.add_argument(
        "--skip-scenarios",
        action="store_true",
        help="Skip the scenario stage for the selected articles.",
    )
    parser.add_argument(
        "--scenario-workers",
        type=int,
        default=None,
        help=(
            "Override the number of workers used by scenario modules that "
            "support parallel execution."
        ),
    )
    return parser.parse_args(argv)


def _resolve_module_source(module: str) -> Path | None:
    """Return the source file executed for the provided module."""

    parts = module.split(".")
    module_path = ROOT / Path(*parts)
    candidates = (
        module_path.with_suffix(".py"),
        module_path / "__main__.py",
        module_path / "__init__.py",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _outputs_are_fresh(task: Task, script_path: Path | None) -> bool:
    """Return whether the task outputs are newer than the script file."""

    if not task.outputs or script_path is None:
        return False

    try:
        script_mtime = script_path.stat().st_mtime
    except FileNotFoundError:
        return False

    for output in task.outputs:
        full_path = ROOT / output
        try:
            if not full_path.exists():
                return False
            if full_path.stat().st_mtime < script_mtime:
                return False
        except FileNotFoundError:
            return False
    return True


def execute_tasks(
    tasks: Iterable[Task],
    heading: str,
    *,
    reuse: bool = False,
    profile: str | None = None,
    scenario_workers: int | None = None,
) -> list[Path]:
    """Run the provided tasks and return the artefact paths they generate."""

    executed_outputs: list[Path] = []
    task_list = list(tasks)
    if not task_list:
        return executed_outputs

    print(f"\n=== {heading} ===")
    for task in task_list:
        print(f"→ {task.description} ({task.module})")
        script_path = _resolve_module_source(task.module)
        if reuse and _outputs_are_fresh(task, script_path):
            print("  ↺ Artefacts à jour, tâche ignorée (--reuse).")
            executed_outputs.extend(task.outputs)
            continue
        command = [PYTHON, "-m", task.module]
        is_scenario_module = ".scenarios." in task.module
        if profile and is_scenario_module:
            command.extend(["--profile", profile])
        if scenario_workers is not None and is_scenario_module:
            command.extend(["--workers", str(scenario_workers)])
        subprocess.run(command, check=True, cwd=ROOT)
        executed_outputs.extend(task.outputs)
    return executed_outputs


def summarise_outputs(paths: Iterable[Path]) -> None:
    """Print a grouped summary of the generated artefact paths."""

    unique_entries: list[tuple[Path, bool]] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        full_path = ROOT / path
        unique_entries.append((path, full_path.exists()))

    if not unique_entries:
        print("\nNo artefacts to report (all stages were skipped).")
        return

    print("\n=== Summary of generated artefacts ===")

    def print_group(title: str, predicate: Callable[[Path], bool]) -> None:
        group = [(p, exists) for p, exists in unique_entries if predicate(p)]
        if not group:
            return
        print(f"\n{title}:")
        for entry, exists in group:
            status = "✓" if exists else "✗"
            print(f"  [{status}] {entry.as_posix()}")

    print_group("CSV files", lambda p: p.suffix.lower() == ".csv")
    print_group("Figures", lambda p: p.suffix.lower() in FIGURE_SUFFIXES)
    print_group("Other artefacts", lambda p: p.suffix.lower() not in {".csv", *FIGURE_SUFFIXES})

    required_energy_files = (
        Path("results/mne3sd/article_a/energy_consumption.csv"),
        Path("results/mne3sd/article_a/energy_consumption_summary.csv"),
    )
    print("\nEnergy consumption files (Article A):")
    for path in required_energy_files:
        exists = (ROOT / path).exists()
        status = "✓" if exists else "✗"
        print(f"  [{status}] {path.as_posix()}")


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the batch execution script."""

    args = parse_args(argv)
    profile = resolve_execution_profile(getattr(args, "profile", None))
    scenario_profile = None if profile == "full" else profile

    if args.skip_plots and args.skip_scenarios:
        print("Both stages were skipped; nothing to do.")
        return

    selected_articles: tuple[str, ...]
    if args.article == "both":
        selected_articles = ("a", "b")
    else:
        selected_articles = (args.article,)

    all_outputs: list[Path] = []

    for article in selected_articles:
        if not args.skip_scenarios:
            tasks = ARTICLE_SCENARIOS.get(article, ())
            heading = f"Article {article.upper()} scenarios"
            all_outputs.extend(
                execute_tasks(
                    tasks,
                    heading,
                    reuse=args.reuse,
                    profile=scenario_profile,
                    scenario_workers=args.scenario_workers,
                )
            )
        else:
            print(f"\nSkipping scenarios for article {article.upper()}.")

        if not args.skip_plots:
            tasks = ARTICLE_PLOTS.get(article, ())
            heading = f"Article {article.upper()} plots"
            all_outputs.extend(
                execute_tasks(tasks, heading, reuse=args.reuse)
            )
        else:
            print(f"\nSkipping plots for article {article.upper()}.")

    summarise_outputs(all_outputs)


if __name__ == "__main__":
    main()
