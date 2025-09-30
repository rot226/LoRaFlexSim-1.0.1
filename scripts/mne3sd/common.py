"""Shared helper utilities for MNE3SD scripts."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import warnings
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable as TypingIterable, Literal, TypeVar


WorkerCount = int | Literal["auto"]


def _parse_worker_argument(value: str) -> WorkerCount:
    """Return a worker specification parsed from ``value``."""

    if value.lower() == "auto":
        return "auto"
    try:
        workers = int(value)
    except ValueError as exc:  # pragma: no cover - defensive programming
        raise argparse.ArgumentTypeError(
            "--workers must be a positive integer or 'auto'"
        ) from exc
    if workers <= 0:
        raise argparse.ArgumentTypeError("--workers must be a positive integer")
    return workers


def _normalise_worker_default(default: WorkerCount) -> WorkerCount:
    """Validate and normalise the default ``--workers`` value."""

    if isinstance(default, str):
        if default.lower() != "auto":
            raise ValueError("default must be a positive integer or 'auto'")
        return "auto"
    if default <= 0:
        raise ValueError("default must be a positive integer")
    return int(default)


def add_worker_argument(parser, *, default: WorkerCount = 1) -> None:
    """Attach a shared ``--workers`` option that accepts integers or ``'auto'``."""

    parser.add_argument(
        "--workers",
        type=_parse_worker_argument,
        default=_normalise_worker_default(default),
        help="Number of parallel worker processes to use (integer or 'auto')",
    )


def resolve_worker_count(workers: WorkerCount, task_count: int) -> int:
    """Return the effective worker count limited by ``task_count``."""

    limited_tasks = max(0, int(task_count))
    if limited_tasks == 0:
        return 0

    if isinstance(workers, str):
        if workers != "auto":  # pragma: no cover - defensive programming
            raise ValueError("workers must be an integer or 'auto'")
        available = os.cpu_count() or 1
        return max(1, min(available, limited_tasks))

    if workers <= 0:  # pragma: no cover - defensive programming
        raise ValueError("workers must be positive")
    return max(1, min(int(workers), limited_tasks))

import matplotlib.pyplot as plt

PROFILE_CHOICES = ("full", "fast", "ci")
PROFILE_ENV_VAR = "MNE3SD_PROFILE"
PROFILE_HELP = (
    "Execution profile preset. 'full' keeps the publication-grade defaults while "
    "'fast' trims simulation sizes for local iteration and 'ci' minimises the "
    "workload for automated smoke tests. The preset can also be supplied through "
    f"the {PROFILE_ENV_VAR} environment variable."
)


T_Task = TypeVar("T_Task")
T_Result = TypeVar("T_Result")


def ensure_directory(path: str | Path) -> Path:
    """Ensure that ``path`` exists and return the created directory."""

    directory = Path(path)
    if directory.suffix and not directory.is_dir():
        directory = directory.parent
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def prepare_figure_directory(
    *,
    article: str,
    scenario: str,
    metric: str,
    base_dir: str | Path | None = None,
) -> Path:
    """Return the output directory for the given article/scenario/metric trio."""

    components = {"article": article, "scenario": scenario, "metric": metric}
    missing = [name for name, value in components.items() if not value]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing figure directory component(s): {joined}")

    if base_dir is None:
        base_path = Path(__file__).resolve().parents[2] / "figures" / "mne3sd"
    else:
        base_path = Path(base_dir)

    return ensure_directory(base_path / article / scenario / metric)


def apply_ieee_style(figsize: tuple[float, float] = (3.5, 2.2)) -> None:
    """Apply a compact IEEE-friendly Matplotlib style."""

    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "figure.figsize": figsize,
        }
    )


def save_figure(
    fig: "plt.Figure",
    basename: str | Path,
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Save ``fig`` as PNG and EPS files inside ``output_dir``."""

    output_base = ensure_directory(output_dir) / Path(basename)
    png_path = output_base.with_suffix(".png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    eps_path = output_base.with_suffix(".eps")
    fig.savefig(eps_path, dpi=dpi, format="eps", bbox_inches="tight")
    return png_path, eps_path


def write_csv(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]] | Iterable[dict[str, Any]],
) -> Path:
    """Write ``rows`` to ``path`` using the provided ``fieldnames``."""

    file_path = Path(path)
    ensure_directory(file_path.parent)
    with file_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return file_path


def filter_completed_tasks(
    csv_path: Path,
    keys: tuple[str, ...],
    tasks: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return ``tasks`` without entries already present in ``csv_path``.

    The CSV is read using :class:`csv.DictReader` and the provided ``keys`` are
    used to identify completed replicates.  Both CSV and task values are
    coerced to strings to ensure consistent comparisons regardless of the
    original types used when scheduling the simulations.
    """

    if not tasks:
        return []

    csv_file = Path(csv_path)
    if not csv_file.exists():
        return tasks

    with csv_file.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        completed = {
            tuple("" if row.get(key) is None else str(row.get(key)) for key in keys)
            for row in reader
        }

    if not completed:
        return tasks

    filtered_tasks: list[dict[str, object]] = []
    for task in tasks:
        signature = tuple(
            "" if task.get(key) is None else str(task.get(key)) for key in keys
        )
        if signature not in completed:
            filtered_tasks.append(task)

    return filtered_tasks


def summarise_metrics(
    data: Iterable[Mapping[str, Any]],
    group_keys: Sequence[str],
    value_keys: Sequence[str],
) -> list[dict[str, Any]]:
    """Return mean and population standard deviation for ``value_keys``."""

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for entry in data:
        key = tuple(entry.get(group_key) for group_key in group_keys)
        grouped[key].append(entry)

    summaries: list[dict[str, Any]] = []
    for key, entries in grouped.items():
        summary = {group_key: value for group_key, value in zip(group_keys, key)}
        for value_key in value_keys:
            values: list[float] = []
            for entry in entries:
                value = entry.get(value_key)
                if isinstance(value, (int, float)):
                    values.append(float(value))
                elif isinstance(value, str):
                    stripped = value.strip()
                    if stripped:
                        try:
                            values.append(float(stripped))
                        except ValueError:
                            continue
            if values:
                summary[f"{value_key}_mean"] = statistics.fmean(values)
                summary[f"{value_key}_std"] = (
                    statistics.pstdev(values) if len(values) > 1 else 0.0
                )
            else:
                summary[f"{value_key}_mean"] = ""
                summary[f"{value_key}_std"] = ""
        summaries.append(summary)
    return summaries


def execute_simulation_tasks(
    tasks: TypingIterable[T_Task],
    worker: Callable[[T_Task], T_Result],
    *,
    max_workers: int = 1,
    progress_callback: Callable[[T_Task, T_Result, int], None] | None = None,
) -> list[T_Result]:
    """Execute ``worker`` for every task with optional ``ProcessPoolExecutor`` support."""

    task_list = list(tasks)
    if not task_list:
        return []

    workers = max(1, int(max_workers))
    workers = min(workers, len(task_list))

    if workers == 1:
        results: list[T_Result] = []
        for index, task in enumerate(task_list):
            result = worker(task)
            results.append(result)
            if progress_callback is not None:
                progress_callback(task, result, index)
        return results

    results_buffer: dict[int, T_Result] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(worker, task): (index, task)
            for index, task in enumerate(task_list)
        }
        for future in as_completed(future_map):
            index, task = future_map[future]
            result = future.result()
            results_buffer[index] = result
            if progress_callback is not None:
                progress_callback(task, result, index)

    return [results_buffer[index] for index in range(len(task_list))]


def add_execution_profile_argument(parser) -> None:
    """Attach a shared ``--profile`` option to ``parser``."""

    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        default=None,
        help=PROFILE_HELP,
    )


def resolve_execution_profile(
    selected: str | None, *, env_var: str = PROFILE_ENV_VAR
) -> str:
    """Return the execution profile resolved from CLI and environment input."""

    candidate = selected or os.getenv(env_var)
    if not candidate:
        return "full"
    profile = candidate.strip().lower()
    if profile in PROFILE_CHOICES:
        return profile
    warnings.warn(
        f"Unknown execution profile '{candidate}', falling back to 'full'.",
        RuntimeWarning,
        stacklevel=2,
    )
    return "full"

