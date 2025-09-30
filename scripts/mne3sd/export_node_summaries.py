"""Generate summary tables across multiple node population experiments."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
import glob
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_METRICS = ("pdr", "collision_rate", "energy_per_node_J")


class SummaryError(RuntimeError):
    """Raised when the summary generation cannot proceed."""


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Return the parsed command line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-node experiment CSV files into publication-ready tables."
        )
    )
    parser.add_argument(
        "--inputs",
        metavar="PATTERN",
        action="append",
        required=True,
        help=(
            "Glob pattern to locate the CSV files to aggregate. The option can be "
            "provided multiple times."
        ),
    )
    parser.add_argument(
        "--group-columns",
        metavar="COLUMN",
        nargs="*",
        default=(),
        help=(
            "Optional column names used to split the results besides the node count."
            " For example 'class' or 'model range_km'."
        ),
    )
    parser.add_argument(
        "--metrics",
        metavar="NAME",
        nargs="*",
        default=DEFAULT_METRICS,
        help=(
            "Metrics to include in the table. Defaults to PDR, collision rate and "
            "energy per node."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Destination path for the aggregated CSV table.",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        required=True,
        help="Destination path for the LaTeX tabular output.",
    )
    parser.add_argument(
        "--tex-caption",
        default="",
        help="Optional caption used in the LaTeX table.",
    )
    parser.add_argument(
        "--tex-label",
        default="",
        help="Optional label attached to the LaTeX table.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Decimal precision for the rendered metrics (default: 3).",
    )
    return parser.parse_args(argv)


def resolve_input_paths(patterns: Iterable[str]) -> list[Path]:
    """Expand glob patterns to a sorted list of distinct paths."""

    seen: set[Path] = set()
    paths: list[Path] = []
    for pattern in patterns:
        for entry in glob.glob(pattern):
            path = Path(entry)
            if path not in seen:
                seen.add(path)
                paths.append(path)
    paths.sort()
    return paths


def _parse_nodes_from_name(path: Path) -> int | None:
    """Extract the node count from ``path`` using the ``_nodes_<N>`` suffix."""

    match = re.search(r"_nodes_(\d+)", path.stem)
    if match:
        try:
            return int(match.group(1))
        except ValueError:  # pragma: no cover - defensive
            return None
    return None


def _to_float(value: str | float | int | None) -> float | None:
    """Return ``value`` converted to ``float`` when possible."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    stripped = str(value).strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def load_measurements(
    files: Iterable[Path],
    metrics: Sequence[str],
    group_columns: Sequence[str],
) -> dict[tuple, dict[str, list[float]]]:
    """Collect metric samples grouped by node count and optional columns."""

    dataset: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for file_path in files:
        node_hint = _parse_nodes_from_name(file_path)
        if not file_path.exists():
            continue
        with file_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue
            for row in reader:
                replicate = str(row.get("replicate", "")).strip().lower()
                if replicate == "aggregate":
                    continue

                node_value: int | None = None
                node_entry = row.get("nodes")
                if node_entry not in (None, ""):
                    try:
                        node_value = int(float(node_entry))
                    except (TypeError, ValueError):
                        node_value = None
                if node_value is None:
                    node_value = node_hint
                if node_value is None:
                    continue

                key_parts: list = [node_value]
                for column in group_columns:
                    key_parts.append(row.get(column, ""))
                key = tuple(key_parts)

                for metric in metrics:
                    value = _to_float(row.get(metric))
                    if value is not None and math.isfinite(value):
                        dataset[key][metric].append(value)
    return dataset


def summarise_measurements(
    dataset: dict[tuple, dict[str, list[float]]],
    metrics: Sequence[str],
    group_columns: Sequence[str],
) -> list[dict[str, float | int | str]]:
    """Return mean and population std-dev for each node/group combination."""

    summaries: list[dict[str, float | int | str]] = []
    for key, metric_samples in dataset.items():
        if not metric_samples:
            continue
        entry: dict[str, float | int | str] = {}
        entry["nodes"] = key[0]
        for index, column in enumerate(group_columns, start=1):
            entry[column] = key[index]
        for metric in metrics:
            samples = metric_samples.get(metric, [])
            if not samples:
                entry[f"{metric}_mean"] = ""
                entry[f"{metric}_std"] = ""
                continue
            entry[f"{metric}_mean"] = statistics.fmean(samples)
            entry[f"{metric}_std"] = (
                statistics.pstdev(samples) if len(samples) > 1 else 0.0
            )
        summaries.append(entry)

    summaries.sort(
        key=lambda row: (
            int(row.get("nodes", 0)),
            *[row.get(column, "") for column in group_columns],
        )
    )
    return summaries


def write_csv_table(
    rows: Sequence[dict[str, float | int | str]],
    metrics: Sequence[str],
    group_columns: Sequence[str],
    destination: Path,
) -> None:
    """Persist ``rows`` as a CSV file."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["nodes", *group_columns]
    for metric in metrics:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])

    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def format_metric(mean: float | str, std: float | str, precision: int) -> str:
    """Return ``mean ± std`` formatted with ``precision`` decimals."""

    if mean in ("", None) or std in ("", None):
        return "--"
    return f"{float(mean):.{precision}f} \\pm {float(std):.{precision}f}"


def write_latex_table(
    rows: Sequence[dict[str, float | int | str]],
    metrics: Sequence[str],
    group_columns: Sequence[str],
    destination: Path,
    *,
    caption: str = "",
    label: str = "",
    precision: int = 3,
) -> None:
    """Render ``rows`` as a LaTeX ``tabular`` environment."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Nodes", *[column.replace("_", " ").title() for column in group_columns]]
    for metric in metrics:
        headers.append(metric.replace("_", " ").title())

    column_spec = "l" + "c" * (len(headers) - 1)
    lines = ["\\begin{table}[ht]", f"\\centering", f"\\begin{{tabular}}{{{column_spec}}}"]
    lines.append("\\toprule")
    lines.append(" & ".join(headers) + " " + "\\\\")
    lines.append("\\midrule")

    for row in rows:
        cells: list[str] = [str(row.get("nodes", ""))]
        for column in group_columns:
            cells.append(str(row.get(column, "")))
        for metric in metrics:
            cells.append(
                format_metric(
                    row.get(f"{metric}_mean", ""),
                    row.get(f"{metric}_std", ""),
                    precision,
                )
            )
        lines.append(" & ".join(cells) + " " + "\\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    destination.write_text("\n".join(lines) + "\n", encoding="utf8")


def generate_tables(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    """Build the summary and persist both CSV and LaTeX artefacts."""

    paths = resolve_input_paths(args.inputs)
    if not paths:
        raise SummaryError("No input CSV files matched the provided patterns.")

    dataset = load_measurements(paths, args.metrics, args.group_columns)
    if not dataset:
        raise SummaryError("No metric samples were extracted from the input files.")

    summaries = summarise_measurements(dataset, args.metrics, args.group_columns)
    if not summaries:
        raise SummaryError("Summary generation resulted in an empty table.")

    write_csv_table(summaries, args.metrics, args.group_columns, args.output_csv)
    write_latex_table(
        summaries,
        args.metrics,
        args.group_columns,
        args.output_tex,
        caption=args.tex_caption,
        label=args.tex_label,
        precision=args.precision,
    )
    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        generate_tables(args)
    except SummaryError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
