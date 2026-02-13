"""Génère tous les graphes de l'article C."""

from __future__ import annotations


LOG_LEVELS = {"quiet": 0, "info": 1, "debug": 2}
_CURRENT_LOG_LEVEL = LOG_LEVELS["info"]


def set_log_level(level: str) -> None:
    global _CURRENT_LOG_LEVEL
    _CURRENT_LOG_LEVEL = LOG_LEVELS[level]


def log_info(message: str) -> None:
    if _CURRENT_LOG_LEVEL >= LOG_LEVELS["info"]:
        print(message)


def log_debug(message: str) -> None:
    if _CURRENT_LOG_LEVEL >= LOG_LEVELS["debug"]:
        print(message)


def log_error(message: str) -> None:
    print(message, file=sys.stderr)


import argparse
import ast
import csv
import importlib
import inspect
import re
import statistics
import sys
import traceback
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


if find_spec("article_c") is None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    if find_spec("article_c") is None:
        raise ModuleNotFoundError(
            "Impossible d'importer 'article_c'. "
            "Ajoutez la racine du dépôt au PYTHONPATH."
        )

from article_c.common.expected_figures import EXPECTED_FIGURES_BY_STEP

ARTICLE_DIR = Path(__file__).resolve().parent
STEP1_RESULTS_DIR = ARTICLE_DIR / "step1" / "results"
STEP2_RESULTS_DIR = ARTICLE_DIR / "step2" / "results"
STEP1_PLOTS_OUTPUT_DIR = ARTICLE_DIR / "step1" / "plots" / "output"
STEP2_PLOTS_OUTPUT_DIR = ARTICLE_DIR / "step2" / "plots" / "output"

PLOT_MODULES = {
    "step1": [
        "article_c.step1.plots.plot_S1",
        "article_c.step1.plots.plot_S2",
        "article_c.step1.plots.plot_S3",
        "article_c.step1.plots.plot_S4",
        "article_c.step1.plots.plot_S5",
        "article_c.step1.plots.plot_S6",
        "article_c.step1.plots.plot_S6_cluster_pdr_vs_density",
        "article_c.step1.plots.plot_S6_cluster_pdr_vs_network_size",
        "article_c.step1.plots.plot_S7_cluster_outage_vs_density",
        "article_c.step1.plots.plot_S7_cluster_outage_vs_network_size",
        "article_c.step1.plots.plot_S8_spreading_factor_distribution",
        "article_c.step1.plots.plot_S9_latency_or_toa_vs_network_size",
        "article_c.step1.plots.plot_S10_rssi_cdf_by_algo",
        "article_c.step1.plots.plot_S10_rssi_or_snr_cdf",
        "article_c.step1.plots.plot_S_new1_pdr_cluster_paper",
        "article_c.step1.plots.plot_S_new2_throughput_cluster_global",
        "article_c.step1.plots.plot_S_new3_energy_per_delivered_packet",
        "article_c.step1.plots.plot_S_new4_interference_realism",
    ],
    "step2": [
        "article_c.step2.plots.plot_RL1",
        "article_c.step2.plots.plot_RL1_learning_curve_reward",
        "article_c.step2.plots.plot_RL2",
        "article_c.step2.plots.plot_RL3",
        "article_c.step2.plots.plot_RL4",
        "article_c.step2.plots.plot_RL5",
        "article_c.step2.plots.plot_RL5_plus",
        "article_c.step2.plots.plot_RL6_cluster_outage_vs_density",
        "article_c.step2.plots.plot_RL7_reward_vs_density",
        "article_c.step2.plots.plot_RL8_reward_distribution",
        "article_c.step2.plots.plot_RL9_sf_selection_entropy",
        "article_c.step2.plots.plot_RL10_reward_vs_pdr_scatter",
        "article_c.step2.plots.plot_R_new1_pdr_global",
        "article_c.step2.plots.plot_R_new2_energy_per_packet",
        "article_c.step2.plots.plot_R_new3_learning_curve",
        "article_c.step2.plots.plot_R_new4_sf_policy",
        "article_c.step2.plots.plot_R_new5_pdr_energy_tradeoff",
        "article_c.step2.plots.plot_R_figure3_ucb1_vs_baselines",
    ],
}

REQUIRED_PLOT_MODULES_BY_STEP: dict[str, set[str]] = {
    step: {module_path for module_path, _ in module_entries}
    for step, module_entries in EXPECTED_FIGURES_BY_STEP.items()
    if step in PLOT_MODULES
}

POST_PLOT_MODULES = [
    "article_c.reproduce_author_results",
    "article_c.compare_with_snir",
    "article_c.plot_cluster_der",
]

MANIFEST_OUTPUT_PATH = ARTICLE_DIR / "figures_manifest.csv"
PLOT_DATA_FILTER_REPORT_OUTPUT_PATH = ARTICLE_DIR / "plot_data_filter_report.csv"
LEGEND_CHECK_REPORT_OUTPUT_PATH = ARTICLE_DIR / "legend_check_report.csv"

ALGO_LEGEND_KEYWORDS = ("adr", "mixra", "ucb1")
STEP2_REQUIRED_LEGEND_MODULES = (
    "article_c.step2.plots.plot_RL1",
    "article_c.step2.plots.plot_RL2",
    "article_c.step2.plots.plot_RL3",
    "article_c.step2.plots.plot_RL4",
    "article_c.step2.plots.plot_RL5",
    "article_c.step2.plots.plot_RL6_cluster_outage_vs_density",
)

MAKE_ALL_PLOTS_PRESETS: dict[str, dict[str, object]] = {
    "ieee-ready-no-titles": {
        "formats": "png,eps,pdf",
        "no_suptitle": True,
    }
}

MANIFEST_STEP_OUTPUT_DIRS = {
    "step1": STEP1_PLOTS_OUTPUT_DIR,
    "step2": STEP2_PLOTS_OUTPUT_DIR,
    "post": ARTICLE_DIR / "plots" / "output",
}

MIN_NETWORK_SIZES_PER_PLOT = {
    "article_c.step2.plots.plot_RL1": 1,
    "article_c.step2.plots.plot_RL1_learning_curve_reward": 1,
    "article_c.step2.plots.plot_RL2": 1,
    "article_c.step2.plots.plot_RL3": 1,
    "article_c.step2.plots.plot_RL4": 1,
    "article_c.step2.plots.plot_RL5": 1,
    "article_c.step2.plots.plot_RL5_plus": 1,
    "article_c.step2.plots.plot_RL6_cluster_outage_vs_density": 1,
    "article_c.step2.plots.plot_RL7_reward_vs_density": 1,
    "article_c.step2.plots.plot_RL8_reward_distribution": 1,
    "article_c.step2.plots.plot_RL9_sf_selection_entropy": 1,
    "article_c.step2.plots.plot_RL10_reward_vs_pdr_scatter": 2,
}

REQUIRED_ALGOS = {
    "step1": ("adr", "mixra_h", "mixra_opt"),
    "step2": ("adr", "mixra_h", "mixra_opt", "ucb1_sf"),
}

REQUIRED_SNIR_MODES = {
    "step1": ("snir_on", "snir_off"),
    "step2": ("snir_on",),
}

ALGO_ALIASES = {
    "adr": "adr",
    "mixra_h": "mixra_h",
    "mixra_hybrid": "mixra_h",
    "mixra_opt": "mixra_opt",
    "mixra_optimal": "mixra_opt",
    "mixraopt": "mixra_opt",
    "ucb1_sf": "ucb1_sf",
    "ucb1sf": "ucb1_sf",
}

SNIR_ALIASES = {
    "snir_on": "snir_on",
    "on": "snir_on",
    "true": "snir_on",
    "1": "snir_on",
    "yes": "snir_on",
    "snir_off": "snir_off",
    "off": "snir_off",
    "false": "snir_off",
    "0": "snir_off",
    "no": "snir_off",
}

RSSI_SNR_COLUMNS = (
    "rssi_dbm",
    "rssi_db",
    "rssi",
    "snr_db",
    "snr_dbm",
    "snr",
)


@dataclass(frozen=True)
class PlotRequirements:
    csv_name: str = "aggregated_results.csv"
    min_network_sizes: int = 2
    require_algo_snir: bool = True
    required_algos: tuple[str, ...] | None = None
    required_snir: tuple[str, ...] | None = None
    required_any_columns: tuple[str, ...] | None = None
    extra_csv_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class CsvDataBundle:
    fieldnames: list[str]
    rows: list[dict[str, str]]
    label: str
    source_paths: tuple[Path, ...]


class MandatoryFigureDataError(RuntimeError):
    """Erreur contrôlée quand une figure obligatoire n'a pas ses données."""


PLOT_REQUIREMENTS = {
    "article_c.step1.plots.plot_S10_rssi_cdf_by_algo": PlotRequirements(
        csv_name="raw_packets.csv",
        min_network_sizes=1,
        require_algo_snir=True,
        required_algos=REQUIRED_ALGOS["step1"],
        required_snir=REQUIRED_SNIR_MODES["step1"],
        required_any_columns=RSSI_SNR_COLUMNS,
    ),
    "article_c.step1.plots.plot_S10_rssi_or_snr_cdf": PlotRequirements(
        csv_name="raw_packets.csv",
        min_network_sizes=1,
        require_algo_snir=True,
        required_algos=(),
        required_snir=(),
        required_any_columns=RSSI_SNR_COLUMNS,
    ),
    "article_c.step2.plots.plot_RL5": PlotRequirements(
        min_network_sizes=1,
        require_algo_snir=False,
        extra_csv_names=("rl5_selection_prob.csv",),
    ),
    "article_c.step2.plots.plot_RL5_plus": PlotRequirements(
        min_network_sizes=1,
        require_algo_snir=False,
        extra_csv_names=("rl5_selection_prob.csv",),
    ),
    "article_c.step2.plots.plot_RL9_sf_selection_entropy": PlotRequirements(
        min_network_sizes=1,
        require_algo_snir=False,
        extra_csv_names=("rl5_selection_prob.csv",),
    ),
    "article_c.step2.plots.plot_R_figure3_ucb1_vs_baselines": PlotRequirements(
        min_network_sizes=1,
        require_algo_snir=True,
    ),
}


def _coerce_csv_value(value: str | None) -> object:
    if value is None or value == "":
        return ""
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


def _load_csv_rows_coerced(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {key: _coerce_csv_value(value) for key, value in row.items()}
            for row in reader
        ]


def _collect_nested_csvs(results_dir: Path, filename: str) -> list[Path]:
    if filename == "aggregated_results.csv":
        return sorted(results_dir.glob("by_size/size_*/aggregated_results.csv"))
    return sorted(results_dir.glob(f"by_size/size_*/rep_*/{filename}"))


def _resolve_step_results_dir(step: str) -> Path:
    return STEP1_RESULTS_DIR if step == "step1" else STEP2_RESULTS_DIR


def _resolve_step_label(step: str) -> str:
    return "Step1" if step == "step1" else "Step2"


def _load_dataset_from_by_size(
    *,
    results_dir: Path,
    csv_name: str,
) -> CsvDataBundle | None:
    nested_paths = _collect_nested_csvs(results_dir, csv_name)
    if not nested_paths:
        return None
    fieldnames: list[str] | None = None
    rows: list[dict[str, str]] = []
    for nested_path in nested_paths:
        nested_fieldnames, nested_rows = _load_csv_data(nested_path)
        if fieldnames is None:
            fieldnames = nested_fieldnames
        elif nested_fieldnames != fieldnames:
            raise ValueError(
                "Colonnes incohérentes entre CSV imbriqués "
                f"({nested_path})."
            )
        rows.extend(nested_rows)
    if fieldnames is None or not rows:
        return None
    return CsvDataBundle(
        fieldnames=fieldnames,
        rows=rows,
        label=f"{results_dir}/by_size/size_*/{csv_name}",
        source_paths=tuple(nested_paths),
    )


def _report_missing_csv(
    *,
    step_label: str,
    results_dir: Path,
    source: str,
    csv_name: str,
) -> None:
    nested_pattern = results_dir / "by_size" / "size_*" / csv_name
    log_debug(f"ERREUR: CSV {step_label} introuvable pour source={source}.")
    log_debug(f"Pattern attendu: {nested_pattern}.")
    log_debug(
        "INFO: vérifiez que les simulations ont bien écrit dans "
        f"{results_dir}."
    )


def _resolve_data_bundle(
    *,
    step: str,
    csv_name: str,
    source: str,
    cache: dict[tuple[str, str], CsvDataBundle],
) -> CsvDataBundle | None:
    cache_key = (step, csv_name)
    if cache_key in cache:
        return cache[cache_key]
    results_dir = _resolve_step_results_dir(step)
    if source != "by_size":
        raise ValueError("Source CSV non supportée. Utilisez uniquement --source by_size.")
    bundle = _load_dataset_from_by_size(results_dir=results_dir, csv_name=csv_name)
    if bundle is not None:
        cache[cache_key] = bundle
    return bundle


def build_arg_parser() -> argparse.ArgumentParser:
    """Construit le parseur d'arguments CLI pour générer les figures."""
    parser = argparse.ArgumentParser(
        description="Génère toutes les figures à partir des CSV agrégés."
    )
    parser.add_argument(
        "--log-level",
        choices=("quiet", "info", "debug"),
        default="info",
        help="Niveau de logs (quiet, info, debug).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Alias de --log-level quiet.",
    )
    parser.add_argument(
        "--preset",
        choices=tuple(sorted(MAKE_ALL_PLOTS_PRESETS)),
        default=None,
        help=(
            "Préremplit un profil documenté. "
            "Preset 'ieee-ready-no-titles' => formats=png,eps,pdf "
            "et suppression du suptitle."
        ),
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="step1,step2",
        help="Étapes à tracer (ex: step1,step2).",
    )
    parser.add_argument(
        "--source",
        choices=("by_size",),
        default="by_size",
        help=(
            "Source des données CSV: by_size fusionne size_*/aggregated_results.csv en mémoire "
            "(défaut). Les variantes flat/historiques sont interdites."
        ),
    )
    parser.add_argument(
        "--network-sizes",
        dest="network_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Tailles de réseau attendues (ex: 50 100 150).",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="png",
        help="Formats d'export des figures (défaut: png, ex: png,pdf,eps).",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help=(
            "Retourne un code non nul si des plots échouent, "
            "sans interrompre l'exécution."
        ),
    )
    parser.add_argument(
        "--no-suptitle",
        action="store_true",
        help="Désactive le titre global (suptitle) des figures.",
    )
    parser.add_argument(
        "--no-figure-clamp",
        action="store_true",
        help="Désactive le clamp de taille des figures.",
    )
    parser.add_argument(
        "--skip-scientific-qa",
        action="store_true",
        help="N'exécute pas les contrôles QA scientifiques avant les plots.",
    )
    parser.add_argument(
        "--allow-scientific-qa-fail",
        action="store_true",
        help="Continue même si les contrôles QA scientifiques retournent FAIL.",
    )
    return parser


def _parse_steps(value: str) -> list[str]:
    steps = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [step for step in steps if step not in PLOT_MODULES]
    if unknown:
        raise ValueError(f"Étape(s) inconnue(s): {', '.join(unknown)}")
    return steps


def _run_plot_module(
    module_path: str,
    *,
    network_sizes: list[int] | None = None,
    allow_sample: bool = True,
    enable_suptitle: bool = True,
) -> object:
    module = importlib.import_module(module_path)
    if not hasattr(module, "main"):
        raise AttributeError(f"Module {module_path} sans fonction main().")
    signature = inspect.signature(module.main)
    kwargs: dict[str, object] = {}
    parameters = signature.parameters
    supports_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in parameters.values()
    )
    if "allow_sample" in parameters or supports_kwargs:
        kwargs["allow_sample"] = allow_sample
    if network_sizes is not None and ("network_sizes" in parameters or supports_kwargs):
        kwargs["network_sizes"] = network_sizes
    if "enable_suptitle" in parameters or supports_kwargs:
        kwargs["enable_suptitle"] = enable_suptitle
    if kwargs:
        module.main(**kwargs)
    else:
        module.main()
    return module


def _figure_has_legend(fig: plt.Figure) -> bool:
    if fig.legends:
        return True
    return any(ax.get_legend() is not None for ax in fig.axes)


def _is_algorithm_legend_label(label: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", label.strip().lower())
    return bool(normalized) and any(
        keyword in normalized for keyword in ALGO_LEGEND_KEYWORDS
    )


def _count_algorithm_legend_entries(fig: plt.Figure) -> int:
    count = 0
    for ax in fig.axes:
        _, labels = ax.get_legend_handles_labels()
        count += sum(
            1 for label in labels if label and _is_algorithm_legend_label(label)
        )
    for legend in fig.legends:
        count += sum(
            1
            for text in legend.get_texts()
            if text.get_text() and _is_algorithm_legend_label(text.get_text())
        )
    return count


def _check_legends_for_module(
    *,
    module_path: str,
    module: object,
    previous_figures: set[int],
    fail_on_missing_legends: bool = False,
    legend_status: dict[str, bool] | None = None,
    legend_report_rows: list[dict[str, object]] | None = None,
    legend_validity_by_module: dict[str, bool] | None = None,
) -> list[str]:
    from article_c.common.plot_helpers import assert_legend_present

    missing_contexts: list[str] = []
    new_fig_numbers = [
        num for num in plt.get_fignums() if num not in previous_figures
    ]
    if not new_fig_numbers:
        return []
    source_file = getattr(module, "__file__", None)
    source_path = (
        Path(source_file).resolve()
        if source_file
        else "source inconnue"
    )
    all_figs_have_legends = True
    module_algorithm_entries = 0
    place_adaptive_legend = getattr(module, "place_adaptive_legend", None)
    for index, fig_number in enumerate(new_fig_numbers, start=1):
        fig = plt.figure(fig_number)
        context = f"{module_path} (figure {index})"
        legend_count = 0
        for ax in fig.axes:
            _, labels = ax.get_legend_handles_labels()
            legend_count += len([label for label in labels if label])
        algo_entry_count = _count_algorithm_legend_entries(fig)
        module_algorithm_entries += algo_entry_count
        log_debug(
            "INFO: "
            f"module {module_path} - {context}: "
            f"{legend_count} légende(s) trouvée(s)."
        )
        if legend_count == 0:
            log_debug(
                "FAIL: "
                f"aucune légende détectée pour {context}. "
                "Chaque figure doit exposer une légende explicite "
                "(labels/legend)."
            )
        assert_legend_present(fig, context)
        has_legend = _figure_has_legend(fig)
        is_valid = has_legend and algo_entry_count > 0
        if not has_legend:
            all_figs_have_legends = False
            log_debug(
                "AVERTISSEMENT: "
                f"légende absente pour {context}. "
                f"Source: {source_path}"
            )
            if callable(place_adaptive_legend):
                best_ax = None
                best_count = 0
                for ax in fig.axes:
                    handles, labels = ax.get_legend_handles_labels()
                    if handles and len(handles) > best_count:
                        best_ax = ax
                        best_count = len(handles)
                if best_ax is None:
                    log_debug(
                        "AVERTISSEMENT: "
                        f"aucune entrée de légende détectée pour {context}; "
                        "place_adaptive_legend ignoré."
                    )
                else:
                    log_debug(
                        "AVERTISSEMENT: "
                        f"tentative de placement automatique de légende pour "
                        f"{context}."
                    )
                    try:
                        place_adaptive_legend(fig, best_ax)
                    except Exception as exc:
                        log_debug(
                            "AVERTISSEMENT: "
                            f"place_adaptive_legend a échoué pour {context}: {exc}"
                        )
                    else:
                        if _figure_has_legend(fig):
                            log_debug(
                                "INFO: "
                                f"légende ajoutée pour {context}."
                            )
                        else:
                            log_debug(
                                "AVERTISSEMENT: "
                                f"place_adaptive_legend n'a pas créé de légende "
                                f"pour {context}."
                            )
            else:
                log_debug(
                    "AVERTISSEMENT: "
                    f"place_adaptive_legend non exposé par {module_path}; "
                    f"légende absente pour {context}."
                )
        if fail_on_missing_legends and not is_valid:
            reason = (
                "légende absente"
                if not has_legend
                else "aucune entrée d'algorithme"
            )
            missing_contexts.append(f"{context} [{source_path}] ({reason})")
    if legend_status is not None:
        module_key = module_path.split(".")[-1]
        legend_status[module_key] = all_figs_have_legends and module_algorithm_entries > 0
    module_is_valid = all_figs_have_legends and module_algorithm_entries > 0
    if legend_report_rows is not None:
        legend_report_rows.append(
            {
                "module": module_path,
                "status": "PASS" if module_is_valid else "FAIL",
                "legend_entries": module_algorithm_entries,
            }
        )
    if legend_validity_by_module is not None:
        legend_validity_by_module[module_path] = module_is_valid
    return missing_contexts


def _write_legend_check_report(rows: list[dict[str, object]]) -> None:
    with LEGEND_CHECK_REPORT_OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["module", "status", "legend_entries"],
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: str(row["module"])))


def _resolve_plot_requirements(step: str, module_path: str) -> PlotRequirements:
    requirements = PLOT_REQUIREMENTS.get(module_path)
    if requirements is None:
        return PlotRequirements(
            required_algos=REQUIRED_ALGOS[step],
            required_snir=REQUIRED_SNIR_MODES[step],
        )
    if requirements.required_algos is None and requirements.require_algo_snir:
        requirements = PlotRequirements(
            **{
                **requirements.__dict__,
                "required_algos": REQUIRED_ALGOS[step],
            }
        )
    if requirements.required_snir is None and requirements.require_algo_snir:
        requirements = PlotRequirements(
            **{
                **requirements.__dict__,
                "required_snir": REQUIRED_SNIR_MODES[step],
            }
        )
    return requirements


def _validate_plot_modules_use_save_figure() -> dict[str, str]:
    missing: dict[str, str] = {}
    missing_save_figure: list[str] = []
    missing_plot_style: list[str] = []
    module_paths = [
        *[path for paths in PLOT_MODULES.values() for path in paths],
        *POST_PLOT_MODULES,
    ]
    for module_path in module_paths:
        spec = find_spec(module_path)
        if spec is None or spec.origin is None:
            missing[module_path] = "module introuvable"
            continue
        source_path = Path(spec.origin)
        try:
            source = source_path.read_text(encoding="utf-8")
        except OSError as exc:
            missing[module_path] = f"lecture impossible: {exc}"
            continue
        issues: list[str] = []
        if "save_figure(" not in source:
            issues.append("ne passe pas par save_figure")
            missing_save_figure.append(module_path)
        if "apply_plot_style(" not in source:
            issues.append("ne respecte pas apply_plot_style")
            missing_plot_style.append(module_path)
        if issues:
            missing[module_path] = ", ".join(issues)
    if missing_save_figure:
        log_debug(
            "ERREUR: certains scripts de plot ne passent pas par save_figure:\n"
            + "\n".join(f"- {item}" for item in missing_save_figure)
        )
    if missing_plot_style:
        log_debug(
            "ERREUR: certains scripts de plot ne respectent pas apply_plot_style:\n"
            + "\n".join(f"- {item}" for item in missing_plot_style)
        )
    return missing




def _validate_plot_modules_no_titles() -> dict[str, str]:
    violations: dict[str, str] = {}
    scoped_modules = [*PLOT_MODULES["step1"], *PLOT_MODULES["step2"], "article_c.reproduce_author_results"]
    for module_path in scoped_modules:
        spec = find_spec(module_path)
        if spec is None or spec.origin is None:
            violations[module_path] = "module introuvable"
            continue
        source_path = Path(spec.origin)
        try:
            source = source_path.read_text(encoding="utf-8")
        except OSError as exc:
            violations[module_path] = f"lecture impossible: {exc}"
            continue
        try:
            module_ast = ast.parse(source)
        except SyntaxError as exc:
            violations[module_path] = f"analyse AST impossible: {exc}"
            continue

        forbidden_lines: list[int] = []
        for node in ast.walk(module_ast):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in {"set_title", "suptitle"}:
                forbidden_lines.append(getattr(node, "lineno", -1))
            elif isinstance(func, ast.Name) and func.id == "suptitle":
                forbidden_lines.append(getattr(node, "lineno", -1))
        if forbidden_lines:
            lines = ", ".join(str(line) for line in sorted({line for line in forbidden_lines if line > 0}))
            violations[module_path] = f"usage interdit de set_title/suptitle (lignes: {lines})"

    if violations:
        log_debug(
            "ERREUR: titres détectés dans des modules interdits (Step1/Step2/reproduction):\n"
            + "\n".join(f"- {module}: {reason}" for module, reason in violations.items())
        )
    return violations


def _preflight_validate_plot_modules() -> dict[str, str]:
    invalid_modules = _validate_plot_modules_use_save_figure()
    invalid_modules.update(_validate_plot_modules_no_titles())
    if invalid_modules:
        log_debug(
            "ERREUR: modules de plots fautifs détectés avant exécution:\n"
            + "\n".join(
                f"- {module}: {reason}"
                for module, reason in sorted(invalid_modules.items())
            )
        )
    return invalid_modules

def _ast_int(node: ast.AST, default: int) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    return default


def _extract_save_figure_stems(module_ast: ast.Module) -> tuple[str, ...]:
    stems: list[str] = []
    for node in ast.walk(module_ast):
        if not isinstance(node, ast.Call):
            continue
        func_name = ""
        if isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id
        if func_name != "save_figure":
            continue
        stem_value: str | None = None
        if len(node.args) >= 3 and isinstance(node.args[2], ast.Constant):
            if isinstance(node.args[2].value, str):
                stem_value = node.args[2].value
        for keyword in node.keywords:
            if keyword.arg == "stem" and isinstance(keyword.value, ast.Constant):
                if isinstance(keyword.value.value, str):
                    stem_value = keyword.value.value
        if stem_value:
            stems.append(stem_value)
    return tuple(dict.fromkeys(stems))


def _infer_panel_count(module_ast: ast.Module) -> int:
    panel_count = 1
    for node in ast.walk(module_ast):
        if not isinstance(node, ast.Call):
            continue
        func_name = ""
        if isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id
        if func_name != "subplots":
            continue
        nrows, ncols = 1, 1
        if node.args:
            nrows = _ast_int(node.args[0], default=1)
        if len(node.args) >= 2:
            ncols = _ast_int(node.args[1], default=1)
        for keyword in node.keywords:
            if keyword.arg == "nrows":
                nrows = _ast_int(keyword.value, default=nrows)
            if keyword.arg == "ncols":
                ncols = _ast_int(keyword.value, default=ncols)
        panel_count = max(panel_count, max(1, nrows * ncols))
    return panel_count


def _infer_metric(short_description: str, module_path: str) -> str:
    lowered = f"{short_description} {module_path}".lower()
    keyword_to_metric = (
        ("pdr", "pdr"),
        ("outage", "outage"),
        ("throughput", "throughput"),
        ("energy", "energy"),
        ("reward", "reward"),
        ("latency", "latency"),
        ("toa", "toa"),
        ("rssi", "rssi"),
        ("snr", "snr"),
        ("entropy", "entropy"),
        ("sf", "sf_policy"),
        ("der", "der"),
    )
    for keyword, metric in keyword_to_metric:
        if keyword in lowered:
            return metric
    match = re.search(r"figure\s+([A-Za-z0-9_+\-]+)", short_description)
    if match:
        return match.group(1).lower()
    return module_path.split(".")[-1]


def _extract_plot_metadata(module_path: str) -> tuple[str, str, int, tuple[str, ...]]:
    spec = importlib.util.find_spec(module_path)
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(f"Module introuvable: {module_path}")
    source = Path(spec.origin).read_text(encoding="utf-8")
    module_ast = ast.parse(source)
    doc = ast.get_docstring(module_ast) or ""
    short_description = doc.strip().splitlines()[0].strip() if doc.strip() else module_path
    metric = _infer_metric(short_description, module_path)
    panel_count = _infer_panel_count(module_ast)
    stems = _extract_save_figure_stems(module_ast)
    if not stems:
        stems = (module_path.split(".")[-1],)
    return metric, short_description, panel_count, stems


def _format_sizes(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def _format_size_dirs(values: tuple[int, ...]) -> str:
    return ";".join(f"size_{value}" for value in values)


def _is_required_plot_module(step: str, module_path: str) -> bool:
    return module_path in REQUIRED_PLOT_MODULES_BY_STEP.get(step, set())


def _compute_size_status(
    *,
    requested_sizes: tuple[int, ...],
    detected_sizes: tuple[int, ...],
    min_network_sizes: int,
) -> tuple[str, str]:
    if not requested_sizes:
        return "OK", "aucune taille demandée"
    missing_sizes = sorted(set(requested_sizes) - set(detected_sizes))
    if not missing_sizes:
        return "OK", "tailles demandées présentes"
    missing_label = ", ".join(str(size) for size in missing_sizes)
    severity = "FAIL" if min_network_sizes > 1 else "WARN"
    return severity, f"tailles demandées manquantes: {missing_label}"


def _write_figures_manifest(
    export_formats: tuple[str, ...],
    manifest_context_by_module: dict[str, ManifestContext],
) -> None:
    rows: list[dict[str, str | int]] = []
    missing_files: list[Path] = []
    for step, module_entries in EXPECTED_FIGURES_BY_STEP.items():
        output_dir = MANIFEST_STEP_OUTPUT_DIRS[step]
        for module_path, stems in module_entries:
            metric, short_description, panel_count, _ = _extract_plot_metadata(module_path)
            context = manifest_context_by_module.get(module_path, ManifestContext())
            for stem in stems:
                for fmt in export_formats:
                    filename = f"{stem}.{fmt}"
                    full_path = output_dir / filename
                    rows.append(
                        {
                            "figure": stem,
                            "module": module_path,
                            "filename": str(full_path.relative_to(ARTICLE_DIR)),
                            "metric": metric,
                            "short_description": short_description,
                            "step": step,
                            "panel_count": panel_count,
                            "csv_source_paths": ";".join(
                                str(path.relative_to(ARTICLE_DIR))
                                for path in context.csv_source_paths
                            ),
                            "size_paths": _format_size_dirs(context.detected_sizes),
                            "detected_sizes": _format_sizes(context.detected_sizes),
                            "requested_sizes": _format_sizes(context.requested_sizes),
                            "filtered_rows": context.filtered_row_count,
                            "size_status": context.size_status,
                            "size_status_detail": context.size_message,
                            "exists": int(full_path.exists()),
                        }
                    )
                    if not full_path.exists():
                        missing_files.append(full_path)
    with MANIFEST_OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "figure",
                "module",
                "filename",
                "metric",
                "short_description",
                "step",
                "panel_count",
                "csv_source_paths",
                "size_paths",
                "detected_sizes",
                "requested_sizes",
                "filtered_rows",
                "size_status",
                "size_status_detail",
                "exists",
            ],
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (str(row["step"]), str(row["filename"]))))
    log_debug(f"Manifest des figures écrit: {MANIFEST_OUTPUT_PATH}")
    if missing_files:
        log_debug(
            "AVERTISSEMENT: certaines figures attendues sont absentes "
            f"({len(sorted(set(missing_files)))}) ; voir la colonne 'exists' du manifest."
        )


def _validate_step2_plot_module_registry() -> list[str]:
    step2_dir = ARTICLE_DIR / "step2" / "plots"
    if not step2_dir.exists():
        log_debug(
            "AVERTISSEMENT: dossier Step2 plots introuvable, "
            "impossible de vérifier la liste PLOT_MODULES."
        )
        return []
    discovered = {
        f"article_c.step2.plots.{path.stem}"
        for path in step2_dir.glob("plot_*.py")
    }
    missing = sorted(discovered - set(PLOT_MODULES["step2"]))
    if missing:
        log_debug(
            "AVERTISSEMENT: certains modules Step2 ne sont pas listés "
            "dans PLOT_MODULES['step2']:\n"
            + "\n".join(f"- {module}" for module in missing)
        )
    return missing


def _inspect_plot_outputs(
    output_dir: Path,
    label: str,
    formats: list[str],
) -> None:
    if not output_dir.exists():
        log_debug(
            "AVERTISSEMENT: "
            f"dossier de sortie absent pour {label}: {output_dir}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        if label == "Step1":
            formats_label = ",".join(formats) if formats else "png"
            log_debug(
                "INFO: relancez la commande suivante pour régénérer "
                "les figures Step1:"
            )
            log_debug(
                "python -m article_c.make_all_plots "
                f"--steps step1 --formats {formats_label}"
            )
        return
    if not formats:
        log_debug(f"INFO: aucun format d'export fourni pour {label}.")
        return
    primary_format = formats[0]
    primary_files = sorted(output_dir.glob(f"*.{primary_format}"))
    if not primary_files:
        log_debug(
            "AVERTISSEMENT: "
            f"aucun {primary_format.upper()} trouvé pour {label} dans {output_dir}."
        )
        return
    missing_variants: list[str] = []
    for primary_path in primary_files:
        stem = primary_path.stem
        for ext in formats:
            candidate = output_dir / f"{stem}.{ext}"
            if not candidate.exists():
                missing_variants.append(str(candidate))
    if missing_variants:
        log_debug(
            "AVERTISSEMENT: sorties manquantes pour le test visuel:\n"
            + "\n".join(f"- {path}" for path in missing_variants)
        )
    else:
        formats_label = "/".join(fmt.upper() for fmt in formats)
        log_debug(
            f"Test visuel: fichiers {formats_label} présents "
            f"pour {label} dans {output_dir}."
        )


def _resolve_module_key_for_stem(
    stem: str,
    known_modules: dict[str, bool],
) -> str | None:
    matches = [
        module_key for module_key in known_modules if stem.startswith(module_key)
    ]
    if not matches:
        return None
    return max(matches, key=len)


def _analyze_step1_pngs(
    output_dir: Path,
    legend_status_by_module: dict[str, bool],
) -> None:
    png_files = sorted(output_dir.glob("*.png"))
    if not png_files:
        log_debug("INFO: aucun PNG Step1 à analyser pour le rapport.")
        return
    sizes = []
    for path in png_files:
        try:
            with Image.open(path) as img:
                sizes.append(img.size)
        except OSError:
            continue
    if not sizes:
        log_debug("AVERTISSEMENT: impossible de lire les tailles des PNG Step1.")
        return
    widths = [width for width, _ in sizes]
    heights = [height for _, height in sizes]
    median_width = statistics.median(widths)
    median_height = statistics.median(heights)
    width_range = (median_width * 0.85, median_width * 1.15)
    height_range = (median_height * 0.85, median_height * 1.15)
    legend_ok = 0
    legend_missing = 0
    legend_unknown = 0
    size_ok = 0
    size_outliers: list[str] = []
    axes_ok = 0
    axes_unknown: list[str] = []
    for path in png_files:
        stem = path.stem
        module_key = _resolve_module_key_for_stem(stem, legend_status_by_module)
        legend_status = (
            legend_status_by_module.get(module_key)
            if module_key is not None
            else None
        )
        if legend_status is True:
            legend_ok += 1
        elif legend_status is False:
            legend_missing += 1
        else:
            legend_unknown += 1
        try:
            with Image.open(path) as img:
                width, height = img.size
                dpi = img.info.get("dpi")
        except OSError:
            size_outliers.append(f"{path.name} (lecture impossible)")
            axes_unknown.append(path.name)
            continue
        if (
            width_range[0] <= width <= width_range[1]
            and height_range[0] <= height <= height_range[1]
        ):
            size_ok += 1
        else:
            size_outliers.append(f"{path.name} ({width}x{height}px)")
        dpi_ok = False
        if dpi:
            try:
                dpi_x, dpi_y = dpi
                dpi_ok = min(float(dpi_x), float(dpi_y)) >= 90
            except (TypeError, ValueError):
                dpi_ok = False
        naming_ok = any(
            token in stem.lower()
            for token in ("axis", "axes", "xlabel", "ylabel")
        )
        if dpi_ok or (width >= 800 and height >= 600) or naming_ok:
            axes_ok += 1
        else:
            axes_unknown.append(path.name)
    total = len(png_files)
    log_debug("\nRapport Step1 (PNG):")
    log_debug(
        f"- Légendes: {legend_ok} OK / {legend_missing} manquantes / "
        f"{legend_unknown} inconnues (total {total})."
    )
    log_debug(
        f"- Tailles: médiane {int(median_width)}x{int(median_height)}px, "
        f"{size_ok} conformes / {len(size_outliers)} atypiques."
    )
    log_debug(
        f"- Axes lisibles: {axes_ok} OK / {len(axes_unknown)} à vérifier."
    )
    if size_outliers:
        log_debug("Détails tailles atypiques:")
        for item in size_outliers:
            log_debug(f"  - {item}")
    if legend_missing:
        log_debug(
            "Détails légendes manquantes: "
            "voir les logs de génération Step1 pour la liste complète."
        )
    if axes_unknown:
        log_debug("Détails axes à vérifier:")
        for item in axes_unknown:
            log_debug(f"  - {item}")


def _pick_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    return None


def _normalize_algo(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().lower().replace(" ", "_").replace("-", "_")
    if not cleaned:
        return None
    return ALGO_ALIASES.get(cleaned)


def _normalize_snir(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    return SNIR_ALIASES.get(cleaned)


def _extract_network_sizes_from_rows(
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> set[int]:
    if "network_size" in fieldnames:
        size_key = "network_size"
    elif "density" in fieldnames:
        size_key = "density"
    else:
        return set()
    sizes: set[int] = set()
    for row in rows:
        raw_value = row.get(size_key)
        if raw_value in (None, ""):
            continue
        try:
            sizes.add(int(float(raw_value)))
        except ValueError:
            continue
    return sizes


def _load_csv_data(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV introuvable: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = [row for row in reader]
    return (fieldnames, rows)


def _load_network_sizes_from_bundles(bundles: list[CsvDataBundle]) -> list[int]:
    sizes: set[int] = set()
    for bundle in bundles:
        log_debug(f"[network_sizes] Lecture de la source: {bundle.label}")
        fieldnames = bundle.fieldnames
        if "network_size" in fieldnames:
            size_column = "network_size"
        elif "density" in fieldnames:
            size_column = "density"
        else:
            raise ValueError(
                f"La source {bundle.label} doit contenir une colonne "
                "'network_size' ou 'density'."
            )
        log_debug(
            "[network_sizes] Colonne utilisée: "
            f"{size_column}"
            + (" (fallback depuis density)" if size_column == "density" else "")
        )
        raw_values = [row.get(size_column) for row in bundle.rows]
        non_null_values = [value for value in raw_values if value not in (None, "")]
        dropped_count = len(raw_values) - len(non_null_values)
        log_debug(
            f"[network_sizes] dropna sur '{size_column}': "
            f"{dropped_count} ligne(s) ignorée(s)."
        )
        unique_raw_values = sorted({str(value).strip() for value in non_null_values if str(value).strip()})
        raw_label = ", ".join(unique_raw_values) if unique_raw_values else "aucune"
        log_debug(f"[network_sizes] Tailles uniques brutes: {raw_label}")

        cast_errors: list[str] = []
        bundle_sizes: set[int] = set()
        for value in non_null_values:
            try:
                bundle_sizes.add(int(float(str(value))))
            except (TypeError, ValueError):
                cast_errors.append(str(value))
        if cast_errors:
            errors_label = ", ".join(sorted(set(cast_errors)))
            log_debug(
                "[network_sizes] Erreurs de cast vers int ignorées "
                f"({len(cast_errors)} ligne(s)): {errors_label}"
            )
        final_bundle_sizes = sorted(bundle_sizes)
        final_label = ", ".join(str(value) for value in final_bundle_sizes) or "aucune"
        log_debug(f"[network_sizes] Tailles finales conservées pour cette source: {final_label}")
        sizes.update(bundle_sizes)
    merged_sizes = sorted(sizes)
    merged_label = ", ".join(str(value) for value in merged_sizes) or "aucune"
    log_debug(f"[network_sizes] Tailles finales agrégées: {merged_label}")
    return merged_sizes


def _suggest_regeneration_command(path: Path, expected_sizes: list[int]) -> str | None:
    sizes = " ".join(str(size) for size in expected_sizes)
    if "step1" in path.parts:
        return (
            "python article_c/step1/run_step1.py "
            f"--network-sizes {sizes} --replications 5 --seeds_base 1000 "
            "--snir_modes snir_on,snir_off"
        )
    if "step2" in path.parts:
        return (
            "python article_c/step2/run_step2.py "
            f"--network-sizes {sizes} --replications 5 --seeds_base 1000"
        )
    return None


def _suggest_step2_resume_command(expected_sizes: list[int]) -> str:
    sizes = " ".join(str(size) for size in expected_sizes)
    return (
        "python article_c/step2/run_step2.py --resume "
        f"--network-sizes {sizes} --replications 5 --seeds_base 1000"
    )


def _validate_network_sizes(
    bundles: list[CsvDataBundle],
    expected_sizes: list[int],
) -> dict[str, list[int]]:
    expected_set = {int(size) for size in expected_sizes}
    missing_by_source: dict[str, list[int]] = {}
    for bundle in bundles:
        found_sizes = _extract_network_sizes_from_rows(bundle.fieldnames, bundle.rows)
        missing = sorted(expected_set - found_sizes)
        if missing:
            missing_by_source[bundle.label] = missing
            missing_list = ", ".join(str(size) for size in missing)
            message_lines = [
                "AVERTISSEMENT: tailles de réseau manquantes dans les résultats.",
                f"Source: {bundle.label}",
                f"Tailles attendues manquantes: {missing_list}.",
                "Les plots compatibles seront générés malgré tout.",
            ]
            log_debug("\n".join(message_lines))
    return missing_by_source


def _validate_plot_data(
    *,
    step: str,
    module_path: str,
    data_bundle: CsvDataBundle,
    requirements: PlotRequirements,
    expected_sizes: list[int] | None,
    source: str,
    required_figure: bool,
) -> tuple[bool, str, dict[str, object]]:
    def _log_filter(filter_name: str, before: int, after: int, detail: str = "") -> None:
        suffix = f" ({detail})" if detail else ""
        log_debug(
            f"[validate_plot_data] Filtre {filter_name}: "
            f"{before} -> {after} ligne(s){suffix}."
        )

    report: dict[str, object] = {
        "step": step,
        "module_path": module_path,
        "csv_path": data_bundle.label,
        "initial_rows": 0,
        "after_cast_type_rows": 0,
        "after_dropna_rows": 0,
        "after_algo_filter_rows": 0,
        "after_snir_filter_rows": 0,
        "after_cluster_filter_rows": 0,
        "detected_sizes": (),
        "requested_sizes": tuple(int(size) for size in expected_sizes or []),
        "status": "PENDING",
        "reason": "",
    }

    log_debug(f"[validate_plot_data] Module: {module_path}")
    log_debug(f"[validate_plot_data] CSV: {data_bundle.label}")
    for extra_name in requirements.extra_csv_names:
        missing_extra = not _collect_nested_csvs(_resolve_step_results_dir(step), extra_name)
        if missing_extra:
            log_debug(
                "AVERTISSEMENT: "
                f"{module_path} nécessite {extra_name}, figure ignorée."
            )
            report["status"] = "SKIP"
            report["reason"] = f"CSV manquant ({extra_name})"
            return False, f"CSV manquant ({extra_name})", report
    fieldnames, rows = data_bundle.fieldnames, data_bundle.rows
    log_debug(
        "[validate_plot_data] Colonnes disponibles: "
        f"{', '.join(fieldnames) if fieldnames else 'aucune'}"
    )
    report["initial_rows"] = len(rows)
    report["after_cast_type_rows"] = len(rows)
    report["after_dropna_rows"] = len(rows)
    report["after_algo_filter_rows"] = len(rows)
    report["after_snir_filter_rows"] = len(rows)
    report["after_cluster_filter_rows"] = len(rows)
    if not fieldnames:
        if required_figure:
            raise MandatoryFigureDataError(
                f"Figure obligatoire {module_path}: CSV vide ({data_bundle.label})."
            )
        log_debug(
            "AVERTISSEMENT: "
            f"CSV vide pour {module_path}, figure ignorée."
        )
        report["status"] = "SKIP"
        report["reason"] = "CSV vide"
        return False, "CSV vide", report
    sizes = _extract_network_sizes_from_rows(fieldnames, rows)
    report["detected_sizes"] = tuple(sorted(int(size) for size in sizes))
    if "network_size" in fieldnames:
        size_col = "network_size"
    elif "density" in fieldnames:
        size_col = "density"
    else:
        size_col = None
    if size_col is None:
        _log_filter("fallback taille", len(rows), len(rows), "aucune colonne network_size/density")
    else:
        fallback_note = "fallback density" if size_col == "density" else "network_size"
        size_raw = [row.get(size_col) for row in rows]
        raw_unique_sizes = sorted(
            {str(value).strip() for value in size_raw if value not in (None, "") and str(value).strip()}
        )
        raw_sizes_label = ", ".join(raw_unique_sizes) if raw_unique_sizes else "aucune"
        final_sizes_label = ", ".join(str(size) for size in sorted(sizes)) or "aucune"
        log_debug(f"[validate_plot_data] Colonne taille utilisée: {size_col} ({fallback_note})")
        log_debug(f"[validate_plot_data] Tailles uniques brutes: {raw_sizes_label}")
        log_debug(f"[validate_plot_data] Tailles finales: {final_sizes_label}")
        dropped_empty = sum(1 for value in size_raw if value in (None, ""))
        cast_errors = 0
        for value in size_raw:
            if value in (None, ""):
                continue
            try:
                int(float(str(value)))
            except (TypeError, ValueError):
                cast_errors += 1
        _log_filter(
            "fallback taille",
            len(rows),
            len(rows) - dropped_empty - cast_errors,
            f"colonne={size_col}, dropna={dropped_empty}, cast_errors={cast_errors}",
        )
        report["after_cast_type_rows"] = len(rows) - cast_errors
        report["after_dropna_rows"] = len(rows) - cast_errors - dropped_empty
    if module_path in MIN_NETWORK_SIZES_PER_PLOT:
        min_network_sizes = MIN_NETWORK_SIZES_PER_PLOT[module_path]
    elif step == "step2":
        min_network_sizes = 1
    else:
        min_network_sizes = requirements.min_network_sizes
    if len(sizes) < min_network_sizes:
        sizes_label = ", ".join(str(size) for size in sorted(sizes)) or "aucune"
        log_debug(
            "Tailles détectées dans "
            f"{data_bundle.label}: {sizes_label}."
        )
        log_debug(
            "WARNING: "
            f"{module_path} nécessite au moins "
            f"{min_network_sizes} taille(s) disponible(s), "
            "figure ignorée."
        )
        log_debug(f"CSV path: {data_bundle.label}")
        report["status"] = "SKIP"
        report["reason"] = "tailles de réseau insuffisantes"
        return False, "tailles de réseau insuffisantes", report
    if expected_sizes:
        expected_set = {int(size) for size in expected_sizes}
        if sizes and sizes < expected_set:
            expected_label = ", ".join(str(size) for size in expected_sizes)
            sizes_label = ", ".join(str(size) for size in sorted(sizes))
            log_debug(
                "AVERTISSEMENT: "
                f"{module_path} est généré avec un jeu réduit "
                f"({sizes_label}) au lieu de {expected_label}."
            )
    if len(sizes) == 1 and min_network_sizes == 1:
        sizes_label = ", ".join(str(size) for size in sorted(sizes)) or "aucune"
        log_debug(
            "AVERTISSEMENT: "
            f"{module_path} est généré avec une seule taille "
            f"({sizes_label})."
        )
    if requirements.required_any_columns:
        metric_col = _pick_column(fieldnames, requirements.required_any_columns)
        if metric_col is None:
            log_debug(
                "AVERTISSEMENT: "
                f"{module_path} nécessite une colonne RSSI/SNR, "
                "figure ignorée."
            )
            report["status"] = "SKIP"
            report["reason"] = "colonne RSSI/SNR manquante"
            return False, "colonne RSSI/SNR manquante", report
    if requirements.require_algo_snir:
        algo_col = _pick_column(fieldnames, ("algo", "algorithm", "method"))
        snir_col = _pick_column(
            fieldnames, ("snir_mode", "snir_state", "snir", "with_snir")
        )
        log_debug(
            "[validate_plot_data] Colonnes utilisées pour filtrage: "
            f"algo={algo_col or 'absente'}, snir={snir_col or 'absente'}"
        )
        if not algo_col or not snir_col:
            log_debug(
                "AVERTISSEMENT: "
                f"{module_path} nécessite les colonnes algo/snir_mode, "
                "figure ignorée."
            )
            report["status"] = "SKIP"
            report["reason"] = "colonnes algo/snir_mode manquantes"
            return False, "colonnes algo/snir_mode manquantes", report
        available_algos = {
            normalized
            for row in rows
            if (normalized := _normalize_algo(row.get(algo_col))) is not None
        }
        available_snir = {
            normalized
            for row in rows
            if (normalized := _normalize_snir(row.get(snir_col))) is not None
        }
        required_algos = requirements.required_algos or ()
        required_snir = requirements.required_snir or ()
        if required_algos:
            algo_filtered_rows = [
                row for row in rows if _normalize_algo(row.get(algo_col)) in required_algos
            ]
        else:
            algo_filtered_rows = list(rows)
        if required_snir:
            snir_filtered_rows = [
                row
                for row in algo_filtered_rows
                if _normalize_snir(row.get(snir_col)) in required_snir
            ]
        else:
            snir_filtered_rows = list(algo_filtered_rows)
        report["after_algo_filter_rows"] = len(algo_filtered_rows)
        report["after_snir_filter_rows"] = len(snir_filtered_rows)
        cluster_col = _pick_column(fieldnames, ("cluster", "cluster_id", "cluster_index"))
        if cluster_col:
            cluster_filtered_rows = [
                row
                for row in snir_filtered_rows
                if row.get(cluster_col) not in (None, "")
            ]
            report["after_cluster_filter_rows"] = len(cluster_filtered_rows)
            _log_filter(
                "cluster",
                len(snir_filtered_rows),
                len(cluster_filtered_rows),
                f"colonne={cluster_col}",
            )
        else:
            report["after_cluster_filter_rows"] = len(snir_filtered_rows)
            _log_filter("cluster", len(snir_filtered_rows), len(snir_filtered_rows), "non appliqué")
        _log_filter(
            "algo",
            len(rows),
            len(algo_filtered_rows),
            f"requis={required_algos or 'aucun (pass-through)'}",
        )
        _log_filter(
            "snir",
            len(algo_filtered_rows),
            len(snir_filtered_rows),
            f"requis={required_snir or 'aucun (pass-through)'}",
        )
        missing_algos = [
            algo for algo in required_algos if algo not in available_algos
        ]
        missing_snir = [
            mode for mode in required_snir if mode not in available_snir
        ]
        if missing_algos or missing_snir:
            sizes_label = ", ".join(str(size) for size in sorted(sizes)) or "aucune"
            log_debug(
                "Tailles détectées dans "
                f"{data_bundle.label}: {sizes_label}."
            )
            details = []
            if missing_algos:
                details.append(f"algos manquants: {', '.join(missing_algos)}")
            if missing_snir:
                details.append(f"SNIR manquants: {', '.join(missing_snir)}")
            log_debug(
                "AVERTISSEMENT: "
                f"{module_path} incomplet ({' ; '.join(details)}), "
                "figure ignorée."
            )
            report["status"] = "SKIP"
            report["reason"] = "données incomplètes"
            return False, "données incomplètes", report
    report["status"] = "OK"
    report["reason"] = "OK"
    return True, "OK", report


def _write_plot_data_filter_report(report_rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "step",
        "module_path",
        "csv_path",
        "initial_rows",
        "after_cast_type_rows",
        "after_dropna_rows",
        "after_algo_filter_rows",
        "after_snir_filter_rows",
        "after_cluster_filter_rows",
        "detected_sizes",
        "requested_sizes",
        "status",
        "reason",
    ]
    with PLOT_DATA_FILTER_REPORT_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)


def _print_plot_data_filter_report_for_module(report_row: dict[str, object]) -> None:
    log_debug("[plot_data_filter_report] Détail module SKIP:")
    log_debug(
        "  - module="
        f"{report_row['module_path']}, status={report_row['status']}, "
        f"reason={report_row['reason']}"
    )
    log_debug(f"  - csv={report_row['csv_path']}")
    log_debug(
        "  - lignes: "
        f"initial={report_row['initial_rows']}, "
        f"après_cast_type={report_row['after_cast_type_rows']}, "
        f"après_dropna={report_row['after_dropna_rows']}, "
        f"après_algo={report_row['after_algo_filter_rows']}, "
        f"après_snir={report_row['after_snir_filter_rows']}, "
        f"après_cluster={report_row['after_cluster_filter_rows']}"
    )


@dataclass
class PlotStatus:
    step: str
    module_path: str
    status: str
    message: str


@dataclass
class ManifestContext:
    csv_source_paths: tuple[Path, ...] = ()
    requested_sizes: tuple[int, ...] = ()
    detected_sizes: tuple[int, ...] = ()
    filtered_row_count: int = 0
    size_status: str = "OK"
    size_message: str = "OK"


def _register_status(
    status_map: dict[str, PlotStatus],
    *,
    step: str,
    module_path: str,
    status: str,
    message: str,
) -> None:
    status_map[module_path] = PlotStatus(
        step=step,
        module_path=module_path,
        status=status,
        message=message,
    )


def _summarize_statuses(
    status_map: dict[str, PlotStatus],
    steps: list[str],
    post_modules: list[str],
) -> dict[str, int]:
    counts = {"OK": 0, "FAIL": 0, "SKIP": 0}
    log_info("\nRésumé d'exécution des plots:")
    for step in steps:
        log_debug(f"\n{step.upper()}:")
        for module_path in PLOT_MODULES[step]:
            entry = status_map.get(module_path)
            if entry is None:
                status_label = "SKIP"
                message = "Non exécuté."
            else:
                status_label = entry.status
                message = entry.message
            counts[status_label] = counts.get(status_label, 0) + 1
            log_debug(f"- {module_path}: {status_label} ({message})")
    if post_modules:
        log_debug("\nPOST:")
        for module_path in post_modules:
            entry = status_map.get(module_path)
            if entry is None:
                status_label = "SKIP"
                message = "Non exécuté."
            else:
                status_label = entry.status
                message = entry.message
            counts[status_label] = counts.get(status_label, 0) + 1
            log_debug(f"- {module_path}: {status_label} ({message})")
    total = sum(counts.values())
    log_debug(
        "\nBilan: "
        f"{counts['OK']} OK / {counts['FAIL']} FAIL / "
        f"{counts['SKIP']} SKIP (total {total})."
    )
    return counts


def _run_post_module(
    module_path: str,
    args_list: list[str],
    *,
    close_figures: bool,
) -> object:
    module = importlib.import_module(module_path)
    if not hasattr(module, "main"):
        raise AttributeError(f"Module {module_path} sans fonction main().")
    signature = inspect.signature(module.main)
    parameters = signature.parameters
    supports_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in parameters.values()
    )
    kwargs: dict[str, object] = {}
    if "argv" in parameters or supports_kwargs:
        kwargs["argv"] = args_list
    if "close_figures" in parameters or supports_kwargs:
        kwargs["close_figures"] = close_figures
    if kwargs:
        module.main(**kwargs)
    else:
        module.main()
    return module


def main(argv: list[str] | None = None) -> None:
    from article_c.common.plot_helpers import (
        parse_export_formats,
        set_default_figure_clamp_enabled,
        set_default_export_formats,
    )

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.quiet:
        args.log_level = "quiet"
    set_log_level(args.log_level)
    if args.preset is not None:
        preset_values = MAKE_ALL_PLOTS_PRESETS[args.preset]
        for key, value in preset_values.items():
            setattr(args, key, value)
    enable_suptitle = not args.no_suptitle
    try:
        export_formats = parse_export_formats(args.formats)
    except ValueError as exc:
        parser.error(str(exc))
    set_default_export_formats(export_formats)
    set_default_figure_clamp_enabled(not args.no_figure_clamp)
    status_map: dict[str, PlotStatus] = {}
    step1_legend_status: dict[str, bool] = {}
    legend_check_report_rows: list[dict[str, object]] = []
    legend_validity_by_module: dict[str, bool] = {}
    invalid_modules = _preflight_validate_plot_modules()
    _validate_step2_plot_module_registry()
    if invalid_modules:
        for step, module_paths in PLOT_MODULES.items():
            for module_path in module_paths:
                if module_path in invalid_modules:
                    _register_status(
                        status_map,
                        step=step,
                        module_path=module_path,
                        status="FAIL",
                        message=invalid_modules[module_path],
                    )
        for module_path in POST_PLOT_MODULES:
            if module_path in invalid_modules:
                _register_status(
                    status_map,
                    step="post",
                    module_path=module_path,
                    status="FAIL",
                    message=invalid_modules[module_path],
                )
    try:
        steps = _parse_steps(args.steps)
    except ValueError as exc:
        parser.error(str(exc))
    step_data_cache: dict[tuple[str, str], CsvDataBundle] = {}
    step_errors: dict[str, str] = {}
    step_primary_bundle: dict[str, CsvDataBundle] = {}
    step1_csv: Path | None = None
    step2_csv: Path | None = None

    for step in steps:
        step_label = _resolve_step_label(step)
        primary_bundle = _resolve_data_bundle(
            step=step,
            csv_name="aggregated_results.csv",
            source=args.source,
            cache=step_data_cache,
        )
        if primary_bundle is None:
            _report_missing_csv(
                step_label=step_label,
                results_dir=_resolve_step_results_dir(step),
                source=args.source,
                csv_name="aggregated_results.csv",
            )
            step_errors[step] = f"CSV {step_label} manquant"
            continue
        step_primary_bundle[step] = primary_bundle
        if primary_bundle.source_paths:
            if step == "step1":
                step1_csv = primary_bundle.source_paths[0]
            else:
                step2_csv = primary_bundle.source_paths[0]

    if (
        step1_csv is not None
        and step2_csv is not None
        and step1_csv.resolve() == step2_csv.resolve()
    ):
        message = (
            "Step1 et Step2 pointent vers le même CSV agrégé. "
            "Vérifiez que chaque étape écrit dans son dossier results."
        )
        log_error(f"ERREUR: {message}")
        step_errors["step1"] = message
        step_errors["step2"] = message

    if not args.skip_scientific_qa and "step1" in steps and "step2" in steps:
        if step1_csv is not None and step2_csv is not None:
            from article_c.qa_scientific_checks import run_scientific_checks

            qa_code, _ = run_scientific_checks(
                step1_csv=step1_csv,
                step2_csv=step2_csv,
                report_txt=ARTICLE_DIR / "scientific_qa_report.txt",
                report_csv=ARTICLE_DIR / "scientific_qa_report.csv",
            )
            if qa_code != 0 and not args.allow_scientific_qa_fail:
                raise SystemExit(
                    "Contrôles QA scientifiques en échec. "
                    "Utilisez --allow-scientific-qa-fail pour forcer la suite."
                )

    step_network_sizes: dict[str, list[int]] = {}
    for step in steps:
        if step in step_errors:
            continue
        primary_bundle = step_primary_bundle.get(step)
        if primary_bundle is not None:
            step_network_sizes[step] = _load_network_sizes_from_bundles([primary_bundle])

    if args.network_sizes:
        network_sizes = args.network_sizes
        _validate_network_sizes(list(step_primary_bundle.values()), network_sizes)
    else:
        network_sizes = sorted(
            {
                size
                for sizes in step_network_sizes.values()
                for size in sizes
            }
        )
        if not network_sizes:
            log_debug(
                "Aucune taille de réseau détectée dans les CSV, "
                "aucun plot n'a été généré."
            )
            step_errors.setdefault("step1", "aucune taille de réseau détectée")
            step_errors.setdefault("step2", "aucune taille de réseau détectée")

    if "step2" in steps:
        step2_sizes = step_network_sizes.get("step2", [])
        if len(step2_sizes) < 2:
            log_debug(
                "AVERTISSEMENT: Step2 contient moins de 2 tailles. "
                "Les plots seront validés individuellement."
            )
            log_debug(f"Tailles Step2 détectées: {step2_sizes or 'aucune'}")
        if args.network_sizes:
            expected_sizes = args.network_sizes
        elif "step1" in step_network_sizes:
            expected_sizes = step_network_sizes["step1"]
        else:
            expected_sizes = []
        if expected_sizes:
            missing_sizes = sorted(set(expected_sizes) - set(step2_sizes))
            if missing_sizes:
                missing_label = ", ".join(str(size) for size in missing_sizes)
                log_debug("WARNING: Step2 ne contient pas toutes les tailles attendues.")
                log_debug(f"Tailles attendues manquantes: {missing_label}")
                log_debug(f"Tailles Step2 détectées: {step2_sizes or 'aucune'}")
                log_debug("Commande PowerShell pour terminer Step2 (mode reprise):")
                log_debug(_suggest_step2_resume_command(expected_sizes))

    plot_data_filter_report_rows: list[dict[str, object]] = []
    manifest_context_by_module: dict[str, ManifestContext] = {}
    for step, module_paths in PLOT_MODULES.items():
        if step not in steps:
            continue
        if step in step_errors:
            for module_path in module_paths:
                expected_sizes = tuple(
                    args.network_sizes
                    or step_network_sizes.get(step)
                    or network_sizes
                )
                requirements = _resolve_plot_requirements(step, module_path)
                min_network_sizes = (
                    MIN_NETWORK_SIZES_PER_PLOT.get(module_path)
                    if module_path in MIN_NETWORK_SIZES_PER_PLOT
                    else (1 if step == "step2" else requirements.min_network_sizes)
                )
                size_status, size_message = _compute_size_status(
                    requested_sizes=tuple(int(size) for size in expected_sizes),
                    detected_sizes=(),
                    min_network_sizes=min_network_sizes,
                )
                manifest_context_by_module.setdefault(
                    module_path,
                    ManifestContext(
                        requested_sizes=tuple(int(size) for size in expected_sizes),
                        detected_sizes=(),
                        filtered_row_count=0,
                        size_status=size_status,
                        size_message=size_message,
                    ),
                )
                if module_path not in status_map:
                    _register_status(
                        status_map,
                        step=step,
                        module_path=module_path,
                        status="SKIP",
                        message=step_errors[step],
                    )
    if "step1" in steps:
        STEP1_PLOTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if "step2" in steps:
        STEP2_PLOTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for step in steps:
        if step in step_errors:
            continue
        for module_path in PLOT_MODULES[step]:
            if module_path in status_map and status_map[module_path].status == "FAIL":
                continue
            requirements = _resolve_plot_requirements(step, module_path)
            required_figure = _is_required_plot_module(step, module_path)
            data_bundle = _resolve_data_bundle(
                step=step,
                csv_name=requirements.csv_name,
                source=args.source,
                cache=step_data_cache,
            )
            if data_bundle is None:
                expected_sizes = tuple(
                    args.network_sizes
                    or step_network_sizes.get(step)
                    or network_sizes
                )
                min_network_sizes = (
                    MIN_NETWORK_SIZES_PER_PLOT.get(module_path)
                    if module_path in MIN_NETWORK_SIZES_PER_PLOT
                    else (1 if step == "step2" else requirements.min_network_sizes)
                )
                size_status, size_message = _compute_size_status(
                    requested_sizes=tuple(int(size) for size in expected_sizes),
                    detected_sizes=(),
                    min_network_sizes=min_network_sizes,
                )
                manifest_context_by_module[module_path] = ManifestContext(
                    requested_sizes=tuple(int(size) for size in expected_sizes),
                    detected_sizes=(),
                    filtered_row_count=0,
                    size_status=size_status,
                    size_message=size_message,
                )
                if required_figure:
                    exc = MandatoryFigureDataError(
                        f"Figure obligatoire {module_path}: CSV manquant ({requirements.csv_name})."
                    )
                    log_error(f"ERREUR: {exc}")
                    _register_status(
                        status_map,
                        step=step,
                        module_path=module_path,
                        status="FAIL",
                        message=str(exc),
                    )
                else:
                    log_debug(
                        "AVERTISSEMENT: "
                        f"CSV manquant ({requirements.csv_name}) pour {module_path}, figure ignorée."
                    )
                    _register_status(
                        status_map,
                        step=step,
                        module_path=module_path,
                        status="SKIP",
                        message=f"CSV manquant ({requirements.csv_name})",
                    )
                continue
            rl10_network_sizes: list[int] | None = None
            if (
                step == "step2"
                and module_path.endswith("plot_RL10_reward_vs_pdr_scatter")
            ):
                if step1_csv is None or step2_csv is None:
                    continue
                step1_sizes = step_network_sizes.get("step1", [])
                step2_sizes = step_network_sizes.get("step2", [])
                intersection = sorted(set(step1_sizes) & set(step2_sizes))
                if len(intersection) < 2:
                    log_debug(
                        "WARNING: "
                        "plot_RL10_reward_vs_pdr_scatter nécessite au moins "
                        "2 tailles communes entre Step1 et Step2."
                    )
                    log_debug(f"Tailles Step1: {step1_sizes or 'aucune'}")
                    log_debug(f"Tailles Step2: {step2_sizes or 'aucune'}")
                    log_debug(f"Intersection: {intersection or 'aucune'}")
                    regen_sizes = step2_sizes or step1_sizes
                    command = (
                        _suggest_regeneration_command(
                            STEP1_RESULTS_DIR / "by_size" / "size_0" / "aggregated_results.csv",
                            regen_sizes,
                        )
                        if regen_sizes
                        else None
                    )
                    if command:
                        log_debug(
                            "Exemple pour régénérer Step1 "
                            "(PowerShell):"
                        )
                        log_debug(command)
                    _register_status(
                        status_map,
                        step=step,
                        module_path=module_path,
                        status="SKIP",
                        message="moins de 2 tailles communes Step1/Step2",
                    )
                    continue
                rl10_network_sizes = intersection
            expected_sizes = (
                args.network_sizes
                or step_network_sizes.get(step)
                or network_sizes
            )
            try:
                is_valid, reason, report_row = _validate_plot_data(
                    step=step,
                    module_path=module_path,
                    data_bundle=data_bundle,
                    requirements=requirements,
                    expected_sizes=expected_sizes,
                    source=args.source,
                    required_figure=required_figure,
                )
            except MandatoryFigureDataError as exc:
                log_error(f"ERREUR: {exc}")
                _register_status(
                    status_map,
                    step=step,
                    module_path=module_path,
                    status="FAIL",
                    message=str(exc),
                )
                continue
            plot_data_filter_report_rows.append(report_row)
            detected_sizes = tuple(int(size) for size in report_row.get("detected_sizes", ()))
            requested_sizes = tuple(int(size) for size in report_row.get("requested_sizes", ()))
            min_network_sizes = (
                MIN_NETWORK_SIZES_PER_PLOT.get(module_path)
                if module_path in MIN_NETWORK_SIZES_PER_PLOT
                else (1 if step == "step2" else requirements.min_network_sizes)
            )
            size_status, size_message = _compute_size_status(
                requested_sizes=requested_sizes,
                detected_sizes=detected_sizes,
                min_network_sizes=min_network_sizes,
            )
            filtered_rows = max(
                0,
                int(report_row.get("initial_rows", 0))
                - int(report_row.get("after_cluster_filter_rows", 0)),
            )
            manifest_context_by_module[module_path] = ManifestContext(
                csv_source_paths=data_bundle.source_paths,
                requested_sizes=requested_sizes,
                detected_sizes=detected_sizes,
                filtered_row_count=filtered_rows,
                size_status=size_status,
                size_message=size_message,
            )
            if not is_valid:
                _register_status(
                    status_map,
                    step=step,
                    module_path=module_path,
                    status="SKIP",
                    message=reason,
                )
                _print_plot_data_filter_report_for_module(report_row)
                continue
            if step == "step2":
                figure = module_path.split(".")[-1]
                step2_sizes = (
                    rl10_network_sizes
                    or (
                        network_sizes
                        if args.network_sizes
                        else step_network_sizes.get("step2") or network_sizes
                    )
                )
                sizes_label = (
                    ", ".join(str(size) for size in step2_sizes)
                    if step2_sizes
                    else "none"
                )
                log_debug(f"Detected sizes: {sizes_label}")
                log_debug(f"Plotting Step2: {figure}")
                if expected_sizes and step2_sizes:
                    expected_set = {int(size) for size in expected_sizes}
                    step2_set = {int(size) for size in step2_sizes}
                    if step2_set < expected_set:
                        expected_label = ", ".join(str(size) for size in expected_sizes)
                        reduced_label = ", ".join(str(size) for size in step2_sizes)
                        log_debug(
                            "WARNING: "
                            f"{module_path} utilise un jeu réduit "
                            f"({reduced_label}) au lieu de {expected_label}."
                        )
            if step == "step1":
                figure = module_path.split(".")[-1]
                step1_network_sizes = (
                    args.network_sizes
                    or step_network_sizes.get("step1")
                )
                sizes_label = (
                    ", ".join(str(size) for size in step1_network_sizes)
                    if step1_network_sizes
                    else "none"
                )
                log_debug(f"Detected sizes: {sizes_label}")
                log_debug(f"Plotting Step1: {figure}")
                try:
                    previous_figures = set(plt.get_fignums())
                    module = _run_plot_module(
                        module_path,
                        network_sizes=step1_network_sizes,
                        allow_sample=False,
                        enable_suptitle=enable_suptitle,
                    )
                    missing_legends = _check_legends_for_module(
                        module_path=module_path,
                        module=module,
                        previous_figures=previous_figures,
                        fail_on_missing_legends=True,
                        legend_status=step1_legend_status,
                        legend_report_rows=legend_check_report_rows,
                        legend_validity_by_module=legend_validity_by_module,
                    )
                    if missing_legends:
                        _register_status(
                            status_map,
                            step=step,
                            module_path=module_path,
                            status="FAIL",
                            message="légende absente: " + "; ".join(missing_legends),
                        )
                        continue
                    _register_status(
                        status_map,
                        step=step,
                        module_path=module_path,
                        status="OK",
                        message="plot généré",
                    )
                except Exception as exc:
                    log_debug(
                        f"ERREUR: échec du plot {module_path}: {exc}"
                    )
                    log_debug("INFO: batch poursuivi (module suivant).")
                    traceback.print_exc()
                    _register_status(
                        status_map,
                        step=step,
                        module_path=module_path,
                        status="FAIL",
                        message=str(exc),
                    )
            elif step == "step2":
                step2_network_sizes = (
                    rl10_network_sizes
                    or (
                        network_sizes
                        if args.network_sizes
                        else step_network_sizes.get("step2") or network_sizes
                    )
                )
                try:
                    previous_figures = set(plt.get_fignums())
                    module = _run_plot_module(
                        module_path,
                        network_sizes=step2_network_sizes,
                        allow_sample=False,
                        enable_suptitle=enable_suptitle,
                    )
                    missing_legends = _check_legends_for_module(
                        module_path=module_path,
                        module=module,
                        previous_figures=previous_figures,
                        fail_on_missing_legends=True,
                        legend_status=step1_legend_status,
                        legend_report_rows=legend_check_report_rows,
                        legend_validity_by_module=legend_validity_by_module,
                    )
                    if missing_legends:
                        _register_status(
                            status_map,
                            step=step,
                            module_path=module_path,
                            status="FAIL",
                            message="légende absente: " + "; ".join(missing_legends),
                        )
                        continue
                    _register_status(
                        status_map,
                        step=step,
                        module_path=module_path,
                        status="OK",
                        message="plot généré",
                    )
                except Exception as exc:
                    log_debug(
                        f"ERREUR: échec du plot {module_path}: {exc}"
                    )
                    log_debug("INFO: batch poursuivi (module suivant).")
                    traceback.print_exc()
                    _register_status(
                        status_map,
                        step=step,
                        module_path=module_path,
                        status="FAIL",
                        message=str(exc),
                    )
    post_ready = (
        "step1" in steps
        and "step2" in steps
        and "step1" not in step_errors
        and "step2" not in step_errors
        and step1_csv is not None
        and step2_csv is not None
    )
    if post_ready:
        post_formats = ",".join(export_formats)
        post_args: dict[str, list[str]] = {
            "article_c.reproduce_author_results": [
                "--step1-results",
                str(step1_csv),
                "--step2-results",
                str(step2_csv),
                "--formats",
                post_formats,
            ],
            "article_c.compare_with_snir": [
                "--step1-csv",
                str(step1_csv),
                "--step2-csv",
                str(step2_csv),
                "--formats",
                post_formats,
            ],
            "article_c.plot_cluster_der": [
                "--formats",
                post_formats,
            ],
        }
        if not enable_suptitle:
            post_args["article_c.reproduce_author_results"].append("--no-header")
            post_args["article_c.compare_with_snir"].append("--no-suptitle")
        if args.no_figure_clamp:
            post_args["article_c.reproduce_author_results"].append(
                "--no-figure-clamp"
            )
        if args.network_sizes:
            post_args["article_c.plot_cluster_der"].extend(
                ["--network-sizes", *map(str, args.network_sizes)]
            )
        for module_path in POST_PLOT_MODULES:
            if (
                module_path in status_map
                and status_map[module_path].status == "FAIL"
            ):
                continue
            try:
                previous_figures = set(plt.get_fignums())
                module = _run_post_module(
                    module_path,
                    post_args.get(module_path, []),
                    close_figures=False,
                )
                missing_legends = _check_legends_for_module(
                    module_path=module_path,
                    module=module,
                    previous_figures=previous_figures,
                    fail_on_missing_legends=True,
                    legend_status=step1_legend_status,
                    legend_report_rows=legend_check_report_rows,
                    legend_validity_by_module=legend_validity_by_module,
                )
                if missing_legends:
                    _register_status(
                        status_map,
                        step="post",
                        module_path=module_path,
                        status="FAIL",
                        message="légende absente: " + "; ".join(missing_legends),
                    )
                    continue
                _register_status(
                    status_map,
                    step="post",
                    module_path=module_path,
                    status="OK",
                    message="plot généré",
                )
            except Exception as exc:
                log_debug(
                    f"ERREUR: échec du plot {module_path}: {exc}"
                )
                log_debug("INFO: batch poursuivi (module suivant).")
                traceback.print_exc()
                _register_status(
                    status_map,
                    step="post",
                    module_path=module_path,
                    status="FAIL",
                    message=str(exc),
                )
    else:
        skip_reason = "comparaisons ignorées (Step1/Step2 indisponibles)"
        for module_path in POST_PLOT_MODULES:
            if module_path not in status_map:
                _register_status(
                    status_map,
                    step="post",
                    module_path=module_path,
                    status="SKIP",
                    message=skip_reason,
                )
    if "step1" in steps:
        _inspect_plot_outputs(
            ARTICLE_DIR / "step1" / "plots" / "output",
            "Step1",
            list(export_formats),
        )
        _analyze_step1_pngs(
            ARTICLE_DIR / "step1" / "plots" / "output",
            step1_legend_status,
        )
    if "step2" in steps:
        _inspect_plot_outputs(
            ARTICLE_DIR / "step2" / "plots" / "output",
            "Step2",
            list(export_formats),
        )
    manifest_module = "article_c.make_all_plots.figures_manifest"
    try:
        _write_figures_manifest(export_formats, manifest_context_by_module)
        _register_status(
            status_map,
            step="post",
            module_path=manifest_module,
            status="OK",
            message=f"manifest écrit ({MANIFEST_OUTPUT_PATH.name})",
        )
    except Exception as exc:
        _register_status(
            status_map,
            step="post",
            module_path=manifest_module,
            status="FAIL",
            message=str(exc),
        )
        log_error(f"ERREUR: génération du manifest impossible: {exc}")
    _write_plot_data_filter_report(plot_data_filter_report_rows)
    log_debug(
        "INFO: rapport de filtrage écrit: "
        f"{PLOT_DATA_FILTER_REPORT_OUTPUT_PATH.resolve()}"
    )
    _write_legend_check_report(legend_check_report_rows)
    log_debug(
        "INFO: rapport des légendes écrit: "
        f"{LEGEND_CHECK_REPORT_OUTPUT_PATH.resolve()}"
    )
    missing_required_step2_legends = [
        module
        for module in STEP2_REQUIRED_LEGEND_MODULES
        if module in PLOT_MODULES["step2"]
        and module in legend_validity_by_module
        and not legend_validity_by_module[module]
    ]
    if missing_required_step2_legends:
        details = ", ".join(missing_required_step2_legends)
        log_error(
            "ERREUR: sortie finale bloquée, légende Step2 invalide "
            f"pour: {details}"
        )
        sys.exit(1)
    counts = _summarize_statuses(status_map, steps, [*POST_PLOT_MODULES, manifest_module])
    if args.fail_on_error and counts.get("FAIL", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
