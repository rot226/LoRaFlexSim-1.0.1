"""Génère tous les graphes de l'article C."""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
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

from article_c.common.csv_io import write_simulation_results, write_step1_results

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
    ],
}

POST_PLOT_MODULES = [
    "article_c.reproduce_author_results",
    "article_c.compare_with_snir",
    "article_c.plot_cluster_der",
]

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


def _ensure_expected_results_dir(csv_path: Path, expected_dir: Path, label: str) -> None:
    try:
        csv_path.resolve().relative_to(expected_dir.resolve())
    except ValueError as exc:
        raise ValueError(
            "Chemin CSV inattendu pour "
            f"{label}: {csv_path} (attendu sous {expected_dir})."
        ) from exc


def _collect_nested_csvs(results_dir: Path, filename: str) -> list[Path]:
    return sorted(results_dir.glob(f"size_*/rep_*/{filename}"))


def _ensure_step1_aggregated(results_dir: Path) -> Path | None:
    aggregated_path = results_dir / "aggregated_results.csv"
    if aggregated_path.exists():
        return aggregated_path
    raw_paths = _collect_nested_csvs(results_dir, "raw_metrics.csv")
    if not raw_paths:
        return None
    raw_rows: list[dict[str, object]] = []
    for path in raw_paths:
        raw_rows.extend(_load_csv_rows_coerced(path))
    if not raw_rows:
        return None
    print("Assemblage des résultats Step1 à partir des sous-dossiers...")
    write_step1_results(results_dir, raw_rows)
    return aggregated_path


def _ensure_step2_aggregated(results_dir: Path) -> Path | None:
    aggregated_path = results_dir / "aggregated_results.csv"
    if aggregated_path.exists():
        return aggregated_path
    raw_paths = _collect_nested_csvs(results_dir, "raw_results.csv")
    if not raw_paths:
        return None
    raw_rows: list[dict[str, object]] = []
    for path in raw_paths:
        raw_rows.extend(_load_csv_rows_coerced(path))
    if not raw_rows:
        return None
    print("Assemblage des résultats Step2 à partir des sous-dossiers...")
    write_simulation_results(results_dir, raw_rows)
    return aggregated_path


def _report_missing_csv(step_label: str, results_dir: Path) -> None:
    aggregated_path = results_dir / "aggregated_results.csv"
    raw_metrics_path = results_dir / "raw_metrics.csv"
    raw_results_path = results_dir / "raw_results.csv"
    print(
        f"ERREUR: CSV {step_label} introuvable. "
        f"Chemin attendu: {aggregated_path}."
    )
    print(
        "INFO: vérifiez que les simulations ont bien écrit dans "
        f"{results_dir}."
    )
    if step_label == "Step1":
        print(f"INFO: exemples attendus: {raw_metrics_path}")
    else:
        print(f"INFO: exemples attendus: {raw_results_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Construit le parseur d'arguments CLI pour générer les figures."""
    parser = argparse.ArgumentParser(
        description="Génère toutes les figures à partir des CSV agrégés."
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="step1,step2",
        help="Étapes à tracer (ex: step1,step2).",
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
        help="Formats d'export des figures (ex: png,eps).",
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


def _check_legends_for_module(
    *,
    module_path: str,
    module: object,
    previous_figures: set[int],
    fail_on_missing_legends: bool = False,
    legend_status: dict[str, bool] | None = None,
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
    place_adaptive_legend = getattr(module, "place_adaptive_legend", None)
    for index, fig_number in enumerate(new_fig_numbers, start=1):
        fig = plt.figure(fig_number)
        context = f"{module_path} (figure {index})"
        legend_count = 0
        for ax in fig.axes:
            _, labels = ax.get_legend_handles_labels()
            legend_count += len([label for label in labels if label])
        print(
            "INFO: "
            f"module {module_path} - {context}: "
            f"{legend_count} légende(s) trouvée(s)."
        )
        if legend_count == 0:
            print(
                "SUGGESTION: "
                f"aucune légende détectée pour {context}; "
                "ajoutez des labels manquants (label=...) aux courbes."
            )
        assert_legend_present(fig, context)
        if not _figure_has_legend(fig):
            all_figs_have_legends = False
            print(
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
                    print(
                        "AVERTISSEMENT: "
                        f"aucune entrée de légende détectée pour {context}; "
                        "place_adaptive_legend ignoré."
                    )
                else:
                    print(
                        "AVERTISSEMENT: "
                        f"tentative de placement automatique de légende pour "
                        f"{context}."
                    )
                    try:
                        place_adaptive_legend(fig, best_ax)
                    except Exception as exc:
                        print(
                            "AVERTISSEMENT: "
                            f"place_adaptive_legend a échoué pour {context}: {exc}"
                        )
                    else:
                        if _figure_has_legend(fig):
                            print(
                                "INFO: "
                                f"légende ajoutée pour {context}."
                            )
                        else:
                            print(
                                "AVERTISSEMENT: "
                                f"place_adaptive_legend n'a pas créé de légende "
                                f"pour {context}."
                            )
            else:
                print(
                    "AVERTISSEMENT: "
                    f"place_adaptive_legend non exposé par {module_path}; "
                    f"légende absente pour {context}."
                )
        if fail_on_missing_legends and not _figure_has_legend(fig):
            missing_contexts.append(context)
    if legend_status is not None:
        module_key = module_path.split(".")[-1]
        legend_status[module_key] = all_figs_have_legends
    return missing_contexts


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


def _resolve_csv_path(
    *,
    step: str,
    requirements: PlotRequirements,
    step1_csv: Path | None,
    step2_csv: Path | None,
) -> Path | None:
    if step == "step1":
        base_dir = STEP1_RESULTS_DIR
        aggregated = step1_csv
    else:
        base_dir = STEP2_RESULTS_DIR
        aggregated = step2_csv
    if requirements.csv_name == "aggregated_results.csv":
        return aggregated
    return base_dir / requirements.csv_name


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
        print(
            "ERREUR: certains scripts de plot ne passent pas par save_figure:\n"
            + "\n".join(f"- {item}" for item in missing_save_figure)
        )
    if missing_plot_style:
        print(
            "ERREUR: certains scripts de plot ne respectent pas apply_plot_style:\n"
            + "\n".join(f"- {item}" for item in missing_plot_style)
        )
    return missing


def _validate_step2_plot_module_registry() -> list[str]:
    step2_dir = ARTICLE_DIR / "step2" / "plots"
    if not step2_dir.exists():
        print(
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
        print(
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
        print(
            "AVERTISSEMENT: "
            f"dossier de sortie absent pour {label}: {output_dir}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        if label == "Step1":
            formats_label = ",".join(formats) if formats else "png"
            print(
                "INFO: relancez la commande suivante pour régénérer "
                "les figures Step1:"
            )
            print(
                "python -m article_c.make_all_plots "
                f"--steps step1 --formats {formats_label}"
            )
        return
    if not formats:
        print(f"INFO: aucun format d'export fourni pour {label}.")
        return
    primary_format = formats[0]
    primary_files = sorted(output_dir.glob(f"*.{primary_format}"))
    if not primary_files:
        print(
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
        print(
            "AVERTISSEMENT: sorties manquantes pour le test visuel:\n"
            + "\n".join(f"- {path}" for path in missing_variants)
        )
    else:
        formats_label = "/".join(fmt.upper() for fmt in formats)
        print(
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
        print("INFO: aucun PNG Step1 à analyser pour le rapport.")
        return
    sizes = []
    for path in png_files:
        try:
            with Image.open(path) as img:
                sizes.append(img.size)
        except OSError:
            continue
    if not sizes:
        print("AVERTISSEMENT: impossible de lire les tailles des PNG Step1.")
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
    print("\nRapport Step1 (PNG):")
    print(
        f"- Légendes: {legend_ok} OK / {legend_missing} manquantes / "
        f"{legend_unknown} inconnues (total {total})."
    )
    print(
        f"- Tailles: médiane {int(median_width)}x{int(median_height)}px, "
        f"{size_ok} conformes / {len(size_outliers)} atypiques."
    )
    print(
        f"- Axes lisibles: {axes_ok} OK / {len(axes_unknown)} à vérifier."
    )
    if size_outliers:
        print("Détails tailles atypiques:")
        for item in size_outliers:
            print(f"  - {item}")
    if legend_missing:
        print(
            "Détails légendes manquantes: "
            "voir les logs de génération Step1 pour la liste complète."
        )
    if axes_unknown:
        print("Détails axes à vérifier:")
        for item in axes_unknown:
            print(f"  - {item}")


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


def _extract_network_sizes(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "network_size" in fieldnames:
            size_key = "network_size"
        elif "density" in fieldnames:
            size_key = "density"
        else:
            return set()
        sizes: set[int] = set()
        for row in reader:
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


def _load_network_sizes_from_csvs(paths: list[Path]) -> list[int]:
    import pandas as pd

    sizes: set[int] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"CSV introuvable: {path}")
        df = pd.read_csv(path)
        if "network_size" in df.columns:
            size_column = "network_size"
        elif "density" in df.columns:
            size_column = "density"
        else:
            raise ValueError(
                f"Le CSV {path} doit contenir une colonne "
                "'network_size' ou 'density'."
            )
        sizes.update(
            int(float(value))
            for value in df[size_column].dropna().unique().tolist()
        )
    return sorted(sizes)


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
    paths: list[Path],
    expected_sizes: list[int],
) -> dict[Path, list[int]]:
    expected_set = {int(size) for size in expected_sizes}
    missing_by_path: dict[Path, list[int]] = {}
    for path in paths:
        found_sizes = _extract_network_sizes(path)
        missing = sorted(expected_set - found_sizes)
        if missing:
            missing_by_path[path] = missing
            missing_list = ", ".join(str(size) for size in missing)
            message_lines = [
                "AVERTISSEMENT: tailles de réseau manquantes dans les résultats.",
                f"CSV: {path}",
                f"Tailles attendues manquantes: {missing_list}.",
            ]
            command = _suggest_regeneration_command(path, expected_sizes)
            if command:
                message_lines.append(
                    "Commande PowerShell pour régénérer les résultats:"
                )
                message_lines.append(command)
            message_lines.append(
                "Les plots compatibles seront générés malgré tout."
            )
            print("\n".join(message_lines))
    return missing_by_path


def _validate_plot_data(
    *,
    step: str,
    module_path: str,
    csv_path: Path,
    requirements: PlotRequirements,
    expected_sizes: list[int] | None,
    cached_data: dict[str, tuple[list[str], list[dict[str, str]]]],
) -> tuple[bool, str]:
    for extra_name in requirements.extra_csv_names:
        extra_path = csv_path.parent / extra_name
        if not extra_path.exists():
            print(
                "AVERTISSEMENT: "
                f"{module_path} nécessite {extra_name}, figure ignorée."
            )
            return False, f"CSV manquant ({extra_name})"
    cache_key = str(csv_path)
    if cache_key not in cached_data:
        cached_data[cache_key] = _load_csv_data(csv_path)
    fieldnames, rows = cached_data[cache_key]
    if not fieldnames:
        print(
            "AVERTISSEMENT: "
            f"CSV vide pour {module_path}, figure ignorée."
        )
        return False, "CSV vide"
    sizes = _extract_network_sizes(csv_path)
    if module_path in MIN_NETWORK_SIZES_PER_PLOT:
        min_network_sizes = MIN_NETWORK_SIZES_PER_PLOT[module_path]
    elif step == "step2":
        min_network_sizes = 1
    else:
        min_network_sizes = requirements.min_network_sizes
    if len(sizes) < min_network_sizes:
        sizes_label = ", ".join(str(size) for size in sorted(sizes)) or "aucune"
        print(
            "Tailles détectées dans "
            f"{csv_path}: {sizes_label}."
        )
        print(
            "WARNING: "
            f"{module_path} nécessite au moins "
            f"{min_network_sizes} taille(s) disponible(s), "
            "figure ignorée."
        )
        print(f"CSV path: {csv_path}")
        return False, "tailles de réseau insuffisantes"
    if expected_sizes:
        expected_set = {int(size) for size in expected_sizes}
        if sizes and sizes < expected_set:
            expected_label = ", ".join(str(size) for size in expected_sizes)
            sizes_label = ", ".join(str(size) for size in sorted(sizes))
            print(
                "AVERTISSEMENT: "
                f"{module_path} est généré avec un jeu réduit "
                f"({sizes_label}) au lieu de {expected_label}."
            )
    if len(sizes) == 1 and min_network_sizes == 1:
        sizes_label = ", ".join(str(size) for size in sorted(sizes)) or "aucune"
        print(
            "AVERTISSEMENT: "
            f"{module_path} est généré avec une seule taille "
            f"({sizes_label})."
        )
    if requirements.required_any_columns:
        metric_col = _pick_column(fieldnames, requirements.required_any_columns)
        if metric_col is None:
            print(
                "AVERTISSEMENT: "
                f"{module_path} nécessite une colonne RSSI/SNR, "
                "figure ignorée."
            )
            return False, "colonne RSSI/SNR manquante"
    if requirements.require_algo_snir:
        algo_col = _pick_column(fieldnames, ("algo", "algorithm", "method"))
        snir_col = _pick_column(
            fieldnames, ("snir_mode", "snir_state", "snir", "with_snir")
        )
        if not algo_col or not snir_col:
            print(
                "AVERTISSEMENT: "
                f"{module_path} nécessite les colonnes algo/snir_mode, "
                "figure ignorée."
            )
            return False, "colonnes algo/snir_mode manquantes"
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
        missing_algos = [
            algo for algo in required_algos if algo not in available_algos
        ]
        missing_snir = [
            mode for mode in required_snir if mode not in available_snir
        ]
        if missing_algos or missing_snir:
            sizes_label = ", ".join(str(size) for size in sorted(sizes)) or "aucune"
            print(
                "Tailles détectées dans "
                f"{csv_path}: {sizes_label}."
            )
            details = []
            if missing_algos:
                details.append(f"algos manquants: {', '.join(missing_algos)}")
            if missing_snir:
                details.append(f"SNIR manquants: {', '.join(missing_snir)}")
            print(
                "AVERTISSEMENT: "
                f"{module_path} incomplet ({' ; '.join(details)}), "
                "figure ignorée."
            )
            return False, "données incomplètes"
    return True, "OK"


@dataclass
class PlotStatus:
    step: str
    module_path: str
    status: str
    message: str


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
    print("\nRésumé d'exécution des plots:")
    for step in steps:
        print(f"\n{step.upper()}:")
        for module_path in PLOT_MODULES[step]:
            entry = status_map.get(module_path)
            if entry is None:
                status_label = "SKIP"
                message = "Non exécuté."
            else:
                status_label = entry.status
                message = entry.message
            counts[status_label] = counts.get(status_label, 0) + 1
            print(f"- {module_path}: {status_label} ({message})")
    if post_modules:
        print("\nPOST:")
        for module_path in post_modules:
            entry = status_map.get(module_path)
            if entry is None:
                status_label = "SKIP"
                message = "Non exécuté."
            else:
                status_label = entry.status
                message = entry.message
            counts[status_label] = counts.get(status_label, 0) + 1
            print(f"- {module_path}: {status_label} ({message})")
    total = sum(counts.values())
    print(
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
    enable_suptitle = not args.no_suptitle
    try:
        export_formats = parse_export_formats(args.formats)
    except ValueError as exc:
        parser.error(str(exc))
    set_default_export_formats(export_formats)
    set_default_figure_clamp_enabled(not args.no_figure_clamp)
    status_map: dict[str, PlotStatus] = {}
    step1_legend_status: dict[str, bool] = {}
    invalid_modules = _validate_plot_modules_use_save_figure()
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
    step1_results_dir = STEP1_RESULTS_DIR
    step2_results_dir = STEP2_RESULTS_DIR
    step1_csv = None
    step2_csv = None
    step_errors: dict[str, str] = {}
    if "step1" in steps:
        step1_csv = _ensure_step1_aggregated(step1_results_dir)
        if step1_csv is None:
            _report_missing_csv("Step1", step1_results_dir)
            step_errors["step1"] = "CSV Step1 manquant"
        else:
            _ensure_expected_results_dir(step1_csv, step1_results_dir, "Step1")
            if not (step1_results_dir / "done.flag").exists():
                print("AVERTISSEMENT: done.flag absent pour Step1, continuation.")
    if "step2" in steps:
        step2_csv = _ensure_step2_aggregated(step2_results_dir)
        if step2_csv is None:
            _report_missing_csv("Step2", step2_results_dir)
            step_errors["step2"] = "CSV Step2 manquant"
        else:
            _ensure_expected_results_dir(step2_csv, step2_results_dir, "Step2")
            if not (step2_results_dir / "done.flag").exists():
                print("AVERTISSEMENT: done.flag absent pour Step2, continuation.")
    if (
        step1_csv is not None
        and step2_csv is not None
        and step1_csv.resolve() == step2_csv.resolve()
    ):
        message = (
            "Step1 et Step2 pointent vers le même CSV agrégé. "
            "Vérifiez que chaque étape écrit dans son dossier results."
        )
        print(f"ERREUR: {message}")
        step_errors["step1"] = message
        step_errors["step2"] = message
    csv_paths: list[Path] = []
    if "step1" in steps and step1_csv is not None:
        csv_paths.append(step1_csv)
    if "step2" in steps and step2_csv is not None:
        csv_paths.append(step2_csv)
    step_network_sizes: dict[str, list[int]] = {}
    if "step1" in steps and step1_csv is not None:
        step_network_sizes["step1"] = _load_network_sizes_from_csvs([step1_csv])
    if "step2" in steps and step2_csv is not None:
        step_network_sizes["step2"] = _load_network_sizes_from_csvs([step2_csv])
    if args.network_sizes:
        network_sizes = args.network_sizes
        _validate_network_sizes(csv_paths, network_sizes)
    else:
        network_sizes = sorted(
            {
                size
                for sizes in step_network_sizes.values()
                for size in sizes
            }
        )
        if not network_sizes:
            print(
                "Aucune taille de réseau détectée dans les CSV, "
                "aucun plot n'a été généré."
            )
            step_errors.setdefault(
                "step1",
                "aucune taille de réseau détectée",
            )
            step_errors.setdefault(
                "step2",
                "aucune taille de réseau détectée",
            )
    if "step2" in steps:
        step2_sizes = step_network_sizes.get("step2", [])
        if len(step2_sizes) < 2:
            print(
                "AVERTISSEMENT: Step2 contient moins de 2 tailles. "
                "Les plots seront validés individuellement."
            )
            print(f"Tailles Step2 détectées: {step2_sizes or 'aucune'}")
        if args.network_sizes:
            expected_sizes = args.network_sizes
        elif step1_csv is not None and step1_csv.exists():
            expected_sizes = _load_network_sizes_from_csvs([step1_csv])
        else:
            expected_sizes = []
        if expected_sizes:
            missing_sizes = sorted(set(expected_sizes) - set(step2_sizes))
            if missing_sizes:
                missing_label = ", ".join(str(size) for size in missing_sizes)
                print(
                    "WARNING: Step2 ne contient pas toutes les tailles attendues."
                )
                print(f"Tailles attendues manquantes: {missing_label}")
                print(f"Tailles Step2 détectées: {step2_sizes or 'aucune'}")
                print("Commande PowerShell pour terminer Step2 (mode reprise):")
                print(_suggest_step2_resume_command(expected_sizes))
    csv_cache: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
    for step, module_paths in PLOT_MODULES.items():
        if step not in steps:
            continue
        if step in step_errors:
            for module_path in module_paths:
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
            csv_path = _resolve_csv_path(
                step=step,
                requirements=requirements,
                step1_csv=step1_csv,
                step2_csv=step2_csv,
            )
            if csv_path is None:
                continue
            if not csv_path.exists():
                _register_status(
                    status_map,
                    step=step,
                    module_path=module_path,
                    status="SKIP",
                    message=f"CSV manquant ({csv_path.name})",
                )
                continue
            rl10_network_sizes: list[int] | None = None
            if (
                step == "step2"
                and module_path.endswith("plot_RL10_reward_vs_pdr_scatter")
            ):
                if step1_csv is None or step2_csv is None:
                    continue
                step1_sizes = _load_network_sizes_from_csvs([step1_csv])
                step2_sizes = _load_network_sizes_from_csvs([step2_csv])
                intersection = sorted(set(step1_sizes) & set(step2_sizes))
                if len(intersection) < 2:
                    print(
                        "WARNING: "
                        "plot_RL10_reward_vs_pdr_scatter nécessite au moins "
                        "2 tailles communes entre Step1 et Step2."
                    )
                    print(f"Tailles Step1: {step1_sizes or 'aucune'}")
                    print(f"Tailles Step2: {step2_sizes or 'aucune'}")
                    print(f"Intersection: {intersection or 'aucune'}")
                    regen_sizes = step2_sizes or step1_sizes
                    command = (
                        _suggest_regeneration_command(
                            step1_csv,
                            regen_sizes,
                        )
                        if regen_sizes
                        else None
                    )
                    if command:
                        print(
                            "Exemple pour régénérer Step1 "
                            "(PowerShell):"
                        )
                        print(command)
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
            is_valid, reason = _validate_plot_data(
                step=step,
                module_path=module_path,
                csv_path=csv_path,
                requirements=requirements,
                expected_sizes=expected_sizes,
                cached_data=csv_cache,
            )
            if not is_valid:
                _register_status(
                    status_map,
                    step=step,
                    module_path=module_path,
                    status="SKIP",
                    message=reason,
                )
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
                print(f"Detected sizes: {sizes_label}")
                print(f"Plotting Step2: {figure}")
                if expected_sizes and step2_sizes:
                    expected_set = {int(size) for size in expected_sizes}
                    step2_set = {int(size) for size in step2_sizes}
                    if step2_set < expected_set:
                        expected_label = ", ".join(str(size) for size in expected_sizes)
                        reduced_label = ", ".join(str(size) for size in step2_sizes)
                        print(
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
                print(f"Detected sizes: {sizes_label}")
                print(f"Plotting Step1: {figure}")
                try:
                    _run_plot_module(
                        module_path,
                        network_sizes=step1_network_sizes,
                        allow_sample=False,
                        enable_suptitle=enable_suptitle,
                    )
                    _register_status(
                        status_map,
                        step=step,
                        module_path=module_path,
                        status="OK",
                        message="plot généré",
                    )
                except Exception as exc:
                    print(
                        f"ERREUR: échec du plot {module_path}: {exc}"
                    )
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
                    _check_legends_for_module(
                        module_path=module_path,
                        module=module,
                        previous_figures=previous_figures,
                        legend_status=step1_legend_status,
                    )
                    _register_status(
                        status_map,
                        step=step,
                        module_path=module_path,
                        status="OK",
                        message="plot généré",
                    )
                except Exception as exc:
                    print(
                        f"ERREUR: échec du plot {module_path}: {exc}"
                    )
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
                _check_legends_for_module(
                    module_path=module_path,
                    module=module,
                    previous_figures=previous_figures,
                    legend_status=step1_legend_status,
                )
                _register_status(
                    status_map,
                    step="post",
                    module_path=module_path,
                    status="OK",
                    message="plot généré",
                )
            except Exception as exc:
                print(
                    f"ERREUR: échec du plot {module_path}: {exc}"
                )
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
    counts = _summarize_statuses(status_map, steps, POST_PLOT_MODULES)
    if args.fail_on_error and counts.get("FAIL", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
