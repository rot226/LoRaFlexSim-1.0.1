"""Point d'entrée pour l'étape 2."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import logging
import math
from multiprocessing import get_context
from pathlib import Path
import shutil
from statistics import median
from typing import Sequence

from article_c.common.config import (
    DEFAULT_CONFIG,
    STEP2_SAFE_CONFIG,
    STEP2_SUPER_SAFE_CONFIG,
)
from article_c.common.csv_io import (
    aggregate_results_by_size,
    write_rows,
    write_simulation_results,
)
from article_c.common.plot_helpers import (
    place_adaptive_legend,
    apply_plot_style,
    parse_export_formats,
    save_figure,
    set_default_figure_clamp_enabled,
    set_default_export_formats,
)
from article_c.common.utils import (
    ensure_dir,
    parse_cli_args,
    parse_network_size_list,
    replication_ids,
    set_deterministic_seed,
    timestamp_tag,
)
from article_c.step2.simulate_step2 import (
    _collision_size_factor,
    _network_load_factor,
    _resolve_collision_clamps,
    _resolve_load_clamps,
    run_simulation,
)
from plot_defaults import resolve_ieee_figsize

logger = logging.getLogger(__name__)
SAFE_PROFILE_SUCCESS_THRESHOLD = 0.2
SUPER_SAFE_PROFILE_SUCCESS_THRESHOLD = 0.05
RX_POWER_DBM_MIN = -120.0
RX_POWER_DBM_MAX = -70.0
AUTO_TUNING_SUCCESS_THRESHOLD = 0.10
AUTO_TUNING_MAX_ATTEMPTS = 3
AUTO_TUNING_MINI_EVAL_NETWORK_SIZE = 80
AUTO_TUNING_MIN_MEASURABLE_GAIN = 1e-4
SUCCESS_ZERO_RATIO_THRESHOLD = 0.95
BY_SIZE_DIRNAME = "by_size"


def _clamp_collision_bounds(config: dict[str, object]) -> None:
    collision_min = float(config["collision_size_min"])
    collision_under = float(config["collision_size_under_max"])
    collision_over = float(config["collision_size_over_max"])
    collision_under = max(collision_under, collision_min)
    collision_over = max(collision_over, collision_under)
    config["collision_size_min"] = collision_min
    config["collision_size_under_max"] = collision_under
    config["collision_size_over_max"] = collision_over


def _soften_collision_config(config: dict[str, object]) -> None:
    config["collision_size_min"] = max(0.55, float(config["collision_size_min"]) * 0.94)
    config["collision_size_under_max"] = max(
        0.75, float(config["collision_size_under_max"]) * 0.92
    )
    config["collision_size_over_max"] = max(
        0.95, float(config["collision_size_over_max"]) * 0.90
    )
    _clamp_collision_bounds(config)


def _extract_auto_tuning_params(config: dict[str, object]) -> dict[str, float]:
    return {
        "traffic_load_scale_step2": float(config["traffic_coeff_scale"]),
        "capture_probability": float(config["capture_probability"]),
        "collision_size_min": float(config["collision_size_min"]),
        "collision_size_under_max": float(config["collision_size_under_max"]),
        "collision_size_over_max": float(config["collision_size_over_max"]),
    }


def _sync_args_from_auto_tuned_config(args: object, config: dict[str, object]) -> None:
    args.traffic_coeff_scale = float(config["traffic_coeff_scale"])
    args.capture_probability = float(config["capture_probability"])
    args.collision_size_min = float(config["collision_size_min"])
    args.collision_size_under_max = float(config["collision_size_under_max"])
    args.collision_size_over_max = float(config["collision_size_over_max"])


def _apply_structured_auto_tuning_adjustments(
    tuned_config: dict[str, object], attempt: int
) -> None:
    traffic_step_by_attempt = {1: 0.78, 2: 0.70}
    capture_step_by_attempt = {1: 0.04, 2: 0.06}

    traffic_factor = traffic_step_by_attempt.get(attempt, 0.65)
    capture_boost = capture_step_by_attempt.get(attempt, 0.08)

    tuned_config["traffic_coeff_scale"] = max(
        0.10, float(tuned_config["traffic_coeff_scale"]) * traffic_factor
    )
    tuned_config["capture_probability"] = min(
        0.95, float(tuned_config["capture_probability"]) + capture_boost
    )

    _soften_collision_config(tuned_config)
    tuned_config["collision_size_min"] = max(
        0.50, float(tuned_config["collision_size_min"]) - 0.03 * attempt
    )
    tuned_config["collision_size_under_max"] = max(
        0.70, float(tuned_config["collision_size_under_max"]) - 0.04 * attempt
    )
    tuned_config["collision_size_over_max"] = max(
        0.90, float(tuned_config["collision_size_over_max"]) - 0.05 * attempt
    )
    _clamp_collision_bounds(tuned_config)


def _run_auto_tuning_before_campaign(
    config: dict[str, object],
    replications: list[int],
    base_results_dir: Path,
) -> dict[str, object]:
    eval_replications = replications[:1] if replications else [1]
    attempt_logs: list[dict[str, object]] = []
    tuned_config = dict(config)
    selected_attempt = AUTO_TUNING_MAX_ATTEMPTS
    success_reached = False
    measurable_gain_detected = False
    previous_success_mean: float | None = None
    auto_tuning_tmp_dir = base_results_dir / "_auto_tuning_tmp"
    if auto_tuning_tmp_dir.exists():
        shutil.rmtree(auto_tuning_tmp_dir)
    ensure_dir(auto_tuning_tmp_dir)
    try:
        for attempt in range(1, AUTO_TUNING_MAX_ATTEMPTS + 1):
            params_before = _extract_auto_tuning_params(tuned_config)
            task = (
                AUTO_TUNING_MINI_EVAL_NETWORK_SIZE,
                0,
                eval_replications,
                tuned_config,
                auto_tuning_tmp_dir,
                None,
                True,
            )
            result = _simulate_density(task)
            diagnostics = dict(result["diagnostics"])
            success_mean = float(
                diagnostics.get(
                    "success_rate_mean",
                    diagnostics.get("success_mean", 0.0),
                )
            )
            delta_success = (
                success_mean - previous_success_mean
                if previous_success_mean is not None
                else None
            )
            if delta_success is not None and delta_success > AUTO_TUNING_MIN_MEASURABLE_GAIN:
                measurable_gain_detected = True

            accepted = success_mean >= AUTO_TUNING_SUCCESS_THRESHOLD
            params_after = dict(params_before)
            adjustment_applied = False
            if not accepted and attempt < AUTO_TUNING_MAX_ATTEMPTS:
                _apply_structured_auto_tuning_adjustments(tuned_config, attempt)
                params_after = _extract_auto_tuning_params(tuned_config)
                adjustment_applied = True

            attempt_logs.append(
                {
                    "attempt": attempt,
                    "network_size": AUTO_TUNING_MINI_EVAL_NETWORK_SIZE,
                    "replications": list(eval_replications),
                    "seed_base": int(config["base_seed"]),
                    "before": {
                        "success_rate_mean": previous_success_mean,
                        "parameters": params_before,
                    },
                    "after": {
                        "success_rate_mean": success_mean,
                        "parameters": params_after,
                    },
                    "success_rate_mean": success_mean,
                    "delta_success_rate_mean": delta_success,
                    "measurable_gain": (
                        delta_success is not None
                        and delta_success > AUTO_TUNING_MIN_MEASURABLE_GAIN
                    ),
                    "accepted": accepted,
                    "adjustment_applied": adjustment_applied,
                }
            )
            print(
                "Mini-évaluation auto-tuning "
                f"(tentative {attempt}/{AUTO_TUNING_MAX_ATTEMPTS}, N={AUTO_TUNING_MINI_EVAL_NETWORK_SIZE}) "
                f"success_rate_mean={success_mean:.4f}, "
                f"delta={delta_success if delta_success is not None else float('nan'):.4f}."
            )
            previous_success_mean = success_mean
            if accepted:
                selected_attempt = attempt
                success_reached = True
                break
        config.update(tuned_config)
    finally:
        shutil.rmtree(auto_tuning_tmp_dir, ignore_errors=True)

    selected_parameters = _extract_auto_tuning_params(config)
    status = "success" if success_reached else "completed_without_threshold"
    if not success_reached and not measurable_gain_detected:
        status = "failed_tuning"

    payload = {
        "status": status,
        "network_size": AUTO_TUNING_MINI_EVAL_NETWORK_SIZE,
        "threshold": AUTO_TUNING_SUCCESS_THRESHOLD,
        "min_measurable_gain": AUTO_TUNING_MIN_MEASURABLE_GAIN,
        "max_attempts": AUTO_TUNING_MAX_ATTEMPTS,
        "selected_attempt": selected_attempt,
        "success_reached": success_reached,
        "measurable_gain_detected": measurable_gain_detected,
        "retained_parameters": selected_parameters,
        "attempts": attempt_logs,
    }
    log_path = base_results_dir / "auto_tuning_log.json"
    log_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Auto-tuning Step2: journal écrit dans {log_path}.")
    return payload


def _clamp_rx_power_dbm(value_dbm: float) -> float:
    clamped = max(RX_POWER_DBM_MIN, min(RX_POWER_DBM_MAX, value_dbm))
    if not math.isclose(clamped, value_dbm, abs_tol=1e-12):
        print(
            "WARNING: rx_power_dbm hors plage admissible "
            f"({value_dbm:.2f} dBm). Valeur clampée à {clamped:.2f} dBm "
            f"(bornes {RX_POWER_DBM_MIN:.2f}..{RX_POWER_DBM_MAX:.2f} dBm)."
        )
    return clamped


def _is_rx_power_clamped(requested_dbm: float, effective_dbm: float) -> bool:
    return not math.isclose(requested_dbm, effective_dbm, abs_tol=1e-12)


def _log_default_profile_if_needed(args: object) -> None:
    if getattr(args, "safe_profile", False):
        return
    defaults = DEFAULT_CONFIG.step2
    print(
        "Profil standard adouci (par défaut) : "
        f"capture_probability={defaults.capture_probability}, "
        f"network_load_min/max={defaults.network_load_min}/{defaults.network_load_max}, "
        f"collision_size_min/under/over="
        f"{defaults.collision_size_min}/"
        f"{defaults.collision_size_under_max}/"
        f"{defaults.collision_size_over_max}."
    )


def _aggregate_selection_probs(
    selection_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    density_by_key: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    for row in selection_rows:
        network_size = int(row["network_size"])
        round_id = int(row["round"])
        sf = int(row["sf"])
        selection_prob = float(row["selection_prob"])
        grouped[(network_size, round_id, sf)].append(selection_prob)
        density_value = row.get("density")
        if isinstance(density_value, (int, float)):
            density_by_key[(network_size, round_id, sf)].append(float(density_value))
    aggregated: list[dict[str, object]] = []
    for (network_size, round_id, sf), values in sorted(grouped.items()):
        avg = sum(values) / len(values) if values else 0.0
        density_values = density_by_key.get((network_size, round_id, sf), [])
        density_value = (
            sum(density_values) / len(density_values)
            if density_values
            else _compute_density(network_size)
        )
        aggregated.append(
            {
                "network_size": network_size,
                "density": density_value,
                "round": round_id,
                "sf": sf,
                "selection_prob": round(avg, 6),
            }
        )
    return aggregated


def _format_size_factor_table(
    sizes: Sequence[int],
    reference_size: int,
    load_clamp_min: float,
    load_clamp_max: float,
    collision_clamp_min: float,
    collision_clamp_under_max: float,
    collision_clamp_over_max: float,
) -> str:
    headers = (
        "Taille",
        "Charge",
        "Charge (legacy)",
        "Collision",
        "Collision (legacy)",
    )
    rows: list[tuple[str, ...]] = [headers]
    for size in sizes:
        load_factor = _network_load_factor(
            size, reference_size, load_clamp_min, load_clamp_max
        )
        legacy_load_factor = _network_load_factor(size, reference_size, 0.6, 2.6)
        collision_factor = _collision_size_factor(
            size,
            reference_size,
            collision_clamp_min,
            collision_clamp_under_max,
            collision_clamp_over_max,
        )
        legacy_collision_factor = _collision_size_factor(
            size, reference_size, 0.6, 1.0, 2.4
        )
        rows.append(
            (
                f"{size}",
                f"{load_factor:.3f}",
                f"{legacy_load_factor:.3f}",
                f"{collision_factor:.3f}",
                f"{legacy_collision_factor:.3f}",
            )
        )
    widths = [max(len(row[idx]) for row in rows) for idx in range(len(headers))]
    lines: list[str] = []
    for idx, row in enumerate(rows):
        line = " | ".join(cell.ljust(widths[col_idx]) for col_idx, cell in enumerate(row))
        lines.append(line)
        if idx == 0:
            lines.append("-+-".join("-" * width for width in widths))
    return "\n".join(lines)


def _log_effective_traffic_scale_for_density(density: int, config: dict[str, object]) -> None:
    load_clamp_min, load_clamp_max = _resolve_load_clamps(
        DEFAULT_CONFIG.step2,
        config.get("network_load_min"),
        config.get("network_load_max"),
        safe_profile=bool(config.get("safe_profile", False)),
        no_clamp=bool(config.get("no_clamp", False)),
    )
    load_factor = _network_load_factor(
        int(density),
        int(config["reference_network_size"]),
        load_clamp_min,
        load_clamp_max,
    )
    traffic_coeff_scale = float(config["traffic_coeff_scale"])
    effective_traffic_scale = traffic_coeff_scale * load_factor
    print(
        "Traffic scale effectif par taille: "
        f"taille={int(density)}, "
        f"traffic_coeff_scale={traffic_coeff_scale:.4f}, "
        f"load_factor={load_factor:.4f}, "
        f"effective_scale={effective_traffic_scale:.4f}"
    )


def _apply_safe_profile_with_log(args: object, reason: str) -> None:
    changes: list[tuple[str, object, object]] = []

    def _set_value(name: str, value: object) -> None:
        previous = getattr(args, name, None)
        if previous != value:
            changes.append((name, previous, value))
        setattr(args, name, value)

    _set_value("safe_profile", True)
    _set_value("capture_probability", STEP2_SAFE_CONFIG.capture_probability)
    _set_value("traffic_coeff_clamp_enabled", STEP2_SAFE_CONFIG.traffic_coeff_clamp_enabled)
    _set_value("traffic_coeff_clamp_min", STEP2_SAFE_CONFIG.traffic_coeff_clamp_min)
    _set_value("traffic_coeff_clamp_max", STEP2_SAFE_CONFIG.traffic_coeff_clamp_max)
    _set_value("network_load_min", STEP2_SAFE_CONFIG.network_load_min)
    _set_value("network_load_max", STEP2_SAFE_CONFIG.network_load_max)
    _set_value("collision_size_min", STEP2_SAFE_CONFIG.collision_size_min)
    _set_value("collision_size_under_max", STEP2_SAFE_CONFIG.collision_size_under_max)
    _set_value("collision_size_over_max", STEP2_SAFE_CONFIG.collision_size_over_max)
    _set_value("reward_floor", STEP2_SAFE_CONFIG.reward_floor)
    _set_value(
        "zero_success_quality_bonus_factor",
        STEP2_SAFE_CONFIG.zero_success_quality_bonus_factor,
    )
    _set_value("max_penalty_ratio", STEP2_SAFE_CONFIG.max_penalty_ratio)
    _set_value("shadowing_sigma_db", STEP2_SAFE_CONFIG.shadowing_sigma_db)

    print(f"Profil sécurisé activé ({reason}).")
    if not changes:
        print("Aucun paramètre modifié par le profil sécurisé.")
        return
    print("Paramètres modifiés par le profil sécurisé:")
    for name, previous, value in sorted(changes):
        print(f"- {name}: {previous} -> {value}")


def _build_safe_profile_config(
    config: dict[str, object]
) -> tuple[dict[str, object], list[tuple[str, object, object]]]:
    updated = dict(config)
    changes: list[tuple[str, object, object]] = []

    def _set_value(name: str, value: object) -> None:
        previous = updated.get(name)
        if previous != value:
            changes.append((name, previous, value))
        updated[name] = value

    _set_value("safe_profile", True)
    _set_value("capture_probability", STEP2_SAFE_CONFIG.capture_probability)
    _set_value("traffic_coeff_clamp_enabled", STEP2_SAFE_CONFIG.traffic_coeff_clamp_enabled)
    _set_value("traffic_coeff_clamp_min", STEP2_SAFE_CONFIG.traffic_coeff_clamp_min)
    _set_value("traffic_coeff_clamp_max", STEP2_SAFE_CONFIG.traffic_coeff_clamp_max)
    _set_value("network_load_min", STEP2_SAFE_CONFIG.network_load_min)
    _set_value("network_load_max", STEP2_SAFE_CONFIG.network_load_max)
    _set_value("collision_size_min", STEP2_SAFE_CONFIG.collision_size_min)
    _set_value("collision_size_under_max", STEP2_SAFE_CONFIG.collision_size_under_max)
    _set_value("collision_size_over_max", STEP2_SAFE_CONFIG.collision_size_over_max)
    _set_value("reward_floor", STEP2_SAFE_CONFIG.reward_floor)
    _set_value(
        "zero_success_quality_bonus_factor",
        STEP2_SAFE_CONFIG.zero_success_quality_bonus_factor,
    )
    _set_value("max_penalty_ratio", STEP2_SAFE_CONFIG.max_penalty_ratio)
    _set_value("shadowing_sigma_db", STEP2_SAFE_CONFIG.shadowing_sigma_db)
    return updated, changes


def _build_super_safe_profile_config(
    config: dict[str, object]
) -> tuple[dict[str, object], list[tuple[str, object, object]]]:
    updated = dict(config)
    changes: list[tuple[str, object, object]] = []

    def _set_value(name: str, value: object) -> None:
        previous = updated.get(name)
        if previous != value:
            changes.append((name, previous, value))
        updated[name] = value

    _set_value("safe_profile", True)
    _set_value("capture_probability", STEP2_SUPER_SAFE_CONFIG.capture_probability)
    _set_value(
        "traffic_coeff_clamp_enabled",
        STEP2_SUPER_SAFE_CONFIG.traffic_coeff_clamp_enabled,
    )
    _set_value("traffic_coeff_clamp_min", STEP2_SUPER_SAFE_CONFIG.traffic_coeff_clamp_min)
    _set_value("traffic_coeff_clamp_max", STEP2_SUPER_SAFE_CONFIG.traffic_coeff_clamp_max)
    _set_value("network_load_min", STEP2_SUPER_SAFE_CONFIG.network_load_min)
    _set_value("network_load_max", STEP2_SUPER_SAFE_CONFIG.network_load_max)
    _set_value("collision_size_min", STEP2_SUPER_SAFE_CONFIG.collision_size_min)
    _set_value(
        "collision_size_under_max", STEP2_SUPER_SAFE_CONFIG.collision_size_under_max
    )
    _set_value("collision_size_over_max", STEP2_SUPER_SAFE_CONFIG.collision_size_over_max)
    _set_value("reward_floor", STEP2_SUPER_SAFE_CONFIG.reward_floor)
    _set_value(
        "zero_success_quality_bonus_factor",
        STEP2_SUPER_SAFE_CONFIG.zero_success_quality_bonus_factor,
    )
    _set_value("max_penalty_ratio", STEP2_SUPER_SAFE_CONFIG.max_penalty_ratio)
    _set_value("shadowing_sigma_db", STEP2_SUPER_SAFE_CONFIG.shadowing_sigma_db)
    return updated, changes


def _log_safe_profile_switch(
    density: int, reason: str, changes: list[tuple[str, object, object]]
) -> None:
    print(f"Bascule profil sécurisé pour la taille {density} ({reason}).")
    if not changes:
        print("Aucun paramètre modifié pour la relance en profil sécurisé.")
        return
    print("Paramètres appliqués pour la relance en profil sécurisé:")
    for name, previous, value in sorted(changes):
        print(f"- {name}: {previous} -> {value}")


def _log_super_safe_profile_switch(
    density: int, reason: str, changes: list[tuple[str, object, object]]
) -> None:
    print(f"Bascule profil super sécurisé pour la taille {density} ({reason}).")
    if not changes:
        print("Aucun paramètre modifié pour la relance super sécurisée.")
        return
    print("Paramètres appliqués pour la relance super sécurisée:")
    for name, previous, value in sorted(changes):
        print(f"- {name}: {previous} -> {value}")


def _update_safe_profile_config(config: dict[str, object], args: object) -> None:
    config.update(
        {
            "safe_profile": bool(getattr(args, "safe_profile", False)),
            "capture_probability": getattr(args, "capture_probability", None),
            "traffic_coeff_clamp_enabled": args.traffic_coeff_clamp_enabled,
            "traffic_coeff_clamp_min": args.traffic_coeff_clamp_min,
            "traffic_coeff_clamp_max": args.traffic_coeff_clamp_max,
            "network_load_min": args.network_load_min,
            "network_load_max": args.network_load_max,
            "collision_size_min": args.collision_size_min,
            "collision_size_under_max": args.collision_size_under_max,
            "collision_size_over_max": args.collision_size_over_max,
            "reward_floor": args.reward_floor,
            "zero_success_quality_bonus_factor": getattr(
                args, "zero_success_quality_bonus_factor", None
            ),
            "max_penalty_ratio": getattr(args, "max_penalty_ratio", None),
            "shadowing_sigma_db": getattr(args, "shadowing_sigma_db", None),
        }
    )


def _aggregate_learning_curve(
    learning_curve_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    density_by_key: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    clamp_rate_by_key: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    for row in learning_curve_rows:
        network_size = int(row["network_size"])
        round_id = int(row["round"])
        algo = str(row["algo"])
        avg_reward = float(row["avg_reward"])
        grouped[(network_size, round_id, algo)].append(avg_reward)
        density_value = row.get("density")
        if isinstance(density_value, (int, float)):
            density_by_key[(network_size, round_id, algo)].append(float(density_value))
        clamp_rate_raw = row.get("traffic_coeff_clamp_rate")
        if isinstance(clamp_rate_raw, (int, float)):
            clamp_rate_by_key[(network_size, round_id, algo)].append(float(clamp_rate_raw))
    aggregated: list[dict[str, object]] = []
    for (network_size, round_id, algo), values in sorted(grouped.items()):
        avg = sum(values) / len(values) if values else 0.0
        density_values = density_by_key.get((network_size, round_id, algo), [])
        density_value = (
            sum(density_values) / len(density_values)
            if density_values
            else _compute_density(network_size)
        )
        clamp_values = clamp_rate_by_key.get((network_size, round_id, algo), [])
        clamp_rate = sum(clamp_values) / len(clamp_values) if clamp_values else 0.0
        aggregated.append(
            {
                "network_size": network_size,
                "density": density_value,
                "round": round_id,
                "algo": algo,
                "avg_reward": round(avg, 6),
                "traffic_coeff_clamp_rate": round(clamp_rate, 6),
            }
        )
    return aggregated




def _ensure_csv_within_scope(csv_path: Path, scope_root: Path) -> Path:
    resolved_csv = csv_path.resolve()
    resolved_scope = scope_root.resolve()
    if resolved_csv.parent != resolved_scope and resolved_scope not in resolved_csv.parents:
        raise RuntimeError(
            "Étape 2: sortie CSV hors périmètre autorisé. "
            f"Fichier: {resolved_csv} ; périmètre attendu: {resolved_scope}."
        )
    return resolved_csv


def _log_step2_key_csv_paths(output_dir: Path) -> None:
    key_csv_names = (
        "run_status_step2.csv",
        "aggregates/aggregated_results.csv",
        "aggregates/diagnostics_step2_by_size.csv",
        "aggregates/loss_causes_histogram.csv",
        "aggregates/snir_distribution_by_sf.csv",
        "aggregates/diagnostics_by_size.csv",
        "aggregates/diagnostics_by_size_algo_sf.csv",
        "traffic_coeff_clamp_rate.csv",
    )
    for csv_name in key_csv_names:
        csv_path = output_dir / csv_name
        resolved_csv = _ensure_csv_within_scope(csv_path, output_dir)
        if csv_path.exists():
            print(f"CSV Step2 écrit: {resolved_csv}")

def _compute_density(network_size: int) -> float:
    radius_m = float(DEFAULT_CONFIG.scenario.radius_m)
    if radius_m <= 0.0:
        raise ValueError("radius_m doit être strictement positif pour calculer la densité.")
    return network_size / (math.pi * radius_m**2)


def _log_results_written(output_dir: Path, row_count: int) -> None:
    raw_path = _ensure_csv_within_scope(output_dir / "raw_results.csv", output_dir)
    aggregated_path = _ensure_csv_within_scope(output_dir / "aggregated_results.csv", output_dir)
    print(f"Append rows: {row_count} -> {raw_path}")
    print(f"Append rows: {row_count} -> {aggregated_path}")


def _assert_flat_output_files(output_dir: Path, density: int | float) -> None:
    raw_path = output_dir / "raw_results.csv"
    aggregated_path = output_dir / "aggregated_results.csv"
    missing = [str(path) for path in (raw_path, aggregated_path) if not path.exists()]
    if missing:
        missing_label = ", ".join(missing)
        message = (
            "ERREUR: fichiers de sortie attendus absents après "
            f"write_simulation_results pour la taille {density}: {missing_label}"
        )
        print(message)
        raise FileNotFoundError(message)


def _log_unique_network_sizes(output_dir: Path) -> None:
    raw_path = output_dir / "raw_results.csv"
    if not raw_path.exists():
        print(f"Aucun raw_results.csv détecté: {raw_path}")
        return
    with raw_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        raw_fieldnames = reader.fieldnames or []
        fieldnames = [
            name.lstrip("\ufeff").strip() for name in raw_fieldnames if name is not None
        ]
        reader.fieldnames = fieldnames
        headers_label = ", ".join(fieldnames or [])
        print(f"Headers lus dans {raw_path}: {headers_label or 'aucun'}")
        if not fieldnames or "network_size" not in fieldnames:
            print(f"Colonne network_size absente dans {raw_path}")
            return
        values: list[float] = []
        lines_read = 0
        valid_lines = 0
        for row in reader:
            lines_read += 1
            if not row or row == {}:
                continue
            if all(value in (None, "") for value in row.values()):
                continue
            valid_lines += 1
            value = row.get("network_size")
            if value in (None, ""):
                continue
            try:
                values.append(float(value))
            except ValueError:
                print(f"Valeur network_size invalide détectée: {value}")
    if any(value == 0.0 for value in values):
        print(
            "ERREUR: network_size à 0.0 détecté dans raw_results.csv, vérifiez la configuration."
        )
    sizes = sorted({int(value) for value in values})
    if not sizes:
        print(
            "Aucune taille détectée dans raw_results.csv "
            f"(lignes lues: {lines_read}, lignes valides: {valid_lines})."
        )
    sizes_label = ", ".join(map(str, sizes)) if sizes else "aucune"
    print(f"Tailles détectées dans raw_results: {sizes_label}")


def _summarize_success_collision(
    raw_rows: list[dict[str, object]]
) -> dict[str, float]:
    success_rates: list[float] = []
    collision_norms: list[float] = []
    reward_values: list[float] = []
    throughput_success_values: list[float] = []
    for row in raw_rows:
        if str(row.get("cluster", "")) != "all":
            continue
        if "success_rate" in row:
            success_rates.append(float(row["success_rate"]))
        if "collision_norm" in row:
            collision_norms.append(float(row["collision_norm"]))
        if "reward" in row:
            reward_values.append(float(row["reward"]))
        if "throughput_success" in row:
            throughput_success_values.append(float(row["throughput_success"]))
    if not success_rates:
        success_rates = [0.0]
    if not collision_norms:
        collision_norms = [0.0]
    if not reward_values:
        reward_values = [0.0]
    if not throughput_success_values:
        throughput_success_values = [0.0]
    return {
        "success_min": min(success_rates),
        "success_max": max(success_rates),
        "success_mean": sum(success_rates) / len(success_rates),
        "collision_min": min(collision_norms),
        "collision_max": max(collision_norms),
        "collision_mean": sum(collision_norms) / len(collision_norms),
        "reward_mean": sum(reward_values) / len(reward_values),
        "throughput_success_mean": sum(throughput_success_values)
        / len(throughput_success_values),
    }


def _assert_flat_output_sizes(
    base_results_dir: Path, simulated_sizes: list[int]
) -> None:
    aggregated_sizes = _read_aggregated_sizes(
        base_results_dir / "aggregates" / "aggregated_results.csv"
    )
    missing_sizes = sorted(set(simulated_sizes) - aggregated_sizes)
    if missing_sizes:
        missing_label = ", ".join(map(str, missing_sizes))
        message = (
            "ERREUR: write_simulation_results manquant pour certaines tailles "
            f"simulées (flat_output=True): {missing_label}"
        )
        print(message)
        raise RuntimeError(message)


def _log_step2_autonomous_inputs(args: object, reference_network_size: int) -> None:
    """Journalise les entrées explicites utilisées par Step2 (sans Step1)."""
    print(
        "Step2 autonome: paramètres explicites uniquement "
        "(network_size, seed, RL, trafic, canal)."
    )
    print(
        "Step2 paramètres explicites: "
        f"seed={getattr(args, 'seeds_base', None)}, "
        f"reference_network_size={reference_network_size}, "
        f"window_size={getattr(args, 'window_size', None)}, "
        f"lambda_collision={getattr(args, 'lambda_collision', None)}, "
        f"traffic_mode={getattr(args, 'traffic_mode', None)}, "
        f"traffic_coeff_scale={getattr(args, 'traffic_coeff_scale', None)}, "
        f"snir_threshold_db={getattr(args, 'snir_threshold_db', None)}, "
        f"noise_floor_dbm={getattr(args, 'noise_floor_dbm', None)}, "
        f"rx_power_dbm_requested={getattr(args, 'rx_power_dbm', None)}."
    )


def _log_rx_power_diagnostics(requested_dbm: float, effective_dbm: float) -> None:
    clamped = _is_rx_power_clamped(requested_dbm, effective_dbm)
    print(
        "Diagnostic rx_power_dbm: "
        f"requested={requested_dbm:.2f} dBm, "
        f"effective={effective_dbm:.2f} dBm, "
        f"clamped={'yes' if clamped else 'no'}."
    )

def _init_collision_histogram() -> dict[str, int]:
    return {"0-0.1": 0, "0.1-0.3": 0, "0.3-0.6": 0, "0.6-1.0": 0}


def _update_collision_histogram(histogram: dict[str, int], value: float) -> None:
    bounded = max(0.0, min(1.0, value))
    if bounded < 0.1:
        histogram["0-0.1"] += 1
    elif bounded < 0.3:
        histogram["0.1-0.3"] += 1
    elif bounded < 0.6:
        histogram["0.3-0.6"] += 1
    else:
        histogram["0.6-1.0"] += 1


def _dominant_loss_cause(
    losses_collisions: int, losses_congestion: int, losses_link_quality: int
) -> str:
    losses = {
        "collisions": losses_collisions,
        "congestion": losses_congestion,
        "link_quality": losses_link_quality,
    }
    if sum(losses.values()) <= 0:
        return "aucune"
    return max(losses, key=losses.get)


def _summarize_post_simulation(
    raw_rows: list[dict[str, object]]
) -> dict[str, object]:
    success_sum = 0.0
    success_count = 0
    success_zero_count = 0
    collision_sum = 0.0
    collision_count = 0
    collision_hist = _init_collision_histogram()
    link_quality_sum = 0.0
    link_quality_count = 0
    link_quality_min: float | None = None
    link_quality_max: float | None = None
    reward_zero_no_success = 0
    reward_zero_clipped = 0
    reward_zero_total = 0
    reward_min: float | None = None
    reward_max: float | None = None
    reward_count = 0
    rx_power_dbm_sum = 0.0
    rx_power_dbm_count = 0
    rx_power_dbm_min: float | None = None
    rx_power_dbm_max: float | None = None
    rx_power_dbm_requested_sum = 0.0
    rx_power_dbm_requested_count = 0
    rx_power_dbm_effective_sum = 0.0
    rx_power_dbm_effective_count = 0
    rx_power_dbm_clamped_count = 0
    traffic_coeff_clamp_rate_sum = 0.0
    traffic_coeff_clamp_rate_count = 0
    traffic_coeff_clamp_alert_count = 0
    losses_collisions_total = 0
    losses_congestion_total = 0
    losses_link_quality_total = 0
    loss_keys_seen: set[tuple[object, object, object]] = set()
    reward_debug_sums: dict[str, dict[str, float]] = {}
    for row in raw_rows:
        if str(row.get("cluster", "")) != "all":
            continue
        loss_key = (
            row.get("replication"),
            row.get("algo"),
            row.get("round"),
        )
        if loss_key not in loss_keys_seen:
            loss_keys_seen.add(loss_key)
            losses_collisions_total += int(row.get("losses_collisions", 0) or 0)
            losses_congestion_total += int(row.get("losses_congestion", 0) or 0)
            losses_link_quality_total += int(row.get("losses_link_quality", 0) or 0)
        success_rate = float(row.get("success_rate", 0.0) or 0.0)
        collision_norm = float(row.get("collision_norm", 0.0) or 0.0)
        link_quality = float(row.get("link_quality", 0.0) or 0.0)
        reward = float(row.get("reward", 0.0) or 0.0)
        success_sum += success_rate
        success_count += 1
        if success_rate <= 1e-9:
            success_zero_count += 1
        collision_sum += collision_norm
        collision_count += 1
        _update_collision_histogram(collision_hist, collision_norm)
        link_quality_sum += link_quality
        link_quality_count += 1
        link_quality_min = (
            link_quality if link_quality_min is None else min(link_quality_min, link_quality)
        )
        link_quality_max = (
            link_quality if link_quality_max is None else max(link_quality_max, link_quality)
        )
        reward_min = reward if reward_min is None else min(reward_min, reward)
        reward_max = reward if reward_max is None else max(reward_max, reward)
        reward_count += 1
        rx_power_dbm = float(row.get("rx_power_dbm", 0.0) or 0.0)
        rx_power_dbm_sum += rx_power_dbm
        rx_power_dbm_count += 1
        rx_power_dbm_min = (
            rx_power_dbm if rx_power_dbm_min is None else min(rx_power_dbm_min, rx_power_dbm)
        )
        rx_power_dbm_max = (
            rx_power_dbm if rx_power_dbm_max is None else max(rx_power_dbm_max, rx_power_dbm)
        )
        rx_power_dbm_requested = float(row.get("rx_power_dbm_requested", rx_power_dbm) or rx_power_dbm)
        rx_power_dbm_effective = float(row.get("rx_power_dbm_effective", rx_power_dbm) or rx_power_dbm)
        rx_power_dbm_requested_sum += rx_power_dbm_requested
        rx_power_dbm_requested_count += 1
        rx_power_dbm_effective_sum += rx_power_dbm_effective
        rx_power_dbm_effective_count += 1
        clamp_flag_raw = row.get("rx_power_dbm_clamped")
        clamp_flag = (
            bool(clamp_flag_raw)
            if clamp_flag_raw is not None
            else _is_rx_power_clamped(rx_power_dbm_requested, rx_power_dbm_effective)
        )
        if clamp_flag:
            rx_power_dbm_clamped_count += 1
        traffic_clamp_rate_raw = row.get("traffic_coeff_clamp_rate")
        if traffic_clamp_rate_raw not in (None, ""):
            try:
                traffic_coeff_clamp_rate_sum += float(traffic_clamp_rate_raw)
                traffic_coeff_clamp_rate_count += 1
            except (TypeError, ValueError):
                pass
        traffic_clamp_alert_raw = row.get("traffic_coeff_clamp_alert_triggered")
        if traffic_clamp_alert_raw not in (None, ""):
            try:
                if int(float(traffic_clamp_alert_raw)) != 0:
                    traffic_coeff_clamp_alert_count += 1
            except (TypeError, ValueError):
                pass
        if reward <= 1e-9:
            reward_zero_total += 1
            if success_rate <= 1e-9:
                reward_zero_no_success += 1
            else:
                reward_zero_clipped += 1
        if "weighted_quality" in row:
            algo_label = str(row.get("algo", ""))
            bucket = reward_debug_sums.setdefault(
                algo_label,
                {
                    "weighted_quality_sum": 0.0,
                    "collision_penalty_sum": 0.0,
                    "success_term_sum": 0.0,
                    "reward_floor_sum": 0.0,
                    "count": 0.0,
                },
            )
            bucket["weighted_quality_sum"] += float(
                row.get("weighted_quality", 0.0) or 0.0
            )
            bucket["collision_penalty_sum"] += float(
                row.get("collision_penalty", 0.0) or 0.0
            )
            bucket["success_term_sum"] += float(row.get("success_term", 0.0) or 0.0)
            bucket["reward_floor_sum"] += float(row.get("reward_floor", 0.0) or 0.0)
            bucket["count"] += 1.0

    reward_debug_summary: dict[str, dict[str, float]] = {}
    for algo_label, values in reward_debug_sums.items():
        count = max(values.get("count", 0.0), 1.0)
        reward_debug_summary[algo_label] = {
            "weighted_quality_mean": values["weighted_quality_sum"] / count,
            "collision_penalty_mean": values["collision_penalty_sum"] / count,
            "success_term_mean": values["success_term_sum"] / count,
            "reward_floor_mean": values["reward_floor_sum"] / count,
            "count": values["count"],
        }
    return {
        "success_sum": success_sum,
        "success_count": success_count,
        "success_zero_count": success_zero_count,
        "collision_sum": collision_sum,
        "collision_count": collision_count,
        "collision_hist": collision_hist,
        "link_quality_sum": link_quality_sum,
        "link_quality_count": link_quality_count,
        "link_quality_min": 0.0 if link_quality_min is None else link_quality_min,
        "link_quality_max": 0.0 if link_quality_max is None else link_quality_max,
        "reward_zero_no_success": reward_zero_no_success,
        "reward_zero_clipped": reward_zero_clipped,
        "reward_zero_total": reward_zero_total,
        "reward_min": 0.0 if reward_min is None else reward_min,
        "reward_max": 0.0 if reward_max is None else reward_max,
        "reward_count": reward_count,
        "rx_power_dbm_sum": rx_power_dbm_sum,
        "rx_power_dbm_count": rx_power_dbm_count,
        "rx_power_dbm_min": 0.0 if rx_power_dbm_min is None else rx_power_dbm_min,
        "rx_power_dbm_max": 0.0 if rx_power_dbm_max is None else rx_power_dbm_max,
        "rx_power_dbm_requested_sum": rx_power_dbm_requested_sum,
        "rx_power_dbm_requested_count": rx_power_dbm_requested_count,
        "rx_power_dbm_effective_sum": rx_power_dbm_effective_sum,
        "rx_power_dbm_effective_count": rx_power_dbm_effective_count,
        "rx_power_dbm_clamped_count": rx_power_dbm_clamped_count,
        "traffic_coeff_clamp_rate_sum": traffic_coeff_clamp_rate_sum,
        "traffic_coeff_clamp_rate_count": traffic_coeff_clamp_rate_count,
        "traffic_coeff_clamp_alert_count": traffic_coeff_clamp_alert_count,
        "losses_collisions_total": losses_collisions_total,
        "losses_congestion_total": losses_congestion_total,
        "losses_link_quality_total": losses_link_quality_total,
        "reward_debug_summary": reward_debug_summary,
    }


def _log_size_diagnostics(density: int, metrics: dict[str, float]) -> None:
    print(
        "Diagnostic taille "
        f"{density}: succès min/max = {metrics['success_min']:.4f}/"
        f"{metrics['success_max']:.4f}, "
        f"collisions min/max = {metrics['collision_min']:.4f}/"
        f"{metrics['collision_max']:.4f}"
    )


def _verify_metric_variation(size_metrics: dict[int, dict[str, float]]) -> None:
    if len(size_metrics) < 2:
        return

    def _has_variation(
        values: list[float], rel_tol: float = 1e-6, abs_tol: float = 1e-9
    ) -> bool:
        if len(values) < 2:
            return False
        min_value = min(values)
        max_value = max(values)
        span = max_value - min_value
        scale = max(abs(max_value), abs(min_value), abs_tol)
        return span > max(abs_tol, scale * rel_tol)

    success_means = [
        metrics.get("success_mean", 0.0) for metrics in size_metrics.values()
    ]
    collision_means = [
        metrics.get("collision_mean", 0.0) for metrics in size_metrics.values()
    ]
    reward_means = [
        metrics.get("reward_mean", 0.0) for metrics in size_metrics.values()
    ]
    throughput_means = [
        metrics.get("throughput_success_mean", 0.0) for metrics in size_metrics.values()
    ]
    if not _has_variation(success_means):
        print(
            "ERREUR: le success_rate moyen ne varie pas avec la taille du réseau."
        )
    if not _has_variation(collision_means):
        print(
            "ERREUR: les collisions moyennes ne varient pas avec la taille du réseau."
        )
    if not _has_variation(reward_means):
        print(
            "ERREUR: le reward_mean moyen ne varie pas avec la taille du réseau."
        )
    if not _has_variation(throughput_means):
        print(
            "ERREUR: le throughput_success_mean ne varie pas avec la taille du réseau."
        )


def _has_traceable_outputs(base_results_dir: Path) -> bool:
    if (base_results_dir / "aggregates" / "aggregated_results.csv").exists():
        return True
    by_size_dir = base_results_dir / BY_SIZE_DIRNAME
    if not by_size_dir.exists():
        return False
    return any(by_size_dir.glob("size_*/rep_*/raw_results.csv"))


def _read_aggregated_sizes(aggregated_path: Path) -> set[int]:
    if not aggregated_path.exists():
        print(f"Aucun aggregated_results.csv détecté: {aggregated_path}")
        return set()
    with aggregated_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        raw_fieldnames = reader.fieldnames or []
        fieldnames = [
            name.lstrip("\ufeff").strip() for name in raw_fieldnames if name is not None
        ]
        reader.fieldnames = fieldnames
        preview = ", ".join(fieldnames[:6]) if fieldnames else "aucun"
        print(f"Premiers headers lus dans {aggregated_path}: {preview}")
        size_key = None
        if "network_size" in fieldnames:
            size_key = "network_size"
        elif "density" in fieldnames:
            size_key = "density"
            print(
                f"Colonne network_size absente dans {aggregated_path}, "
                "fallback sur density."
            )
        else:
            print(
                f"Colonnes network_size/density absentes dans {aggregated_path}"
            )
            return set()
        sizes: set[int] = set()
        lines_read = 0
        valid_lines = 0
        for row in reader:
            lines_read += 1
            if not row or row == {}:
                continue
            if all(value in (None, "") for value in row.values()):
                continue
            valid_lines += 1
            value = row.get(size_key)
            if value in (None, ""):
                continue
            try:
                sizes.add(int(float(value)))
            except ValueError:
                print(
                    f"Valeur {size_key} invalide détectée: {value}"
                )
        if not sizes:
            print(
                "Aucune taille détectée dans aggregated_results.csv "
                f"(lignes lues: {lines_read}, lignes valides: {valid_lines})."
            )
        return sizes


def _read_nested_sizes(base_results_dir: Path, replications: list[int]) -> set[int]:
    sizes: set[int] = set()
    by_size_dir = base_results_dir / BY_SIZE_DIRNAME
    for size_dir in sorted(by_size_dir.glob("size_*")):
        if not size_dir.is_dir():
            continue
        try:
            size = int(size_dir.name.split("size_", 1)[1])
        except (IndexError, ValueError):
            continue
        rep_paths = [
            size_dir / f"rep_{replication}" / "aggregated_results.csv"
            for replication in replications
        ]
        if rep_paths and all(path.exists() for path in rep_paths):
            sizes.add(size)
    if not sizes:
        print(
            "Aucune taille complète détectée dans les sous-dossiers "
            f"{base_results_dir / BY_SIZE_DIRNAME / 'size_<N>/rep_<R>'}."
        )
    return sizes


def _compose_post_simulation_report(
    per_size_stats: dict[int, dict[str, object]],
    per_size_diagnostics: dict[int, dict[str, float]],
    safe_profile_sizes: Sequence[int],
) -> str:
    if not per_size_stats:
        return "Rapport post-simulation indisponible (aucune statistique collectée)."
    overall_success_sum = 0.0
    overall_success_count = 0
    overall_success_zero_count = 0
    overall_collision_sum = 0.0
    overall_collision_count = 0
    overall_collision_hist = _init_collision_histogram()
    overall_link_quality_sum = 0.0
    overall_link_quality_count = 0
    overall_link_quality_min: float | None = None
    overall_link_quality_max: float | None = None
    reward_zero_no_success = 0
    reward_zero_clipped = 0
    reward_zero_total = 0
    reward_min: float | None = None
    reward_max: float | None = None
    reward_count = 0
    overall_losses_collisions = 0
    overall_losses_congestion = 0
    overall_losses_link_quality = 0
    overall_rx_power_dbm_sum = 0.0
    overall_rx_power_dbm_count = 0
    overall_rx_power_dbm_min: float | None = None
    overall_rx_power_dbm_max: float | None = None
    overall_rx_power_dbm_requested_sum = 0.0
    overall_rx_power_dbm_requested_count = 0
    overall_rx_power_dbm_effective_sum = 0.0
    overall_rx_power_dbm_effective_count = 0
    overall_rx_power_dbm_clamped_count = 0

    per_size_link_quality_mean: dict[int, float] = {}
    for size, stats in per_size_stats.items():
        success_sum = float(stats.get("success_sum", 0.0))
        success_count = int(stats.get("success_count", 0))
        success_zero_count = int(stats.get("success_zero_count", 0))
        collision_sum = float(stats.get("collision_sum", 0.0))
        collision_count = int(stats.get("collision_count", 0))
        link_quality_sum = float(stats.get("link_quality_sum", 0.0))
        link_quality_count = int(stats.get("link_quality_count", 0))
        link_quality_min = float(stats.get("link_quality_min", 0.0))
        link_quality_max = float(stats.get("link_quality_max", 0.0))
        overall_success_sum += success_sum
        overall_success_count += success_count
        overall_success_zero_count += success_zero_count
        overall_collision_sum += collision_sum
        overall_collision_count += collision_count
        for bucket, count in dict(stats.get("collision_hist", {})).items():
            overall_collision_hist[bucket] = overall_collision_hist.get(bucket, 0) + int(
                count
            )
        overall_link_quality_sum += link_quality_sum
        overall_link_quality_count += link_quality_count
        overall_link_quality_min = (
            link_quality_min
            if overall_link_quality_min is None
            else min(overall_link_quality_min, link_quality_min)
        )
        overall_link_quality_max = (
            link_quality_max
            if overall_link_quality_max is None
            else max(overall_link_quality_max, link_quality_max)
        )
        reward_zero_no_success += int(stats.get("reward_zero_no_success", 0))
        reward_zero_clipped += int(stats.get("reward_zero_clipped", 0))
        reward_zero_total += int(stats.get("reward_zero_total", 0))
        reward_min = (
            float(stats.get("reward_min", 0.0))
            if reward_min is None
            else min(reward_min, float(stats.get("reward_min", 0.0)))
        )
        reward_max = (
            float(stats.get("reward_max", 0.0))
            if reward_max is None
            else max(reward_max, float(stats.get("reward_max", 0.0)))
        )
        reward_count += int(stats.get("reward_count", 0))
        rx_power_dbm_sum = float(stats.get("rx_power_dbm_sum", 0.0))
        rx_power_dbm_count = int(stats.get("rx_power_dbm_count", 0))
        rx_power_dbm_min = float(stats.get("rx_power_dbm_min", 0.0))
        rx_power_dbm_max = float(stats.get("rx_power_dbm_max", 0.0))
        rx_power_dbm_requested_sum = float(stats.get("rx_power_dbm_requested_sum", 0.0))
        rx_power_dbm_requested_count = int(stats.get("rx_power_dbm_requested_count", 0))
        rx_power_dbm_effective_sum = float(stats.get("rx_power_dbm_effective_sum", 0.0))
        rx_power_dbm_effective_count = int(stats.get("rx_power_dbm_effective_count", 0))
        rx_power_dbm_clamped_count = int(stats.get("rx_power_dbm_clamped_count", 0))
        overall_rx_power_dbm_sum += rx_power_dbm_sum
        overall_rx_power_dbm_count += rx_power_dbm_count
        overall_rx_power_dbm_min = (
            rx_power_dbm_min
            if overall_rx_power_dbm_min is None
            else min(overall_rx_power_dbm_min, rx_power_dbm_min)
        )
        overall_rx_power_dbm_max = (
            rx_power_dbm_max
            if overall_rx_power_dbm_max is None
            else max(overall_rx_power_dbm_max, rx_power_dbm_max)
        )
        overall_rx_power_dbm_requested_sum += rx_power_dbm_requested_sum
        overall_rx_power_dbm_requested_count += rx_power_dbm_requested_count
        overall_rx_power_dbm_effective_sum += rx_power_dbm_effective_sum
        overall_rx_power_dbm_effective_count += rx_power_dbm_effective_count
        overall_rx_power_dbm_clamped_count += rx_power_dbm_clamped_count
        overall_losses_collisions += int(stats.get("losses_collisions_total", 0))
        overall_losses_congestion += int(stats.get("losses_congestion_total", 0))
        overall_losses_link_quality += int(stats.get("losses_link_quality_total", 0))
        if link_quality_count > 0:
            per_size_link_quality_mean[size] = link_quality_sum / link_quality_count

    success_mean = (
        overall_success_sum / overall_success_count
        if overall_success_count > 0
        else 0.0
    )
    collision_mean = (
        overall_collision_sum / overall_collision_count
        if overall_collision_count > 0
        else 0.0
    )
    link_quality_mean = (
        overall_link_quality_sum / overall_link_quality_count
        if overall_link_quality_count > 0
        else 0.0
    )
    link_quality_min = 0.0 if overall_link_quality_min is None else overall_link_quality_min
    link_quality_max = 0.0 if overall_link_quality_max is None else overall_link_quality_max
    rx_power_dbm_mean = (
        overall_rx_power_dbm_sum / overall_rx_power_dbm_count
        if overall_rx_power_dbm_count > 0
        else 0.0
    )
    rx_power_dbm_min = (
        0.0 if overall_rx_power_dbm_min is None else overall_rx_power_dbm_min
    )
    rx_power_dbm_max = (
        0.0 if overall_rx_power_dbm_max is None else overall_rx_power_dbm_max
    )
    rx_power_dbm_requested_mean = (
        overall_rx_power_dbm_requested_sum / overall_rx_power_dbm_requested_count
        if overall_rx_power_dbm_requested_count > 0
        else 0.0
    )
    rx_power_dbm_effective_mean = (
        overall_rx_power_dbm_effective_sum / overall_rx_power_dbm_effective_count
        if overall_rx_power_dbm_effective_count > 0
        else 0.0
    )
    rx_power_dbm_clamped_ratio = (
        overall_rx_power_dbm_clamped_count / overall_rx_power_dbm_effective_count
        if overall_rx_power_dbm_effective_count > 0
        else 0.0
    )

    lines = [
        "Rapport post-simulation (étape 2)",
        "",
        "Taux de succès moyen:",
        f"- success_rate moyen global: {success_mean:.4f}",
    ]
    zero_success_ratio = (
        overall_success_zero_count / overall_success_count
        if overall_success_count > 0
        else 0.0
    )
    if overall_success_count > 0 and zero_success_ratio > 0.95:
        lines.extend(
            [
                "",
                (
                    "AVERTISSEMENT: plus de 95% des fenêtres ont un success_rate "
                    "nul. Vérifiez la configuration (trafic, SNIR, collisions)."
                ),
                "Simulation invalide : success_rate trop faible.",
            ]
        )
    if per_size_diagnostics:
        lines.append("- success_rate moyen par taille:")
        for size in sorted(per_size_diagnostics):
            metrics = per_size_diagnostics[size]
            lines.append(
                f"  - taille {size}: {metrics['success_mean']:.4f} "
                f"(min {metrics['success_min']:.4f}, max {metrics['success_max']:.4f})"
            )

    total_collisions = sum(overall_collision_hist.values()) or 1
    lines.extend(
        [
            "",
            "Distribution des collisions (collision_norm):",
            f"- moyenne globale: {collision_mean:.4f}",
        ]
    )
    for bucket in ("0-0.1", "0.1-0.3", "0.3-0.6", "0.6-1.0"):
        count = overall_collision_hist.get(bucket, 0)
        percent = 100.0 * count / total_collisions
        lines.append(f"  - {bucket}: {count} fenêtres ({percent:.1f}%)")

    lines.extend(
        [
            "",
            "Variation de link_quality:",
            (
                f"- moyenne globale: {link_quality_mean:.4f} "
                f"(min {link_quality_min:.4f}, max {link_quality_max:.4f})"
            ),
        ]
    )
    lines.extend(
        [
            "",
            "Puissance Rx effective (dBm):",
            (
                f"- requested moyen: {rx_power_dbm_requested_mean:.2f}, "
                f"effective moyen: {rx_power_dbm_effective_mean:.2f}"
            ),
            (
                f"- valeur finale moyenne: {rx_power_dbm_mean:.2f} "
                f"(min {rx_power_dbm_min:.2f}, max {rx_power_dbm_max:.2f})"
            ),
            (
                f"- clamps globaux: {overall_rx_power_dbm_clamped_count}/"
                f"{overall_rx_power_dbm_effective_count} ({rx_power_dbm_clamped_ratio:.1%})"
            ),
            (
                f"- plage admissible: {RX_POWER_DBM_MIN:.2f}..{RX_POWER_DBM_MAX:.2f} dBm"
            ),
        ]
    )
    lines.append("- clamps Rx par taille:")
    for size in sorted(per_size_stats):
        stats = per_size_stats[size]
        size_clamped = int(stats.get("rx_power_dbm_clamped_count", 0))
        size_total = int(stats.get("rx_power_dbm_effective_count", 0))
        size_requested_mean = (
            float(stats.get("rx_power_dbm_requested_sum", 0.0)) / size_total
            if size_total > 0
            else 0.0
        )
        size_effective_mean = (
            float(stats.get("rx_power_dbm_effective_sum", 0.0)) / size_total
            if size_total > 0
            else 0.0
        )
        size_ratio = size_clamped / size_total if size_total > 0 else 0.0
        lines.append(
            f"  - taille {size}: requested={size_requested_mean:.2f}, "
            f"effective={size_effective_mean:.2f}, clamps={size_clamped}/{size_total} "
            f"({size_ratio:.1%})"
        )
        if size_total > 0 and size_ratio > 0.05:
            lines.append(
                f"    AVERTISSEMENT: plus de 5% des échantillons sont clampés "
                f"pour la taille {size}."
            )
    if rx_power_dbm_clamped_ratio > 0.05:
        lines.append(
            "AVERTISSEMENT global: plus de 5% des échantillons Rx sont clampés."
        )
    if per_size_link_quality_mean:
        lq_means = list(per_size_link_quality_mean.values())
        lq_delta = max(lq_means) - min(lq_means) if lq_means else 0.0
        variation_label = (
            "variation détectée"
            if lq_delta >= 1e-3
            else "variation très faible (quasi stable)"
        )
        lines.append(f"- amplitude inter-tailles: {lq_delta:.4f} ({variation_label})")
        lines.append("- moyenne link_quality par taille:")
        for size in sorted(per_size_link_quality_mean):
            lines.append(
                f"  - taille {size}: {per_size_link_quality_mean[size]:.4f}"
            )

    reward_zero_ratio = reward_zero_total / reward_count if reward_count > 0 else 0.0
    reward_min_value = 0.0 if reward_min is None else reward_min
    reward_max_value = 0.0 if reward_max is None else reward_max
    lines.extend(
        [
            "",
            "Analyse reward nul:",
            (
                f"- reward min/max observé: {reward_min_value:.4f}/"
                f"{reward_max_value:.4f}"
            ),
            f"- part de reward nul: {reward_zero_total}/{reward_count} "
            f"({reward_zero_ratio:.1%})",
        ]
    )
    if reward_zero_total > 0:
        no_success_ratio = reward_zero_no_success / reward_zero_total
        clipped_ratio = reward_zero_clipped / reward_zero_total
        lines.append(
            f"- reward nul sans succès: {reward_zero_no_success} "
            f"({no_success_ratio:.1%})"
        )
        lines.append(
            f"- reward nul malgré succès (>0): {reward_zero_clipped} "
            f"({clipped_ratio:.1%})"
        )
        if no_success_ratio > 0.6:
            conclusion = "Le reward nul provient majoritairement d'une absence de succès."
            lines.append(
                "- message: rewards nuls majoritairement dus à un success_rate nul."
            )
        elif clipped_ratio > 0.6:
            conclusion = (
                "Le reward nul provient majoritairement d'un écrêtage (pénalité collision)."
            )
        else:
            conclusion = (
                "Le reward nul est mixte: absence de succès et écrêtage contribuent."
            )
        lines.append(f"- conclusion: {conclusion}")
    else:
        lines.append("- conclusion: aucun reward nul détecté.")

    total_losses = (
        overall_losses_collisions
        + overall_losses_congestion
        + overall_losses_link_quality
    )
    lines.extend(["", "Analyse des pertes (compteurs T07):"])
    if total_losses > 0:
        dominant_cause = _dominant_loss_cause(
            overall_losses_collisions,
            overall_losses_congestion,
            overall_losses_link_quality,
        )
        lines.extend(
            [
                f"- pertes collisions: {overall_losses_collisions}",
                f"- pertes congestion: {overall_losses_congestion}",
                f"- pertes link_quality: {overall_losses_link_quality}",
                f"- cause dominante: {dominant_cause}",
            ]
        )
    else:
        lines.append("- aucune perte détectée via les compteurs T07.")

    reward_debug_entries: list[tuple[int, str, dict[str, float]]] = []
    for size in sorted(per_size_stats):
        summary = per_size_stats[size].get("reward_debug_summary", {})
        if not isinstance(summary, dict):
            continue
        for algo_label, values in sorted(summary.items()):
            if isinstance(values, dict):
                reward_debug_entries.append((size, str(algo_label), values))

    if reward_debug_entries:
        lines.extend(["", "Résumé reward debug (par algo/taille):"])
        for size, algo_label, values in reward_debug_entries:
            weighted_quality_mean = float(values.get("weighted_quality_mean", 0.0))
            collision_penalty_mean = float(values.get("collision_penalty_mean", 0.0))
            success_term_mean = float(values.get("success_term_mean", 0.0))
            reward_floor_mean = float(values.get("reward_floor_mean", 0.0))
            component_mean = (
                weighted_quality_mean
                + collision_penalty_mean
                + success_term_mean
                + reward_floor_mean
            ) / 4.0
            lines.append(
                f"- taille {size} | {algo_label}: "
                f"weighted_quality={weighted_quality_mean:.4f}, "
                f"collision_penalty={collision_penalty_mean:.4f}, "
                f"success_term={success_term_mean:.4f}, "
                f"reward_floor={reward_floor_mean:.4f}, "
                f"moyenne={component_mean:.4f}"
            )

    lines.extend(["", "Relances en profil sécurisé:"])
    if safe_profile_sizes:
        sizes_label = ", ".join(str(size) for size in sorted(set(safe_profile_sizes)))
        lines.append(f"- tailles relancées: {sizes_label}")
    else:
        lines.append("- aucune taille relancée en profil sécurisé.")

    return "\n".join(lines)


def _assert_success_rate_threshold(
    per_size_stats: dict[int, dict[str, object]],
    threshold: float = SUCCESS_ZERO_RATIO_THRESHOLD,
    strict: bool = False,
) -> dict[str, object]:
    overall_success_sum = sum(
        float(stats.get("success_sum", 0.0)) for stats in per_size_stats.values()
    )
    overall_success_count = sum(
        int(stats.get("success_count", 0)) for stats in per_size_stats.values()
    )
    overall_success_zero_count = sum(
        int(stats.get("success_zero_count", 0)) for stats in per_size_stats.values()
    )
    assessment: dict[str, object] = {
        "simulation_quality": "ok",
        "thresholds": {"success_zero_ratio_max": float(threshold)},
        "reasons": [],
    }
    if overall_success_count <= 0:
        reasons = assessment["reasons"]
        if isinstance(reasons, list):
            reasons.append("Aucune statistique success_rate disponible.")
        assessment["simulation_quality"] = "low"
        if strict:
            raise RuntimeError("Aucune statistique success_rate disponible.")
        return assessment

    zero_ratio = overall_success_zero_count / overall_success_count
    assessment["success_zero_ratio"] = zero_ratio
    assessment["success_rate_mean"] = overall_success_sum / overall_success_count
    if zero_ratio > threshold:
        success_mean = overall_success_sum / overall_success_count
        per_size_lines: list[str] = []
        for size in sorted(per_size_stats):
            stats = per_size_stats[size]
            size_count = int(stats.get("success_count", 0))
            size_zero = int(stats.get("success_zero_count", 0))
            size_sum = float(stats.get("success_sum", 0.0))
            size_mean = size_sum / size_count if size_count > 0 else 0.0
            size_zero_ratio = size_zero / size_count if size_count > 0 else 0.0
            per_size_lines.append(
                f"  - taille {size}: mean={size_mean:.4f}, zéros={size_zero_ratio:.1%}"
            )
        summary = "\n".join(per_size_lines) if per_size_lines else "  - aucune statistique"
        message = (
            "Simulation invalide : success_rate trop faible.\n"
            f"- success_rate moyen global: {success_mean:.4f}\n"
            f"- part de zéros: {zero_ratio:.1%} (seuil {threshold:.1%})\n"
            "- résumé par taille:\n"
            f"{summary}"
        )
        reasons = assessment["reasons"]
        if isinstance(reasons, list):
            reasons.append(
                "Part de success_rate nuls trop élevée: "
                f"{zero_ratio:.1%} > seuil {threshold:.1%}."
            )
        assessment["simulation_quality"] = "low"
        logging.warning(message)
        if strict:
            raise RuntimeError(message)
    return assessment


def _write_post_simulation_report(
    output_dir: Path,
    per_size_stats: dict[int, dict[str, object]],
    per_size_diagnostics: dict[int, dict[str, float]],
    safe_profile_sizes: Sequence[int],
) -> None:
    report = _compose_post_simulation_report(
        per_size_stats, per_size_diagnostics, safe_profile_sizes
    )
    report_path = output_dir / "post_simulation_report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    print(report)


def _compute_quantiles(values: list[float], quantiles: Sequence[float]) -> dict[float, float]:
    if not values:
        return {float(q): 0.0 for q in quantiles}
    sorted_values = sorted(values)
    n_values = len(sorted_values)
    if n_values == 1:
        single_value = float(sorted_values[0])
        return {float(q): single_value for q in quantiles}

    result: dict[float, float] = {}
    for quantile in quantiles:
        q = max(0.0, min(1.0, float(quantile)))
        position = q * (n_values - 1)
        low_index = int(math.floor(position))
        high_index = int(math.ceil(position))
        if low_index == high_index:
            result[q] = float(sorted_values[low_index])
            continue
        low_value = float(sorted_values[low_index])
        high_value = float(sorted_values[high_index])
        fraction = position - low_index
        result[q] = low_value + (high_value - low_value) * fraction
    return result


def _load_raw_rows_for_snir_distribution(
    base_results_dir: Path, flat_output: bool
) -> list[dict[str, str]]:
    raw_paths: list[Path] = []
    if flat_output:
        raw_path = base_results_dir / "raw_results.csv"
        if raw_path.exists():
            raw_paths.append(raw_path)
    else:
        raw_paths.extend(
            sorted((base_results_dir / BY_SIZE_DIRNAME).glob("size_*/rep_*/raw_results.csv"))
        )

    rows: list[dict[str, str]] = []
    for raw_path in raw_paths:
        with raw_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(dict(row))
    return rows




def _write_step2_diagnostics_exports(
    output_dir: Path,
    per_size_diagnostics: dict[int, dict[str, float]],
    per_size_stats: dict[int, dict[str, object]],
    flat_output: bool,
) -> None:
    diagnostics_header = [
        "network_size",
        "success_rate_mean",
        "collision_mean",
        "link_quality_mean",
        "reward_mean",
        "traffic_coeff_clamp_rate_mean",
    ]
    diagnostics_values: list[list[object]] = []
    for size in sorted(per_size_diagnostics):
        diagnostics = per_size_diagnostics[size]
        stats = per_size_stats.get(size, {})
        link_quality_count = int(stats.get("link_quality_count", 0))
        link_quality_sum = float(stats.get("link_quality_sum", 0.0))
        link_quality_mean = (
            link_quality_sum / link_quality_count if link_quality_count > 0 else 0.0
        )
        clamp_count = int(stats.get("traffic_coeff_clamp_rate_count", 0))
        clamp_sum = float(stats.get("traffic_coeff_clamp_rate_sum", 0.0))
        clamp_mean = clamp_sum / clamp_count if clamp_count > 0 else 0.0
        diagnostics_values.append(
            [
                size,
                round(float(diagnostics.get("success_mean", 0.0)), 6),
                round(float(diagnostics.get("collision_mean", 0.0)), 6),
                round(link_quality_mean, 6),
                round(float(diagnostics.get("reward_mean", 0.0)), 6),
                round(clamp_mean, 6),
            ]
        )
    diagnostics_path = _ensure_csv_within_scope(
        output_dir / "aggregates" / "diagnostics_step2_by_size.csv", output_dir
    )
    write_rows(
        diagnostics_path,
        diagnostics_header,
        diagnostics_values,
    )

    losses_header = ["cause", "count"]
    losses_values = [
        [
            "collisions",
            sum(int(stats.get("losses_collisions_total", 0)) for stats in per_size_stats.values()),
        ],
        [
            "congestion",
            sum(int(stats.get("losses_congestion_total", 0)) for stats in per_size_stats.values()),
        ],
        [
            "link_quality",
            sum(int(stats.get("losses_link_quality_total", 0)) for stats in per_size_stats.values()),
        ],
    ]
    write_rows(_ensure_csv_within_scope(output_dir / "aggregates" / "loss_causes_histogram.csv", output_dir), losses_header, losses_values)

    raw_rows = _load_raw_rows_for_snir_distribution(output_dir, flat_output)
    snir_candidates = ["snir_db", "snir", "snir_value", "snr_db", "snir_threshold_db"]
    snir_column = ""
    if raw_rows:
        first_row_keys = set(raw_rows[0])
        for candidate in snir_candidates:
            if candidate in first_row_keys:
                snir_column = candidate
                break
    quantiles = (0.1, 0.25, 0.5, 0.75, 0.9)
    snir_values_by_sf: dict[int, list[float]] = defaultdict(list)
    if snir_column:
        for row in raw_rows:
            if str(row.get("cluster", "")) != "all":
                continue
            sf_value = row.get("sf")
            snir_value = row.get(snir_column)
            if sf_value in (None, "") or snir_value in (None, ""):
                continue
            try:
                snir_values_by_sf[int(float(sf_value))].append(float(snir_value))
            except (TypeError, ValueError):
                continue

    snir_header = [
        "sf",
        "snir_metric",
        "count",
        "q10",
        "q25",
        "q50",
        "q75",
        "q90",
    ]
    snir_rows: list[list[object]] = []
    for sf in sorted(snir_values_by_sf):
        values = snir_values_by_sf[sf]
        quantiles_map = _compute_quantiles(values, quantiles)
        snir_rows.append(
            [
                sf,
                snir_column,
                len(values),
                round(quantiles_map[0.1], 6),
                round(quantiles_map[0.25], 6),
                round(quantiles_map[0.5], 6),
                round(quantiles_map[0.75], 6),
                round(quantiles_map[0.9], 6),
            ]
        )
    write_rows(_ensure_csv_within_scope(output_dir / "aggregates" / "snir_distribution_by_sf.csv", output_dir), snir_header, snir_rows)

    def _min_mean_max(values: list[float]) -> tuple[float, float, float]:
        if not values:
            return 0.0, 0.0, 0.0
        return min(values), sum(values) / len(values), max(values)

    rx_requested_by_size: dict[int, list[float]] = defaultdict(list)
    rx_effective_by_size: dict[int, list[float]] = defaultdict(list)
    overlap_by_size: dict[int, list[float]] = defaultdict(list)
    snir_by_size_sf: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    losses_by_size: dict[int, dict[str, int]] = defaultdict(
        lambda: {"collisions": 0, "congestion": 0, "link_quality": 0}
    )
    loss_keys_seen_by_size: dict[int, set[tuple[object, object, object]]] = defaultdict(set)

    for row in raw_rows:
        if str(row.get("cluster", "")) != "all":
            continue
        size_raw = row.get("network_size", row.get("density"))
        try:
            size = int(float(size_raw))
        except (TypeError, ValueError):
            continue

        requested_raw = row.get("rx_power_dbm_requested")
        if requested_raw not in (None, ""):
            try:
                rx_requested_by_size[size].append(float(requested_raw))
            except (TypeError, ValueError):
                pass

        effective_raw = row.get("rx_power_dbm_effective", row.get("rx_power_dbm"))
        if effective_raw not in (None, ""):
            try:
                rx_effective_by_size[size].append(float(effective_raw))
            except (TypeError, ValueError):
                pass

        overlap_raw = row.get("mean_temporal_overlap")
        if overlap_raw not in (None, ""):
            try:
                overlap_by_size[size].append(float(overlap_raw))
            except (TypeError, ValueError):
                pass

        if snir_column:
            sf_raw = row.get("sf")
            snir_raw = row.get(snir_column)
            if sf_raw not in (None, "") and snir_raw not in (None, ""):
                try:
                    sf_value = int(float(sf_raw))
                    snir_by_size_sf[size][sf_value].append(float(snir_raw))
                except (TypeError, ValueError):
                    pass

        loss_key = (row.get("replication"), row.get("algo"), row.get("round"))
        if loss_key in loss_keys_seen_by_size[size]:
            continue
        loss_keys_seen_by_size[size].add(loss_key)
        for cause, key in (
            ("collisions", "losses_collisions"),
            ("congestion", "losses_congestion"),
            ("link_quality", "losses_link_quality"),
        ):
            try:
                losses_by_size[size][cause] += int(float(row.get(key, 0) or 0))
            except (TypeError, ValueError):
                continue

    diagnostics_dedicated_header = [
        "network_size",
        "sf",
        "snir_metric",
        "snir_min",
        "snir_mean",
        "snir_max",
        "rx_power_dbm_requested_min",
        "rx_power_dbm_requested_mean",
        "rx_power_dbm_requested_max",
        "rx_power_dbm_effective_min",
        "rx_power_dbm_effective_mean",
        "rx_power_dbm_effective_max",
        "losses_collisions",
        "losses_congestion",
        "losses_link_quality",
        "losses_collisions_ratio",
        "losses_congestion_ratio",
        "losses_link_quality_ratio",
        "losses_dominant_cause",
        "mean_temporal_overlap",
    ]
    diagnostics_dedicated_rows: list[list[object]] = []
    all_sizes = sorted(
        set(per_size_diagnostics)
        | set(rx_requested_by_size)
        | set(rx_effective_by_size)
        | set(snir_by_size_sf)
        | set(losses_by_size)
        | set(overlap_by_size)
    )
    for size in all_sizes:
        req_min, req_mean, req_max = _min_mean_max(rx_requested_by_size.get(size, []))
        eff_min, eff_mean, eff_max = _min_mean_max(rx_effective_by_size.get(size, []))
        overlap_values = overlap_by_size.get(size, [])
        mean_temporal_overlap = (
            sum(overlap_values) / len(overlap_values) if overlap_values else 0.0
        )
        losses = losses_by_size.get(
            size,
            {"collisions": 0, "congestion": 0, "link_quality": 0},
        )
        total_losses = sum(losses.values())
        collisions_ratio = (
            losses["collisions"] / total_losses if total_losses > 0 else 0.0
        )
        congestion_ratio = (
            losses["congestion"] / total_losses if total_losses > 0 else 0.0
        )
        link_quality_ratio = (
            losses["link_quality"] / total_losses if total_losses > 0 else 0.0
        )
        dominant_cause = _dominant_loss_cause(
            losses["collisions"],
            losses["congestion"],
            losses["link_quality"],
        )
        sf_values = sorted(snir_by_size_sf.get(size, {}))
        if not sf_values:
            sf_values = [-1]
        for sf_value in sf_values:
            snir_values = snir_by_size_sf.get(size, {}).get(sf_value, []) if sf_value >= 0 else []
            snir_min, snir_mean, snir_max = _min_mean_max(snir_values)
            diagnostics_dedicated_rows.append(
                [
                    size,
                    "all" if sf_value < 0 else sf_value,
                    snir_column,
                    round(snir_min, 6),
                    round(snir_mean, 6),
                    round(snir_max, 6),
                    round(req_min, 6),
                    round(req_mean, 6),
                    round(req_max, 6),
                    round(eff_min, 6),
                    round(eff_mean, 6),
                    round(eff_max, 6),
                    int(losses["collisions"]),
                    int(losses["congestion"]),
                    int(losses["link_quality"]),
                    round(collisions_ratio, 6),
                    round(congestion_ratio, 6),
                    round(link_quality_ratio, 6),
                    dominant_cause,
                    round(mean_temporal_overlap, 6),
                ]
            )
    write_rows(
        _ensure_csv_within_scope(output_dir / "aggregates" / "diagnostics_by_size.csv", output_dir),
        diagnostics_dedicated_header,
        diagnostics_dedicated_rows,
    )

    algo_sf_header = [
        "network_size",
        "algo",
        "sf",
        "snir_metric",
        "success_rate_mean",
        "success_rate_min",
        "success_rate_max",
        "collisions_ratio",
        "link_quality_rejects_ratio",
        "congestion_rejects_ratio",
        "capture_success_ratio",
        "snir_mean",
        "snir_min",
        "snir_max",
        "rx_power_dbm_requested_mean",
        "rx_power_dbm_requested_min",
        "rx_power_dbm_requested_max",
        "rx_power_dbm_effective_mean",
        "rx_power_dbm_effective_min",
        "rx_power_dbm_effective_max",
    ]
    algo_sf_grouped: dict[tuple[int, str, int], dict[str, object]] = {}
    for row in raw_rows:
        if str(row.get("cluster", "")) != "all":
            continue
        size_raw = row.get("network_size", row.get("density"))
        algo_label = str(row.get("algo", ""))
        sf_raw = row.get("sf")
        try:
            size = int(float(size_raw))
            sf_value = int(float(sf_raw))
        except (TypeError, ValueError):
            continue

        group_key = (size, algo_label, sf_value)
        group = algo_sf_grouped.setdefault(
            group_key,
            {
                "success_rates": [],
                "snir_values": [],
                "rx_requested": [],
                "rx_effective": [],
                "round_keys": set(),
                "losses_collisions": 0,
                "losses_congestion": 0,
                "losses_link_quality": 0,
                "capture_ratios": [],
            },
        )

        try:
            group["success_rates"].append(float(row.get("success_rate", 0.0) or 0.0))
        except (TypeError, ValueError):
            pass

        if snir_column:
            snir_raw = row.get(snir_column)
            if snir_raw not in (None, ""):
                try:
                    group["snir_values"].append(float(snir_raw))
                except (TypeError, ValueError):
                    pass

        requested_raw = row.get("rx_power_dbm_requested")
        if requested_raw not in (None, ""):
            try:
                group["rx_requested"].append(float(requested_raw))
            except (TypeError, ValueError):
                pass

        effective_raw = row.get("rx_power_dbm_effective", row.get("rx_power_dbm"))
        if effective_raw not in (None, ""):
            try:
                group["rx_effective"].append(float(effective_raw))
            except (TypeError, ValueError):
                pass

        round_key = (
            row.get("replication"),
            row.get("algo"),
            row.get("round"),
            row.get("sf"),
        )
        round_keys = group["round_keys"]
        if isinstance(round_keys, set) and round_key not in round_keys:
            round_keys.add(round_key)
            for key, target in (
                ("losses_collisions", "losses_collisions"),
                ("losses_congestion", "losses_congestion"),
                ("losses_link_quality", "losses_link_quality"),
            ):
                try:
                    group[target] = int(group.get(target, 0)) + int(float(row.get(key, 0) or 0))
                except (TypeError, ValueError):
                    continue
            try:
                group["capture_ratios"].append(float(row.get("capture_ratio", 0.0) or 0.0))
            except (TypeError, ValueError):
                pass

    algo_sf_rows: list[list[object]] = []
    compact_by_size: dict[int, list[dict[str, float]]] = defaultdict(list)
    for size, algo_label, sf_value in sorted(algo_sf_grouped):
        group = algo_sf_grouped[(size, algo_label, sf_value)]
        success_values = list(group.get("success_rates", []))
        snir_values = list(group.get("snir_values", []))
        rx_requested_values = list(group.get("rx_requested", []))
        rx_effective_values = list(group.get("rx_effective", []))
        capture_ratios = list(group.get("capture_ratios", []))
        success_min, success_mean, success_max = _min_mean_max(success_values)
        snir_min, snir_mean, snir_max = _min_mean_max(snir_values)
        req_min, req_mean, req_max = _min_mean_max(rx_requested_values)
        eff_min, eff_mean, eff_max = _min_mean_max(rx_effective_values)
        losses_collisions = int(group.get("losses_collisions", 0))
        losses_congestion = int(group.get("losses_congestion", 0))
        losses_link_quality = int(group.get("losses_link_quality", 0))
        losses_total = losses_collisions + losses_congestion + losses_link_quality
        collisions_ratio = losses_collisions / losses_total if losses_total > 0 else 0.0
        link_quality_rejects_ratio = (
            losses_link_quality / losses_total if losses_total > 0 else 0.0
        )
        congestion_rejects_ratio = (
            losses_congestion / losses_total if losses_total > 0 else 0.0
        )
        capture_success_ratio = (
            sum(capture_ratios) / len(capture_ratios) if capture_ratios else 0.0
        )
        compact_by_size[size].append(
            {
                "success_rate_mean": success_mean,
                "collisions_ratio": collisions_ratio,
                "capture_success_ratio": capture_success_ratio,
                "snir_mean": snir_mean,
            }
        )
        algo_sf_rows.append(
            [
                size,
                algo_label,
                sf_value,
                snir_column,
                round(success_mean, 6),
                round(success_min, 6),
                round(success_max, 6),
                round(collisions_ratio, 6),
                round(link_quality_rejects_ratio, 6),
                round(congestion_rejects_ratio, 6),
                round(capture_success_ratio, 6),
                round(snir_mean, 6),
                round(snir_min, 6),
                round(snir_max, 6),
                round(req_mean, 6),
                round(req_min, 6),
                round(req_max, 6),
                round(eff_mean, 6),
                round(eff_min, 6),
                round(eff_max, 6),
            ]
        )

    write_rows(
        _ensure_csv_within_scope(output_dir / "aggregates" / "diagnostics_by_size_algo_sf.csv", output_dir),
        algo_sf_header,
        algo_sf_rows,
    )

    print("Résumé compact diagnostics_by_size_algo_sf (Step2):")
    if not compact_by_size:
        print("- aucun échantillon cluster=all exploitable.")
    for size in sorted(compact_by_size):
        rows = compact_by_size[size]
        if not rows:
            continue
        success_mean = sum(row["success_rate_mean"] for row in rows) / len(rows)
        collisions_ratio = sum(row["collisions_ratio"] for row in rows) / len(rows)
        capture_ratio_mean = (
            sum(row["capture_success_ratio"] for row in rows) / len(rows)
        )
        snir_mean = sum(row["snir_mean"] for row in rows) / len(rows)
        print(
            f"- taille {size}: success={success_mean:.4f}, "
            f"collisions={collisions_ratio:.3f}, capture={capture_ratio_mean:.3f}, "
            f"snir={snir_mean:.2f}"
        )


def _detect_low_success_first_size(
    density: int,
    diagnostics: dict[str, float],
    threshold: float = 0.2,
) -> bool:
    success_mean = float(diagnostics.get("success_mean", 0.0))
    if success_mean >= threshold:
        return False
    print(
        "AVERTISSEMENT: première taille avec success_rate moyen trop bas "
        f"détectée ({density})."
    )
    print(
        "Résumé première taille sous le seuil: success_mean="
        f"{success_mean:.4f}, min={diagnostics.get('success_min', 0.0):.4f}, "
        f"max={diagnostics.get('success_max', 0.0):.4f} "
        f"(seuil {threshold:.2f})."
    )
    return True


def _needs_safe_profile_rerun(
    diagnostics: dict[str, float], threshold: float = SAFE_PROFILE_SUCCESS_THRESHOLD
) -> tuple[bool, float]:
    success_mean = float(diagnostics.get("success_mean", 0.0))
    return success_mean < threshold, success_mean


def _needs_super_safe_profile_rerun(
    diagnostics: dict[str, float],
    threshold: float = SUPER_SAFE_PROFILE_SUCCESS_THRESHOLD,
) -> tuple[bool, float]:
    success_mean = float(diagnostics.get("success_mean", 0.0))
    return success_mean < threshold, success_mean


def _replace_rows_for_size(
    rows: list[dict[str, object]],
    size: int,
    new_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    filtered = []
    for row in rows:
        value = row.get("network_size", row.get("density", -1))
        try:
            size_value = int(value)
        except (TypeError, ValueError):
            size_value = -1
        if size_value != int(size):
            filtered.append(row)
    filtered.extend(new_rows)
    return filtered


def _is_non_empty_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _load_step2_aggregated_with_errors(
    aggregated_path: Path,
) -> list[dict[str, object]]:
    if not aggregated_path.exists():
        print(f"Aucun aggregated_results.csv détecté: {aggregated_path}")
        return []
    rows: list[dict[str, object]] = []
    with aggregated_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            network_size_value = row.get("network_size") or row.get("density")
            if network_size_value in (None, ""):
                continue
            rows.append(
                {
                    "network_size": int(float(network_size_value)),
                    "algo": str(row.get("algo", "")),
                    "snir_mode": str(row.get("snir_mode", "")),
                    "cluster": str(row.get("cluster", "all")),
                    "reward_mean": float(row.get("reward_mean", 0.0) or 0.0),
                    "reward_std": float(row.get("reward_std", 0.0) or 0.0),
                    "reward_ci95": float(row.get("reward_ci95", 0.0) or 0.0),
                }
            )
    return rows


def _plot_summary_reward(output_dir: Path) -> None:
    aggregated_path = output_dir / "aggregates" / "aggregated_results.csv"
    rows = _load_step2_aggregated_with_errors(aggregated_path)
    if not rows:
        print("Aucune ligne agrégée disponible pour le plot de synthèse.")
        return
    rows = [
        row
        for row in rows
        if row.get("cluster") == "all" and row.get("snir_mode") == "snir_on"
    ]
    if not rows:
        print("Aucune ligne agrégée filtrée pour le plot de synthèse.")
        return
    apply_plot_style()
    import matplotlib.pyplot as plt

    network_sizes = sorted({row["network_size"] for row in rows})
    algorithms = sorted({row["algo"] for row in rows})
    fig, ax = plt.subplots(figsize=resolve_ieee_figsize(len(algorithms)))
    error_key = "reward_ci95" if any(row.get("reward_ci95") for row in rows) else "reward_std"
    for algo in algorithms:
        algo_rows = [row for row in rows if row["algo"] == algo]
        points = {row["network_size"]: row["reward_mean"] for row in algo_rows}
        errors = {row["network_size"]: row.get(error_key, 0.0) for row in algo_rows}
        values = [points.get(size, float("nan")) for size in network_sizes]
        yerr = [errors.get(size, 0.0) for size in network_sizes]
        ax.errorbar(
            network_sizes,
            values,
            yerr=yerr,
            marker="o",
            capsize=3,
            label=algo,
        )
    ax.set_xlabel("Network size (number of nodes)")
    ax.set_ylabel("Mean Reward")
    ax.set_xticks(network_sizes)
    place_adaptive_legend(fig, ax)
    output_plot_dir = output_dir / "plots"
    save_figure(fig, output_plot_dir, "summary_reward", use_tight=False)
    plt.close(fig)


def _simulate_density(
    task: tuple[int, int, list[int], dict[str, object], Path, Path | None, bool]
) -> dict[str, object]:
    (
        density,
        density_idx,
        replications,
        config,
        base_results_dir,
        timestamp_dir,
        flat_output,
    ) = task
    if config.get("debug_step2") or config.get("reward_debug"):
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s:%(name)s:%(message)s",
        )
    raw_rows: list[dict[str, object]] = []
    per_rep_rows: dict[int, list[dict[str, object]]] = {
        replication: [] for replication in replications
    }
    selection_rows: list[dict[str, object]] = []
    learning_curve_rows: list[dict[str, object]] = []
    algorithms = ("adr", "mixra_h", "mixra_opt", "ucb1_sf")
    algo_offsets = {algo: idx * 10000 for idx, algo in enumerate(algorithms)}
    jitter_range_s = float(config.get("jitter_range_s", 30.0))
    print(f"Jitter range utilisé (s): {jitter_range_s}")
    offsets_label = ", ".join(
        f"{algo}={offset}" for algo, offset in algo_offsets.items()
    )
    print(f"Offsets de seed par algorithme: {offsets_label}")
    status_csv_path = base_results_dir / "run_status_step2.csv"

    _log_effective_traffic_scale_for_density(int(density), config)

    for replication in replications:
        seed = int(config["base_seed"]) + density_idx * 1000 + replication
        for algorithm in algorithms:
            algorithm_seed = seed + algo_offsets[algorithm]
            result = None
            for attempt in (1, 2):
                try:
                    set_deterministic_seed(algorithm_seed)
                    result = run_simulation(
                        algorithm=algorithm,
                        n_nodes=int(density),
                        density=density,
                        snir_mode="snir_on",
                        seed=algorithm_seed,
                        snir_threshold_db=float(config["snir_threshold_db"]),
                        snir_threshold_min_db=float(config["snir_threshold_min_db"]),
                        snir_threshold_max_db=float(config["snir_threshold_max_db"]),
                        traffic_mode=str(config["traffic_mode"]),
                        jitter_range_s=jitter_range_s,
                        window_duration_s=float(config["window_duration_s"]),
                        window_size=int(config["window_size"]),
                        lambda_collision=(
                            float(config["lambda_collision"])
                            if config.get("lambda_collision") is not None
                            else None
                        ),
                        traffic_coeff_min=float(config["traffic_coeff_min"]),
                        traffic_coeff_max=float(config["traffic_coeff_max"]),
                        traffic_coeff_enabled=bool(config["traffic_coeff_enabled"]),
                        traffic_coeff_scale=float(config["traffic_coeff_scale"]),
                        auto_collision_control=bool(config.get("auto_collision_control", False)),
                        capture_probability=float(config["capture_probability"]),
                        congestion_coeff=float(config["congestion_coeff"]),
                        congestion_coeff_base=float(config["congestion_coeff_base"]),
                        congestion_coeff_growth=float(config["congestion_coeff_growth"]),
                        congestion_coeff_max=float(config["congestion_coeff_max"]),
                        network_load_min=float(config["network_load_min"]),
                        network_load_max=float(config["network_load_max"]),
                        collision_size_min=float(config["collision_size_min"]),
                        collision_size_under_max=float(config["collision_size_under_max"]),
                        collision_size_over_max=float(config["collision_size_over_max"]),
                        collision_size_factor=(
                            float(config["collision_size_factor"])
                            if config.get("collision_size_factor") is not None
                            else None
                        ),
                        max_penalty_ratio=(
                            float(config["max_penalty_ratio"])
                            if config.get("max_penalty_ratio") is not None
                            else None
                        ),
                        traffic_coeff_clamp_min=float(config["traffic_coeff_clamp_min"]),
                        traffic_coeff_clamp_max=float(config["traffic_coeff_clamp_max"]),
                        traffic_coeff_clamp_enabled=bool(config["traffic_coeff_clamp_enabled"]),
                        traffic_coeff_clamp_alert_threshold=float(
                            config.get("traffic_coeff_clamp_alert_threshold", 0.45)
                        ),
                        window_delay_enabled=bool(config["window_delay_enabled"]),
                        window_delay_range_s=float(config["window_delay_range_s"]),
                        shadowing_sigma_db=(
                            float(config["shadowing_sigma_db"])
                            if config.get("shadowing_sigma_db") is not None
                            else None
                        ),
                        rx_power_dbm=float(config["rx_power_dbm"]),
                        reference_network_size=int(config["reference_network_size"]),
                        reward_floor=(
                            float(config["reward_floor"])
                            if config.get("reward_floor") is not None
                            else None
                        ),
                        zero_success_quality_bonus_factor=(
                            float(config["zero_success_quality_bonus_factor"])
                            if config.get("zero_success_quality_bonus_factor") is not None
                            else None
                        ),
                        floor_on_zero_success=bool(config["floor_on_zero_success"]),
                        debug_step2=bool(config.get("debug_step2", False)),
                        reward_debug=bool(config.get("reward_debug", False)),
                        reward_alert_level=str(config.get("reward_alert_level", "WARNING")),
                        safe_profile=bool(config.get("safe_profile", False)),
                        no_clamp=bool(config.get("no_clamp", False)),
                    )
                    break
                except Exception as exc:
                    print(
                        "Échec simulation step2 "
                        f"(size={density}, rep={replication}, seed={algorithm_seed}, "
                        f"algo={algorithm}, step=step2, attempt={attempt}/2): {exc}"
                    )
                    if attempt == 1:
                        print("Retry immédiat (1/1) pour cette simulation unitaire.")
                    else:
                        with status_csv_path.open("a", newline="", encoding="utf-8") as handle:
                            writer = csv.DictWriter(
                                handle,
                                fieldnames=[
                                    "status",
                                    "step",
                                    "network_size",
                                    "replication",
                                    "seed",
                                    "algorithm",
                                    "error",
                                ],
                            )
                            writer.writerow(
                                {
                                    "status": "failed",
                                    "step": "step2",
                                    "network_size": density,
                                    "replication": replication,
                                    "seed": algorithm_seed,
                                    "algorithm": algorithm,
                                    "error": str(exc),
                                }
                            )
            if result is None:
                continue
            for row in result.raw_rows:
                row["replication"] = replication
            raw_rows.extend(result.raw_rows)
            per_rep_rows[replication].extend(result.raw_rows)
            if algorithm == "ucb1_sf":
                selection_rows.extend(result.selection_prob_rows)
            learning_curve_rows.extend(result.learning_curve_rows)

    invalid_sizes = {
        int(row.get("network_size", -1))
        for row in raw_rows
        if int(row.get("network_size", -1)) != int(density)
    }
    if invalid_sizes:
        raise ValueError(
            "network_size différent de n_nodes détecté pour la taille "
            f"{density}: {sorted(invalid_sizes)}"
        )
    diagnostics = _summarize_success_collision(raw_rows)
    post_stats = _summarize_post_simulation(raw_rows)
    _log_size_diagnostics(int(density), diagnostics)

    for replication, rows in per_rep_rows.items():
        rep_dir = (
            base_results_dir
            / BY_SIZE_DIRNAME
            / f"size_{density}"
            / f"rep_{replication}"
        )
        write_simulation_results(rep_dir, rows, network_size=density)
        _log_results_written(rep_dir, len(rows))
        _log_unique_network_sizes(rep_dir)
        if timestamp_dir is not None:
            rep_timestamp_dir = (
                timestamp_dir
                / BY_SIZE_DIRNAME
                / f"size_{density}"
                / f"rep_{replication}"
            )
            write_simulation_results(rep_timestamp_dir, rows, network_size=density)
            _log_results_written(rep_timestamp_dir, len(rows))
            _log_unique_network_sizes(rep_timestamp_dir)
    return {
        "density": density,
        "row_count": len(raw_rows),
        "selection_rows": selection_rows,
        "learning_curve_rows": learning_curve_rows,
        "diagnostics": diagnostics,
        "post_stats": post_stats,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_cli_args(argv)
    if getattr(args, "no_clamp", False):
        if getattr(args, "traffic_coeff_clamp_enabled", False):
            print("Option --no-clamp: désactivation du clamp traffic_coeff.")
        args.traffic_coeff_clamp_enabled = False
    _log_default_profile_if_needed(args)
    if getattr(args, "auto_safe_profile", False) and not getattr(args, "safe_profile", False):
        _apply_safe_profile_with_log(
            args,
            "--auto-safe-profile activé par défaut (application avant simulation)",
        )
    elif getattr(args, "safe_profile", False):
        _apply_safe_profile_with_log(args, "demande explicite --safe-profile")
    try:
        export_formats = parse_export_formats(args.formats)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    set_default_export_formats(export_formats)
    set_default_figure_clamp_enabled(not args.no_figure_clamp)
    if args.debug_step2 or getattr(args, "reward_debug", False):
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s:%(name)s:%(message)s",
        )
    base_seed = set_deterministic_seed(args.seeds_base)
    densities = parse_network_size_list(args.network_sizes)
    requested_sizes = list(densities)
    flat_output = False
    if bool(args.flat_output):
        print(
            "Option --flat-output ignorée: écriture primaire imposée sous by_size/."
        )
    if getattr(args, "reference_network_size", None) is not None:
        reference_network_size = int(args.reference_network_size)
        reference_source = "argument --reference-network-size"
    elif len(requested_sizes) == 1:
        reference_network_size = int(requested_sizes[0])
        reference_source = "taille unique demandée"
    else:
        reference_network_size = int(round(median(requested_sizes)))
        reference_source = "médiane des tailles demandées"
    print(
        "Référence réseau utilisée pour l'étape 2: "
        f"{reference_network_size} ({reference_source})."
    )
    _log_step2_autonomous_inputs(args, reference_network_size)
    replications = replication_ids(args.replications)
    simulated_sizes: list[int] = []

    base_results_dir = Path(__file__).resolve().parent / "results"
    _ensure_csv_within_scope(base_results_dir / "run_status_step2.csv", base_results_dir)
    ensure_dir(base_results_dir)
    status_csv_path = base_results_dir / "run_status_step2.csv"
    status_fieldnames = [
        "status",
        "step",
        "network_size",
        "replication",
        "seed",
        "algorithm",
        "error",
    ]
    if args.reset_status or not status_csv_path.exists():
        with status_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=status_fieldnames)
            writer.writeheader()
        action = "réinitialisé" if args.reset_status else "initialisé"
        print(f"Statut Step2 {action}: {status_csv_path.resolve()}")
    else:
        print(f"Statut Step2 conservé (mode campagne): {status_csv_path.resolve()}")
    timestamp_dir: Path | None = None
    if args.timestamp:
        timestamp_dir = base_results_dir / timestamp_tag(with_timezone=True)
        ensure_dir(timestamp_dir)
    aggregated_path = base_results_dir / "aggregates" / "aggregated_results.csv"
    if _is_non_empty_file(aggregated_path):
        size_bytes = aggregated_path.stat().st_size
        print(
            "aggregated_results.csv existant détecté "
            f"({size_bytes} octets) : aucune réinitialisation."
        )

    selection_rows: list[dict[str, object]] = []
    learning_curve_rows: list[dict[str, object]] = []
    size_diagnostics: dict[int, dict[str, float]] = {}
    size_post_stats: dict[int, dict[str, object]] = {}
    safe_profile_sizes: set[int] = set()
    total_rows = 0

    config: dict[str, object] = {
        "base_seed": base_seed,
        "no_clamp": getattr(args, "no_clamp", False),
        "safe_profile": bool(getattr(args, "safe_profile", False)),
        "traffic_mode": args.traffic_mode,
        "jitter_range_s": args.jitter_range_s,
        "window_duration_s": args.window_duration_s,
        "window_size": args.window_size,
        "lambda_collision": args.lambda_collision,
        "snir_threshold_db": args.snir_threshold_db,
        "snir_threshold_min_db": args.snir_threshold_min_db,
        "snir_threshold_max_db": args.snir_threshold_max_db,
        "traffic_coeff_min": args.traffic_coeff_min,
        "traffic_coeff_max": args.traffic_coeff_max,
        "traffic_coeff_enabled": args.traffic_coeff_enabled,
        "traffic_coeff_scale": args.traffic_coeff_scale,
        "auto_collision_control": args.auto_collision_control,
        "capture_probability": args.capture_probability,
        "congestion_coeff": args.congestion_coeff,
        "congestion_coeff_base": args.congestion_coeff_base,
        "congestion_coeff_growth": args.congestion_coeff_growth,
        "congestion_coeff_max": args.congestion_coeff_max,
        "network_load_min": args.network_load_min,
        "network_load_max": args.network_load_max,
        "collision_size_min": args.collision_size_min,
        "collision_size_under_max": args.collision_size_under_max,
        "collision_size_over_max": args.collision_size_over_max,
        "collision_size_factor": args.collision_size_factor,
        "max_penalty_ratio": getattr(args, "max_penalty_ratio", None),
        "traffic_coeff_clamp_min": args.traffic_coeff_clamp_min,
        "traffic_coeff_clamp_max": args.traffic_coeff_clamp_max,
        "traffic_coeff_clamp_enabled": args.traffic_coeff_clamp_enabled,
        "traffic_coeff_clamp_alert_threshold": 0.45,
        "window_delay_enabled": args.window_delay_enabled,
        "window_delay_range_s": args.window_delay_range_s,
        "reference_network_size": max(1, reference_network_size),
        "reward_floor": args.reward_floor,
        "zero_success_quality_bonus_factor": getattr(
            args, "zero_success_quality_bonus_factor", None
        ),
        "floor_on_zero_success": args.floor_on_zero_success,
        "shadowing_sigma_db": getattr(args, "shadowing_sigma_db", None),
        "debug_step2": args.debug_step2,
        "reward_debug": getattr(args, "reward_debug", False),
        "reward_alert_level": args.reward_alert_level,
    }
    requested_rx_power_dbm = float(args.rx_power_dbm)
    config["rx_power_dbm"] = _clamp_rx_power_dbm(requested_rx_power_dbm)
    _log_rx_power_diagnostics(requested_rx_power_dbm, float(config["rx_power_dbm"]))

    auto_tuning_payload = _run_auto_tuning_before_campaign(
        config, replications, base_results_dir
    )
    if str(auto_tuning_payload.get("status", "")) == "failed_tuning":
        raise SystemExit("failed_tuning")
    _sync_args_from_auto_tuned_config(args, config)
    print(
        "Paramètres Step2 retenus après auto-tuning: "
        f"traffic_load_scale_step2={float(config['traffic_coeff_scale']):.4f}, "
        f"collision_size_min={float(config['collision_size_min']):.4f}, "
        f"collision_size_under_max={float(config['collision_size_under_max']):.4f}, "
        f"collision_size_over_max={float(config['collision_size_over_max']):.4f}."
    )

    aggregated_sizes = _read_nested_sizes(base_results_dir, replications)
    merge_stats = aggregate_results_by_size(
        base_results_dir,
        write_global_aggregated=bool(args.global_aggregated),
    )
    print(
        "Agrégation Step2 par taille: "
        f"{merge_stats['size_count']} dossier(s) size_<N>, "
        f"{merge_stats['size_row_count']} ligne(s) consolidée(s)."
    )
    if bool(args.global_aggregated):
        print(
            "Agrégation Step2 globale: "
            f"{merge_stats['global_row_count']} ligne(s) écrite(s) "
            "dans results/aggregates/aggregated_results.csv."
        )
    requested_set = set(requested_sizes)
    existing_sizes = sorted(requested_set & aggregated_sizes)
    remaining_sizes = sorted(requested_set - aggregated_sizes)
    existing_label = ", ".join(map(str, existing_sizes)) if existing_sizes else "aucune"
    if args.resume:
        densities = remaining_sizes
        print("Mode reprise activé: exclusion des tailles déjà agrégées.")
    simulated_targets = densities if args.resume else requested_sizes
    simulated_label = (
        ", ".join(map(str, simulated_targets)) if simulated_targets else "aucune"
    )
    print(f"Tailles déjà présentes dans les sous-dossiers by_size/: {existing_label}")
    print(f"Tailles à simuler: {simulated_label}")
    safe_profile_active = bool(getattr(args, "safe_profile", False))
    load_clamp_min, load_clamp_max = _resolve_load_clamps(
        DEFAULT_CONFIG.step2,
        args.network_load_min,
        args.network_load_max,
        safe_profile=safe_profile_active,
        no_clamp=getattr(args, "no_clamp", False),
    )
    (
        collision_clamp_min,
        collision_clamp_under_max,
        collision_clamp_over_max,
    ) = _resolve_collision_clamps(
        DEFAULT_CONFIG.step2,
        args.collision_size_min,
        args.collision_size_under_max,
        args.collision_size_over_max,
        safe_profile=safe_profile_active,
        no_clamp=getattr(args, "no_clamp", False),
    )
    table = _format_size_factor_table(
        sorted(set(requested_sizes)),
        reference_network_size,
        load_clamp_min,
        load_clamp_max,
        collision_clamp_min,
        collision_clamp_under_max,
        collision_clamp_over_max,
    )
    logger.info(
        "Tableau comparaison facteurs par taille (réf=%s):\n%s",
        reference_network_size,
        table,
    )

    worker_count = max(1, int(args.workers))
    low_success_detected = False
    if getattr(args, "auto_safe_profile", False) and not getattr(
        args, "safe_profile", False
    ):
        if worker_count > 1:
            print(
                "Auto-safe-profile actif: exécution séquentielle pour détecter "
                "la première taille sous le seuil."
            )
        worker_count = 1

    if densities and worker_count == 1:
        for density_idx, density in enumerate(densities):
            task = (
                density,
                density_idx,
                replications,
                config,
                base_results_dir,
                timestamp_dir,
                flat_output,
            )
            result = _simulate_density(task)
            diagnostics = dict(result["diagnostics"])
            diagnostics_for_threshold = dict(diagnostics)
            low_success = False
            rerun_safe = False
            rerun_super_safe = False
            if not config.get("safe_profile", False):
                should_rerun, success_mean = _needs_safe_profile_rerun(diagnostics)
                if should_rerun:
                    safe_config, changes = _build_safe_profile_config(config)
                    reason = (
                        f"success_mean={success_mean:.4f} "
                        f"< seuil {SAFE_PROFILE_SUCCESS_THRESHOLD:.2f}"
                    )
                    _log_safe_profile_switch(int(density), reason, changes)
                    safe_task = (
                        density,
                        density_idx,
                        replications,
                        safe_config,
                        base_results_dir,
                        timestamp_dir,
                        flat_output,
                    )
                    result = _simulate_density(safe_task)
                    diagnostics = dict(result["diagnostics"])
                    rerun_safe = True
                    should_super_safe, super_success_mean = (
                        _needs_super_safe_profile_rerun(diagnostics)
                    )
                    if should_super_safe:
                        super_config, super_changes = _build_super_safe_profile_config(
                            safe_config
                        )
                        super_reason = (
                            f"success_mean={super_success_mean:.4f} "
                            f"< seuil {SUPER_SAFE_PROFILE_SUCCESS_THRESHOLD:.2f}"
                        )
                        _log_super_safe_profile_switch(
                            int(density), super_reason, super_changes
                        )
                        super_task = (
                            density,
                            density_idx,
                            replications,
                            super_config,
                            base_results_dir,
                            timestamp_dir,
                            flat_output,
                        )
                        result = _simulate_density(super_task)
                        diagnostics = dict(result["diagnostics"])
                        rerun_super_safe = True
            if not low_success_detected:
                low_success = _detect_low_success_first_size(
                    density, diagnostics_for_threshold
                )
                if low_success and not getattr(args, "safe_profile", False):
                    if getattr(args, "auto_safe_profile", False):
                        _apply_safe_profile_with_log(
                            args,
                            "auto-safe-profile (success_rate faible détecté)",
                        )
                        _update_safe_profile_config(config, args)
                        if not rerun_safe:
                            print(
                                "Relance de la taille "
                                f"{density} avec le profil sécurisé."
                            )
                            result = _simulate_density(task)
                            diagnostics = dict(result["diagnostics"])
                            rerun_safe = True
                            should_super_safe, super_success_mean = (
                                _needs_super_safe_profile_rerun(diagnostics)
                            )
                            if should_super_safe:
                                super_config, super_changes = (
                                    _build_super_safe_profile_config(config)
                                )
                                super_reason = (
                                    f"success_mean={super_success_mean:.4f} "
                                    f"< seuil {SUPER_SAFE_PROFILE_SUCCESS_THRESHOLD:.2f}"
                                )
                                _log_super_safe_profile_switch(
                                    int(density), super_reason, super_changes
                                )
                                super_task = (
                                    density,
                                    density_idx,
                                    replications,
                                    super_config,
                                    base_results_dir,
                                    timestamp_dir,
                                    flat_output,
                                )
                                result = _simulate_density(super_task)
                                diagnostics = dict(result["diagnostics"])
                                rerun_super_safe = True
                            safe_profile_sizes.add(int(result["density"]))
                    else:
                        print(
                            "Astuce: utilisez --auto-safe-profile pour basculer "
                            "automatiquement vers STEP2_SAFE_CONFIG."
                        )
            if low_success:
                low_success_detected = True
            if int(result["density"]) not in simulated_sizes:
                simulated_sizes.append(int(result["density"]))
            total_rows += int(result["row_count"])
            if rerun_safe or rerun_super_safe:
                selection_rows[:] = _replace_rows_for_size(
                    selection_rows, int(result["density"]), result["selection_rows"]
                )
                learning_curve_rows[:] = _replace_rows_for_size(
                    learning_curve_rows,
                    int(result["density"]),
                    result["learning_curve_rows"],
                )
                safe_profile_sizes.add(int(result["density"]))
            else:
                selection_rows.extend(result["selection_rows"])
                learning_curve_rows.extend(result["learning_curve_rows"])
            size_diagnostics[int(result["density"])] = dict(result["diagnostics"])
            size_post_stats[int(result["density"])] = dict(result["post_stats"])
    elif densities:
        first_task = (
            densities[0],
            0,
            replications,
            config,
            base_results_dir,
            timestamp_dir,
            flat_output,
        )
        first_result = _simulate_density(first_task)
        simulated_sizes.append(int(first_result["density"]))
        total_rows += int(first_result["row_count"])
        selection_rows.extend(first_result["selection_rows"])
        learning_curve_rows.extend(first_result["learning_curve_rows"])
        size_diagnostics[int(first_result["density"])] = dict(
            first_result["diagnostics"]
        )
        size_post_stats[int(first_result["density"])] = dict(
            first_result["post_stats"]
        )
        diagnostics = dict(first_result["diagnostics"])
        low_success_detected = _detect_low_success_first_size(
            int(first_result["density"]), diagnostics
        )
        remaining_tasks = [
            (
                density,
                density_idx,
                replications,
                config,
                base_results_dir,
                timestamp_dir,
                flat_output,
            )
            for density_idx, density in enumerate(densities)
            if density_idx != 0
        ]

        if remaining_tasks:
            ctx = get_context("spawn")
            with ctx.Pool(processes=worker_count) as pool:
                for result in pool.imap_unordered(_simulate_density, remaining_tasks):
                    simulated_sizes.append(int(result["density"]))
                    total_rows += int(result["row_count"])
                    selection_rows.extend(result["selection_rows"])
                    learning_curve_rows.extend(result["learning_curve_rows"])
                    size_diagnostics[int(result["density"])] = dict(
                        result["diagnostics"]
                    )
                    size_post_stats[int(result["density"])] = dict(
                        result["post_stats"]
                    )

        if not low_success_detected:
            for density in densities:
                diagnostics = size_diagnostics.get(int(density))
                if diagnostics and _detect_low_success_first_size(
                    int(density), diagnostics
                ):
                    low_success_detected = True
                    break
        if not config.get("safe_profile", False):
            low_sizes = []
            for density in densities:
                diagnostics = size_diagnostics.get(int(density))
                if diagnostics is None:
                    continue
                should_rerun, success_mean = _needs_safe_profile_rerun(diagnostics)
                if should_rerun:
                    low_sizes.append((int(density), success_mean))
            if low_sizes:
                print(
                    "Relance séquentielle en profil sécurisé pour les tailles "
                    "sous le seuil."
                )
                density_to_idx = {int(value): idx for idx, value in enumerate(densities)}
                for density, success_mean in low_sizes:
                    density_idx = density_to_idx.get(int(density), 0)
                    safe_config, changes = _build_safe_profile_config(config)
                    reason = (
                        f"success_mean={success_mean:.4f} "
                        f"< seuil {SAFE_PROFILE_SUCCESS_THRESHOLD:.2f}"
                    )
                    _log_safe_profile_switch(int(density), reason, changes)
                    safe_task = (
                        density,
                        density_idx,
                        replications,
                        safe_config,
                        base_results_dir,
                        timestamp_dir,
                        flat_output,
                    )
                    safe_result = _simulate_density(safe_task)
                    rerun_super_safe = False
                    super_result = None
                    should_super_safe, super_success_mean = _needs_super_safe_profile_rerun(
                        safe_result["diagnostics"]
                    )
                    if should_super_safe:
                        super_config, super_changes = _build_super_safe_profile_config(
                            safe_config
                        )
                        super_reason = (
                            f"success_mean={super_success_mean:.4f} "
                            f"< seuil {SUPER_SAFE_PROFILE_SUCCESS_THRESHOLD:.2f}"
                        )
                        _log_super_safe_profile_switch(
                            int(density), super_reason, super_changes
                        )
                        super_task = (
                            density,
                            density_idx,
                            replications,
                            super_config,
                            base_results_dir,
                            timestamp_dir,
                            flat_output,
                        )
                        super_result = _simulate_density(super_task)
                        rerun_super_safe = True
                    final_result = super_result if rerun_super_safe else safe_result
                    total_rows += int(final_result["row_count"])
                    selection_rows[:] = _replace_rows_for_size(
                        selection_rows,
                        int(final_result["density"]),
                        final_result["selection_rows"],
                    )
                    learning_curve_rows[:] = _replace_rows_for_size(
                        learning_curve_rows,
                        int(final_result["density"]),
                        final_result["learning_curve_rows"],
                    )
                    size_diagnostics[int(final_result["density"])] = dict(
                        final_result["diagnostics"]
                    )
                    size_post_stats[int(final_result["density"])] = dict(
                        final_result["post_stats"]
                    )
                    safe_profile_sizes.add(int(final_result["density"]))

    print(f"Rows written: {total_rows}")
    if flat_output and simulated_sizes:
        _assert_flat_output_sizes(base_results_dir, simulated_sizes)
    _verify_metric_variation(size_diagnostics)
    _write_post_simulation_report(
        base_results_dir,
        size_post_stats,
        size_diagnostics,
        sorted(safe_profile_sizes),
    )
    _write_step2_diagnostics_exports(
        base_results_dir,
        size_diagnostics,
        size_post_stats,
        flat_output,
    )
    strict_mode = bool(getattr(args, "strict", False)) or not bool(
        args.allow_low_success_rate
    )
    try:
        quality_summary = _assert_success_rate_threshold(
            size_post_stats,
            strict=strict_mode,
        )
    except RuntimeError as exc:
        if _has_traceable_outputs(base_results_dir):
            warning_message = (
                "Contrôle qualité strict ignoré: sorties traçables détectées, "
                "la génération des figures continue."
            )
            print(f"ATTENTION: {warning_message}")
            quality_summary = _assert_success_rate_threshold(
                size_post_stats,
                strict=False,
            )
            quality_summary["strict_quality_error"] = str(exc)
            reasons = quality_summary.get("reasons")
            if isinstance(reasons, list):
                reasons.append(warning_message)
        else:
            raise
    quality_path = base_results_dir / "simulation_quality_step2.json"
    quality_path.write_text(
        json.dumps(quality_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if selection_rows:
        rl5_rows = _aggregate_selection_probs(selection_rows)
        rl5_path = _ensure_csv_within_scope(
            base_results_dir / "rl5_selection_prob.csv", base_results_dir
        )
        rl5_header = ["network_size", "density", "round", "sf", "selection_prob"]
        rl5_values = [
            [
                row["network_size"],
                row["density"],
                row["round"],
                row["sf"],
                row["selection_prob"],
            ]
            for row in rl5_rows
        ]
        write_rows(rl5_path, rl5_header, rl5_values)
        if timestamp_dir is not None:
            rl5_timestamp_path = _ensure_csv_within_scope(
                timestamp_dir / "rl5_selection_prob.csv", base_results_dir
            )
            write_rows(rl5_timestamp_path, rl5_header, rl5_values)

    if learning_curve_rows:
        learning_curve = _aggregate_learning_curve(learning_curve_rows)
        learning_curve_header = ["network_size", "density", "round", "algo", "avg_reward"]
        learning_curve_values = [
            [
                row["network_size"],
                row["density"],
                row["round"],
                row["algo"],
                row["avg_reward"],
            ]
            for row in learning_curve
        ]
        learning_curve_path = _ensure_csv_within_scope(
            base_results_dir / "learning_curve.csv", base_results_dir
        )
        write_rows(learning_curve_path, learning_curve_header, learning_curve_values)
        if timestamp_dir is not None:
            learning_curve_timestamp_path = _ensure_csv_within_scope(
                timestamp_dir / "learning_curve.csv", base_results_dir
            )
            write_rows(
                learning_curve_timestamp_path,
                learning_curve_header,
                learning_curve_values,
            )
        clamp_rate_header = [
            "network_size",
            "density",
            "round",
            "algo",
            "traffic_coeff_clamp_rate",
        ]
        clamp_rate_values = [
            [
                row["network_size"],
                row["density"],
                row["round"],
                row["algo"],
                row.get("traffic_coeff_clamp_rate", 0.0),
            ]
            for row in learning_curve
        ]
        clamp_rate_path = _ensure_csv_within_scope(
            base_results_dir / "traffic_coeff_clamp_rate.csv", base_results_dir
        )
        write_rows(clamp_rate_path, clamp_rate_header, clamp_rate_values)
        if timestamp_dir is not None:
            clamp_rate_timestamp_path = _ensure_csv_within_scope(
                timestamp_dir / "traffic_coeff_clamp_rate.csv", base_results_dir
            )
            write_rows(clamp_rate_timestamp_path, clamp_rate_header, clamp_rate_values)

    aggregated_sizes = _read_nested_sizes(base_results_dir, replications)
    merge_stats = aggregate_results_by_size(
        base_results_dir,
        write_global_aggregated=bool(args.global_aggregated),
    )
    print(
        "Agrégation Step2 par taille: "
        f"{merge_stats['size_count']} dossier(s) size_<N>, "
        f"{merge_stats['size_row_count']} ligne(s) consolidée(s)."
    )
    if bool(args.global_aggregated):
        print(
            "Agrégation Step2 globale: "
            f"{merge_stats['global_row_count']} ligne(s) écrite(s) "
            "dans results/aggregates/aggregated_results.csv."
        )
    missing_sizes = sorted(set(requested_sizes) - aggregated_sizes)
    if missing_sizes:
        missing_label = ", ".join(map(str, missing_sizes))
        print(
            "ATTENTION: tailles manquantes dans by_size/, "
            f"done.flag non écrit. Manquantes: {missing_label}"
        )
    else:
        (base_results_dir / "done.flag").write_text("done\n", encoding="utf-8")
        print("done.flag écrit (agrégation complète).")
    if simulated_sizes:
        sizes_label = ",".join(str(size) for size in simulated_sizes)
        print(f"Tailles simulées: {sizes_label}")
    aggregated_path = base_results_dir / "aggregates" / "aggregated_results.csv"
    if aggregated_path.exists():
        if args.plot_summary:
            _plot_summary_reward(base_results_dir)
    elif args.plot_summary:
        print(
            "Plot de synthèse ignoré: aggregated_results.csv absent "
            "(utilisez make_all_plots.py)."
        )

    _log_step2_key_csv_paths(base_results_dir)


if __name__ == "__main__":
    main()
