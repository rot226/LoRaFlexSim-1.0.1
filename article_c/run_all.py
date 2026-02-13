"""Exécute toutes les étapes de l'article C."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from importlib.util import find_spec
from pathlib import Path
from statistics import median
from time import perf_counter, sleep

if find_spec("article_c") is None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    if find_spec("article_c") is None:
        raise ModuleNotFoundError(
            "Impossible d'importer 'article_c'. "
            "Ajoutez la racine du dépôt au PYTHONPATH."
        )

from article_c.common.config import DEFAULT_CONFIG
from article_c.common.csv_io import aggregate_results_by_size
from article_c.common.utils import parse_network_size_list, replication_dirnames
from article_c.step1.run_step1 import main as run_step1
from article_c.step2.run_step2 import main as run_step2
from article_c.validate_results import main as validate_results

DEFAULT_REPLICATIONS = 10
STEP2_SUCCESS_RATE_MEAN_LOW_THRESHOLD = 0.20


LOG_LEVELS = {"quiet": 0, "info": 1, "debug": 2}
_CURRENT_LOG_LEVEL = LOG_LEVELS["info"]

PREFIX_CONTRACT_ERROR = "[CONTRACT_ERROR]"
PREFIX_IO_ERROR = "[IO_ERROR]"


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




def _assert_path_within_scope(path: Path, scope_root: Path, context: str) -> Path:
    resolved_path = path.resolve()
    resolved_scope = scope_root.resolve()
    if resolved_path.parent != resolved_scope and resolved_scope not in resolved_path.parents:
        raise RuntimeError(
            f"{context}: sortie hors périmètre autorisé. "
            f"Fichier: {resolved_path} ; périmètre attendu: {resolved_scope}."
        )
    return resolved_path


def _log_existing_key_csv_paths(step_label: str, results_dir: Path) -> None:
    key_csv_names = ("run_status", "raw_results", "aggregated_results", "raw_metrics")
    for csv_path in sorted(results_dir.glob("**/*.csv")):
        _assert_path_within_scope(csv_path, results_dir, step_label)
        if any(csv_path.name.startswith(prefix) for prefix in key_csv_names):
            log_info(f"{step_label}: CSV clé détecté {csv_path.resolve()}")


def _cleanup_size_directory(results_dir: Path, network_size: int, step_label: str) -> None:
    """Supprime l'ancien dossier `by_size/size_<N>` avant une relance isolée."""
    size_dir = results_dir / "by_size" / f"size_{network_size}"
    if size_dir.exists():
        _assert_path_within_scope(size_dir, results_dir, step_label)
        shutil.rmtree(size_dir)
        log_debug(f"{step_label}: dossier nettoyé avant simulation isolée: {size_dir.resolve()}")


def _remove_global_aggregation_artifacts(results_dir: Path, step_label: str) -> None:
    """Retire les artefacts globaux avant campagne pour garantir une agrégation finale unique."""
    for relative in (Path("aggregates") / "aggregated_results.csv", Path("aggregates") / "diagnostics_step2_by_size.csv", Path("aggregates") / "diagnostics_by_size.csv", Path("aggregates") / "diagnostics_by_size_algo_sf.csv"):
        candidate = results_dir / relative
        _assert_path_within_scope(candidate, results_dir, step_label)
        if candidate.exists():
            candidate.unlink()
            log_debug(f"{step_label}: artefact global supprimé avant campagne: {candidate.resolve()}")


def _assert_no_global_writes_during_simulation(results_dir: Path, step_label: str) -> None:
    """Échoue si des CSV globaux sont écrits dans `results/` pendant la simulation."""
    forbidden = [
        results_dir / "aggregates" / "aggregated_results.csv",
        results_dir / "raw_results.csv",
        results_dir / "raw_metrics.csv",
    ]
    written = [
        str(_assert_path_within_scope(path, results_dir, step_label))
        for path in forbidden
        if path.exists()
    ]
    if written:
        raise RuntimeError(
            f"{step_label}: écriture globale directe interdite pendant simulation: {written}"
        )


def _clean_run_artifacts(*, hard: bool) -> None:
    """Nettoie les artefacts run_all puis recrée l'arborescence minimale requise."""
    base_dir = Path(__file__).resolve().parent
    results_dirs = [
        (base_dir / "step1" / "results").resolve(),
        (base_dir / "step2" / "results").resolve(),
    ]
    plots_output_dirs = [
        (base_dir / "step1" / "plots" / "output").resolve(),
        (base_dir / "step2" / "plots" / "output").resolve(),
        (base_dir / "plots" / "output").resolve(),
    ]

    dirs_to_purge = list(results_dirs)
    if hard:
        dirs_to_purge.extend(plots_output_dirs)

    for directory in dirs_to_purge:
        _assert_path_within_scope(directory, base_dir, "RunAllClean")
        if directory.exists():
            shutil.rmtree(directory)
            log_info(f"[CLEAN] dossier supprimé: {directory.resolve()}")

    required_dirs = list(results_dirs)
    if hard:
        required_dirs.extend(plots_output_dirs)

    for directory in required_dirs:
        _assert_path_within_scope(directory, base_dir, "RunAllClean")
        directory.mkdir(parents=True, exist_ok=True)
        log_info(f"[CLEAN] dossier prêt: {directory.resolve()}")


def _write_campaign_state(
    state_path: Path,
    *,
    size: int | None,
    rep: int | None,
    step: str | None,
    status: str,
) -> None:
    payload = {
        "size": size,
        "rep": rep,
        "step": step,
        "status": status,
    }
    state_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_campaign_state(state_path: Path) -> dict[str, object] | None:
    if not state_path.exists():
        return None
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def _find_first_missing_rep(
    results_dir: Path,
    size: int,
    replications_total: int,
) -> int | None:
    missing = _missing_replications_by_size(results_dir, [size], replications_total)
    reps = missing.get(int(size), [])
    if not reps:
        return None
    return int(min(reps))


def _remove_done_flag(results_dir: Path, step_label: str) -> None:
    done_flag = results_dir / "done.flag"
    _assert_path_within_scope(done_flag, results_dir, step_label)
    if done_flag.exists():
        done_flag.unlink()
        log_debug(f"{step_label}: done.flag supprimé (campagne incomplète).")


def _self_check_replication_layout(
    size_dir: Path,
    expected_rep_dirs: list[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Self-check statique: compare les dossiers `rep_*` attendus vs présents."""
    expected_sorted = sorted(expected_rep_dirs)
    actual_sorted = sorted(
        path.name for path in size_dir.glob("rep_*") if path.is_dir()
    )
    missing = sorted(set(expected_sorted) - set(actual_sorted))
    unexpected = sorted(set(actual_sorted) - set(expected_sorted))
    return expected_sorted, actual_sorted, missing, unexpected

def _assert_output_layout_compliant(
    results_dir: Path,
    expected_sizes: list[int],
    replications_total: int,
    step_label: str,
) -> None:
    """Vérifie la conformité stricte du layout `by_size/size_<N>/rep_<R>`."""
    by_size_dir = results_dir / "by_size"
    if not by_size_dir.exists():
        raise RuntimeError(
            f"{PREFIX_CONTRACT_ERROR} {step_label}: dossier manquant {by_size_dir.resolve()}."
        )
    expected_rep_dirs = replication_dirnames(replications_total)
    for size in expected_sizes:
        size_dir = by_size_dir / f"size_{size}"
        if not size_dir.is_dir():
            raise RuntimeError(
                f"{PREFIX_CONTRACT_ERROR} {step_label}: layout invalide, dossier manquant {size_dir.resolve()}."
            )
        expected_dirs_sorted, actual_dirs_sorted, missing_reps, unexpected_reps = _self_check_replication_layout(
            size_dir,
            expected_rep_dirs,
        )
        if missing_reps or unexpected_reps:
            missing_rep_paths = [
                str((size_dir / rep_dir).resolve()) for rep_dir in missing_reps
            ]
            expected_rep_paths = [str((size_dir / rep_dir).resolve()) for rep_dir in expected_dirs_sorted]
            raise RuntimeError(
                f"{PREFIX_CONTRACT_ERROR} {step_label}: layout invalide. Taille concernée=size_{size}. "
                f"Réplications manquantes={missing_reps or 'aucune'}. "
                f"Réplications inattendues={unexpected_reps or 'aucune'}. "
                f"Rep attendus={expected_dirs_sorted}, rep réels={actual_dirs_sorted}. "
                f"Chemin(s) absolu(s) attendu(s) pour les réplications manquantes: "
                f"{missing_rep_paths or 'aucun'}. "
                f"Tous les chemins absolus attendus pour cette taille: {expected_rep_paths}."
            )
        _assert_cumulative_sizes_nested(results_dir, {int(size)}, step_label)


def _missing_replications_by_size(
    results_dir: Path,
    expected_sizes: list[int],
    replications_total: int,
) -> dict[int, list[int]]:
    """Scanne `by_size/size_<N>/rep_<R>` et retourne les réplications manquantes."""
    by_size_dir = results_dir / "by_size"
    expected_reps = set(range(1, int(replications_total) + 1))
    missing_by_size: dict[int, list[int]] = {}
    for size in expected_sizes:
        size_dir = by_size_dir / f"size_{int(size)}"
        existing_reps: set[int] = set()
        if size_dir.is_dir():
            for rep_dir in size_dir.glob("rep_*"):
                if not rep_dir.is_dir():
                    continue
                try:
                    rep_id = int(rep_dir.name.split("rep_", 1)[1])
                except (IndexError, ValueError):
                    continue
                if rep_id > 0:
                    existing_reps.add(rep_id)
        missing = sorted(expected_reps - existing_reps)
        if missing:
            missing_by_size[int(size)] = missing
    return missing_by_size


def _restore_preserved_rep_dirs(
    preserved_root: Path,
    size_dir: Path,
) -> None:
    for preserved_rep_dir in sorted(preserved_root.glob("rep_*")):
        target_rep_dir = size_dir / preserved_rep_dir.name
        if target_rep_dir.exists():
            shutil.rmtree(target_rep_dir)
        shutil.copytree(preserved_rep_dir, target_rep_dir)


def _relaunch_missing_replications(
    *,
    step_label: str,
    results_dir: Path,
    missing_by_size: dict[int, list[int]],
    build_args,
    runner,
    base_args: argparse.Namespace,
) -> list[tuple[int, int]]:
    """Relance uniquement les couples (size, rep) absents via filtres de taille/réplication."""
    relaunched: list[tuple[int, int]] = []
    for size in sorted(missing_by_size):
        size_dir = results_dir / "by_size" / f"size_{size}"
        size_dir.mkdir(parents=True, exist_ok=True)
        for rep in sorted(missing_by_size[size]):
            with tempfile.TemporaryDirectory(prefix=f"run_all_{step_label.lower()}_") as tmp_dir_str:
                tmp_dir = Path(tmp_dir_str)
                preserved_root = tmp_dir / "preserved"
                preserved_root.mkdir(parents=True, exist_ok=True)
                for rep_to_preserve in range(1, int(rep) + 1):
                    if rep_to_preserve == int(rep):
                        continue
                    existing_rep_dir = size_dir / f"rep_{rep_to_preserve}"
                    if existing_rep_dir.is_dir():
                        shutil.copytree(existing_rep_dir, preserved_root / existing_rep_dir.name)

                relaunch_args = argparse.Namespace(**vars(base_args))
                relaunch_args.network_sizes = [int(size)]
                relaunch_args.replications = int(rep)
                relaunch_args.flat_output = False
                relaunch_args.reset_status = False
                runner(build_args(relaunch_args))
                _restore_preserved_rep_dirs(preserved_root, size_dir)

            generated_rep_dir = size_dir / f"rep_{rep}"
            if not generated_rep_dir.is_dir():
                raise RuntimeError(
                    f"{step_label}: relance ciblée échouée, dossier manquant {generated_rep_dir.resolve()}."
                )
            relaunched.append((int(size), int(rep)))
    return relaunched



def _assert_aggregation_contract_consistent(
    results_dir: Path,
    expected_sizes: list[int],
    step_label: str,
) -> None:
    """Vérifie le contrat d'agrégation by_size (format unique autorisé)."""
    global_csv = results_dir / "aggregates" / "aggregated_results.csv"
    if global_csv.exists():
        raise RuntimeError(
            f"{step_label}: variante flat interdite détectée: {global_csv.resolve()}."
        )

    reference_fieldnames: list[str] | None = None
    total_by_size_rows = 0
    for size in expected_sizes:
        size_csv = results_dir / "by_size" / f"size_{size}" / "aggregated_results.csv"
        if not size_csv.exists():
            raise RuntimeError(
                f"{step_label}: agrégat par taille manquant: {size_csv.resolve()}."
            )
        with size_csv.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            size_fieldnames = list(reader.fieldnames or [])
            if reference_fieldnames is None and size_fieldnames:
                reference_fieldnames = size_fieldnames
            elif reference_fieldnames and size_fieldnames and size_fieldnames != reference_fieldnames:
                raise RuntimeError(
                    f"{step_label}: schéma incohérent entre agrégats by_size ({size_csv.resolve()})."
                )
            total_by_size_rows += sum(1 for _ in reader)

    if total_by_size_rows == 0:
        raise RuntimeError(
            f"{step_label}: incohérence d'agrégation finale, aucune ligne dans by_size."
        )


def _assert_required_aggregates_present(
    results_dir: Path,
    expected_sizes: list[int],
    step_label: str,
) -> None:
    """Valide la présence (et non-vacuité) des agrégats par taille obligatoires."""
    missing_paths: list[str] = []
    empty_paths: list[str] = []
    for size in expected_sizes:
        aggregate_path = results_dir / "by_size" / f"size_{int(size)}" / "aggregated_results.csv"
        if not aggregate_path.exists():
            missing_paths.append(str(aggregate_path.resolve()))
            continue
        with aggregate_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if sum(1 for _ in reader) == 0:
                empty_paths.append(str(aggregate_path.resolve()))
    if missing_paths or empty_paths:
        raise RuntimeError(
            f"{PREFIX_CONTRACT_ERROR} {step_label}: agrégats obligatoires invalides. "
            f"Manquants={missing_paths or 'aucun'}. "
            f"Vides={empty_paths or 'aucun'}."
        )


def _run_verify_all_strict(replications_total: int) -> None:
    """Exécute verify_all en mode strict et propage toute exception."""
    command = [
        sys.executable,
        "-m",
        "article_c.tools.verify_all",
        "--replications",
        str(int(replications_total)),
    ]
    log_info("[STRICT_PIPELINE] Exécution finale verify_all...")
    subprocess.run(command, check=True)
RUN_ALL_PRESETS: dict[str, dict[str, object]] = {
    "article-c": {
        "network_sizes": list(DEFAULT_CONFIG.scenario.network_sizes),
        "replications": 5,
        "seeds_base": 1,
        "snir_modes": "snir_on,snir_off",
        "snir_threshold_db": float(DEFAULT_CONFIG.snir.snir_threshold_db),
        "snir_threshold_min_db": float(DEFAULT_CONFIG.snir.snir_threshold_min_db),
        "snir_threshold_max_db": float(DEFAULT_CONFIG.snir.snir_threshold_max_db),
        "noise_floor_dbm": float(DEFAULT_CONFIG.snir.noise_floor_dbm),
    }
}


def _count_failed_runs(status_csv_path: Path, network_size: int) -> int:
    """Compte les exécutions marquées `failed` pour une taille donnée."""
    if not status_csv_path.exists():
        return 0
    failed = 0
    with status_csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("status", "")).strip().lower() != "failed":
                continue
            size_value = row.get("network_size")
            try:
                row_size = int(float(str(size_value)))
            except (TypeError, ValueError):
                continue
            if row_size == int(network_size):
                failed += 1
    return failed


def _read_step2_success_rate_mean(results_dir: Path, network_size: int) -> float | None:
    """Lit le success_rate moyen d'une taille depuis aggregates/diagnostics_step2_by_size.csv."""
    diagnostics_path = results_dir / "aggregates" / "diagnostics_step2_by_size.csv"
    if not diagnostics_path.exists():
        return None
    with diagnostics_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_size = row.get("network_size") or row.get("density")
            if raw_size in (None, ""):
                continue
            try:
                row_size = int(float(str(raw_size)))
            except (TypeError, ValueError):
                continue
            if row_size != int(network_size):
                continue
            try:
                return float(row.get("success_rate_mean", 0.0) or 0.0)
            except (TypeError, ValueError):
                return None
    return None


def _build_step2_quality_summary(
    results_dir: Path,
    network_size: int,
    failed_runs: int,
) -> dict[str, object]:
    """Évalue la qualité de simulation step2 (ok/low) et explique les raisons."""
    reasons: list[str] = []
    success_rate_mean = _read_step2_success_rate_mean(results_dir, network_size)
    if failed_runs > 0:
        reasons.append(
            f"run_status_step2.csv contient {failed_runs} exécution(s) en échec pour la taille {network_size}."
        )
    if success_rate_mean is None:
        reasons.append(
            "Impossible de lire success_rate_mean dans aggregates/diagnostics_step2_by_size.csv."
        )
    elif success_rate_mean < STEP2_SUCCESS_RATE_MEAN_LOW_THRESHOLD:
        reasons.append(
            "success_rate_mean "
            f"{success_rate_mean:.4f} < seuil {STEP2_SUCCESS_RATE_MEAN_LOW_THRESHOLD:.2f}."
        )

    quality = "low" if reasons else "ok"
    return {
        "simulation_quality": quality,
        "success_rate_mean": success_rate_mean,
        "thresholds": {
            "success_rate_mean_min": STEP2_SUCCESS_RATE_MEAN_LOW_THRESHOLD,
            "failed_runs_max": 0,
        },
        "reasons": reasons,
    }


def _assert_cumulative_sizes(
    csv_path: Path,
    expected_sizes_so_far: set[int],
    step_label: str,
) -> None:
    """Valide que le CSV contient bien toutes les tailles attendues jusque-là."""
    if not csv_path.exists():
        raise RuntimeError(
            f"{PREFIX_IO_ERROR} {step_label}: CSV introuvable pour validation cumulative: {csv_path}"
        )
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(
                f"{step_label}: en-têtes CSV absents dans {csv_path.resolve()}."
            )
        fieldnames = {
            name.lstrip("\ufeff").strip() for name in reader.fieldnames if name is not None
        }
        size_key = "network_size" if "network_size" in fieldnames else None
        if size_key is None and "density" in fieldnames:
            size_key = "density"
        if size_key is None:
            raise RuntimeError(
                f"{step_label}: colonnes network_size/density absentes dans {csv_path.resolve()}."
            )
        found_sizes: set[int] = set()
        for row in reader:
            value = row.get(size_key)
            if value in (None, ""):
                continue
            try:
                found_sizes.add(int(float(str(value))))
            except ValueError:
                continue
    if not expected_sizes_so_far.issubset(found_sizes):
        raise RuntimeError(
            f"{step_label}: validation cumulative échouée pour {csv_path.resolve()}. "
            f"Tailles attendues={sorted(expected_sizes_so_far)}, "
            f"tailles trouvées={sorted(found_sizes)}"
        )


def _assert_cumulative_sizes_nested(
    base_results_dir: Path,
    expected_sizes_so_far: set[int],
    step_label: str,
) -> None:
    """Valide le mode non-flat via `by_size/size_*/rep_*` et leurs CSV."""

    by_size_dir = base_results_dir / "by_size"
    max_attempts = 3
    retry_delay_s = 0.25
    found_sizes: set[int] = set()
    valid_files_count = 0
    scan_debug: list[dict[str, object]] = []

    for attempt in range(1, max_attempts + 1):
        size_pattern = str(by_size_dir / "size_*")
        log_debug(
            f"{step_label}: scan cumulatif tentative {attempt}/{max_attempts} "
            f"via le pattern {size_pattern}"
        )

        current_scan: dict[str, object] = {
            "attempt": attempt,
            "size_pattern": size_pattern,
            "size_dirs": [],
        }
        size_dirs = sorted(path for path in by_size_dir.glob("size_*") if path.is_dir())
        current_scan["size_dirs"] = [str(path.resolve()) for path in size_dirs]
        log_debug(
            f"{step_label}: dossiers size scannés: "
            f"{current_scan['size_dirs']}"
        )

        found_sizes = set()
        valid_files_count = 0
        size_entries: list[dict[str, object]] = []
        for size_dir in size_dirs:
            try:
                size_value = int(size_dir.name.split("size_", 1)[1])
            except (IndexError, ValueError):
                continue

            rep_pattern = str(size_dir / "rep_*")
            log_debug(f"{step_label}: scan des réplications via le pattern {rep_pattern}")
            rep_dirs = sorted(path for path in size_dir.glob("rep_*") if path.is_dir())
            rep_resolved = [str(path.resolve()) for path in rep_dirs]
            log_debug(
                f"{step_label}: dossiers rep scannés pour size_{size_value}: "
                f"{rep_resolved}"
            )

            size_entries.append(
                {
                    "size": size_value,
                    "size_dir": str(size_dir.resolve()),
                    "rep_pattern": rep_pattern,
                    "rep_dirs": rep_resolved,
                }
            )
            if not rep_dirs:
                continue
            for rep_dir in rep_dirs:
                _assert_cumulative_sizes(
                    rep_dir / "aggregated_results.csv",
                    {size_value},
                    step_label,
                )
                valid_files_count += 1
            found_sizes.add(size_value)

        current_scan["size_entries"] = size_entries
        current_scan["found_sizes"] = sorted(found_sizes)
        current_scan["valid_files_count"] = valid_files_count
        scan_debug.append(current_scan)

        status = "OK" if expected_sizes_so_far.issubset(found_sizes) else "FAIL"
        log_info(
            f"{step_label}: scan cumulatif tentative {attempt}/{max_attempts} | "
            f"tailles trouvées={sorted(found_sizes)} | "
            f"fichiers valides={valid_files_count} | statut={status}"
        )

        if expected_sizes_so_far.issubset(found_sizes):
            return
        if attempt < max_attempts:
            log_debug(
                f"{step_label}: validation cumulative incomplète "
                f"(attendues={sorted(expected_sizes_so_far)}, trouvées={sorted(found_sizes)}). "
                f"Nouvelle tentative dans {retry_delay_s:.2f}s."
            )
            sleep(retry_delay_s)

    error_message = (
        f"{step_label}: validation cumulative échouée après {max_attempts} tentatives pour "
        f"{base_results_dir.resolve()}. Tailles attendues={sorted(expected_sizes_so_far)}, "
        f"tailles trouvées={sorted(found_sizes)}. "
        f"fichiers valides={valid_files_count}. statut=FAIL"
    )
    if _CURRENT_LOG_LEVEL >= LOG_LEVELS["debug"]:
        error_message += (
            f" Diagnostic scan complet={json.dumps(scan_debug, ensure_ascii=False)}"
        )
    raise RuntimeError(error_message)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construit le parseur d'arguments CLI pour l'exécution complète."""
    parser = argparse.ArgumentParser(
        description="Exécute les étapes 1 et 2 avec des arguments communs."
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
        choices=tuple(sorted(RUN_ALL_PRESETS)),
        default=None,
        help=(
            "Préremplit un profil documenté. "
            "Preset 'article-c' => network_sizes=50 100 150, replications=5, "
            "seeds_base=1 et options SNIR (modes + seuils + noise floor)."
        ),
    )
    parser.add_argument(
        "--allow-non-article-c",
        action="store_true",
        help=(
            "Bypass explicite du garde-fou de branche Git "
            "(autorise une branche différente de 'article_c')."
        ),
    )
    parser.add_argument(
        "--network-sizes",
        dest="network_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Tailles de réseau (nombre de nœuds entiers, ex: 50 100 150).",
    )
    parser.add_argument(
        "--densities",
        dest="network_sizes",
        type=int,
        nargs="+",
        default=argparse.SUPPRESS,
        help="Alias de --network-sizes (déprécié).",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=None,
        help="Nombre de réplications par configuration.",
    )
    parser.add_argument(
        "--seeds_base",
        type=int,
        default=None,
        help="Seed de base commune aux étapes 1 et 2.",
    )
    parser.add_argument(
        "--seed",
        dest="seeds_base",
        type=int,
        default=argparse.SUPPRESS,
        help="Alias de --seeds_base (déprécié).",
    )
    parser.add_argument(
        "--snir_modes",
        type=str,
        default=None,
        help="Liste des modes SNIR pour l'étape 1 (ex: snir_on,snir_off).",
    )
    parser.add_argument(
        "--snir-threshold-db",
        type=float,
        default=None,
        help="Seuil SNIR (dB).",
    )
    parser.add_argument(
        "--snir-threshold-min-db",
        type=float,
        default=None,
        help="Borne basse de clamp du seuil SNIR (dB).",
    )
    parser.add_argument(
        "--snir-threshold-max-db",
        type=float,
        default=None,
        help="Borne haute de clamp du seuil SNIR (dB).",
    )
    parser.add_argument(
        "--noise-floor-dbm",
        type=float,
        default=None,
        help="Bruit thermique (dBm).",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Ajoute un timestamp dans les sorties de l'étape 2.",
    )
    parser.add_argument(
        "--safe-profile",
        action="store_true",
        help="Active le profil sécurisé pour l'étape 2.",
    )
    parser.add_argument(
        "--auto-safe-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Active/désactive le profil sécurisé automatique pour l'étape 2 "
            "(activé par défaut)."
        ),
    )
    parser.add_argument(
        "--allow-low-success-rate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Autorise un success_rate global trop faible à l'étape 2 "
            "(log un avertissement au lieu d'arrêter)."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Active le mode strict pour l'étape 2 (arrêt si success_rate trop faible)."
        ),
    )
    parser.add_argument(
        "--traffic-mode",
        type=str,
        default=None,
        choices=("periodic", "poisson"),
        help="Modèle de trafic pour les étapes 1 et 2 (periodic ou poisson).",
    )
    parser.add_argument(
        "--jitter-range-s",
        dest="jitter_range_s",
        type=float,
        default=30.0,
        help="Amplitude du jitter pour l'étape 2 (secondes).",
    )
    parser.add_argument(
        "--jitter-range",
        dest="jitter_range_s",
        type=float,
        default=argparse.SUPPRESS,
        help="Alias de --jitter-range-s (déprécié).",
    )
    parser.add_argument(
        "--window-duration-s",
        type=float,
        default=None,
        help="Durée d'une fenêtre de simulation (secondes).",
    )
    parser.add_argument(
        "--reward-floor",
        type=float,
        default=None,
        help=(
            "Plancher de récompense appliqué dès que success_rate > 0 "
            "(étape 2)."
        ),
    )
    parser.add_argument(
        "--floor-on-zero-success",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Applique un plancher minimal si success_rate == 0 "
            "(utile pour éviter des rewards uniformes en conditions extrêmes)."
        ),
    )
    parser.add_argument(
        "--traffic-coeff-min",
        type=float,
        default=None,
        help="Coefficient de trafic minimal par nœud.",
    )
    parser.add_argument(
        "--traffic-coeff-max",
        type=float,
        default=None,
        help="Coefficient de trafic maximal par nœud.",
    )
    parser.add_argument(
        "--traffic-coeff-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Active/désactive la variabilité de trafic par nœud.",
    )
    parser.add_argument(
        "--traffic-load-scale-step2",
        dest="traffic_coeff_scale",
        type=float,
        default=None,
        help="Alias de --traffic-coeff-scale, appliqué uniquement à l'étape 2.",
    )
    parser.add_argument(
        "--capture-probability",
        type=float,
        default=None,
        help="Probabilité de capture lors d'une collision (0 à 1).",
    )
    parser.add_argument(
        "--congestion-coeff",
        type=float,
        default=None,
        help=(
            "Coefficient multiplicatif appliqué à la probabilité de congestion "
            "(1.0 pour garder la valeur calculée)."
        ),
    )
    parser.add_argument(
        "--congestion-coeff-base",
        type=float,
        default=None,
        help="Coefficient de base de la probabilité de congestion (0 à 1).",
    )
    parser.add_argument(
        "--congestion-coeff-growth",
        type=float,
        default=None,
        help="Coefficient de croissance de la probabilité de congestion.",
    )
    parser.add_argument(
        "--congestion-coeff-max",
        type=float,
        default=None,
        help="Plafond de probabilité de congestion (0 à 1).",
    )
    parser.add_argument(
        "--network-load-min",
        type=float,
        default=None,
        help="Borne minimale du facteur de charge réseau.",
    )
    parser.add_argument(
        "--network-load-max",
        type=float,
        default=None,
        help="Borne maximale du facteur de charge réseau.",
    )
    parser.add_argument(
        "--collision-size-min",
        type=float,
        default=None,
        help="Borne minimale du facteur de taille des collisions.",
    )
    parser.add_argument(
        "--collision-size-under-max",
        type=float,
        default=None,
        help="Borne max (sous-charge) du facteur de taille des collisions.",
    )
    parser.add_argument(
        "--collision-size-over-max",
        type=float,
        default=None,
        help="Borne max (surcharge) du facteur de taille des collisions.",
    )
    parser.add_argument(
        "--collision-size-factor",
        type=float,
        default=None,
        help=(
            "Facteur de taille appliqué aux collisions (override du calcul "
            "par taille de réseau si fourni)."
        ),
    )
    parser.add_argument(
        "--traffic-coeff-clamp-min",
        type=float,
        default=None,
        help="Borne minimale du clamp appliqué aux coefficients de trafic.",
    )
    parser.add_argument(
        "--traffic-coeff-clamp-max",
        type=float,
        default=None,
        help="Borne maximale du clamp appliqué aux coefficients de trafic.",
    )
    parser.add_argument(
        "--traffic-coeff-clamp-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Active/désactive le clamp des coefficients de trafic (diagnostic).",
    )
    parser.add_argument(
        "--window-delay-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Active/désactive le délai aléatoire entre fenêtres.",
    )
    parser.add_argument(
        "--window-delay-range-s",
        type=float,
        default=None,
        help="Amplitude du délai aléatoire entre fenêtres (secondes).",
    )
    parser.add_argument(
        "--step1-outdir",
        type=str,
        default=None,
        help="Répertoire de sortie de l'étape 1.",
    )
    parser.add_argument(
        "--strict-pipeline",
        action="store_true",
        help=(
            "Active les contrôles stricts pipeline: vérification stricte du layout, "
            "agrégats obligatoires, puis verify_all final avec exception propagée."
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help=(
            "Nettoie step1/results et step2/results avant exécution, "
            "puis recrée les dossiers requis."
        ),
    )
    parser.add_argument(
        "--clean-hard",
        action="store_true",
        default=False,
        help=(
            "Purge totale: --clean + suppression de step1/plots/output, "
            "step2/plots/output et article_c/plots/output."
        ),
    )
    parser.add_argument(
        "--skip-step1",
        action="store_true",
        help="Ignore l'exécution de l'étape 1.",
    )
    parser.add_argument(
        "--skip-step2",
        action="store_true",
        help="Ignore l'exécution de l'étape 2.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Affiche la progression de l'étape 1.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Durée de la simulation pour l'étape 1 (secondes).",
    )
    parser.add_argument(
        "--mixra-opt-max-iterations",
        type=int,
        default=None,
        help="Nombre maximal d'itérations pour MixRA-Opt.",
    )
    parser.add_argument(
        "--mixra-opt-candidate-subset-size",
        type=int,
        default=None,
        help="Nombre maximal de nœuds optimisés par itération en MixRA-Opt.",
    )
    parser.add_argument(
        "--mixra-opt-epsilon",
        type=float,
        default=None,
        help="Seuil d'amélioration pour la convergence MixRA-Opt.",
    )
    parser.add_argument(
        "--mixra-opt-max-evals",
        type=int,
        default=None,
        help="Nombre maximal d'évaluations pour MixRA-Opt.",
    )
    parser.add_argument(
        "--mixra-opt-budget",
        type=int,
        default=None,
        help="Budget cible d'évaluations pour MixRA-Opt (max d'évaluations).",
    )
    parser.add_argument(
        "--mixra-opt-budget-base",
        type=int,
        default=None,
        help="Offset additif appliqué au budget MixRA-Opt calculé.",
    )
    parser.add_argument(
        "--mixra-opt-budget-scale",
        type=float,
        default=None,
        help="Facteur multiplicatif appliqué au budget MixRA-Opt calculé.",
    )
    parser.add_argument(
        "--mixra-opt-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Active ou désactive MixRA-Opt.",
    )
    parser.add_argument(
        "--mixra-opt-mode",
        choices=("fast", "balanced", "full"),
        default=None,
        help="Mode MixRA-Opt (balanced par défaut).",
    )
    parser.add_argument(
        "--mixra-opt-no-fallback",
        "--mixra-opt-hard",
        dest="mixra_opt_no_fallback",
        action="store_true",
        default=False,
        help=(
            "Désactive explicitement le fallback MixRA-H pour MixRA-Opt, "
            "même en mode balanced/fast."
        ),
    )
    parser.add_argument(
        "--mixra-opt-timeout",
        type=float,
        default=None,
        help="Timeout (secondes) pour MixRA-Opt afin d'éviter les blocages.",
    )
    parser.add_argument(
        "--plot-summary",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Génère un plot de synthèse avec barres d'erreur à l'étape 1.",
    )
    parser.add_argument(
        "--flat-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Option historique désormais interdite. "
            "run_all impose exclusivement by_size/size_*/rep_*."
        ),
    )
    parser.add_argument(
        "--profile-timing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Affiche les durées des étapes internes pour l'étape 1.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Nombre de processus worker pour paralléliser l'étape 1.",
    )
    return parser


def _get_current_git_branch() -> str | None:
    """Retourne le nom de la branche Git courante ou ``None`` si indisponible."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    branch = result.stdout.strip()
    if not branch:
        return None
    return branch


def _enforce_article_c_branch(allow_non_article_c: bool) -> None:
    """Informe sur la branche courante sans bloquer l'exécution utilisateur."""
    if allow_non_article_c:
        return
    current_branch = _get_current_git_branch()
    if current_branch is None:
        log_debug(
            "AVERTISSEMENT: impossible de déterminer la branche Git courante "
            "(archive ZIP, Git absent ou dépôt non initialisé). "
            "Contrôle de branche ignoré."
        )
        return
    expected_branch = "article_c"
    if current_branch != expected_branch:
        log_debug(
            "AVERTISSEMENT: branche Git attendue 'article_c', "
            f"branche détectée: '{current_branch}'. "
            "Exécution poursuivie pour compatibilité locale/Windows."
        )


def _build_step1_args(args: argparse.Namespace) -> list[str]:
    step1_args: list[str] = []
    if args.network_sizes:
        step1_args.append("--network-sizes")
        step1_args.extend([str(size) for size in args.network_sizes])
    if args.replications is not None:
        step1_args.extend(["--replications", str(args.replications)])
    if args.seeds_base is not None:
        step1_args.extend(["--seeds_base", str(args.seeds_base)])
    if args.snir_modes:
        step1_args.extend(["--snir_modes", args.snir_modes])
    if args.snir_threshold_db is not None:
        step1_args.extend(["--snir-threshold-db", str(args.snir_threshold_db)])
    if args.snir_threshold_min_db is not None:
        step1_args.extend(
            ["--snir-threshold-min-db", str(args.snir_threshold_min_db)]
        )
    if args.snir_threshold_max_db is not None:
        step1_args.extend(
            ["--snir-threshold-max-db", str(args.snir_threshold_max_db)]
        )
    if args.noise_floor_dbm is not None:
        step1_args.extend(["--noise-floor-dbm", str(args.noise_floor_dbm)])
    if args.traffic_mode is not None:
        step1_args.extend(["--traffic-mode", args.traffic_mode])
    if args.jitter_range_s is not None:
        step1_args.extend(["--jitter-range-s", str(args.jitter_range_s)])
    if args.duration_s is not None:
        step1_args.extend(["--duration-s", str(args.duration_s)])
    if args.step1_outdir:
        step1_args.extend(["--outdir", args.step1_outdir])
    else:
        step1_args.extend(["--outdir", "article_c/step1/results"])
    if args.progress is not None:
        step1_args.append("--progress" if args.progress else "--no-progress")
    if args.mixra_opt_max_iterations is not None:
        step1_args.extend(
            ["--mixra-opt-max-iterations", str(args.mixra_opt_max_iterations)]
        )
    if args.mixra_opt_candidate_subset_size is not None:
        step1_args.extend(
            [
                "--mixra-opt-candidate-subset-size",
                str(args.mixra_opt_candidate_subset_size),
            ]
        )
    if args.mixra_opt_epsilon is not None:
        step1_args.extend(["--mixra-opt-epsilon", str(args.mixra_opt_epsilon)])
    if args.mixra_opt_max_evals is not None:
        step1_args.extend(["--mixra-opt-max-evals", str(args.mixra_opt_max_evals)])
    if args.mixra_opt_budget is not None:
        step1_args.extend(["--mixra-opt-budget", str(args.mixra_opt_budget)])
    if args.mixra_opt_budget_base is not None:
        step1_args.extend(["--mixra-opt-budget-base", str(args.mixra_opt_budget_base)])
    if args.mixra_opt_budget_scale is not None:
        step1_args.extend(["--mixra-opt-budget-scale", str(args.mixra_opt_budget_scale)])
    if args.mixra_opt_enabled is not None:
        step1_args.append(
            "--mixra-opt-enabled"
            if args.mixra_opt_enabled
            else "--no-mixra-opt-enabled"
        )
    if args.mixra_opt_mode is not None:
        step1_args.extend(["--mixra-opt-mode", args.mixra_opt_mode])
    if args.mixra_opt_no_fallback:
        step1_args.append("--mixra-opt-no-fallback")
    if args.mixra_opt_timeout is not None:
        step1_args.extend(["--mixra-opt-timeout", str(args.mixra_opt_timeout)])
    if args.plot_summary is not None:
        step1_args.append(
            "--plot-summary" if args.plot_summary else "--no-plot-summary"
        )
    if args.flat_output is not None:
        step1_args.append("--flat-output" if args.flat_output else "--no-flat-output")
    if args.profile_timing is not None:
        step1_args.append(
            "--profile-timing" if args.profile_timing else "--no-profile-timing"
        )
    if args.workers is not None:
        step1_args.extend(["--workers", str(args.workers)])
    if getattr(args, "reset_status", False):
        step1_args.append("--reset-status")
    return step1_args


def _build_step2_args(args: argparse.Namespace) -> list[str]:
    step2_args: list[str] = []
    if args.network_sizes:
        step2_args.append("--network-sizes")
        step2_args.extend([str(size) for size in args.network_sizes])
    if getattr(args, "reference_network_size", None) is not None:
        step2_args.extend(
            ["--reference-network-size", str(args.reference_network_size)]
        )
    if args.replications is not None:
        step2_args.extend(["--replications", str(args.replications)])
    if args.seeds_base is not None:
        step2_args.extend(["--seeds_base", str(args.seeds_base)])
    if args.timestamp:
        step2_args.append("--timestamp")
    if args.safe_profile:
        step2_args.append("--safe-profile")
    if args.auto_safe_profile is not None:
        step2_args.append(
            "--auto-safe-profile"
            if args.auto_safe_profile
            else "--no-auto-safe-profile"
        )
    if args.strict:
        step2_args.append("--strict")
    elif args.allow_low_success_rate is False:
        step2_args.append("--no-allow-low-success-rate")
    if args.snir_threshold_db is not None:
        step2_args.extend(["--snir-threshold-db", str(args.snir_threshold_db)])
    if args.snir_threshold_min_db is not None:
        step2_args.extend(
            ["--snir-threshold-min-db", str(args.snir_threshold_min_db)]
        )
    if args.snir_threshold_max_db is not None:
        step2_args.extend(
            ["--snir-threshold-max-db", str(args.snir_threshold_max_db)]
        )
    if args.noise_floor_dbm is not None:
        step2_args.extend(["--noise-floor-dbm", str(args.noise_floor_dbm)])
    if args.traffic_mode is not None:
        step2_args.extend(["--traffic-mode", args.traffic_mode])
    if args.jitter_range_s is not None:
        step2_args.extend(["--jitter-range-s", str(args.jitter_range_s)])
    if args.window_duration_s is not None:
        step2_args.extend(["--window-duration-s", str(args.window_duration_s)])
    if args.traffic_coeff_min is not None:
        step2_args.extend(["--traffic-coeff-min", str(args.traffic_coeff_min)])
    if args.traffic_coeff_max is not None:
        step2_args.extend(["--traffic-coeff-max", str(args.traffic_coeff_max)])
    if args.traffic_coeff_enabled is not None:
        step2_args.append(
            "--traffic-coeff-enabled"
            if args.traffic_coeff_enabled
            else "--no-traffic-coeff-enabled"
        )
    if args.traffic_coeff_scale is not None:
        step2_args.extend(["--traffic-coeff-scale", str(args.traffic_coeff_scale)])
    if args.capture_probability is not None:
        step2_args.extend(["--capture-probability", str(args.capture_probability)])
    if args.congestion_coeff is not None:
        step2_args.extend(["--congestion-coeff", str(args.congestion_coeff)])
    if args.congestion_coeff_base is not None:
        step2_args.extend(["--congestion-coeff-base", str(args.congestion_coeff_base)])
    if args.congestion_coeff_growth is not None:
        step2_args.extend(
            ["--congestion-coeff-growth", str(args.congestion_coeff_growth)]
        )
    if args.congestion_coeff_max is not None:
        step2_args.extend(["--congestion-coeff-max", str(args.congestion_coeff_max)])
    if args.network_load_min is not None:
        step2_args.extend(["--network-load-min", str(args.network_load_min)])
    if args.network_load_max is not None:
        step2_args.extend(["--network-load-max", str(args.network_load_max)])
    if args.collision_size_min is not None:
        step2_args.extend(["--collision-size-min", str(args.collision_size_min)])
    if args.collision_size_under_max is not None:
        step2_args.extend(
            ["--collision-size-under-max", str(args.collision_size_under_max)]
        )
    if args.collision_size_over_max is not None:
        step2_args.extend(
            ["--collision-size-over-max", str(args.collision_size_over_max)]
        )
    if args.collision_size_factor is not None:
        step2_args.extend(["--collision-size-factor", str(args.collision_size_factor)])
    if args.traffic_coeff_clamp_min is not None:
        step2_args.extend(
            ["--traffic-coeff-clamp-min", str(args.traffic_coeff_clamp_min)]
        )
    if args.traffic_coeff_clamp_max is not None:
        step2_args.extend(
            ["--traffic-coeff-clamp-max", str(args.traffic_coeff_clamp_max)]
        )
    if args.traffic_coeff_clamp_enabled is not None:
        step2_args.append(
            "--traffic-coeff-clamp-enabled"
            if args.traffic_coeff_clamp_enabled
            else "--no-traffic-coeff-clamp-enabled"
        )
    if args.window_delay_enabled is not None:
        step2_args.append(
            "--window-delay-enabled"
            if args.window_delay_enabled
            else "--no-window-delay-enabled"
        )
    if args.window_delay_range_s is not None:
        step2_args.extend(["--window-delay-range-s", str(args.window_delay_range_s)])
    if args.reward_floor is not None:
        step2_args.extend(["--reward-floor", str(args.reward_floor)])
    if args.floor_on_zero_success is not None:
        step2_args.append(
            "--floor-on-zero-success"
            if args.floor_on_zero_success
            else "--no-floor-on-zero-success"
        )
    if args.flat_output is not None:
        step2_args.append("--flat-output" if args.flat_output else "--no-flat-output")
    if getattr(args, "reset_status", False):
        step2_args.append("--reset-status")
    return step2_args


def _build_step2_explicit_config(args: argparse.Namespace) -> dict[str, object]:
    """Construit une configuration Step2 autonome (sans lecture Step1)."""
    return {
        "replications": args.replications,
        "seeds_base": args.seeds_base,
        "timestamp": args.timestamp,
        "safe_profile": args.safe_profile,
        "auto_safe_profile": args.auto_safe_profile,
        "strict": args.strict,
        "allow_low_success_rate": args.allow_low_success_rate,
        "snir_threshold_db": args.snir_threshold_db,
        "snir_threshold_min_db": args.snir_threshold_min_db,
        "snir_threshold_max_db": args.snir_threshold_max_db,
        "noise_floor_dbm": args.noise_floor_dbm,
        "traffic_mode": args.traffic_mode,
        "jitter_range_s": args.jitter_range_s,
        "window_duration_s": args.window_duration_s,
        "traffic_coeff_min": args.traffic_coeff_min,
        "traffic_coeff_max": args.traffic_coeff_max,
        "traffic_coeff_enabled": args.traffic_coeff_enabled,
        "traffic_coeff_scale": args.traffic_coeff_scale,
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
        "traffic_coeff_clamp_min": args.traffic_coeff_clamp_min,
        "traffic_coeff_clamp_max": args.traffic_coeff_clamp_max,
        "traffic_coeff_clamp_enabled": args.traffic_coeff_clamp_enabled,
        "window_delay_enabled": args.window_delay_enabled,
        "window_delay_range_s": args.window_delay_range_s,
        "reward_floor": args.reward_floor,
        "floor_on_zero_success": args.floor_on_zero_success,
        "flat_output": args.flat_output,
        "reset_status": getattr(args, "reset_status", False),
    }


def _validate_step2_explicit_config_startup(
    args: argparse.Namespace,
    step2_explicit_config: dict[str, object],
) -> None:
    """Valide la config explicite Step2 et détecte les attributs manquants."""
    explicit_keys = sorted(step2_explicit_config)
    log_debug("Step2: clés explicites construites = " + ", ".join(explicit_keys))

    allowed_internal_keys = {"reset_status"}
    missing_from_args = [
        key
        for key in explicit_keys
        if key not in allowed_internal_keys and not hasattr(args, key)
    ]
    if missing_from_args:
        raise RuntimeError(
            "Step2: incohérence parser/config explicite, clés absentes du namespace CLI: "
            f"{missing_from_args}"
        )

    probe_args = argparse.Namespace(**step2_explicit_config)
    probe_args.network_sizes = []
    try:
        _build_step2_args(probe_args)
    except AttributeError as exc:
        raise RuntimeError(
            "Step2: validation explicite échouée, un attribut attendu par "
            f"_build_step2_args est manquant: {exc}"
        ) from exc

    log_debug(
        "Step2: validation explicite OK, aucune clé ne provoque d'AttributeError."
    )


def _build_report_metadata(skip_step1: bool, skip_step2: bool) -> dict[str, object]:
    """Construit les métadonnées du rapport final (complet vs partiel)."""
    skipped_steps: list[str] = []
    if skip_step1:
        skipped_steps.append("step1")
    if skip_step2:
        skipped_steps.append("step2")

    executed_steps = [step for step in ("step1", "step2") if step not in skipped_steps]
    if skipped_steps:
        return {
            "kind": "partial",
            "executed_steps": executed_steps,
            "skipped_steps": skipped_steps,
            "note": (
                "Rapport partiel: certaines étapes ont été explicitement ignorées "
                f"({', '.join(skipped_steps)})."
            ),
        }

    return {
        "kind": "full",
        "executed_steps": executed_steps,
        "skipped_steps": [],
        "note": "Rapport complet: step1 et step2 ont été exécutées.",
    }


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.quiet:
        args.log_level = "quiet"
    if args.flat_output:
        parser.error("--flat-output est interdit: utilisez exclusivement by_size/size_*/rep_*.")
    set_log_level(args.log_level)
    if args.clean and args.clean_hard:
        raise ValueError("Options incompatibles: utilisez --clean ou --clean-hard, pas les deux.")
    if args.clean or args.clean_hard:
        _clean_run_artifacts(hard=args.clean_hard)
    if args.preset is not None:
        preset_values = RUN_ALL_PRESETS[args.preset]
        for key, value in preset_values.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
    _enforce_article_c_branch(args.allow_non_article_c)
    if args.auto_safe_profile:
        log_debug(
            "Auto-safe-profile activé par défaut: le profil sécurisé sera appliqué "
            "avant la simulation de l'étape 2."
        )
    else:
        log_debug(
            "Recommandation: activez --auto-safe-profile pour éviter un "
            "success_rate trop faible à l'étape 2."
        )
    if args.step1_outdir is not None:
        default_step1_dir = (
            Path(__file__).resolve().parent / "step1" / "results"
        ).resolve()
        requested_dir = Path(args.step1_outdir).resolve()
        if requested_dir != default_step1_dir:
            raise ValueError(
                "Étape 1: le répertoire de sortie doit être "
                f"{default_step1_dir}."
            )
    requested_sizes = (
        parse_network_size_list(args.network_sizes)
        if args.network_sizes
        else list(DEFAULT_CONFIG.scenario.network_sizes)
    )
    replications_total = (
        int(args.replications)
        if args.replications is not None
        else DEFAULT_REPLICATIONS
    )
    step1_results_dir = (Path(__file__).resolve().parent / "step1" / "results").resolve()
    step2_results_dir = (Path(__file__).resolve().parent / "step2" / "results").resolve()
    campaign_state_path = (Path(__file__).resolve().parent / "campaign_state.json").resolve()
    _assert_path_within_scope(step1_results_dir / "run_status_step1.csv", step1_results_dir, "Step1")
    _assert_path_within_scope(step2_results_dir / "run_status_step2.csv", step2_results_dir, "Step2")
    campaign_summary_path = (Path(__file__).resolve().parent / "campaign_summary.json").resolve()
    _remove_global_aggregation_artifacts(step1_results_dir, "Step1")
    _remove_global_aggregation_artifacts(step2_results_dir, "Step2")
    campaign_summary: dict[str, object] = {
        "network_sizes": requested_sizes,
        "replications_total": replications_total,
        "simulation_quality": "ok",
        "quality_thresholds": {
            "step2": {
                "success_rate_mean_min": STEP2_SUCCESS_RATE_MEAN_LOW_THRESHOLD,
                "failed_runs_max": 0,
            }
        },
        "quality_reasons": [],
        "total_elapsed_seconds": 0.0,
        "sizes": [],
        "output_paths": {
            "step1_results": str(step1_results_dir),
            "step2_results": str(step2_results_dir),
        },
        "report": _build_report_metadata(args.skip_step1, args.skip_step2),
    }
    campaign_start = perf_counter()
    reference_network_size = int(round(median(requested_sizes)))
    args.reference_network_size = reference_network_size
    step2_explicit_config = _build_step2_explicit_config(args)
    _validate_step2_explicit_config_startup(args, step2_explicit_config)
    step2_explicit_config["flat_output"] = False

    # Évite tout faux positif de complétion avant la fin effective de la campagne.
    if not args.skip_step1:
        _remove_done_flag(step1_results_dir, "Step1")
    if not args.skip_step2:
        _remove_done_flag(step2_results_dir, "Step2")

    start_index = 0
    current_step = "step1"
    resume_rep = 1
    previous_state = _read_campaign_state(campaign_state_path)
    if previous_state is not None:
        raw_size = previous_state.get("size")
        raw_step = previous_state.get("step")
        raw_rep = previous_state.get("rep")
        if isinstance(raw_size, int) and raw_size in requested_sizes:
            start_index = requested_sizes.index(raw_size)
        if isinstance(raw_step, str) and raw_step in {"step1", "step2"}:
            current_step = raw_step
        if isinstance(raw_rep, int) and raw_rep > 0:
            resume_rep = raw_rep

    found_pending = False
    for idx in range(start_index, len(requested_sizes)):
        size = int(requested_sizes[idx])
        steps_to_check = ["step1", "step2"]
        if idx == start_index and current_step == "step2":
            steps_to_check = ["step2"]
        for step_name in steps_to_check:
            if step_name == "step1" and args.skip_step1:
                continue
            if step_name == "step2" and args.skip_step2:
                continue
            step_dir = step1_results_dir if step_name == "step1" else step2_results_dir
            first_missing = _find_first_missing_rep(step_dir, size, replications_total)
            if first_missing is not None:
                start_index = idx
                current_step = step_name
                resume_rep = first_missing
                found_pending = True
                break
        if found_pending:
            break

    if not found_pending and previous_state is not None and not (args.skip_step1 and args.skip_step2):
        log_info("[RESUME] Aucun travail partiel détecté, campagne déjà complète.")
        start_index = len(requested_sizes)

    if found_pending or start_index > 0 or previous_state is not None:
        if start_index < len(requested_sizes):
            log_info(
                f"[RESUME] Reprise campagne depuis size={requested_sizes[start_index]} "
                f"rep={resume_rep} step={current_step}."
            )

    step1_status_reset_pending = True
    step2_status_reset_pending = True
    for size_index, size in enumerate(requested_sizes):
        if size_index < start_index:
            continue
        step1_size_args = argparse.Namespace(**vars(args))
        step1_size_args.network_sizes = [size]
        step2_size_args = argparse.Namespace(**step2_explicit_config)
        step2_size_args.network_sizes = [size]
        size_summary: dict[str, object] = {
            "network_size": size,
            "replications_total": replications_total,
            "failed": 0,
            "elapsed_seconds": 0.0,
            "step1": {
                "status": "skipped" if step1_size_args.skip_step1 else "pending",
                "failed": 0,
                "elapsed_seconds": 0.0,
                "output_path": str(step1_results_dir),
                "status_file": str(step1_results_dir / "run_status_step1.csv"),
            },
            "step2": {
                "status": "skipped" if step1_size_args.skip_step2 else "pending",
                "failed": 0,
                "elapsed_seconds": 0.0,
                "output_path": str(step2_results_dir),
                "status_file": str(step2_results_dir / "run_status_step2.csv"),
            },
        }
        size_start = perf_counter()
        if not step1_size_args.skip_step1 and (current_step == "step1"):
            step1_first_missing = _find_first_missing_rep(step1_results_dir, int(size), replications_total)
            _write_campaign_state(
                campaign_state_path,
                size=int(size),
                rep=step1_first_missing or 1,
                step="step1",
                status="in_progress",
            )
            log_info(f"[PHASE] step1-simulation size={size}")
            _cleanup_size_directory(step1_results_dir, int(size), "Step1")
            step1_size_args.flat_output = False
            step1_size_args.reset_status = step1_status_reset_pending
            step_start = perf_counter()
            run_step1(_build_step1_args(step1_size_args))
            step1_status_reset_pending = False
            _assert_no_global_writes_during_simulation(step1_results_dir, "Step1")
            _assert_cumulative_sizes_nested(
                step1_results_dir,
                {int(size)},
                "Step1",
            )
            _log_existing_key_csv_paths("Step1", step1_results_dir)
            step1_elapsed = perf_counter() - step_start
            step1_failed = _count_failed_runs(step1_results_dir / "run_status_step1.csv", size)
            size_summary["step1"] = {
                **size_summary["step1"],
                "status": "failed" if step1_failed > 0 else "ok",
                "failed": step1_failed,
                "elapsed_seconds": round(step1_elapsed, 3),
            }
            if not step1_size_args.skip_step2:
                _write_campaign_state(
                    campaign_state_path,
                    size=int(size),
                    rep=1,
                    step="step2",
                    status="in_progress",
                )

        if not step1_size_args.skip_step2:
            if current_step == "step2":
                step2_first_missing = _find_first_missing_rep(step2_results_dir, int(size), replications_total)
                _write_campaign_state(
                    campaign_state_path,
                    size=int(size),
                    rep=step2_first_missing or 1,
                    step="step2",
                    status="in_progress",
                )
            _cleanup_size_directory(step2_results_dir, int(size), "Step2")
            step2_size_args.flat_output = False
            step2_size_args.reset_status = step2_status_reset_pending
            step_start = perf_counter()
            run_step2(_build_step2_args(step2_size_args))
            step2_status_reset_pending = False
            _assert_no_global_writes_during_simulation(step2_results_dir, "Step2")
            _assert_cumulative_sizes_nested(
                step2_results_dir,
                {int(size)},
                "Step2",
            )
            _log_existing_key_csv_paths("Step2", step2_results_dir)
            step2_elapsed = perf_counter() - step_start
            step2_failed = _count_failed_runs(step2_results_dir / "run_status_step2.csv", size)
            step2_quality = _build_step2_quality_summary(
                step2_results_dir,
                size,
                step2_failed,
            )
            size_summary["step2"] = {
                **size_summary["step2"],
                "status": "failed" if step2_failed > 0 else "ok",
                "failed": step2_failed,
                "elapsed_seconds": round(step2_elapsed, 3),
                "quality": step2_quality,
            }
            if str(step2_quality.get("simulation_quality")) == "low":
                campaign_summary["simulation_quality"] = "low"
                reasons = campaign_summary["quality_reasons"]
                if isinstance(reasons, list):
                    for reason in step2_quality.get("reasons", []):
                        reasons.append(f"taille {size}: {reason}")
        current_step = "step1"

        next_index = size_index + 1
        if next_index < len(requested_sizes):
            _write_campaign_state(
                campaign_state_path,
                size=int(requested_sizes[next_index]),
                rep=1,
                step="step1",
                status="in_progress",
            )
        size_summary["elapsed_seconds"] = round(perf_counter() - size_start, 3)
        size_summary["failed"] = int(size_summary["step1"]["failed"]) + int(
            size_summary["step2"]["failed"]
        )
        cast_sizes = campaign_summary["sizes"]
        if isinstance(cast_sizes, list):
            cast_sizes.append(size_summary)
        log_info(f"Résumé: taille de réseau {size} terminée.")

        if size_index < len(requested_sizes) - 1:
            if not args.skip_step1:
                _remove_done_flag(step1_results_dir, "Step1")
            if not args.skip_step2:
                _remove_done_flag(step2_results_dir, "Step2")

    if not args.skip_step1:
        step1_missing = _missing_replications_by_size(
            step1_results_dir,
            requested_sizes,
            replications_total,
        )
        if step1_missing:
            step1_pairs = [
                (size, rep)
                for size in sorted(step1_missing)
                for rep in step1_missing[size]
            ]
            log_info(
                "[RECOVERY] Step1: réplications manquantes détectées, "
                f"relance ciblée de {step1_pairs}"
            )
            relaunched_step1_pairs = _relaunch_missing_replications(
                step_label="Step1",
                results_dir=step1_results_dir,
                missing_by_size=step1_missing,
                build_args=_build_step1_args,
                runner=run_step1,
                base_args=argparse.Namespace(**vars(args)),
            )
            log_info(
                "[RECOVERY] Step1: couples relancés exactement = "
                f"{relaunched_step1_pairs}"
            )

    if not args.skip_step2:
        step2_missing = _missing_replications_by_size(
            step2_results_dir,
            requested_sizes,
            replications_total,
        )
        if step2_missing:
            step2_pairs = [
                (size, rep)
                for size in sorted(step2_missing)
                for rep in step2_missing[size]
            ]
            log_info(
                "[RECOVERY] Step2: réplications manquantes détectées, "
                f"relance ciblée de {step2_pairs}"
            )
            relaunched_step2_pairs = _relaunch_missing_replications(
                step_label="Step2",
                results_dir=step2_results_dir,
                missing_by_size=step2_missing,
                build_args=_build_step2_args,
                runner=run_step2,
                base_args=argparse.Namespace(**step2_explicit_config),
            )
            log_info(
                "[RECOVERY] Step2: couples relancés exactement = "
                f"{relaunched_step2_pairs}"
            )

    campaign_summary["total_elapsed_seconds"] = round(perf_counter() - campaign_start, 3)
    if not args.skip_step1:
        # 3) Validation du layout by_size/rep immédiatement après la simulation.
        _assert_output_layout_compliant(
            step1_results_dir,
            requested_sizes,
            replications_total,
            "Step1",
        )
        # 4) Agrégation globale finale.
        log_info("[PHASE] step1-aggregation")
        step1_merge_stats = aggregate_results_by_size(
            step1_results_dir,
            write_global_aggregated=False,
        )
        log_debug(
            "Step1: consolidation by_size finale exécutée "
            f"({step1_merge_stats['global_row_count']} lignes)."
        )
    if not args.skip_step2:
        step2_merge_stats = aggregate_results_by_size(
            step2_results_dir,
            write_global_aggregated=False,
        )
        log_debug(
            "Step2: consolidation by_size finale exécutée "
            f"({step2_merge_stats['global_row_count']} lignes)."
        )
    if not args.skip_step1:
        # 5) Validation post-agrégation.
        _assert_aggregation_contract_consistent(
            step1_results_dir,
            requested_sizes,
            "Step1",
        )
    if not args.skip_step2:
        _assert_output_layout_compliant(
            step2_results_dir,
            requested_sizes,
            replications_total,
            "Step2",
        )
        _assert_aggregation_contract_consistent(
            step2_results_dir,
            requested_sizes,
            "Step2",
        )

    if args.strict_pipeline:
        log_info("[STRICT_PIPELINE] Contrôles stricts supplémentaires activés.")
        if not args.skip_step1:
            _assert_output_layout_compliant(
                step1_results_dir,
                requested_sizes,
                replications_total,
                "Step1",
            )
            _assert_required_aggregates_present(
                step1_results_dir,
                requested_sizes,
                "Step1",
            )
        if not args.skip_step2:
            _assert_output_layout_compliant(
                step2_results_dir,
                requested_sizes,
                replications_total,
                "Step2",
            )
            _assert_required_aggregates_present(
                step2_results_dir,
                requested_sizes,
                "Step2",
            )
    report_meta = campaign_summary.get("report", {})
    if isinstance(report_meta, dict) and report_meta.get("kind") == "partial":
        log_info(f"[REPORT] {report_meta.get('note', 'Rapport partiel demandé.')}")

    log_info("Validation des résultats (article C) en cours...")
    validation_args: list[str] = []
    if args.skip_step1:
        validation_args.append("--skip-step1")
    if args.skip_step2:
        validation_args.append("--skip-step2")
    validation_code = validate_results(validation_args)
    campaign_summary["validation"] = {
        "status": "ok" if validation_code == 0 else "failed",
        "exit_code": validation_code,
        "scope": {
            "skip_step1": bool(args.skip_step1),
            "skip_step2": bool(args.skip_step2),
        },
    }
    if args.strict_pipeline:
        _run_verify_all_strict(replications_total)
        campaign_summary["strict_pipeline_verify_all"] = "ok"
    campaign_summary_path.write_text(
        json.dumps(campaign_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_campaign_state(
        campaign_state_path,
        size=None,
        rep=None,
        step=None,
        status="completed",
    )
    log_info(f"Résumé de campagne écrit: {campaign_summary_path}")
    if validation_code != 0:
        raise SystemExit(validation_code)


if __name__ == "__main__":
    main()
