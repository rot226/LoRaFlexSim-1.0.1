"""Point d'entrée CLI: lancement de campagne multi-runs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import math
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from time import perf_counter

from sfrd.parse.aggregate import aggregate_logs
from sfrd.parse.parse_run import parse_run

_PRECHECK_NETWORK_SIZES: tuple[int, ...] = (80, 160)
_PRECHECK_SNIR_MODES: tuple[str, ...] = ("OFF", "ON")
_PRECHECK_SEED = 1
_TIME_DRIFT_ALERT_RATIO = 10_000.0
_TIME_DRIFT_ALERT_HEARTBEATS = 3
_TIME_DRIFT_CRITICAL_RATIO = 100_000.0
_TIME_DRIFT_CRITICAL_HEARTBEATS = 3
_AUTO_PRECHECK_MIN_TOTAL_RUNS = 24


class MetricsInconsistentError(RuntimeError):
    """Erreur levée quand les métriques de run sont incohérentes."""


class _CampaignContextFilter(logging.Filter):
    """Injecte des champs de contexte par défaut pour le formatage des logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        defaults = {
            "run_id": "-",
            "snir": "-",
            "algo": "-",
            "network_size": "-",
            "seed": "-",
            "statut": "-",
        }
        for key, value in defaults.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


def _run_key(
    *,
    snir_mode: str,
    network_size: int,
    algorithm: str,
    seed: int,
) -> str:
    """Construit une clé stable pour indexer l'état d'un run."""

    return json.dumps([snir_mode, int(network_size), str(algorithm), int(seed)], ensure_ascii=False)


def _find_existing_run_entry(
    runs_state: dict[str, dict[str, object]],
    *,
    snir_mode: str,
    network_size: int,
    algorithm: str,
    seed: int,
) -> tuple[str, dict[str, object]] | None:
    """Retrouve une entrée existante y compris en cas d'ancien format de clé."""

    expected = (snir_mode, int(network_size), str(algorithm), int(seed))
    for key, entry in runs_state.items():
        if (
            entry.get("snir"),
            entry.get("network_size"),
            entry.get("algo"),
            entry.get("seed"),
        ) == expected:
            return key, entry
    return None


def _is_run_completed(run_dir: Path) -> bool:
    """Vérifie si un run est déjà finalisé via ses artefacts essentiels."""

    required_files = [
        run_dir / "campaign_summary.json",
        run_dir / "raw_packets.csv",
        run_dir / "raw_energy.csv",
    ]
    return all(path.is_file() for path in required_files)


def _normalize_run_entry(run_id: str, payload: object) -> dict[str, object] | None:
    """Normalise une entrée de run du state file, ancien ou nouveau format."""

    if isinstance(payload, str):
        return {
            "status": payload,
            "snir": None,
            "network_size": None,
            "algo": None,
            "seed": None,
            "paths": {},
            "duration_s": None,
            "failure_reason": None,
        }

    if not isinstance(payload, dict):
        return None

    status = payload.get("status")
    if not isinstance(status, str):
        return None

    paths = payload.get("paths", {})
    if not isinstance(paths, dict):
        paths = {}

    duration_s = payload.get("duration_s")
    if not isinstance(duration_s, (float, int)):
        duration_s = None

    return {
        "status": status,
        "snir": payload.get("snir"),
        "network_size": payload.get("network_size"),
        "algo": payload.get("algo"),
        "seed": payload.get("seed"),
        "paths": dict(paths),
        "duration_s": float(duration_s) if duration_s is not None else None,
        "failure_reason": payload.get("failure_reason"),
        "suspect": bool(payload.get("suspect", False)),
        "time_drift": payload.get("time_drift"),
    }


def _load_campaign_state(state_path: Path) -> dict[str, dict[str, object]]:
    """Charge l'index de progression des runs."""

    if not state_path.exists():
        return {}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    runs = payload.get("runs")
    if not isinstance(runs, dict):
        return {}
    valid_status = {"pending", "running", "done", "failed", "failed_timeout", "incomplete"}
    state: dict[str, dict[str, object]] = {}
    for run_id, run_payload in runs.items():
        if not isinstance(run_id, str):
            continue
        entry = _normalize_run_entry(run_id, run_payload)
        if entry is None:
            continue
        if entry["status"] not in valid_status:
            continue
        state[run_id] = entry
    return state


def _write_campaign_state(state_path: Path, runs_state: dict[str, dict[str, object]]) -> None:
    """Persist l'index de progression des runs."""

    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "runs": dict(sorted(runs_state.items())),
    }
    serialized = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    with state_path.open("w", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def _new_run_entry(
    *,
    snir_mode: str,
    network_size: int,
    algorithm: str,
    seed: int,
    run_dir: Path,
) -> dict[str, object]:
    return {
        "snir": snir_mode,
        "network_size": network_size,
        "algo": algorithm,
        "seed": seed,
        "status": "pending",
        "paths": {
            "run_dir": str(run_dir),
            "summary": str(run_dir / "campaign_summary.json"),
            "raw_packets": str(run_dir / "raw_packets.csv"),
            "raw_energy": str(run_dir / "raw_energy.csv"),
        },
        "duration_s": None,
        "failure_reason": None,
        "suspect": False,
        "time_drift": None,
    }


def _configure_logging(logs_root: Path) -> tuple[logging.Logger, Path]:
    """Configure le logger campagne: console INFO + fichier DEBUG."""

    logger = logging.getLogger("sfrd.campaign")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    logs_dir = logs_root
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | run_id=%(run_id)s | snir=%(snir)s | algo=%(algo)s | "
            "network_size=%(network_size)s | seed=%(seed)s | statut=%(statut)s | "
            "%(levelname)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    context_filter = _CampaignContextFilter()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(context_filter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(context_filter)
    logger.addHandler(file_handler)

    return logger, log_path


def _new_campaign_id() -> str:
    """Génère un identifiant de campagne unique pour l'exécution courante."""

    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _load_run_single_campaign():
    """Charge le point d'entrée de simulation défini dans ``sfrd/cli.py``."""

    module_path = Path(__file__).resolve().parents[1] / "cli.py"
    spec = importlib.util.spec_from_file_location("_sfrd_core_cli", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossible de charger {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_campaign


def _parse_snir_modes(raw_values: str) -> list[str]:
    """Parse la liste SNIR CSV et retourne des valeurs normalisées OFF/ON."""

    mapping = {
        "off": "OFF",
        "snir_off": "OFF",
        "on": "ON",
        "snir_on": "ON",
    }
    modes: list[str] = []
    for value in raw_values.split(","):
        key = value.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise argparse.ArgumentTypeError(
                f"Mode SNIR invalide: {value!r}. Attendu: OFF ou ON"
            )
        normalized = mapping[key]
        if normalized not in modes:
            modes.append(normalized)

    if not modes:
        raise argparse.ArgumentTypeError("--snir ne contient aucune valeur valide")
    return modes


def _parse_algorithms(raw_values: list[str]) -> list[str]:
    """Parse les algorithmes en préservant strictement l'ordre fourni en CLI."""

    algorithms: list[str] = []
    for value in raw_values:
        # Autorise une saisie mixte espace/CSV sans modifier la séquence.
        for token in value.split(","):
            algo = token.strip()
            if algo:
                algorithms.append(algo)

    if not algorithms:
        raise argparse.ArgumentTypeError("--algos ne contient aucune valeur valide")
    return algorithms


def _filter_skipped_algorithms(algorithms: list[str], skipped: list[str]) -> list[str]:
    """Retire les algorithmes demandés dans ``skipped`` (comparaison tolérante)."""

    def _normalize(value: str) -> str:
        return "".join(ch for ch in value.lower() if ch.isalnum())

    skipped_keys = {_normalize(algo) for algo in skipped if algo.strip()}
    if not skipped_keys:
        return algorithms
    return [algo for algo in algorithms if _normalize(algo) not in skipped_keys]

def _format_seconds(value: float | None) -> str:
    """Formate une durée en secondes pour les logs de heartbeat."""

    if value is None:
        return "n/a"
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(seconds) or seconds < 0.0:
        return "n/a"
    return f"{seconds:.1f}s"


def _build_expected_runs(args: argparse.Namespace) -> set[tuple[str, int, str, int]]:
    expected: set[tuple[str, int, str, int]] = set()
    for snir_mode in args.snir:
        for network_size in args.network_sizes:
            for algorithm in args.algos:
                for replication_index in range(1, args.replications + 1):
                    seed = args.seeds_base + replication_index - 1
                    expected.add((snir_mode, int(network_size), str(algorithm), int(seed)))
    return expected


def _count_raw_packets_rows(raw_packets_path: Path, *, warmup_s: float) -> dict[str, int | bool]:
    """Retourne des statistiques de lignes lues/retenues pour un raw_packets.csv."""

    stats: dict[str, int | bool] = {
        "source_rows": 0,
        "retained_rows": 0,
        "sf_column_present": False,
        "sf_valid_rows": 0,
    }
    if not raw_packets_path.exists():
        return stats

    with raw_packets_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        stats["sf_column_present"] = "sf" in fieldnames

        for row in reader:
            stats["source_rows"] += 1

            timestamp: float | None = None
            raw_time = (row.get("time") or "").strip()
            if raw_time:
                try:
                    timestamp = float(raw_time)
                except ValueError:
                    timestamp = None
            if timestamp is not None and timestamp < warmup_s:
                continue

            stats["retained_rows"] += 1
            if stats["sf_column_present"]:
                raw_sf = (row.get("sf") or "").strip()
                if not raw_sf:
                    continue
                try:
                    float(raw_sf)
                except ValueError:
                    continue
                stats["sf_valid_rows"] += 1

    return stats


def _format_precheck_stats(precheck_stats: dict[str, int]) -> str:
    return (
        f"runs_exécutés={precheck_stats['runs_executed']} | "
        f"runs_réussis={precheck_stats['runs_success']} | "
        f"runs_échoués={precheck_stats['runs_failed']} | "
        f"raw_packets_lignes_lues={precheck_stats['raw_packets_source_rows']} | "
        f"raw_packets_lignes_après_warmup={precheck_stats['raw_packets_retained_rows']} | "
        f"csv_finaux_lignes_écrites={precheck_stats['final_csv_rows_written']}"
    )


def _describe_empty_sf_distribution(precheck_stats: dict[str, int]) -> str:
    if precheck_stats["raw_packets_files_found"] == 0:
        return "aucun raw_packets.csv trouvé"
    if precheck_stats["raw_packets_retained_rows"] == 0:
        return "toutes les lignes ont été filtrées"
    if precheck_stats["raw_packets_sf_column_missing_or_invalid"] > 0:
        return "colonne sf absente/invalide"
    return "cause indéterminée"


def _validate_precheck_csv(
    csv_path: Path,
    expected_columns: tuple[str, ...],
    *,
    precheck_stats: dict[str, int],
) -> int:
    csv_path = csv_path.resolve()
    stats_context = _format_precheck_stats(precheck_stats)
    if not csv_path.exists():
        raise ValueError(f"CSV attendu absent: {csv_path} | {stats_context}")
    if csv_path.stat().st_size == 0:
        raise ValueError(f"CSV attendu vide: {csv_path} | {stats_context}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        if columns != expected_columns:
            raise ValueError(
                (
                    f"Colonnes invalides pour {csv_path}: attendu={list(expected_columns)} "
                    f"obtenu={list(columns)} | {stats_context}"
                )
            )
        rows = list(reader)

    if not rows:
        if csv_path.name == "sf_distribution.csv":
            reason = _describe_empty_sf_distribution(precheck_stats)
            raise ValueError(f"CSV sans données: {csv_path} | raison={reason} | {stats_context}")
        raise ValueError(f"CSV sans données: {csv_path} | {stats_context}")

    for row_number, row in enumerate(rows, start=2):
        for field_name, raw_value in row.items():
            text = (raw_value or "").strip()
            if not text:
                continue
            try:
                parsed = float(text)
            except ValueError:
                continue
            if math.isnan(parsed):
                raise ValueError(
                    f"NaN détecté dans {csv_path} ligne {row_number} colonne {field_name} | {stats_context}"
                )

        if "pdr" in row:
            pdr = float(row["pdr"])
            if not (0.0 <= pdr <= 1.0):
                raise ValueError(
                    f"pdr hors bornes [0,1] dans {csv_path} ligne {row_number}: {pdr} | {stats_context}"
                )
        if "throughput_packets_per_s" in row:
            throughput = float(row["throughput_packets_per_s"])
            if throughput < 0.0:
                raise ValueError(
                    (
                        f"throughput_packets_per_s négatif dans {csv_path} ligne {row_number}: {throughput} "
                        f"| {stats_context}"
                    )
                )
        if "energy_joule_per_packet" in row:
            energy = float(row["energy_joule_per_packet"])
            if energy < 0.0:
                raise ValueError(
                    (
                        f"energy_joule_per_packet négatif dans {csv_path} ligne {row_number}: {energy} "
                        f"| {stats_context}"
                    )
                )

    return len(rows)


def _run_precheck(*, args: argparse.Namespace, logger: logging.Logger, run_single_campaign) -> None:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    precheck_root = Path("sfrd") / ".precheck" / timestamp
    suffix = 1
    while precheck_root.exists():
        precheck_root = Path("sfrd") / ".precheck" / f"{timestamp}-{suffix:02d}"
        suffix += 1
    precheck_logs_root = precheck_root / "logs"
    precheck_logs_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        (
            "Précheck démarré: mini matrice "
            f"tailles={list(_PRECHECK_NETWORK_SIZES)}, snir={list(_PRECHECK_SNIR_MODES)}, "
            f"seed={_PRECHECK_SEED}, algos={args.algos}, workspace={precheck_root.resolve()}"
        ),
        extra={"statut": "precheck_start"},
    )

    try:
        precheck_stats: dict[str, int] = {
            "runs_executed": 0,
            "runs_success": 0,
            "runs_failed": 0,
            "raw_packets_files_found": 0,
            "raw_packets_source_rows": 0,
            "raw_packets_retained_rows": 0,
            "raw_packets_sf_column_missing_or_invalid": 0,
            "final_csv_rows_written": 0,
        }
        for snir_mode in _PRECHECK_SNIR_MODES:
            snir_folder = f"SNIR_{snir_mode}"
            snir_cli_value = "snir_on" if snir_mode == "ON" else "snir_off"
            for network_size in _PRECHECK_NETWORK_SIZES:
                for algorithm in args.algos:
                    precheck_stats["runs_executed"] += 1
                    run_dir = (
                        precheck_logs_root
                        / snir_folder
                        / f"ns_{network_size}"
                        / f"algo_{algorithm}"
                        / f"seed_{_PRECHECK_SEED}"
                    )
                    run_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        run_single_campaign(
                            network_size=network_size,
                            algorithm=str(algorithm),
                            snir_mode=snir_cli_value,
                            seed=_PRECHECK_SEED,
                            warmup_s=float(args.warmup_s),
                            output_dir=run_dir,
                            ucb_config_path=args.ucb_config,
                        )
                        parse_run(run_dir / "raw_packets.csv", warmup_s=0.0)

                        raw_packets_stats = _count_raw_packets_rows(
                            run_dir / "raw_packets.csv",
                            warmup_s=float(args.warmup_s),
                        )
                        if (run_dir / "raw_packets.csv").exists():
                            precheck_stats["raw_packets_files_found"] += 1
                        precheck_stats["raw_packets_source_rows"] += int(raw_packets_stats["source_rows"])
                        precheck_stats["raw_packets_retained_rows"] += int(raw_packets_stats["retained_rows"])
                        if (
                            not bool(raw_packets_stats["sf_column_present"])
                            or int(raw_packets_stats["sf_valid_rows"]) == 0
                        ):
                            precheck_stats["raw_packets_sf_column_missing_or_invalid"] += 1

                        precheck_stats["runs_success"] += 1
                    except Exception:
                        precheck_stats["runs_failed"] += 1
                        raise RuntimeError(
                            f"Précheck NO-GO pendant exécution run | {_format_precheck_stats(precheck_stats)}"
                        )

        aggregate_root = Path(aggregate_logs(precheck_logs_root, allow_partial=False)).resolve()
        expected_csvs: list[tuple[Path, tuple[str, ...]]] = [
            (aggregate_root / "SNIR_OFF" / "pdr_results.csv", ("network_size", "algorithm", "snir", "pdr")),
            (
                aggregate_root / "SNIR_OFF" / "throughput_results.csv",
                ("network_size", "algorithm", "snir", "throughput_packets_per_s"),
            ),
            (
                aggregate_root / "SNIR_OFF" / "energy_results.csv",
                ("network_size", "algorithm", "snir", "energy_joule_per_packet"),
            ),
            (aggregate_root / "SNIR_OFF" / "sf_distribution.csv", ("network_size", "algorithm", "snir", "sf", "count")),
            (aggregate_root / "SNIR_ON" / "pdr_results.csv", ("network_size", "algorithm", "snir", "pdr")),
            (
                aggregate_root / "SNIR_ON" / "throughput_results.csv",
                ("network_size", "algorithm", "snir", "throughput_packets_per_s"),
            ),
            (
                aggregate_root / "SNIR_ON" / "energy_results.csv",
                ("network_size", "algorithm", "snir", "energy_joule_per_packet"),
            ),
            (aggregate_root / "SNIR_ON" / "sf_distribution.csv", ("network_size", "algorithm", "snir", "sf", "count")),
        ]
        if any(str(algo).upper() == "UCB" for algo in args.algos):
            expected_csvs.append((aggregate_root / "learning_curve_ucb.csv", ("episode", "reward")))

        for csv_path, expected_columns in expected_csvs:
            resolved_csv_path = csv_path.resolve()
            logger.info(
                f"Précheck CSV attendu: {resolved_csv_path}",
                extra={"statut": "precheck_csv_path"},
            )
            csv_rows = _validate_precheck_csv(
                resolved_csv_path,
                expected_columns,
                precheck_stats=precheck_stats,
            )
            precheck_stats["final_csv_rows_written"] += csv_rows

        logger.info(
            (
                f"Précheck GO ✅ | {_format_precheck_stats(precheck_stats)} | csv_validés={len(expected_csvs)} | "
                f"logs_root={precheck_logs_root.resolve()} | aggregate_root={aggregate_root}"
            ),
            extra={"statut": "precheck_go"},
        )
    except Exception:
        logger.error(
            (
                "Précheck NO-GO ❌ | artefacts conservés pour debug: "
                f"{precheck_root.resolve()}"
            ),
            extra={"statut": "precheck_artifacts_kept"},
        )
        raise

    if args.keep_precheck_artifacts:
        logger.info(
            f"Précheck terminé: conservation explicite des artefacts ({precheck_root.resolve()}).",
            extra={"statut": "precheck_artifacts_kept"},
        )
    else:
        shutil.rmtree(precheck_root)
        logger.info(
            "Précheck terminé: artefacts nettoyés automatiquement (succès).",
            extra={"statut": "precheck_artifacts_cleaned"},
        )


def _collect_completed_runs(
    logs_root: Path,
    *,
    runs_state: dict[str, dict[str, object]] | None = None,
    logger: logging.Logger | None = None,
    progress_every: int = 500,
) -> set[tuple[str, int, str, int]]:
    """Collecte les runs complétés en priorisant ``campaign_state.json``.

    Le scan disque des ``campaign_summary.json`` n'est utilisé qu'en fallback.
    """

    completed: set[tuple[str, int, str, int]] = set()

    state_path = logs_root / "campaign_state.json"
    use_state_as_source = False
    if isinstance(runs_state, dict):
        state_payload = runs_state
        use_state_as_source = True
    else:
        state_payload = _load_campaign_state(state_path)
        if state_path.exists():
            try:
                raw_payload = json.loads(state_path.read_text(encoding="utf-8"))
                use_state_as_source = isinstance(raw_payload, dict) and isinstance(raw_payload.get("runs"), dict)
            except (json.JSONDecodeError, OSError, ValueError, TypeError):
                use_state_as_source = False

    if use_state_as_source:
        for run_payload in state_payload.values():
            if not isinstance(run_payload, dict):
                continue
            try:
                status = str(run_payload.get("status", "")).strip().lower()
                snir = str(run_payload.get("snir", "")).strip().upper()
                network_size = int(run_payload.get("network_size", 0))
                algorithm = str(run_payload.get("algo", "")).strip()
                seed = int(run_payload.get("seed", -1))
            except (TypeError, ValueError):
                continue

            if status != "done":
                continue
            if snir in {"ON", "OFF"} and network_size > 0 and algorithm and seed >= 0:
                completed.add((snir, network_size, algorithm, seed))

        if logger is not None:
            logger.info(
                (
                    "Collecte des runs complétés via campaign_state.json: "
                    f"{len(completed)} entrées done identifiées."
                ),
                extra={"statut": "completed_from_state"},
            )
        return completed

    if logger is not None:
        logger.warning(
            "campaign_state.json absent/invalide/vide: fallback sur scan disque des campaign_summary.json.",
            extra={"statut": "completed_scan_fallback"},
        )

    scanned = 0
    corrupted = 0
    try:
        for summary_path in logs_root.glob("SNIR_*/ns_*/algo_*/seed_*/campaign_summary.json"):
            scanned += 1
            if progress_every > 0 and scanned % progress_every == 0 and logger is not None:
                logger.info(
                    (
                        "Scan fallback campaign_summary.json en cours: "
                        f"fichiers_scannés={scanned}, runs_validés={len(completed)}, corrompus={corrupted}"
                    ),
                    extra={"statut": "completed_scan_progress"},
                )
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                contract = payload.get("contract", {})
                snir_mode = str(contract.get("snir_mode", "")).strip().upper()
                network_size = int(contract.get("network_size", 0))
                algorithm = str(contract.get("algorithm", "")).strip()
                seed = int(contract.get("seed", -1))
            except (json.JSONDecodeError, OSError, ValueError, TypeError):
                corrupted += 1
                continue

            if snir_mode in {"ON", "OFF"} and network_size > 0 and algorithm and seed >= 0:
                completed.add((snir_mode, network_size, algorithm, seed))
    except KeyboardInterrupt:
        if logger is not None:
            logger.warning(
                (
                    "Interruption pendant le scan fallback: sauvegarde de l'état partiel. "
                    f"fichiers_scannés={scanned}, runs_validés={len(completed)}"
                ),
                extra={"statut": "completed_scan_interrupted"},
            )

    if logger is not None:
        logger.info(
            (
                "Scan fallback terminé: "
                f"fichiers_scannés={scanned}, runs_validés={len(completed)}, corrompus={corrupted}"
            ),
            extra={"statut": "completed_scan_done"},
        )

    if completed:
        fallback_state = _load_campaign_state(state_path)
        for snir_mode, network_size, algorithm, seed in completed:
            run_key = _run_key(
                snir_mode=snir_mode,
                network_size=network_size,
                algorithm=algorithm,
                seed=seed,
            )
            entry = fallback_state.get(run_key)
            if not isinstance(entry, dict):
                run_dir = (
                    logs_root
                    / f"SNIR_{snir_mode}"
                    / f"ns_{network_size}"
                    / f"algo_{algorithm}"
                    / f"seed_{seed}"
                )
                entry = _new_run_entry(
                    snir_mode=snir_mode,
                    network_size=network_size,
                    algorithm=algorithm,
                    seed=seed,
                    run_dir=run_dir,
                )
                fallback_state[run_key] = entry
            entry["status"] = "done"

        _write_campaign_state(state_path, fallback_state)

    return completed


def _write_missing_combinations_report(
    logs_root: Path,
    expected_runs: set[tuple[str, int, str, int]],
    completed_runs: set[tuple[str, int, str, int]],
    runs_state: dict[str, dict[str, object]] | None = None,
) -> Path:
    status_by_run: dict[tuple[str, int, str, int], str] = {}
    if isinstance(runs_state, dict):
        for run_payload in runs_state.values():
            if not isinstance(run_payload, dict):
                continue
            try:
                snir = str(run_payload.get("snir", "")).strip().upper()
                network_size = int(run_payload.get("network_size", 0))
                algorithm = str(run_payload.get("algo", "")).strip()
                seed = int(run_payload.get("seed", -1))
            except (TypeError, ValueError):
                continue
            if snir in {"ON", "OFF"} and network_size > 0 and algorithm and seed >= 0:
                status = str(run_payload.get("status", "")).strip().lower() or "missing"
                status_by_run[(snir, network_size, algorithm, seed)] = status

    report_path = logs_root / "campaign_missing_combinations.csv"
    missing = sorted(expected_runs - completed_runs)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        csv_writer = csv.DictWriter(
            handle,
            fieldnames=["snir", "network_size", "algorithm", "seed", "status"],
        )
        csv_writer.writeheader()
        for snir_mode, network_size, algorithm, seed in missing:
            csv_writer.writerow(
                {
                    "snir": snir_mode,
                    "network_size": network_size,
                    "algorithm": algorithm,
                    "seed": seed,
                    "status": status_by_run.get(
                        (snir_mode, network_size, algorithm, seed),
                        "missing",
                    ),
                }
            )
    return report_path


def _write_timeout_campaign_summary(
    *,
    run_dir: Path,
    network_size: int,
    algorithm: str,
    snir_cli_value: str,
    seed: int,
    warmup_s: float,
    duration_s: float,
    max_run_seconds: float,
    cause: str,
) -> Path:
    """Écrit un ``campaign_summary.json`` explicite pour un run interrompu par timeout."""

    summary_payload = {
        "contract": {
            "network_size": int(network_size),
            "algorithm": str(algorithm),
            "snir_mode": str(snir_cli_value),
            "seed": int(seed),
            "warmup_s": float(warmup_s),
            "output_dir": str(run_dir),
        },
        "metrics": {},
        "status": "failed_timeout",
        "failure_reason": str(cause),
        "max_run_seconds": float(max_run_seconds),
        "duration_s": float(duration_s),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    summary_path = run_dir / "campaign_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return summary_path


def _resolve_precheck_mode(args: argparse.Namespace) -> str:
    """Résout le mode effectif de précheck, y compris la logique automatique."""

    if args.precheck != "auto":
        return str(args.precheck)

    total_runs = len(args.network_sizes) * len(args.algos) * len(args.snir) * args.replications
    is_large_full_campaign = (
        len(args.network_sizes) >= 2
        and len(args.algos) >= 2
        and len(args.snir) >= 2
        and args.replications >= 2
        and total_runs >= _AUTO_PRECHECK_MIN_TOTAL_RUNS
    )
    return "on" if is_large_full_campaign else "off"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lance une matrice de runs SFRD (SNIR x taille x algo x réplication)."
    )
    parser.add_argument(
        "--network-sizes",
        nargs="+",
        type=int,
        required=True,
        help="Tailles réseau, ex: --network-sizes 80 160 320 640 1280",
    )
    parser.add_argument(
        "--replications",
        type=int,
        required=True,
        help="Nombre de réplications par combinaison (>=1)",
    )
    parser.add_argument(
        "--seeds-base",
        type=int,
        required=True,
        help="Base de seed déterministe (seed = seeds_base + replication_index - 1)",
    )
    parser.add_argument(
        "--snir",
        type=_parse_snir_modes,
        default=_parse_snir_modes("OFF,ON"),
        help="Modes SNIR CSV, ex: --snir OFF,ON",
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        required=True,
        help=(
            "Algorithmes (ordre strict conservé), ex: "
            "--algos ADR MixRA-H MixRA-Opt UCB"
        ),
    )
    parser.add_argument(
        "--skip-algos",
        nargs="+",
        default=[],
        help=(
            "Algorithmes à exclure de --algos (utile pour debug rapide), ex: "
            "--skip-algos MixRA-H"
        ),
    )
    parser.add_argument(
        "--warmup-s",
        type=float,
        required=True,
        help="Durée warmup en secondes",
    )
    parser.add_argument(
        "--ucb-config",
        type=Path,
        default=Path("sfrd/config/ucb_config.json"),
        help="Fichier JSON de configuration UCB (lambda_E, exploration, épisode).",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("sfrd/logs"),
        help="Dossier racine contenant un sous-dossier dédié par campaign_id",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Autorise l'agrégation partielle d'une campagne incomplète. "
            "Par défaut, l'agrégation exige la matrice complète attendue."
        ),
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Désactive l'agrégation automatique en fin de campagne.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Reprend une campagne en sautant les runs déjà terminés (par défaut).",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Relance tous les runs même si les artefacts existent.",
    )
    parser.add_argument(
        "--precheck",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Mode de précheck GO/NO-GO avant la campagne complète. "
            "auto=on pour campagne large complète, off pour runs ciblés/debug."
        ),
    )
    parser.add_argument(
        "--keep-precheck-artifacts",
        action="store_true",
        help="Conserve les artefacts du précheck même en cas de succès.",
    )
    parser.add_argument(
        "--skip-precheck",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-run-seconds",
        type=float,
        default=900.0,
        help=(
            "Durée murale maximale d'un run (secondes). "
            "Au-delà, le run est arrêté et marqué failed_timeout."
        ),
    )
    args = parser.parse_args()

    if args.skip_precheck:
        args.precheck = "off"

    if args.replications <= 0:
        parser.error("--replications doit être >= 1")
    if any(size <= 0 for size in args.network_sizes):
        parser.error("Toutes les valeurs de --network-sizes doivent être > 0")
    if args.max_run_seconds is not None and args.max_run_seconds <= 0:
        parser.error("--max-run-seconds doit être > 0")

    try:
        args.algos = _parse_algorithms(args.algos)
        args.skip_algos = _parse_algorithms(args.skip_algos) if args.skip_algos else []
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    args.algos = _filter_skipped_algorithms(args.algos, args.skip_algos)
    if not args.algos:
        parser.error("Tous les algorithmes ont été retirés via --skip-algos")

    return args


def main() -> None:
    """Exécution principale."""

    args = _parse_args()
    run_single_campaign = _load_run_single_campaign()

    base_logs_root: Path = args.logs_root
    base_logs_root.mkdir(parents=True, exist_ok=True)
    campaign_id = _new_campaign_id()
    logs_root = base_logs_root / campaign_id
    logs_root.mkdir(parents=True, exist_ok=True)

    expected_runs = _build_expected_runs(args)
    expected_runs_rows = [
        {
            "snir": snir,
            "network_size": int(network_size),
            "algorithm": algorithm,
            "seed": int(seed),
        }
        for snir, network_size, algorithm, seed in sorted(expected_runs)
    ]
    campaign_manifest = {
        "campaign_id": campaign_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_logs_root": str(base_logs_root.resolve()),
        "logs_root": str(logs_root.resolve()),
        "allow_partial": bool(args.allow_partial),
        "args": _jsonable(vars(args)),
        "expected_runs": expected_runs_rows,
    }
    manifest_path = logs_root / "campaign_manifest.json"
    manifest_path.write_text(
        json.dumps(campaign_manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    manifests_root = base_logs_root / "campaign_manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)
    (manifests_root / f"{campaign_id}.json").write_text(
        json.dumps(campaign_manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    logger, campaign_log_path = _configure_logging(logs_root)
    state_path = logs_root / "campaign_state.json"
    runs_state = _load_campaign_state(state_path)

    logger.info(
        "Démarrage de campagne.",
        extra={"statut": "start"},
    )
    logger.info(
        f"Journal campagne: {campaign_log_path.resolve()}",
        extra={"statut": "log_path"},
    )
    logger.info(
        f"campaign_id={campaign_id} | logs_root={logs_root.resolve()} | manifest={manifest_path.resolve()}",
        extra={"statut": "campaign_context"},
    )
    logger.info(
        f"Séquence algorithmes retenue (ordre d'exécution): {args.algos}",
        extra={"statut": "algo_sequence"},
    )

    precheck_mode = _resolve_precheck_mode(args)
    if args.precheck == "auto":
        logger.info(
            (
                "Précheck auto résolu. "
                f"mode_effectif={precheck_mode} | tailles={len(args.network_sizes)} | "
                f"algos={len(args.algos)} | snir={len(args.snir)} | replications={args.replications}"
            ),
            extra={"statut": "precheck_auto_resolved"},
        )

    if precheck_mode == "off":
        logger.warning(
            "Précheck désactivé (mode=off): validation stricte de fin de campagne conservée.",
            extra={"statut": "precheck_skipped"},
        )
    else:
        try:
            _run_precheck(args=args, logger=logger, run_single_campaign=run_single_campaign)
        except Exception as exc:
            logger.error(
                f"Précheck NO-GO ❌ | cause={exc}",
                extra={"statut": "precheck_nogo"},
            )
            raise RuntimeError("Campagne bloquée: précheck NO-GO.") from exc

    run_index: list[dict[str, object]] = []
    total_runs = len(args.network_sizes) * len(args.algos) * len(args.snir) * args.replications
    completed_runs = 0
    failed_runs = 0
    skipped_runs = 0
    current_run = 0

    epsilon = 1e-9

    stop_campaign = False
    interrupted_by_user = False
    expected_runs = _build_expected_runs(args)

    for snir_mode in args.snir:
        snir_folder = f"SNIR_{snir_mode}"
        snir_cli_value = "snir_on" if snir_mode == "ON" else "snir_off"

        for network_size in args.network_sizes:
            for algorithm in args.algos:
                for replication_index in range(1, args.replications + 1):
                    current_run += 1
                    seed = args.seeds_base + replication_index - 1
                    run_dir = (
                        logs_root
                        / snir_folder
                        / f"ns_{network_size}"
                        / f"algo_{algorithm}"
                        / f"seed_{seed}"
                    )
                    run_dir.mkdir(parents=True, exist_ok=True)

                    run_input = {
                        "snir_mode": snir_mode,
                        "network_size": int(network_size),
                        "algorithm": algorithm,
                        "replication_index": int(replication_index),
                        "seed": int(seed),
                        "warmup_s": float(args.warmup_s),
                        "seed_formula": "seed = seeds_base + replication_index - 1",
                    }
                    (run_dir / "run_input.json").write_text(
                        json.dumps(run_input, indent=2, ensure_ascii=False, sort_keys=True),
                        encoding="utf-8",
                    )

                    run_context = {
                        "run_id": f"{current_run}/{total_runs}",
                        "snir": snir_mode,
                        "algo": algorithm,
                        "network_size": network_size,
                        "seed": seed,
                    }
                    run_key = _run_key(
                        snir_mode=snir_mode,
                        network_size=int(network_size),
                        algorithm=str(algorithm),
                        seed=int(seed),
                    )

                    run_entry = runs_state.get(run_key)
                    migrated_entry = False
                    if run_entry is None:
                        existing = _find_existing_run_entry(
                            runs_state,
                            snir_mode=snir_mode,
                            network_size=int(network_size),
                            algorithm=str(algorithm),
                            seed=int(seed),
                        )
                        if existing is not None:
                            previous_key, run_entry = existing
                            if previous_key != run_key:
                                runs_state.pop(previous_key, None)
                                runs_state[run_key] = run_entry
                                migrated_entry = True

                    if run_entry is None:
                        run_entry = _new_run_entry(
                            snir_mode=snir_mode,
                            network_size=int(network_size),
                            algorithm=str(algorithm),
                            seed=int(seed),
                            run_dir=run_dir,
                        )
                        runs_state[run_key] = run_entry
                        _write_campaign_state(state_path, runs_state)
                    else:
                        run_entry.update(
                            {
                                "snir": snir_mode,
                                "network_size": int(network_size),
                                "algo": str(algorithm),
                                "seed": int(seed),
                            }
                        )
                        run_paths = run_entry.get("paths")
                        if not isinstance(run_paths, dict):
                            run_paths = {}
                        run_paths.update(
                            {
                                "run_dir": str(run_dir),
                                "summary": str(run_dir / "campaign_summary.json"),
                                "raw_packets": str(run_dir / "raw_packets.csv"),
                                "raw_energy": str(run_dir / "raw_energy.csv"),
                            }
                        )
                        run_entry["paths"] = run_paths
                        if migrated_entry:
                            _write_campaign_state(state_path, runs_state)

                    if args.force_rerun:
                        run_entry["status"] = "pending"
                        run_entry["duration_s"] = None
                        run_entry["failure_reason"] = None
                        _write_campaign_state(state_path, runs_state)
                        logger.info(
                            "Run marqué pour relance forcée (--force-rerun).",
                            extra={**run_context, "statut": "rerun"},
                        )
                    elif args.resume:
                        run_status = str(run_entry.get("status", "pending")).strip().lower()
                        has_valid_artifacts = _is_run_completed(run_dir)
                        if run_status == "done" and has_valid_artifacts:
                            skipped_runs += 1
                            run_index.append(
                                {
                                    **run_input,
                                    "run_dir": str(run_dir),
                                    "summary_path": str(run_dir / "campaign_summary.json"),
                                }
                            )
                            logger.info(
                                "Run déjà terminé: SKIPPED.",
                                extra={**run_context, "statut": "skipped"},
                            )
                            continue
                        if run_status == "done" and not has_valid_artifacts:
                            logger.warning(
                                "Run marqué done mais artefacts invalides: relance.",
                                extra={**run_context, "statut": "resume_invalid_artifacts"},
                            )
                        if run_status in {"failed", "incomplete", "pending", "running"} or (
                            run_status == "done" and not has_valid_artifacts
                        ):
                            run_entry["status"] = "pending"
                            run_entry["duration_s"] = None
                            run_entry["failure_reason"] = None
                            _write_campaign_state(state_path, runs_state)

                    logger.info(
                        "Run démarré.",
                        extra={**run_context, "statut": "start"},
                    )
                    logger.debug(
                        (
                            "Chemins fichiers bruts attendus: "
                            f"raw_packets.csv={run_dir / 'raw_packets.csv'} ; "
                            f"raw_energy.csv={run_dir / 'raw_energy.csv'}"
                        ),
                        extra={**run_context, "statut": "raw_paths"},
                    )

                    t0 = perf_counter()
                    run_entry["status"] = "running"
                    run_entry["duration_s"] = None
                    run_entry["failure_reason"] = None
                    run_entry["suspect"] = False
                    run_entry["time_drift"] = None
                    _write_campaign_state(state_path, runs_state)
                    try:
                        heartbeat_interval_s = 45.0
                        drift_state = {
                            "max_ratio": 0.0,
                            "consecutive_alert": 0,
                            "consecutive_critical": 0,
                            "suspect": False,
                        }

                        def _heartbeat(status: dict[str, object]) -> None:
                            elapsed_s = perf_counter() - t0
                            safe_elapsed_s = max(elapsed_s, 1e-9)
                            raw_sim_time = status.get("sim_time_s")
                            try:
                                sim_time_s = float(raw_sim_time)
                            except (TypeError, ValueError):
                                sim_time_s = 0.0
                            if not math.isfinite(sim_time_s) or sim_time_s < 0.0:
                                sim_time_s = 0.0
                            drift_ratio = sim_time_s / safe_elapsed_s
                            drift_state["max_ratio"] = max(float(drift_state["max_ratio"]), drift_ratio)
                            if drift_ratio > _TIME_DRIFT_ALERT_RATIO:
                                drift_state["consecutive_alert"] = int(drift_state["consecutive_alert"]) + 1
                            else:
                                drift_state["consecutive_alert"] = 0

                            if drift_ratio > _TIME_DRIFT_CRITICAL_RATIO:
                                drift_state["consecutive_critical"] = int(drift_state["consecutive_critical"]) + 1
                            else:
                                drift_state["consecutive_critical"] = 0

                            if (
                                int(drift_state["consecutive_alert"]) >= _TIME_DRIFT_ALERT_HEARTBEATS
                                and not bool(drift_state["suspect"])
                            ):
                                drift_state["suspect"] = True
                                run_entry["suspect"] = True
                                run_entry["time_drift"] = {
                                    "max_ratio": float(drift_state["max_ratio"]),
                                    "alert_ratio": _TIME_DRIFT_ALERT_RATIO,
                                    "alert_heartbeats": _TIME_DRIFT_ALERT_HEARTBEATS,
                                }
                                _write_campaign_state(state_path, runs_state)
                                logger.warning(
                                    (
                                        "anomalous_time_drift détecté: run marqué suspect | "
                                        f"ratio={drift_ratio:.2f} | "
                                        f"consecutive_alert={drift_state['consecutive_alert']}"
                                    ),
                                    extra={**run_context, "statut": "anomalous_time_drift"},
                                )

                            if int(drift_state["consecutive_critical"]) >= _TIME_DRIFT_CRITICAL_HEARTBEATS:
                                logger.error(
                                    (
                                        "Dérive temporelle critique persistante: arrêt du run | "
                                        f"ratio={drift_ratio:.2f} | "
                                        f"consecutive_critical={drift_state['consecutive_critical']}"
                                    ),
                                    extra={**run_context, "statut": "anomalous_time_drift"},
                                )
                                raise RuntimeError(
                                    "anomalous_time_drift: seuil critique dépassé "
                                    f"({_TIME_DRIFT_CRITICAL_RATIO:.0f}) pendant "
                                    f"{_TIME_DRIFT_CRITICAL_HEARTBEATS} heartbeats"
                                )

                            sim_time = _format_seconds(status.get("sim_time_s"))
                            events_processed = int(status.get("events_processed") or 0)
                            last_qos_refresh = _format_seconds(status.get("last_qos_refresh_sim_time"))
                            logger.info(
                                (
                                    f"still running | run_id={run_context['run_id']} | "
                                    f"elapsed={elapsed_s:.1f}s | sim_time={sim_time} | "
                                    f"time_drift_ratio={drift_ratio:.2f} | "
                                    f"events_processed={events_processed} | "
                                    f"last_qos_refresh={last_qos_refresh}"
                                ),
                                extra={**run_context, "statut": "heartbeat"},
                            )

                        result = run_single_campaign(
                            network_size=int(network_size),
                            algorithm=str(algorithm),
                            snir_mode=snir_cli_value,
                            seed=int(seed),
                            warmup_s=float(args.warmup_s),
                            output_dir=run_dir,
                            ucb_config_path=args.ucb_config,
                            heartbeat_callback=_heartbeat,
                            heartbeat_interval_s=heartbeat_interval_s,
                            max_run_seconds=args.max_run_seconds,
                        )
                        duration_s = perf_counter() - t0
                        metrics = result.get("summary", {}).get("metrics", {})
                        tx = int(metrics.get("tx_attempted", 0))
                        success = int(metrics.get("rx_delivered", metrics.get("delivered", 0)))
                        pdr = (success / tx) if tx > 0 else 0.0
                        summary_pdr = float(metrics.get("pdr", 0.0))

                        raw_packets_path = run_dir / "raw_packets.csv"
                        raw_energy_path = run_dir / "raw_energy.csv"
                        parsed_metrics = parse_run(raw_packets_path, warmup_s=0.0)
                        raw_tx = int(parsed_metrics.get("tx_count") or 0)
                        raw_success = int(parsed_metrics.get("success_count") or 0)

                        if raw_tx > 0 and raw_success == 0:
                            logger.warning(
                                (
                                    "Aucune trame marquée succès dans le log brut "
                                    f"(tx_count={raw_tx}, raw={raw_packets_path.resolve()})."
                                ),
                                extra={**run_context, "statut": "raw_warning"},
                            )

                        raw_pdr = (raw_success / raw_tx) if raw_tx > 0 else 0.0
                        if raw_tx > 0 and not math.isclose(summary_pdr, raw_pdr, rel_tol=0.0, abs_tol=epsilon):
                            run_status = {
                                "status": "metrics_inconsistent",
                                "duration_s": duration_s,
                                "summary_path": str(result["summary_path"]),
                                "raw_packets_path": str(raw_packets_path),
                                "raw_energy_path": str(raw_energy_path),
                                "summary_pdr": summary_pdr,
                                "raw_tx": raw_tx,
                                "raw_success": raw_success,
                                "raw_pdr": raw_pdr,
                                "epsilon": epsilon,
                            }
                            (run_dir / "run_status.json").write_text(
                                json.dumps(run_status, indent=2, ensure_ascii=False, sort_keys=True),
                                encoding="utf-8",
                            )
                            raise MetricsInconsistentError(
                                "Métriques incohérentes entre résumé et log brut: "
                                f"pdr_summary={summary_pdr:.12f}, pdr_raw={raw_pdr:.12f}, "
                                f"raw={raw_packets_path}"
                            )

                        if tx > 0 and abs(summary_pdr - (success / tx)) >= epsilon:
                            run_status = {
                                "status": "metrics_inconsistent",
                                "duration_s": duration_s,
                                "summary_path": str(result["summary_path"]),
                                "tx": tx,
                                "success": success,
                                "pdr": summary_pdr,
                                "expected_pdr": success / tx,
                                "epsilon": epsilon,
                            }
                            (run_dir / "run_status.json").write_text(
                                json.dumps(run_status, indent=2, ensure_ascii=False, sort_keys=True),
                                encoding="utf-8",
                            )
                            raise MetricsInconsistentError(
                                "Métriques incohérentes: "
                                f"pdr={summary_pdr:.12f}, success/tx={(success / tx):.12f}, "
                                f"epsilon={epsilon}"
                            )

                        run_status = {
                            "status": "completed",
                            "duration_s": duration_s,
                            "summary_path": str(result["summary_path"]),
                            "suspect": bool(drift_state["suspect"]),
                            "time_drift": {
                                "max_ratio": float(drift_state["max_ratio"]),
                                "alert_ratio": _TIME_DRIFT_ALERT_RATIO,
                                "critical_ratio": _TIME_DRIFT_CRITICAL_RATIO,
                            },
                        }
                        (run_dir / "run_status.json").write_text(
                            json.dumps(run_status, indent=2, ensure_ascii=False, sort_keys=True),
                            encoding="utf-8",
                        )

                        run_index.append(
                            {
                                **run_input,
                                "run_dir": str(run_dir),
                                "summary_path": str(result["summary_path"]),
                            }
                        )
                        completed_runs += 1
                        run_entry["status"] = "done"
                        run_entry["duration_s"] = duration_s
                        run_entry["failure_reason"] = None
                        run_entry["suspect"] = bool(drift_state["suspect"])
                        run_entry["time_drift"] = {
                            "max_ratio": float(drift_state["max_ratio"]),
                            "alert_ratio": _TIME_DRIFT_ALERT_RATIO,
                            "critical_ratio": _TIME_DRIFT_CRITICAL_RATIO,
                        }
                        _write_campaign_state(state_path, runs_state)
                        logger.info(
                            (
                                f"Run terminé: duration={duration_s:.1f}s | tx={tx} | "
                                f"success={success} | pdr={pdr:.4f}"
                            ),
                            extra={**run_context, "statut": "completed"},
                        )
                        logger.debug(
                            (
                                f"Résumé run: {Path(result['summary_path']).resolve()} | "
                                f"raw_packets.csv={run_dir / 'raw_packets.csv'} | "
                                f"raw_energy.csv={run_dir / 'raw_energy.csv'}"
                            ),
                            extra={**run_context, "statut": "artifacts"},
                        )
                    except MetricsInconsistentError as exc:
                        failed_runs += 1
                        run_entry["status"] = "failed"
                        run_entry["duration_s"] = None
                        run_entry["failure_reason"] = str(exc)
                        _write_campaign_state(state_path, runs_state)
                        logger.error(
                            f"Run échoué: {exc}",
                            extra={**run_context, "statut": "metrics_inconsistent"},
                        )
                        stop_campaign = True
                        break
                    except KeyboardInterrupt:
                        duration_s = perf_counter() - t0
                        run_entry["status"] = "incomplete"
                        run_entry["duration_s"] = duration_s
                        _write_campaign_state(state_path, runs_state)
                        logger.warning(
                            "Interruption clavier: run marqué incomplete.",
                            extra={**run_context, "statut": "incomplete"},
                        )
                        interrupted_by_user = True
                        break
                    except TimeoutError as exc:
                        duration_s = perf_counter() - t0
                        summary_path = _write_timeout_campaign_summary(
                            run_dir=run_dir,
                            network_size=int(network_size),
                            algorithm=str(algorithm),
                            snir_cli_value=snir_cli_value,
                            seed=int(seed),
                            warmup_s=float(args.warmup_s),
                            duration_s=duration_s,
                            max_run_seconds=float(args.max_run_seconds),
                            cause=str(exc),
                        )
                        run_status = {
                            "status": "failed_timeout",
                            "duration_s": duration_s,
                            "error": str(exc),
                            "max_run_seconds": args.max_run_seconds,
                            "summary_path": str(summary_path),
                        }
                        (run_dir / "run_status.json").write_text(
                            json.dumps(run_status, indent=2, ensure_ascii=False, sort_keys=True),
                            encoding="utf-8",
                        )
                        failed_runs += 1
                        run_entry["status"] = "failed_timeout"
                        run_entry["duration_s"] = duration_s
                        run_entry["failure_reason"] = str(exc)
                        _write_campaign_state(state_path, runs_state)
                        logger.error(
                            (
                                "Run timeout: "
                                f"duration={duration_s:.1f}s | max={args.max_run_seconds}s | "
                                f"cause={exc}"
                            ),
                            extra={**run_context, "statut": "failed_timeout"},
                        )
                    except Exception as exc:  # pragma: no cover - robustesse CLI
                        duration_s = perf_counter() - t0
                        run_status = {
                            "status": "failed",
                            "duration_s": duration_s,
                            "error": str(exc),
                        }
                        (run_dir / "run_status.json").write_text(
                            json.dumps(run_status, indent=2, ensure_ascii=False, sort_keys=True),
                            encoding="utf-8",
                        )
                        failed_runs += 1
                        run_entry["status"] = "failed"
                        run_entry["duration_s"] = duration_s
                        run_entry["failure_reason"] = str(exc)
                        _write_campaign_state(state_path, runs_state)
                        logger.error(
                            (
                                "Run échoué: "
                                f"duration={duration_s:.1f}s | tx=NA | success=NA | pdr=NA"
                            ),
                            extra={**run_context, "statut": "failed"},
                        )
                        logger.exception(
                            f"Erreur run: {exc}",
                            extra={**run_context, "statut": "failed"},
                        )
                    finally:
                        _write_campaign_state(state_path, runs_state)
                if stop_campaign:
                    break
                if interrupted_by_user:
                    break
            if stop_campaign:
                break
            if interrupted_by_user:
                break
        if stop_campaign:
            break
        if interrupted_by_user:
            break

    if stop_campaign:
        logger.error(
            "Campagne interrompue prématurément suite à une incohérence de diagnostic.",
            extra={"statut": "aborted"},
        )
        raise MetricsInconsistentError(
            "Campagne interrompue: incohérence détectée entre métriques agrégées et logs bruts."
        )

    (logs_root / "campaign_runs.json").write_text(
        json.dumps(run_index, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    if interrupted_by_user:
        completed_set = _collect_completed_runs(
            logs_root,
            runs_state=runs_state,
            logger=logger,
        )
        missing_report = _write_missing_combinations_report(
            logs_root,
            expected_runs,
            completed_set,
            runs_state,
        )
        logger.warning(
            (
                "Ctrl+C détecté: agrégation de fin de campagne. "
                f"Combinaisons manquantes listées dans {missing_report.resolve()}"
            ),
            extra={"statut": "partial_suggested"},
        )
        if not args.skip_aggregate:
            if not args.allow_partial:
                logger.warning(
                    "Agrégation ignorée: campagne incomplète et --allow-partial absent.",
                    extra={"statut": "partial_blocked"},
                )
            else:
                aggregate_path = aggregate_logs(
                    logs_root,
                    allow_partial=True,
                    manifest_path=manifest_path,
                )
                logger.info(
                    f"Agrégation partielle terminée: {Path(aggregate_path).resolve()}",
                    extra={"statut": "partial_aggregated"},
                )
        return

    if not args.skip_aggregate:
        aggregate_path = aggregate_logs(
            logs_root,
            allow_partial=args.allow_partial,
            manifest_path=manifest_path,
        )
        aggregate_root = Path(aggregate_path)
        final_csv_paths = [
            aggregate_root / "SNIR_OFF" / "pdr_results.csv",
            aggregate_root / "SNIR_OFF" / "throughput_results.csv",
            aggregate_root / "SNIR_OFF" / "energy_results.csv",
            aggregate_root / "SNIR_OFF" / "sf_distribution.csv",
            aggregate_root / "SNIR_ON" / "pdr_results.csv",
            aggregate_root / "SNIR_ON" / "throughput_results.csv",
            aggregate_root / "SNIR_ON" / "energy_results.csv",
            aggregate_root / "SNIR_ON" / "sf_distribution.csv",
            aggregate_root / "learning_curve_ucb.csv",
        ]
        logger.info(
            f"Agrégation terminée: {aggregate_root.resolve()}",
            extra={"statut": "aggregated"},
        )
        for csv_path in final_csv_paths:
            logger.info(
                f"CSV final généré: {csv_path.resolve()}",
                extra={"statut": "final_csv"},
            )
    else:
        logger.info(
            "Agrégation automatique ignorée (--skip-aggregate).",
            extra={"statut": "skipped"},
        )

    logger.info(
        (
            "Bilan campagne: "
            f"completed={completed_runs}, skipped={skipped_runs}, "
            f"failed={failed_runs}, total={total_runs}"
        ),
        extra={"statut": "done"},
    )


if __name__ == "__main__":
    main()
