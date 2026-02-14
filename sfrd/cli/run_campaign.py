"""Point d'entrée CLI: lancement de campagne multi-runs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from datetime import datetime
from pathlib import Path
from time import perf_counter

from sfrd.parse.aggregate import aggregate_logs


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


def _configure_logging() -> tuple[logging.Logger, Path]:
    """Configure le logger campagne: console INFO + fichier DEBUG."""

    logger = logging.getLogger("sfrd.campaign")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    logs_dir = Path("sfrd/logs")
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
        help="Algorithmes, ex: --algos UCB ADR MixRA-H MixRA-Opt",
    )
    parser.add_argument(
        "--warmup-s",
        type=float,
        required=True,
        help="Durée warmup en secondes",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("sfrd/logs"),
        help="Dossier racine des logs de campagne",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Désactive l'agrégation automatique en fin de campagne.",
    )
    args = parser.parse_args()

    if args.replications <= 0:
        parser.error("--replications doit être >= 1")
    if any(size <= 0 for size in args.network_sizes):
        parser.error("Toutes les valeurs de --network-sizes doivent être > 0")
    return args


def main() -> None:
    """Exécution principale."""

    args = _parse_args()
    logger, campaign_log_path = _configure_logging()
    run_single_campaign = _load_run_single_campaign()
    logs_root: Path = args.logs_root
    logs_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Démarrage de campagne.",
        extra={"statut": "start"},
    )
    logger.info(
        f"Journal campagne: {campaign_log_path.resolve()}",
        extra={"statut": "log_path"},
    )

    run_index: list[dict[str, object]] = []
    total_runs = len(args.network_sizes) * len(args.algos) * len(args.snir) * args.replications
    completed_runs = 0
    failed_runs = 0
    current_run = 0

    epsilon = 1e-9

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
                    try:
                        result = run_single_campaign(
                            network_size=int(network_size),
                            algorithm=str(algorithm),
                            snir_mode=snir_cli_value,
                            seed=int(seed),
                            warmup_s=float(args.warmup_s),
                            output_dir=run_dir,
                        )
                        duration_s = perf_counter() - t0
                        metrics = result.get("summary", {}).get("metrics", {})
                        tx = int(metrics.get("tx_attempted", 0))
                        success = int(metrics.get("rx_delivered", 0))
                        pdr = (success / tx) if tx > 0 else 0.0
                        summary_pdr = float(metrics.get("pdr", 0.0))

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
                        logger.error(
                            f"Run échoué: {exc}",
                            extra={**run_context, "statut": "metrics_inconsistent"},
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

    (logs_root / "campaign_runs.json").write_text(
        json.dumps(run_index, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    if not args.skip_aggregate:
        aggregate_path = aggregate_logs(logs_root)
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
        f"Bilan campagne: completed={completed_runs}, failed={failed_runs}, total={total_runs}",
        extra={"statut": "done"},
    )


if __name__ == "__main__":
    main()
