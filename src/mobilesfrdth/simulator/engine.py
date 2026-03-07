"""Moteur de simulation event-driven pour uplinks périodiques."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import json
import logging
from pathlib import Path
import random
from time import monotonic
from typing import Any, Callable

from .io import write_run_outputs


@dataclass
class Node:
    node_id: int
    period_s: float
    next_uplink_s: float = 0.0
    payload_size: int = 12
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(order=True)
class Event:
    time_s: float
    kind: str
    node_id: int


@dataclass
class SimulationResult:
    uplink_count: int = 0
    events: list[Event] = field(default_factory=list)


class EventDrivenEngine:
    """Boucle event-driven basée sur une file de priorité.

    Chaque nœud planifie un événement ``uplink`` périodique.
    """

    def __init__(self, *, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def _schedule_initial_events(self, nodes: list[Node]) -> list[Event]:
        queue: list[Event] = []
        for node in nodes:
            jitter = self.rng.uniform(0.0, min(node.period_s, 1.0))
            node.next_uplink_s = max(0.0, jitter)
            heapq.heappush(queue, Event(time_s=node.next_uplink_s, kind="uplink", node_id=node.node_id))
        return queue

    def run(
        self,
        *,
        nodes: list[Node],
        until_s: float,
        progress_callback: Callable[[float], None] | None = None,
    ) -> SimulationResult:
        if until_s <= 0:
            return SimulationResult()

        node_by_id = {n.node_id: n for n in nodes}
        queue = self._schedule_initial_events(nodes)
        result = SimulationResult()
        while queue:
            event = heapq.heappop(queue)
            if event.time_s > until_s:
                break
            result.events.append(event)

            if progress_callback is not None:
                progress_callback(min(max(event.time_s / until_s, 0.0), 1.0))

            if event.kind == "uplink":
                result.uplink_count += 1
                node = node_by_id[event.node_id]
                next_time = event.time_s + max(node.period_s, 1e-6)
                node.next_uplink_s = next_time
                heapq.heappush(queue, Event(time_s=next_time, kind="uplink", node_id=node.node_id))

        if progress_callback is not None:
            progress_callback(1.0)

        return result


@dataclass
class RunExecutionReport:
    run_id: str
    success: bool
    run_dir: Path
    error: str | None = None


@dataclass
class BatchExecutionReport:
    reports: list[RunExecutionReport]
    total_jobs: int = 0
    skipped_runs: int = 0
    scheduled_runs: int = 0
    interrupted: bool = False

    @property
    def failed_reports(self) -> list[RunExecutionReport]:
        return [report for report in self.reports if not report.success]


class GridRunOrchestrator:
    """Orchestre l'exécution d'une grille de runs et la persistance des artefacts."""

    def __init__(self, *, output_root: Path) -> None:
        self.output_root = output_root

    def _build_nodes(self, params: dict[str, Any]) -> list[Node]:
        node_count = int(params["N"])
        period_s = float(params.get("period_s", 60.0))
        payload_size = int(params.get("payload_size", 12))
        return [
            Node(node_id=node_id, period_s=period_s, payload_size=payload_size)
            for node_id in range(1, node_count + 1)
        ]

    def _build_run_config(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "N": int(params["N"]),
            "speed": float(params.get("speed", 0.0)),
            "mobility_model": str(params.get("model", "RWP")).lower(),
            "mode": str(params.get("mode", "SNIR_OFF")).lower(),
            "algo": str(params.get("algo", "ADR")).lower(),
            "gateways": int(params.get("gateways", 1)),
            "sigma": float(params.get("sigma", 0.0)),
            "seed": int(params.get("seed", 0)),
            "rep": int(params.get("rep", 1)),
            **params,
        }

    def _logger_for_run(
        self,
        run_id: str,
    ) -> tuple[logging.Logger, list[logging.Handler], Path]:
        run_dir = self.output_root / "results" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(f"mobilesfrdth.run.{run_id}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        log_path = run_dir / "run.log"
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger, [file_handler, console_handler], run_dir

    def _is_run_completed(self, run_id: str) -> bool:
        run_dir = self.output_root / "results" / run_id
        required = [
            run_dir / "run_config.json",
            run_dir / "events.csv",
            run_dir / "node_timeseries.csv",
            run_dir / "summary.csv",
        ]
        return all(path.is_file() for path in required)

    def _write_campaign_progress(
        self,
        *,
        progress_path: Path,
        total_runs: int,
        completed_runs: int,
        skipped_runs: int,
        error_reports: list[RunExecutionReport],
        status: str,
    ) -> None:
        payload = {
            "status": status,
            "total_runs": total_runs,
            "runs_completed": completed_runs,
            "runs_skipped": skipped_runs,
            "runs_remaining": max(total_runs - completed_runs - skipped_runs, 0),
            "errors_count": len(error_reports),
            "errors": [
                {
                    "run_id": report.run_id,
                    "run_dir": str(report.run_dir),
                    "error": report.error,
                }
                for report in error_reports
            ],
        }
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def execute_jobs(
        self,
        jobs: list[dict[str, Any]],
        *,
        resume: bool = False,
        max_runs: int | None = None,
        max_walltime_s: float | None = None,
        progress_path: Path | None = None,
        progress_interval_s: float = 30.0,
        on_run_complete: Callable[[RunExecutionReport, int, int, int, int, float | None], None] | None = None,
    ) -> BatchExecutionReport:
        if max_runs is not None and max_runs < 1:
            raise ValueError("max_runs doit être >= 1")
        if max_walltime_s is not None and max_walltime_s <= 0:
            raise ValueError("max_walltime_s doit être > 0")
        if progress_interval_s <= 0:
            raise ValueError("progress_interval_s doit être > 0")

        reports: list[RunExecutionReport] = []
        pending_jobs: list[dict[str, Any]] = []
        skipped_runs = 0

        for job in jobs:
            params = dict(job.get("params", {}))
            run_id = str(params.get("run_id", job.get("job_id", "run")))
            if resume and self._is_run_completed(run_id):
                skipped_runs += 1
                continue
            pending_jobs.append(job)

        if max_runs is not None:
            pending_jobs = pending_jobs[:max_runs]

        total_runs = len(jobs)
        walltime_start_s = monotonic()
        progress_target = progress_path or (self.output_root / "campaign_progress.json")
        scheduled_runs = len(pending_jobs)
        interrupted = False
        self._write_campaign_progress(
            progress_path=progress_target,
            total_runs=total_runs,
            completed_runs=0,
            skipped_runs=skipped_runs,
            error_reports=[],
            status="running",
        )

        for job in pending_jobs:
            params = dict(job.get("params", {}))
            run_id = str(params.get("run_id", job.get("job_id", "run")))
            logger, handlers, run_dir = self._logger_for_run(run_id)
            try:
                elapsed_walltime_s = monotonic() - walltime_start_s
                if max_walltime_s is not None and elapsed_walltime_s >= max_walltime_s:
                    logger.warning(
                        "Arrêt campagne: plafond walltime atteint (%.1fs/%.1fs).",
                        elapsed_walltime_s,
                        max_walltime_s,
                    )
                    break

                seed = int(params.get("seed", 0))
                duration_s = float(params.get("duration_s", 3600.0))
                logger.info("Démarrage run_id=%s seed=%s", run_id, seed)
                logger.info("Paramètres: %s", params)

                engine = EventDrivenEngine(seed=seed)
                nodes = self._build_nodes(params)
                next_progress_log_at = monotonic() + progress_interval_s

                def _progress(progress: float) -> None:
                    nonlocal next_progress_log_at
                    now = monotonic()
                    if now >= next_progress_log_at:
                        next_progress_log_at = now + progress_interval_s
                        logger.info(
                            "Progression périodique run_id=%s: %s%%",
                            run_id,
                            int(progress * 100),
                        )

                result = engine.run(
                    nodes=nodes,
                    until_s=duration_s,
                    progress_callback=_progress,
                )
                run_config = self._build_run_config(params)
                write_run_outputs(
                    output_root=self.output_root,
                    run_id=run_id,
                    run_config=run_config,
                    events=result.events,
                    duration_s=duration_s,
                    time_bin_s=float(params.get("time_bin_s", 10.0)),
                )
                logger.info("Run terminé: uplinks=%s", result.uplink_count)
                reports.append(RunExecutionReport(run_id=run_id, success=True, run_dir=run_dir))
            except KeyboardInterrupt:
                logger.warning("Interruption utilisateur détectée (Ctrl+C). Arrêt propre après ce run.")
                reports.append(
                    RunExecutionReport(
                        run_id=run_id,
                        success=False,
                        run_dir=run_dir,
                        error="Interruption utilisateur (Ctrl+C)",
                    )
                )
                interrupted = True
            except Exception as exc:
                logger.exception("Run en erreur: %s", exc)
                reports.append(RunExecutionReport(run_id=run_id, success=False, run_dir=run_dir, error=str(exc)))
            finally:
                for handler in handlers:
                    logger.removeHandler(handler)
                    handler.close()

            run_index = len(reports)
            if on_run_complete is not None:
                elapsed_s = monotonic() - walltime_start_s
                remaining_runs = max(scheduled_runs - run_index, 0)
                eta_s = None
                if run_index > 0 and remaining_runs > 0:
                    eta_s = (elapsed_s / run_index) * remaining_runs
                last_report = reports[-1]
                on_run_complete(
                    last_report,
                    run_index,
                    scheduled_runs,
                    len([report for report in reports if report.success]),
                    len([report for report in reports if not report.success]),
                    eta_s,
                )

            self._write_campaign_progress(
                progress_path=progress_target,
                total_runs=total_runs,
                completed_runs=len([report for report in reports if report.success]),
                skipped_runs=skipped_runs,
                error_reports=[report for report in reports if not report.success],
                status="interrupted" if interrupted else "running",
            )

            if interrupted:
                break

        self._write_campaign_progress(
            progress_path=progress_target,
            total_runs=total_runs,
            completed_runs=len([report for report in reports if report.success]),
            skipped_runs=skipped_runs,
            error_reports=[report for report in reports if not report.success],
            status="interrupted" if interrupted else "finished",
        )

        return BatchExecutionReport(
            reports=reports,
            total_jobs=total_runs,
            skipped_runs=skipped_runs,
            scheduled_runs=scheduled_runs,
            interrupted=interrupted,
        )
