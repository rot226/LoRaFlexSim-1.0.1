"""Moteur de simulation event-driven pour uplinks périodiques."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import logging
import json
from pathlib import Path
import random
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
        thresholds = [0.25, 0.5, 0.75, 1.0]
        threshold_idx = 0

        while queue:
            event = heapq.heappop(queue)
            if event.time_s > until_s:
                break
            result.events.append(event)

            if progress_callback is not None:
                while threshold_idx < len(thresholds) and event.time_s >= (until_s * thresholds[threshold_idx]):
                    progress_callback(thresholds[threshold_idx])
                    threshold_idx += 1

            if event.kind == "uplink":
                result.uplink_count += 1
                node = node_by_id[event.node_id]
                next_time = event.time_s + max(node.period_s, 1e-6)
                node.next_uplink_s = next_time
                heapq.heappush(queue, Event(time_s=next_time, kind="uplink", node_id=node.node_id))

        if progress_callback is not None:
            while threshold_idx < len(thresholds):
                progress_callback(thresholds[threshold_idx])
                threshold_idx += 1

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

    def _logger_for_run(self, run_id: str) -> tuple[logging.Logger, logging.Handler, Path]:
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
        logger.addHandler(file_handler)
        return logger, file_handler, run_dir

    def _write_run_status(self, run_dir: Path, run_id: str, status: str, error: str | None = None) -> None:
        payload: dict[str, Any] = {"run_id": run_id, "status": status}
        if error:
            payload["error"] = error
        (run_dir / "run_status.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def execute_jobs(
        self,
        jobs: list[dict[str, Any]],
        progress_callback: Callable[[int, int, RunExecutionReport], None] | None = None,
    ) -> BatchExecutionReport:
        reports: list[RunExecutionReport] = []
        total_jobs = len(jobs)
        for index, job in enumerate(jobs, start=1):
            params = dict(job.get("params", {}))
            run_id = str(params.get("run_id", job.get("job_id", "run")))
            logger, handler, run_dir = self._logger_for_run(run_id)
            self._write_run_status(run_dir, run_id, "running")
            try:
                seed = int(params.get("seed", 0))
                duration_s = float(params.get("duration_s", 3600.0))
                logger.info("Démarrage run_id=%s seed=%s", run_id, seed)
                logger.info("Paramètres: %s", params)

                engine = EventDrivenEngine(seed=seed)
                nodes = self._build_nodes(params)
                result = engine.run(
                    nodes=nodes,
                    until_s=duration_s,
                    progress_callback=lambda progress: logger.info("Progression: %s%%", int(progress * 100)),
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
                self._write_run_status(run_dir, run_id, "completed")
                report = RunExecutionReport(run_id=run_id, success=True, run_dir=run_dir)
                reports.append(report)
                if progress_callback is not None:
                    progress_callback(index, total_jobs, report)
            except Exception as exc:
                logger.exception("Run en erreur: %s", exc)
                self._write_run_status(run_dir, run_id, "failed", str(exc))
                report = RunExecutionReport(run_id=run_id, success=False, run_dir=run_dir, error=str(exc))
                reports.append(report)
                if progress_callback is not None:
                    progress_callback(index, total_jobs, report)
            finally:
                logger.removeHandler(handler)
                handler.close()

        return BatchExecutionReport(reports=reports)
