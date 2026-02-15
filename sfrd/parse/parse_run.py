"""Parsing d'un run individuel à partir des logs bruts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _to_float(value: Any) -> float | None:
    """Convertit ``value`` en ``float`` si possible, sinon ``None``."""

    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            result = float(text)
        except ValueError:
            return None
        return result if math.isfinite(result) else None
    return None


def _to_int(value: Any) -> int | None:
    """Convertit ``value`` en entier si possible, sinon ``None``."""

    number = _to_float(value)
    if number is None:
        return None
    return int(number)


def _to_bool(value: Any) -> bool | None:
    """Convertit ``value`` en booléen si possible."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if not isinstance(value, str):
        return None

    key = value.strip().lower()
    true_values = {"1", "true", "ok", "success", "succès", "delivered", "yes"}
    false_values = {"0", "false", "ko", "failed", "collision", "drop", "no"}
    if key in true_values:
        return True
    if key in false_values:
        return False
    return None


def _read_raw_events(raw_log_path: str | Path) -> list[dict[str, Any]]:
    """Lit des événements depuis JSON, JSONL ou CSV."""

    path = Path(raw_log_path)
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        events: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            striped = line.strip()
            if not striped:
                continue
            item = json.loads(striped)
            if isinstance(item, dict):
                events.append(item)
        return events

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("events", "logs", "records", "rows"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        return []

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]

    raise ValueError(f"Format de log brut non supporté: {path}")


def _extract_time(event: dict[str, Any]) -> float | None:
    """Retourne le timestamp de l'événement si disponible."""

    for key in ("time", "timestamp", "event_time", "start_time", "tx_time"):
        value = _to_float(event.get(key))
        if value is not None:
            return value
    return None


def _is_tx_event(event: dict[str, Any]) -> bool:
    """Détermine si une ligne correspond à une transmission."""

    event_kind = str(event.get("event_type") or event.get("type") or event.get("action") or "").strip().lower()
    if event_kind in {"tx", "transmit", "transmission", "uplink_tx", "packet_tx"}:
        return True
    if event_kind in {"rx", "receive", "ack", "downlink"}:
        return False

    # Les logs LoRaFlexSim contiennent généralement un champ ``result`` par
    # tentative d'uplink; sa présence est un signal fort d'une transmission.
    if "result" in event:
        return True

    # Fallback: si un SF explicite est présent, il s'agit très probablement
    # d'une tentative TX dans les logs d'événements.
    if any(key in event for key in ("sf", "spreading_factor", "final_sf", "tx_sf")):
        return True

    return False


def _is_success_event(event: dict[str, Any]) -> bool:
    """Détermine si la transmission est reçue avec succès."""

    for key in ("success", "is_success", "delivered", "rx_ok"):
        parsed = _to_bool(event.get(key))
        if parsed is not None:
            return parsed

    status = str(event.get("result") or event.get("status") or event.get("outcome") or "").strip().lower()
    if status in {"success", "ok", "delivered", "received", "rx_ok", "pass"}:
        return True
    if status in {"collision", "failed", "drop", "timeout", "error", "rx_ko"}:
        return False

    return False


def _extract_sf(event: dict[str, Any]) -> int | None:
    """Extrait le spreading factor s'il est disponible."""

    def _parse_sf(value: Any) -> int | None:
        number = _to_float(value)
        if number is None:
            return None
        sf = int(number)
        if not math.isclose(number, float(sf), rel_tol=0.0, abs_tol=1e-9):
            return None
        if sf not in {7, 8, 9, 10, 11, 12}:
            return None
        return sf

    for key in ("sf", "spreading_factor", "final_sf", "tx_sf", "initial_sf"):
        value = _parse_sf(event.get(key))
        if value is not None:
            return value
    return None


def _compute_effective_duration(
    *,
    timestamps_after_warmup: list[float],
    sim_duration_s: float | None,
    warmup_s: float,
) -> float | None:
    """Calcule une durée exploitable hors warm-up."""

    duration_from_contract = None
    if sim_duration_s is not None and math.isfinite(sim_duration_s):
        candidate = sim_duration_s - warmup_s
        if candidate > 0:
            duration_from_contract = candidate

    duration_from_timestamps = None
    if timestamps_after_warmup:
        t_min = min(timestamps_after_warmup)
        t_max = max(timestamps_after_warmup)
        span = t_max - t_min
        if span > 0:
            duration_from_timestamps = span

    if duration_from_timestamps is None:
        return duration_from_contract
    if duration_from_contract is None:
        return duration_from_timestamps

    # Préférence à la durée contractuelle quand elle est valide;
    # les timestamps servent de fallback si cette donnée est absente/invalide.
    return duration_from_contract


def parse_run(
    raw_log_path: str | Path,
    *,
    warmup_s: float = 0.0,
    sim_duration_s: float | None = None,
    total_energy_joule: float | None = None,
) -> dict[str, Any]:
    """Parse un run brut et calcule des métriques robustes hors warm-up.

    Stratégie cas limites (pas d'extrapolation):
    - ``pdr`` vaut ``None`` si ``tx_count == 0``.
    - ``throughput_packets_per_s`` vaut ``None`` si durée absente/invalide.
    - ``energy_joule_per_packet`` vaut ``None`` si énergie absente ou
      ``success_count == 0``.
    """

    if warmup_s < 0 or not math.isfinite(warmup_s):
        raise ValueError("warmup_s doit être un flottant fini >= 0")

    events = _read_raw_events(raw_log_path)

    tx_count = 0
    success_count = 0
    sf_distribution = {sf: 0 for sf in range(7, 13)}
    timestamps_after_warmup: list[float] = []

    for event in events:
        timestamp = _extract_time(event)
        if timestamp is not None and timestamp < warmup_s:
            continue

        if timestamp is not None:
            timestamps_after_warmup.append(timestamp)

        if not _is_tx_event(event):
            continue

        tx_count += 1
        if _is_success_event(event):
            success_count += 1

        sf = _extract_sf(event)
        if sf in sf_distribution:
            sf_distribution[sf] += 1

    effective_duration_s = _compute_effective_duration(
        timestamps_after_warmup=timestamps_after_warmup,
        sim_duration_s=sim_duration_s,
        warmup_s=warmup_s,
    )

    pdr = (success_count / tx_count) if tx_count > 0 else None
    throughput_packets_per_s = (
        success_count / effective_duration_s
        if effective_duration_s is not None and effective_duration_s > 0
        else None
    )
    energy_joule_per_packet = (
        total_energy_joule / success_count
        if total_energy_joule is not None and success_count > 0
        else None
    )

    return {
        "tx_count": tx_count,
        "success_count": success_count,
        "effective_duration_s": effective_duration_s,
        "pdr": pdr,
        "throughput_packets_per_s": throughput_packets_per_s,
        "energy_joule_per_packet": energy_joule_per_packet,
        "sf_distribution": sf_distribution,
        "warmup_s": warmup_s,
        "events_total": len(events),
        "events_used": tx_count,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse un run brut SFRD.")
    parser.add_argument("raw_log_path", type=Path, help="Chemin du log brut (JSON/JSONL/CSV)")
    parser.add_argument("--warmup-s", type=float, default=0.0, help="Durée warm-up à exclure")
    parser.add_argument("--sim-duration-s", type=float, default=None, help="Durée simulation contractuelle")
    parser.add_argument("--total-energy-joule", type=float, default=None, help="Énergie totale du run")
    return parser.parse_args()


def main() -> None:
    """Entrée CLI utilitaire."""

    args = _parse_args()
    metrics = parse_run(
        args.raw_log_path,
        warmup_s=args.warmup_s,
        sim_duration_s=args.sim_duration_s,
        total_energy_joule=args.total_energy_joule,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
