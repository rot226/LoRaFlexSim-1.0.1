"""Validation matrix scenarios comparing LoRaFlexSim with FLoRa outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from functools import partial

from loraflexsim.launcher import Simulator, MultiChannel, Channel
from loraflexsim.launcher import adr_ml, explora_at
from loraflexsim.launcher.compare_flora import (
    load_flora_metrics,
    load_flora_rx_stats,
)
from loraflexsim.launcher.smooth_mobility import SmoothMobility
from loraflexsim.scenarios.long_range import (
    LONG_RANGE_RECOMMENDATIONS,
    create_long_range_channels,
)


@dataclass(frozen=True)
class ScenarioTolerance:
    """Accepted deviation between simulator and FLoRa metrics."""

    pdr: float = 0.02
    collisions: int = 1
    snr: float = 1.0


@dataclass(frozen=True)
class ValidationScenario:
    """Describe a validation case for regression testing."""

    name: str
    description: str
    flora_config: Path
    flora_reference: Path
    sim_kwargs: dict[str, Any]
    channel_plan: Iterable[float] | None = None
    run_steps: int | None = None
    tolerances: ScenarioTolerance = field(default_factory=ScenarioTolerance)
    setup: Sequence[Callable[[Simulator], None]] = field(default_factory=tuple)
    channels_factory: Callable[[], list[Channel]] | None = None

    def build_simulator(self, **overrides: Any) -> Simulator:
        """Instantiate :class:`Simulator` for the scenario.

        Parameters passed via ``overrides`` allow callers to tweak specific
        simulator keyword arguments (e.g. ``seed``) without mutating the
        scenario definition.  They take precedence over the baseline
        ``sim_kwargs`` supplied when constructing the scenario.
        """

        kwargs = dict(self.sim_kwargs)
        kwargs.update(overrides)
        if self.channels_factory is not None:
            kwargs["channels"] = self.channels_factory()
        elif self.channel_plan is not None:
            kwargs["channels"] = MultiChannel(list(self.channel_plan))
        kwargs.setdefault("validation_mode", "flora")
        sim = Simulator(**kwargs)
        for hook in self.setup:
            hook(sim)
        return sim


def compute_average_snr(sim: Simulator) -> float:
    """Return the average SNR of successfully delivered packets."""

    snrs: list[float] = []
    for entry in sim.events_log:
        if entry.get("result") == "Success" and entry.get("snr_dB") is not None:
            snrs.append(float(entry["snr_dB"]))
    return sum(snrs) / len(snrs) if snrs else 0.0


def load_flora_reference(path: Path) -> dict[str, float]:
    """Load PDR, collisions and SNR metrics from a FLoRa export."""

    flora_metrics = load_flora_metrics(path)
    rx_stats = load_flora_rx_stats(path)
    return {
        "PDR": float(flora_metrics["PDR"]),
        "collisions": float(rx_stats["collisions"]),
        "snr": float(rx_stats["snr"]),
    }


def run_validation(sim: Simulator, max_steps: int | None = None) -> dict[str, float]:
    """Execute the scenario and compute validation metrics."""

    sim.run(max_steps)
    metrics = sim.get_metrics()
    avg_snr = compute_average_snr(sim)
    return {
        "PDR": float(metrics.get("PDR", 0.0)),
        "collisions": float(metrics.get("collisions", 0.0)),
        "snr": avg_snr,
    }


def compare_to_reference(sim_metrics: dict[str, float], reference: dict[str, float], tolerances: ScenarioTolerance) -> dict[str, float]:
    """Return absolute differences between simulator and reference metrics."""

    return {
        "PDR": abs(sim_metrics["PDR"] - reference["PDR"]),
        "collisions": abs(sim_metrics["collisions"] - reference["collisions"]),
        "snr": abs(sim_metrics["snr"] - reference["snr"]),
    }


# Matrix of reproducible scenarios derived from FLoRa configurations.
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "tests" / "integration" / "data"
FLORA_DIR = BASE_DIR / "flora-master" / "simulations" / "examples"

SCENARIOS: list[ValidationScenario] = [
    ValidationScenario(
        name="long_range",
        description="Scénario longue portée 12 km dérivé du preset FLoRa.",
        flora_config=FLORA_DIR / "long_range_flora.ini",
        flora_reference=DATA_DIR / "long_range_flora.sca",
        sim_kwargs=dict(
            flora_mode=True,
            config_file=str(FLORA_DIR / "long_range_flora.ini"),
            seed=3,
            packets_to_send=LONG_RANGE_RECOMMENDATIONS["flora"].packets_per_node,
            mobility=False,
            transmission_mode="Periodic",
        ),
        channels_factory=partial(create_long_range_channels, "flora"),
        tolerances=ScenarioTolerance(pdr=0.015, collisions=0, snr=0.22),
    ),
    ValidationScenario(
        name="mono_gw_single_channel_class_a",
        description="Mono-passerelle, canal unique EU868, classes A statiques avec ADR nœud+serveur.",
        flora_config=FLORA_DIR / "n100-gw1.ini",
        flora_reference=DATA_DIR / "mono_gw_single_channel_class_a.sca",
        sim_kwargs=dict(
            flora_mode=True,
            config_file=str(FLORA_DIR / "n100-gw1.ini"),
            seed=1,
            packets_to_send=2,
            mobility=False,
            adr_node=True,
            adr_server=True,
            adr_method="avg",
        ),
        channel_plan=[868.1e6],
        run_steps=None,
        tolerances=ScenarioTolerance(pdr=0.02, collisions=2, snr=1.5),
    ),
    ValidationScenario(
        name="mono_gw_multichannel_node_adr",
        description="Mono-passerelle, 3 canaux EU868, ADR côté nœud uniquement (classe A).",
        flora_config=FLORA_DIR / "n100-gw1.ini",
        flora_reference=DATA_DIR / "mono_gw_multichannel_node_adr.sca",
        sim_kwargs=dict(
            flora_mode=True,
            config_file=str(FLORA_DIR / "n100-gw1.ini"),
            seed=2,
            packets_to_send=2,
            mobility=False,
            adr_node=True,
            adr_server=False,
            adr_method="avg",
        ),
        channel_plan=[868.1e6, 868.3e6, 868.5e6],
        run_steps=None,
        tolerances=ScenarioTolerance(pdr=0.02, collisions=2, snr=1.5),
    ),
    ValidationScenario(
        name="multi_gw_multichannel_server_adr",
        description="Deux passerelles, multi-canaux, ADR serveur uniquement (classe A).",
        flora_config=FLORA_DIR / "n1000-gw2.ini",
        flora_reference=DATA_DIR / "multi_gw_multichannel_server_adr.sca",
        sim_kwargs=dict(
            flora_mode=True,
            config_file=str(FLORA_DIR / "n1000-gw2.ini"),
            seed=3,
            packets_to_send=1,
            mobility=False,
            adr_node=False,
            adr_server=True,
            adr_method="avg",
        ),
        channel_plan=[868.1e6, 868.3e6, 868.5e6],
        run_steps=None,
        tolerances=ScenarioTolerance(pdr=0.03, collisions=3, snr=2.0),
    ),
    ValidationScenario(
        name="class_b_beacon_scheduling",
        description="Classe B avec synchronisation beacon, canal unique, topologie statique.",
        flora_config=FLORA_DIR / "n100-gw1.ini",
        flora_reference=DATA_DIR / "class_b_beacon_scheduling.sca",
        sim_kwargs=dict(
            flora_mode=True,
            config_file=str(FLORA_DIR / "n100-gw1.ini"),
            seed=4,
            packets_to_send=1,
            mobility=False,
            adr_node=False,
            adr_server=False,
            node_class="B",
            adr_method="avg",
        ),
        channel_plan=[868.1e6],
        run_steps=None,
        tolerances=ScenarioTolerance(pdr=0.05, collisions=2, snr=2.5),
    ),
    ValidationScenario(
        name="class_c_mobility_multichannel",
        description="Classe C mobile avec 3 canaux et ADR serveur.",
        flora_config=FLORA_DIR / "n100-gw1.ini",
        flora_reference=DATA_DIR / "class_c_mobility_multichannel.sca",
        sim_kwargs=dict(
            flora_mode=True,
            config_file=str(FLORA_DIR / "n100-gw1.ini"),
            seed=5,
            packets_to_send=1,
            mobility=True,
            adr_node=False,
            adr_server=True,
            node_class="C",
            adr_method="avg",
            mobility_model=SmoothMobility(
                2376.0,
                1.0,
                3.0,
                rng=np.random.Generator(np.random.MT19937(5)),
            ),
        ),
        channel_plan=[868.1e6, 868.3e6, 868.5e6],
        run_steps=None,
        tolerances=ScenarioTolerance(pdr=0.05, collisions=3, snr=3.0),
    ),
    ValidationScenario(
        name="duty_cycle_enforcement_class_a",
        description="Duty cycle 1 % appliqué explicitement (classe A).",
        flora_config=FLORA_DIR / "n100-gw1.ini",
        flora_reference=DATA_DIR / "duty_cycle_enforcement_class_a.sca",
        sim_kwargs=dict(
            flora_mode=True,
            config_file=str(FLORA_DIR / "n100-gw1.ini"),
            seed=6,
            packets_to_send=1,
            mobility=False,
            adr_node=False,
            adr_server=False,
            adr_method="avg",
            duty_cycle=0.01,
        ),
        channel_plan=[868.1e6],
        run_steps=None,
        tolerances=ScenarioTolerance(pdr=0.02, collisions=1, snr=2.0),
    ),
    ValidationScenario(
        name="dynamic_multichannel_random_assignment",
        description="Multi-canaux avec répartition aléatoire et ADR combiné.",
        flora_config=FLORA_DIR / "n100-gw1.ini",
        flora_reference=DATA_DIR / "dynamic_multichannel_random_assignment.sca",
        sim_kwargs=dict(
            flora_mode=True,
            config_file=str(FLORA_DIR / "n100-gw1.ini"),
            seed=7,
            packets_to_send=1,
            mobility=False,
            adr_node=True,
            adr_server=True,
            adr_method="avg",
            channel_distribution="random",
        ),
        channel_plan=[868.1e6, 868.3e6, 868.5e6],
        run_steps=None,
        tolerances=ScenarioTolerance(pdr=0.03, collisions=2, snr=2.5),
    ),
    ValidationScenario(
        name="class_b_mobility_multichannel",
        description="Classe B mobile avec SmoothMobility et plan tri-canal.",
        flora_config=FLORA_DIR / "n100-gw1.ini",
        flora_reference=DATA_DIR / "class_b_mobility_multichannel.sca",
        sim_kwargs=dict(
            flora_mode=True,
            config_file=str(FLORA_DIR / "n100-gw1.ini"),
            seed=8,
            packets_to_send=1,
            mobility=True,
            adr_node=False,
            adr_server=True,
            node_class="B",
            adr_method="avg",
            mobility_model=SmoothMobility(
                2376.0,
                1.0,
                3.0,
                rng=np.random.Generator(np.random.MT19937(8)),
            ),
        ),
        channel_plan=[868.1e6, 868.3e6, 868.5e6],
        run_steps=None,
        tolerances=ScenarioTolerance(pdr=0.05, collisions=3, snr=3.0),
    ),
    ValidationScenario(
        name="explora_at_balanced_airtime",
        description="EXPLoRa-AT active l'équilibrage airtime ADR.",
        flora_config=FLORA_DIR / "n100-gw1.ini",
        flora_reference=DATA_DIR / "explora_at_balanced_airtime.sca",
        sim_kwargs=dict(
            flora_mode=True,
            config_file=str(FLORA_DIR / "n100-gw1.ini"),
            seed=9,
            packets_to_send=1,
            mobility=False,
            adr_node=False,
            adr_server=False,
            adr_method="avg",
        ),
        channel_plan=[868.1e6, 868.3e6, 868.5e6],
        run_steps=None,
        tolerances=ScenarioTolerance(pdr=0.05, collisions=3, snr=3.0),
        setup=(explora_at.apply,),
    ),
    ValidationScenario(
        name="adr_ml_adaptive_strategy",
        description="ADR-ML appliqué aux nœuds tri-canaux (classe A).",
        flora_config=FLORA_DIR / "n100-gw1.ini",
        flora_reference=DATA_DIR / "adr_ml_adaptive_strategy.sca",
        sim_kwargs=dict(
            flora_mode=True,
            config_file=str(FLORA_DIR / "n100-gw1.ini"),
            seed=10,
            packets_to_send=1,
            mobility=False,
            adr_node=False,
            adr_server=False,
            adr_method="avg",
        ),
        channel_plan=[868.1e6, 868.3e6, 868.5e6],
        run_steps=None,
        tolerances=ScenarioTolerance(pdr=0.05, collisions=3, snr=3.0),
        setup=(adr_ml.apply,),
    ),
]


__all__ = [
    "ScenarioTolerance",
    "ValidationScenario",
    "SCENARIOS",
    "compute_average_snr",
    "compare_to_reference",
    "load_flora_reference",
    "run_validation",
]
