import numpy as np

from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.gateway import Gateway


class _StubServer:
    def __init__(self) -> None:
        self.received: list[int] = []
        self.collision_reasons: dict[int, str] = {}

    def schedule_receive(self, event_id: int, *_args, **_kwargs) -> None:
        self.received.append(event_id)

    def register_collision_reason(self, event_id: int, reason: str) -> None:
        self.collision_reasons[event_id] = reason


def _channel_kwargs(rng) -> dict:
    return dict(
        shadowing_std=0.0,
        fast_fading_std=0.0,
        time_variation_std=0.0,
        noise_floor_std=0.0,
        fine_fading_std=0.0,
        variable_noise_std=0.0,
        processing_gain=False,
        phy_model="",
        rng=rng,
    )


def test_compute_snir_injects_fading_variability() -> None:
    rng = np.random.default_rng(1234)
    channel = Channel(snir_fading_std=2.0, **_channel_kwargs(rng))
    rssi, snr, snir, noise = channel.compute_snir(
        14.0,
        distance=10.0,
        sf=7,
        interferers_mW=1e-13,
    )

    baseline_rng = np.random.default_rng(1234)
    baseline_channel = Channel(
        snir_fading_std=0.0, **_channel_kwargs(baseline_rng)
    )
    _, _, baseline_snir, baseline_noise = baseline_channel.compute_snir(
        14.0,
        distance=10.0,
        sf=7,
        interferers_mW=1e-13,
    )

    assert snir != baseline_snir
    assert noise == baseline_noise
    assert rssi != 0.0
    assert snr != 0.0


def test_margin_collision_drop_marks_losses() -> None:
    gw = Gateway(1, 0.0, 0.0, rng=np.random.default_rng(0))
    server = _StubServer()

    base_kwargs = dict(
        sf=7,
        capture_threshold=0.5,
        required_snr_db_by_sf={7: 0.5},
        frequency=868e6,
        min_interference_time=0.0,
        noise_floor=-120.0,
        capture_mode="basic",
        flora_phy=None,
        orthogonal_sf=True,
        capture_window_symbols=0,
        non_orth_delta=None,
        snir_fading_std=0.0,
        marginal_snir_db=0.5,
        marginal_drop_prob=1.0,
    )

    gw.start_reception(
        event_id=1,
        node_id=101,
        rssi=-116.0,
        end_time=1.5,
        current_time=0.0,
        snir=0.5,
        **base_kwargs,
    )
    gw.start_reception(
        event_id=2,
        node_id=202,
        rssi=-114.0,
        end_time=1.5,
        current_time=0.0,
        snir=0.5,
        **base_kwargs,
    )

    gw.end_reception(1, server, 101)
    gw.end_reception(2, server, 202)

    assert server.received == []
    assert server.collision_reasons[1] == "snir_marginal"
    assert server.collision_reasons[2] == "snir_marginal"


def test_marginal_snir_drop_remains_probabilistic() -> None:
    def _run(seed: int):
        gw = Gateway(1, 0.0, 0.0, rng=np.random.default_rng(seed))
        server = _StubServer()
        common_kwargs = dict(
            sf=7,
            capture_threshold=0.0,
            required_snr_db_by_sf={7: 0.0},
            frequency=868e6,
            min_interference_time=0.0,
            noise_floor=-120.0,
            capture_mode="basic",
            flora_phy=None,
            orthogonal_sf=True,
            capture_window_symbols=0,
            non_orth_delta=None,
            snir_fading_std=0.0,
            marginal_snir_db=1.0,
            marginal_drop_prob=0.5,
        )

        gw.start_reception(
            event_id=10,
            node_id=1,
            rssi=-119.5,
            end_time=1.0,
            current_time=0.0,
            snir=0.2,
            **common_kwargs,
        )
        gw.start_reception(
            event_id=11,
            node_id=2,
            rssi=-118.5,
            end_time=1.0,
            current_time=0.0,
            snir=0.2,
            **common_kwargs,
        )

        gw.end_reception(10, server, 1)
        gw.end_reception(11, server, 2)
        return server

    dropped = _run(3)
    kept = _run(4)

    assert dropped.received == []
    assert set(dropped.collision_reasons.values()) == {"snir_marginal"}

    assert kept.received == [11]
    assert kept.collision_reasons == {}
