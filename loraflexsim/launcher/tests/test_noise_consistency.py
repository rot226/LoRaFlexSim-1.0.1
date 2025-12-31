import math
from types import MethodType

from loraflexsim.launcher.channel import Channel
from loraflexsim.launcher.simulator import EventType, Simulator


def test_noise_sample_reuse_keeps_heard_and_collision_stable():
    channel = Channel(noise_floor_std=5.0, phy_model="")
    channel.snir_fading_std = 0.0

    noise_values = iter([-120.0, -118.0, -140.0])
    used_noise: list[float] = []

    def fake_noise(self, freq_offset_hz: float = 0.0):
        try:
            value = next(noise_values)
        except StopIteration as exc:  # pragma: no cover - unexpected path
            raise AssertionError("noise_floor_dBm sampled too many times") from exc
        used_noise.append(value)
        self.last_noise_dBm = value
        return value

    snr_margin = -4.0

    def fake_compute(self, tx_power_dBm, distance, sf, **kwargs):
        noise = self.noise_floor_dBm(kwargs.get("freq_offset_hz", 0.0))
        rssi = noise + snr_margin
        self.last_rssi_dBm = rssi
        self.last_filter_att_dB = 0.0
        self.last_freq_hz = self.frequency_hz
        return rssi, snr_margin

    channel.noise_floor_dBm = MethodType(fake_noise, channel)
    channel.compute_rssi = MethodType(fake_compute, channel)

    sim = Simulator(
        num_nodes=2,
        num_gateways=1,
        area_size=1.0,
        transmission_mode="Periodic",
        packet_interval=1.0,
        packets_to_send=1,
        mobility=False,
        channels=[channel],
        seed=123,
    )

    for node in sim.nodes:
        node.sf = 7
        node.tx_power = 14.0
        node.channel = channel
    threshold = Channel.flora_detection_threshold(7, channel.bandwidth) + channel.sensitivity_margin_dB
    channel.detection_threshold_dBm = threshold

    sim.event_queue = []
    sim.event_id_counter = 0
    for node in sim.nodes:
        eid = sim.event_id_counter
        sim.event_id_counter += 1
        sim._push_event(0.0, EventType.TX_START, eid, node.id)

    sim.run()

    assert used_noise == [-120.0, -118.0]
    assert len(sim.events_log) == 2
    assert all(entry["heard"] for entry in sim.events_log)
    assert {entry["result"] for entry in sim.events_log} == {"CollisionLoss"}
    assert sim.packets_lost_collision == 2
    assert sim.packets_lost_no_signal == 0


def test_collision_events_include_snir_db():
    channel = Channel(noise_floor_std=0.0, phy_model="")
    channel.snir_fading_std = 0.0

    def fake_noise(self, freq_offset_hz: float = 0.0):
        value = -120.0
        self.last_noise_dBm = value
        return value

    def fake_compute(self, tx_power_dBm, distance, sf, **kwargs):
        noise = self.noise_floor_dBm(kwargs.get("freq_offset_hz", 0.0))
        rssi = noise - 2.0
        self.last_rssi_dBm = rssi
        self.last_filter_att_dB = 0.0
        self.last_freq_hz = self.frequency_hz
        return rssi, rssi - noise

    channel.noise_floor_dBm = MethodType(fake_noise, channel)
    channel.compute_rssi = MethodType(fake_compute, channel)

    sim = Simulator(
        num_nodes=2,
        num_gateways=1,
        area_size=1.0,
        transmission_mode="Periodic",
        packet_interval=1.0,
        packets_to_send=1,
        mobility=False,
        channels=[channel],
        seed=456,
    )

    for node in sim.nodes:
        node.sf = 7
        node.tx_power = 14.0
        node.channel = channel
    threshold = Channel.flora_detection_threshold(7, channel.bandwidth) + channel.sensitivity_margin_dB
    channel.detection_threshold_dBm = threshold

    sim.event_queue = []
    sim.event_id_counter = 0
    for node in sim.nodes:
        eid = sim.event_id_counter
        sim.event_id_counter += 1
        sim._push_event(0.0, EventType.TX_START, eid, node.id)

    sim.run()

    collisions = [entry for entry in sim.events_log if entry["result"] == "CollisionLoss"]
    assert collisions
    for entry in collisions:
        snir = entry.get("snir_dB")
        assert snir is not None
        assert math.isfinite(float(snir))
