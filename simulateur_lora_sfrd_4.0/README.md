# LoRa Network Simulator 4.0

This repository contains a lightweight LoRa network simulator implemented in Python. The latest code resides in the `VERSION_4` directory and now integrates a simplified OMNeT++ physical layer for frequency and clock drifts as well as thermal noise. It remains independent from the full simulator stack.

## Features
- Duty cycle enforcement to mimic real LoRa constraints
- Optional node mobility with Bezier interpolation or terrain-aware random
  waypoint movement, plus path-based navigation avoiding obstacles
- Multi-channel radio support
- Advanced channel model with loss and noise parameters
- Optional multipath fading with synchronised paths and external interference modeling
- Correlated fading and 3D obstacle maps with automatic calibration. Obstacle
  or height maps can be loaded from JSON or plain text matrices
- Antenna gains and cable losses for accurate link budgets
- Optional LoRa spreading gain applied to SNR
- Additional COST231 path loss, Okumura‑Hata model and 3D propagation via
  `AdvancedChannel`
- Terrain and weather based attenuation parameters
- Time‑varying frequency and synchronisation offsets for realistic collisions
- Optional frequency‑selective interference bands for jamming scenarios
- Configurable bandwidth and coding rate per channel
- Preset propagation environments (urban/suburban/rural) for quick channel setup
- Capture effect and a minimum interference time to ignore very short overlaps
- Initial spreading factor and power selection
- Full LoRaWAN ADR layer following the official specification (LinkADRReq/Ans,
  ADRACKReq, channel mask, NbTrans, ADR_ACK_DELAY fallback) and derived from the
  FLoRa model
- Optional battery model to track remaining energy per node
  (FLoRa energy profile)
- Regional channel plans for EU868, US915, AU915 and Asian bands
- Customizable energy profiles with a simple registry
- Beacon loss and clock drift simulation for Class B nodes
- Energy-aware quasi-continuous listening for Class C nodes

## Quick start

```bash
# Install dependencies
cd VERSION_4
python3 -m venv env
source env/bin/activate  # On Windows use env\Scripts\activate
pip install -r requirements.txt

# Launch the dashboard
panel serve launcher/dashboard.py --show

# From the repository root use
panel serve VERSION_4/launcher/dashboard.py --show
```

The dashboard now exposes a **Seed** input. Set the same value on
subsequent runs to keep the node placement identical.
Enable **Manual positions** to override node or gateway coordinates. Each line
should follow `node,id=3,x=120,y=40` or `gw,id=1,x=10,y=80`.

# Run a simulation
```bash
python run.py --nodes 20 --steps 100
```

Add `--seed <n>` to obtain the same node placement on each run.
Use `--runs <n>` to repeat the simulation several times and average the metrics.

You can also execute the simulator directly from the repository root:

```bash
python VERSION_4/run.py --nodes 20 --steps 100
```

For a detailed description of all options, see `VERSION_4/README.md`.

## Advanced usage

Here are some commands to explore more simulator features:

```bash
# Multi-channel simulation with node mobility
python VERSION_4/run.py --nodes 50 --gateways 2 --channels 3 \
  --mobility --steps 500 --output advanced.csv

# LoRaWAN demo with downlinks
python VERSION_4/run.py --lorawan-demo --steps 100 --output lorawan.csv
```

### LoRaWAN class B/C examples

The Python API exposes additional parameters to experiment with class B or
class C behaviours. Below are minimal code snippets:

```python
from launcher import Simulator

# Class B nodes with periodic ping slots
sim_b = Simulator(num_nodes=10, node_class="B", beacon_interval=128,
                  ping_slot_interval=1.0)
sim_b.run(1000)

# Class C nodes listening almost continuously
sim_c = Simulator(num_nodes=5, node_class="C", class_c_rx_interval=0.5)
sim_c.run(500)

```

### Realistic mobility scenario

You can model smoother movements by enabling mobility and adjusting the speed
range:

```python
from launcher import Simulator

sim = Simulator(num_nodes=20, num_gateways=3, area_size=2000.0, mobility=True,
                mobility_speed=(1.0, 5.0))
sim.run(1000)
```

### FLoRa INI scenario example

To reproduce a typical FLoRa configuration and check the SF distribution per
node, run the helper script below. It applies the "Mode FLoRa complet" settings
with ADR variant 1 and fixed node positions:

```bash
python examples/run_flora_example.py --runs 5 --seed 123
```

The script prints the Packet Delivery Ratio and spreading factor histogram for
each run. ``Simulator`` now accepts ``flora_mode=True`` which enables the
official detection threshold, interference window and a ``flora`` propagation
  profile. Pass the `--flora-csv <file>` option to automatically compare these
  metrics with an official FLoRa export using `compare_with_sim`. FLoRa stores
  its results in `.sca` and `.vec` files. Use `tools/convert_flora_results.py`
  to turn these into CSV files if needed.
Passing `--degrade` to the script enables a harsh propagation profile that
significantly lowers the Packet Delivery Ratio.  Additional interference,
stronger fast fading and a higher path loss exponent are applied together with a
slow varying noise term.  The resulting PDR usually settles around **35–40 %**,
providing a challenging baseline for protocol comparisons.

An example configuration file named `examples/flora_full.ini` reproduces the
official positions used by FLoRa. Load it with
``Simulator(config_file="examples/flora_full.ini")`` to start from the exact
  same coordinates. Several sample exports from the OMNeT++ model are also
  provided in `examples/` (`flora_full.csv`, `flora_collisions.csv`, etc.) to
  help check your results.

Below is an example INI scenario that can be used with the FLoRa mode. It
defines fixed node and gateway coordinates along with the ADR settings:

```ini
[General]
network = flora.simulations.LoRaNetworkTest
**.maxTransmissionDuration = 4s
**.energyDetection = -110dBm
cmdenv-output-file = cmd_env_log.txt
**.vector-recording = true

rng-class = "cMersenneTwister"
**.loRaGW[*].numUdpApps = 1
**.loRaGW[*].packetForwarder.localPort = 2000
**.loRaGW[*].packetForwarder.destPort = 1000
**.loRaGW[*].packetForwarder.destAddresses = "networkServer"
**.loRaGW[*].packetForwarder.indexNumber = 1

**.networkServer.numApps = 1
**.networkServer.**.evaluateADRinServer = true
**.networkServer.app[0].typename = "NetworkServerApp"
**.networkServer.app[0].destAddresses = "loRaGW[0]"
**.networkServer.app[0].destPort = 2000
**.networkServer.app[0].localPort = 1000
**.networkServer.app[0].adrMethod = ${"avg"}

**.numberOfPacketsToSend = 80
sim-time-limit = 1d
repeat = 5
**.timeToFirstPacket = exponential(100s)
**.timeToNextPacket = exponential(100s)
**.alohaChannelModel = true
**.mobility = false

**.loRaNodes[*].**.evaluateADRinNode = true
**.loRaNodes[*].**initialLoRaBW = 125 kHz
**.loRaNodes[*].**initialLoRaCR = 4
**.loRaNodes[*].**initialLoRaSF = 12

output-scalar-file = ../results/novo5-80-gw1-s${runnumber}.ini.sca
```

Any INI file must define ``[gateways]`` and ``[nodes]`` sections listing the
coordinates (and optionally SF and power) of each entity.  The loader also
accepts a JSON document containing ``gateways`` and ``nodes`` lists to
describe more complex scenarios.

Use a preset propagation environment from Python:

```python
from VERSION_4.launcher.channel import Channel
suburban = Channel(environment="suburban")
```

You can analyze the resulting CSV file with:

```bash
python examples/analyse_resultats.py advanced.csv
```

If you collected several runs into one CSV via the dashboard or
`python VERSION_4/run.py --runs <n> --output results.csv`, use
`analyse_runs.py` to compute the average metrics for each run:

```bash
python examples/analyse_runs.py results.csv
```

## Cleaning results

Remove duplicated rows and `NaN` values from a CSV file using
`clean_results.py`. The cleaned file is saved as `<file>_clean.csv`:

```bash
python VERSION_4/launcher/clean_results.py results.csv
```

## Validating results

Run the test suite to ensure everything works as expected:

```bash
pytest -q
```

The tests compare RSSI and airtime calculations against theoretical values and check collision handling.

### Cross-check with FLoRa

  The module `VERSION_4/launcher/compare_flora.py` can read exports from
  the FLoRa simulator. It accepts raw `.sca` result files or CSV files produced
  with `tools/convert_flora_results.py` and extracts several metrics: Packet Delivery Ratio,
spreading factor histogram, energy consumption, throughput and packet
collisions.  The test file `tests/test_flora_comparison.py` demonstrates
how to compare these values with those returned by
`Simulator.get_metrics` to validate the Python implementation against
OMNeT++ runs.
  You can also supply a reference CSV or a directory containing `.sca` files to
  `examples/run_flora_example.py` via `--flora-csv` to perform this check outside
  the test suite.

### Generating a FLoRa comparison report

  The helper `tools/compare_flora_report.py` compares a FLoRa export
  (after conversion to CSV) with the results produced by `VERSION_4/run.py` and creates a short
report. It prints a table with the metrics from both sources and saves
bar charts highlighting the differences:

```bash
python tools/compare_flora_report.py flora.csv results.csv
```

Two images named `report_metrics.png` and `report_sf.png` are written in
the current directory.

### Calibration systématique avec FLoRa

  The script `tools/calibrate_flora.py` automates the search of channel
  parameters that best reproduce a reference export from FLoRa. The reference
  data should come from FLoRa results converted to CSV with
  `tools/convert_flora_results.py`. It launches
  several runs with different propagation settings and reports the combination
  yielding the smallest PDR difference. From the repository root run:

  ```bash
  python tools/calibrate_flora.py examples/flora_full.csv
  ```

The resulting parameters typically give a correspondence above **99 %** with
FLoRa on the provided dataset. The calibration is also executed during the
test suite to ensure the simulator stays in sync with the reference model.

Pass ``--advanced`` to also optimise the correlated fading coefficient and
the obstacle attenuation when using ``AdvancedChannel``.

To check several FLoRa exports at once and average the error over all of them
use:

```bash
python tools/calibrate_flora.py examples/flora_full.csv examples/flora_interference.csv
```

In this cross-validation mode the script evaluates the Packet Delivery Ratio
and spreading factor histograms across all inputs to pick the most consistent
propagation parameters.

### Comparer plusieurs références FLoRa

The helper `tools/compare_flora_multi.py` prints a summary of the metric
difference between a simulation CSV and several FLoRa exports:

```bash
python tools/compare_flora_multi.py results.csv examples/flora_full.csv \
    examples/flora_interference.csv
```

Add `--output diff.csv` to save the table for later analysis.

## Versioning

The current package version is defined in `pyproject.toml`.
See `CHANGELOG.md` for a summary of releases.
This project is licensed under the [MIT License](LICENSE).

## Current limitations

- This simulator aims to remain lightweight and therefore omits several advanced concepts:

- The physical layer is greatly simplified and does not reproduce hardware
  imperfections found in real devices.
- By default mobility relies on Bezier paths; an optional RandomWaypoint model
  can use terrain maps to handle obstacles.
- Basic LoRaWAN security is enabled. A dedicated join server validates
  JoinRequests and derives session keys, and all LoRaWAN frames are checked for
  a valid MIC before decryption.

Contributions are welcome to improve these areas or add missing features.
