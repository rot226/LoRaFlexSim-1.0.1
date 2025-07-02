# LoRa Network Simulator 4.0

This repository contains a lightweight LoRa network simulator implemented in Python. The latest code resides in the `VERSION_4` directory and is based on a simplified version of the FLoRa model so it can run without OMNeT++.

## Features
- Duty cycle enforcement to mimic real LoRa constraints
- Optional node mobility with Bezier interpolation
- Multi-channel radio support
- Advanced channel model with loss and noise parameters
- Configurable bandwidth and coding rate per channel
- Initial spreading factor and power selection
- Full LoRaWAN ADR layer following the official specification (LinkADRReq/Ans,
  ADRACKReq, channel mask, NbTrans, ADR_ACK_DELAY fallback) and derived from the
  FLoRa model
- Optional battery model to track remaining energy per node (FLoRa energy profile)

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

The dashboard now exposes a **Graine** input. Set the same value on
subsequent runs to keep the node placement identical.

# Run a simulation
python run.py --nodes 20 --steps 100
```

Add `--seed <n>` to obtain the same node placement on each run.

You can also execute the simulator directly from the repository root:

```bash
python VERSION_4/run.py --nodes 20 --steps 100
```

For a detailed description of all options, see `VERSION_4/README.md`.

## Advanced usage

Here are some commands to explore more simulator features:

```bash
# Multi-channel simulation with node mobility
python VERSION_4/run.py --nodes 50 --gateways 2 --area 2000 --channels 3 \
  --mobility --steps 500 --output advanced.csv

# LoRaWAN demo with downlinks
python VERSION_4/run.py --lorawan-demo --steps 100 --output lorawan.csv
```

You can analyse the resulting CSV file with:

```bash
python examples/analyse_resultats.py advanced.csv
```

## Validating results

Run the test suite to ensure everything works as expected:

```bash
pytest -q
```

The tests compare RSSI and airtime calculations against theoretical values and check collision handling.

## Versioning

The current package version is defined in `pyproject.toml`.
See `CHANGELOG.md` for a summary of releases.

This project is licensed under the [MIT License](LICENSE).
