#!/bin/bash
# Build the native FLoRa physical layer library used by the flora_cpp model

set -e

# Navigate to the repository root even if script executed elsewhere
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

FLORA_DIR="$ROOT_DIR/flora-master"

if [ ! -d "$FLORA_DIR" ]; then
    echo "Directory '$FLORA_DIR' not found" >&2
    exit 1
fi

cd "$FLORA_DIR"

# Generate makefiles if missing
if [ ! -f src/Makefile ]; then
    make makefiles
fi

make libflora_phy.so -j$(nproc)

echo "Built library at $FLORA_DIR/libflora_phy.so"
