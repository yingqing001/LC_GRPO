#!/usr/bin/env bash
# Multi-node wrapper for the FLUX Flow-GRPO baseline.
# Set MASTER_ADDR and pass MACHINE_RANK through the environment or as $1.

set -euo pipefail

if [[ -z "${MACHINE_RANK:-}" ]]; then
    if (( $# == 0 )); then
        echo "Set MACHINE_RANK or pass the machine rank as the first argument." >&2
        exit 2
    fi
    MACHINE_RANK="$1"
    shift
fi
: "${MASTER_ADDR:?Set MASTER_ADDR to the rank-0 host address}"

export MACHINE_RANK
export NUM_MACHINES="${NUM_MACHINES:-2}"
export GPUS_PER_MACHINE="${GPUS_PER_MACHINE:-4}"
export NUM_PROCESSES="${NUM_PROCESSES:-$((NUM_MACHINES * GPUS_PER_MACHINE))}"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-scripts/accelerate_configs/deepspeed_zero2.yaml}"

exec bash scripts/single_node/flux_flow_grpo.sh "$@"
