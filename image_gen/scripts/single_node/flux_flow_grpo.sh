#!/usr/bin/env bash
# Flow-GRPO baseline for FLUX.1-Dev (paper-matched 8-GPU image setting).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TASK="${TASK:-hps}"
NUM_MACHINES="${NUM_MACHINES:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29501}"
MODEL="${MODEL:-black-forest-labs/FLUX.1-dev}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-scripts/accelerate_configs/deepspeed_zero2.yaml}"

if [[ "$NUM_PROCESSES" != "8" ]]; then
    echo "Paper presets require 8 total processes; got $NUM_PROCESSES." >&2
    exit 2
fi

case "$TASK" in
    hps)
        CONFIG_NAME="hps_flux_8gpu"
        ADVANTAGE_AGGREGATION="reward_sum"
        ;;
    hps_and_clip)
        CONFIG_NAME="hps_and_clip_flux_8gpu"
        ADVANTAGE_AGGREGATION="per_reward_adv"
        ;;
    *)
        echo "Unsupported TASK=$TASK. Choose hps or hps_and_clip." >&2
        exit 2
        ;;
esac

LAUNCH_ARGS=(
    --config_file "$ACCELERATE_CONFIG"
    --num_processes "$NUM_PROCESSES"
    --main_process_port "$MAIN_PROCESS_PORT"
)
if (( NUM_MACHINES > 1 )); then
    : "${MASTER_ADDR:?Set MASTER_ADDR for multi-node training}"
    : "${MACHINE_RANK:?Set MACHINE_RANK for multi-node training}"
    LAUNCH_ARGS+=(
        --num_machines "$NUM_MACHINES"
        --machine_rank "$MACHINE_RANK"
        --main_process_ip "$MASTER_ADDR"
    )
fi

accelerate launch "${LAUNCH_ARGS[@]}" \
    scripts/train_flux.py \
    --config "config/grpo.py:${CONFIG_NAME}" \
    --config.pretrained.model="$MODEL" \
    --config.sample.noise_level=0.9 \
    --config.sample.sde_type=sde \
    --config.train.beta=0 \
    --config.adv_aggregation="$ADVANTAGE_AGGREGATION" \
    "$@"
