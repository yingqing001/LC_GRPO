#!/usr/bin/env bash
# LC-GRPO for SD3.5-Medium (paper-matched 8-GPU image setting).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TASK="${TASK:-hps}"
NUM_MACHINES="${NUM_MACHINES:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29501}"
MODEL="${MODEL:-stabilityai/stable-diffusion-3.5-medium}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-scripts/accelerate_configs/multi_gpu.yaml}"

if [[ "$NUM_PROCESSES" != "8" ]]; then
    echo "Paper presets require 8 total processes; got $NUM_PROCESSES." >&2
    exit 2
fi

case "$TASK" in
    hps)
        CONFIG_NAME="hps_sd3_lc_grpo_8gpu"
        NOISE_LEVEL="0.7"
        KL_BETA="0"
        ADVANTAGE_AGGREGATION="reward_sum"
        ;;
    hps_and_clip)
        CONFIG_NAME="hps_and_clip_sd3_lc_grpo_8gpu"
        NOISE_LEVEL="0.7"
        KL_BETA="0"
        ADVANTAGE_AGGREGATION="per_reward_adv"
        ;;
    ocr)
        CONFIG_NAME="ocr_sd3_lc_grpo_8gpu"
        NOISE_LEVEL="0.8"
        KL_BETA="0.004"
        ADVANTAGE_AGGREGATION="reward_sum"
        ;;
    *)
        echo "Unsupported TASK=$TASK. Choose hps, hps_and_clip, or ocr." >&2
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
    scripts/train_sd3_langevin_correct.py \
    --config "config/grpo.py:${CONFIG_NAME}" \
    --config.pretrained.model="$MODEL" \
    --config.sample.noise_level="$NOISE_LEVEL" \
    --config.sample.ode_rate=1.0 \
    --config.sample.langevin_coeff_type=base_const \
    --config.train.beta="$KL_BETA" \
    --config.train.correction_step_grad_mode=train \
    --config.train.correction_cfg=False \
    --config.adv_aggregation="$ADVANTAGE_AGGREGATION" \
    "$@"
