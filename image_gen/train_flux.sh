#!/usr/bin/env bash
# LC-GRPO paper reproduction launcher for FLUX.1-Dev image generation.
# Paper: https://arxiv.org/abs/2608.05600
#
# Usage:
#   TASK=hps bash flux.sh
#   TASK=hps_and_clip bash flux.sh
#
# Extra arguments are forwarded to the training script, for example:
#   TASK=hps bash flux.sh --config.run_name=my-run

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

TASK="${TASK:-hps}"
NUM_GPUS="${NUM_GPUS:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29501}"
MODEL="${MODEL:-black-forest-labs/FLUX.1-dev}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-scripts/accelerate_configs/deepspeed_zero2.yaml}"

# The public configs below reproduce the paper's 48 prompts per epoch and
# 24 generations per prompt on 8 GPUs.
if [[ "$NUM_GPUS" != "8" ]]; then
    echo "This reproduction launcher requires NUM_GPUS=8; got NUM_GPUS=$NUM_GPUS." >&2
    echo "For another GPU count, add a matching preset in config/grpo.py." >&2
    exit 2
fi

case "$TASK" in
    hps)
        CONFIG_NAME="hps_flux_correct_8gpu"
        NOISE_LEVEL="0.8"
        ADVANTAGE_AGGREGATION="reward_sum"
        ;;
    hps_and_clip)
        CONFIG_NAME="hps_and_clip_flux_correct_8gpu"
        NOISE_LEVEL="0.9"
        ADVANTAGE_AGGREGATION="per_reward_adv"
        ;;
    *)
        echo "Unsupported TASK=$TASK. Choose hps or hps_and_clip." >&2
        exit 2
        ;;
esac

RUN_NAME="${RUN_NAME:-lc-grpo-flux-${TASK}}"
SAVE_DIR="${SAVE_DIR:-logs/flux/${TASK}/${RUN_NAME}}"

# FLUX paper settings: 512x512, 6 rollout steps, LoRA r=64/alpha=128
# (set in train_flux_langevin_correct.py), lr=3e-4, beta=0, and no CFG.
# HPS uses eta=0.8; joint HPS-CLIP uses eta=0.9 with 0.7/0.3 weights.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Launching LC-GRPO FLUX reproduction"
echo "  task=$TASK config=$CONFIG_NAME gpus=$NUM_GPUS eta=$NOISE_LEVEL"
echo "  model=$MODEL output=$SAVE_DIR"

accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    --num_processes "$NUM_GPUS" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    scripts/train_flux_langevin_correct.py \
    --config "config/grpo.py:${CONFIG_NAME}" \
    --config.pretrained.model="$MODEL" \
    --config.run_name="$RUN_NAME" \
    --config.save_dir="$SAVE_DIR" \
    --config.resolution=512 \
    --config.sample.num_steps=6 \
    --config.sample.eval_num_steps=28 \
    --config.sample.guidance_scale=3.5 \
    --config.sample.noise_level="$NOISE_LEVEL" \
    --config.sample.ode_rate=1.0 \
    --config.sample.langevin_coeff_type=base_const \
    --config.sample.log_mean=True \
    --config.adv_aggregation="$ADVANTAGE_AGGREGATION" \
    --config.train.learning_rate=3e-4 \
    --config.train.beta=0 \
    --config.train.clip_range=1e-4 \
    --config.train.timestep_fraction=0.99 \
    --config.train.correction_step_grad_mode=train \
    "$@"
