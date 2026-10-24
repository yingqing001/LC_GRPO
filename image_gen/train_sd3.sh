#!/usr/bin/env bash
# LC-GRPO paper reproduction launcher for SD3.5-Medium image generation.
# Paper: https://arxiv.org/abs/2608.05600
#
# Usage:
#   TASK=hps bash sd_general.sh
#   TASK=hps_and_clip bash sd_general.sh
#   TASK=ocr bash sd_general.sh
#
# Extra arguments are forwarded to the training script, for example:
#   TASK=ocr bash sd_general.sh --config.run_name=my-run

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

TASK="${TASK:-hps}"
NUM_GPUS="${NUM_GPUS:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29501}"
MODEL="${MODEL:-stabilityai/stable-diffusion-3.5-medium}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-scripts/accelerate_configs/multi_gpu.yaml}"

# The public configs below reproduce the paper's 48 prompts per epoch and
# 24 generations per prompt on 8 GPUs.
if [[ "$NUM_GPUS" != "8" ]]; then
    echo "This reproduction launcher requires NUM_GPUS=8; got NUM_GPUS=$NUM_GPUS." >&2
    echo "For another GPU count, add a matching preset in config/grpo.py." >&2
    exit 2
fi

case "$TASK" in
    hps)
        CONFIG_NAME="hps_sd3_correct_8gpu"
        NOISE_LEVEL="0.7"
        KL_BETA="0"
        ADVANTAGE_AGGREGATION="reward_sum"
        ;;
    hps_and_clip)
        CONFIG_NAME="hps_and_clip_sd3_correct_8gpu"
        NOISE_LEVEL="0.7"
        KL_BETA="0"
        ADVANTAGE_AGGREGATION="per_reward_adv"
        ;;
    ocr)
        CONFIG_NAME="ocr_sd3_correct_8gpu"
        NOISE_LEVEL="0.8"
        KL_BETA="0.004"
        ADVANTAGE_AGGREGATION="reward_sum"
        ;;
    *)
        echo "Unsupported TASK=$TASK. Choose hps, hps_and_clip, or ocr." >&2
        exit 2
        ;;
esac

RUN_NAME="${RUN_NAME:-lc-grpo-sd35m-${TASK}}"
SAVE_DIR="${SAVE_DIR:-logs/sd3/${TASK}/${RUN_NAME}}"

# SD3.5-M paper settings: 512x512, 10 rollout steps, CFG scale 4.5,
# LoRA r=32/alpha=64 (set in train_sd3_langevin_correct.py), and lr=3e-4.
# CFG is deliberately disabled only inside the Langevin correction.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Launching LC-GRPO SD3.5-Medium reproduction"
echo "  task=$TASK config=$CONFIG_NAME gpus=$NUM_GPUS eta=$NOISE_LEVEL beta=$KL_BETA"
echo "  model=$MODEL output=$SAVE_DIR"

accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    --num_processes "$NUM_GPUS" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    scripts/train_sd3_langevin_correct.py \
    --config "config/grpo.py:${CONFIG_NAME}" \
    --config.pretrained.model="$MODEL" \
    --config.run_name="$RUN_NAME" \
    --config.save_dir="$SAVE_DIR" \
    --config.resolution=512 \
    --config.sample.num_steps=10 \
    --config.sample.eval_num_steps=40 \
    --config.sample.guidance_scale=4.5 \
    --config.sample.noise_level="$NOISE_LEVEL" \
    --config.sample.ode_rate=1.0 \
    --config.sample.langevin_coeff_type=base_const \
    --config.sample.log_mean=True \
    --config.adv_aggregation="$ADVANTAGE_AGGREGATION" \
    --config.train.learning_rate=3e-4 \
    --config.train.beta="$KL_BETA" \
    --config.train.clip_range=1e-4 \
    --config.train.timestep_fraction=0.99 \
    --config.train.correction_step_grad_mode=train \
    --config.train.correction_cfg=False \
    "$@"
