source activate dance-grpo

# export WANDB_MODE=offline
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- knobs ----
batch_size=2          # train_batch_size == train_sp_batch_size (groups per GPU)
grpo_group_size=8     # num_generations (group size per prompt)
grad_accum_steps=$((batch_size * 4))    # gradient accumulation (in samples)
bestofn=8

prompt_size=$((batch_size * 8))



coeff_type=base_const


noise_level=0.7
sampling_steps=8

reward_distibute=V

# map reward_distibute -> per-channel coefficients (equal 1/n weighting)
vq_coef=0.0; mq_coef=0.0; ta_coef=0.0
n=0
[[ $reward_distibute == *V* ]] && n=$((n+1))
[[ $reward_distibute == *M* ]] && n=$((n+1))
[[ $reward_distibute == *T* ]] && n=$((n+1))
if (( n == 0 )); then
    echo "ERROR: reward_distibute='$reward_distibute' selects no channels (need some of V/M/T)" >&2
    exit 1
fi
coef=$(awk "BEGIN{printf \"%.4f\", 1.0/$n}")
[[ $reward_distibute == *V* ]] && vq_coef=$coef
[[ $reward_distibute == *M* ]] && mq_coef=$coef
[[ $reward_distibute == *T* ]] && ta_coef=$coef

# vq_coef=0.4; mq_coef=0.4; ta_coef=0.2

echo "reward_distibute=$reward_distibute -> vq_coef=$vq_coef mq_coef=$mq_coef ta_coef=$ta_coef"


# run name encodes the settings so A/B runs don't clobber each other's output_dir / wandb run
name=correct_${coeff_type}_noise${noise_level}_sample${sampling_steps}_prompt${prompt_size}_${reward_distibute}

# run name encodes the settings so A/B runs don't clobber each other's output_dir
# name=baseline_orig_sde_test

torchrun --nproc_per_node=8  --master_port=29500 \
    fastvideo/train_grpo_hunyuan_orig_correct.py \
    --seed 42 \
    --model_type "hunyuan_hf" \
    --pretrained_model_name_or_path data/HunyuanVideo \
    --vae_model_path data/HunyuanVideo \
    --cache_dir data/.cache \
    --data_json_path data/rl_embeddings/videos2caption.json \
    --validation_prompt_dir data/Mochi-Black-Myth/validation \
    --gradient_checkpointing \
    --train_batch_size $batch_size \
    --sp_size 1 \
    --train_sp_batch_size $batch_size \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps $grad_accum_steps \
    --max_train_steps 202 \
    --learning_rate 1e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 10 \
    --validation_steps 100000000 \
    --validation_sampling_steps 8 \
    --checkpoints_total_limit 10 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --ema_decay 0.999 \
    --log_validation \
    --output_dir "outputs_res/${name}" \
    --tracker_project_name hunyuan-grpo \
    --wandb_run_name "${name}" \
    --h 480 \
    --w 480 \
    --t 53 \
    --sampling_steps $sampling_steps \
    --eta $noise_level \
    --lr_warmup_steps 0 \
    --fps 8 \
    --sampler_seed 1237 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --num_generations $grpo_group_size \
    --shift 5 \
    --use_group \
    --use_videoalign \
    --timestep_fraction 0.6 \
    --use_same_noise \
    --bestofn $bestofn \
    --coeff_type $coeff_type \
    --vq_coef $vq_coef \
    --mq_coef $mq_coef \
    --ta_coef $ta_coef
