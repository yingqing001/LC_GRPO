"""Generate videos from a VBench prompt suite for downstream VBench evaluation.

This is a sibling of ``fastvideo/eval_hunyuan.py``: it reuses the exact same
HunyuanVideo denoising rollout (``run_eval_sample_step`` / ``sd3_time_shift``),
but instead of scoring each video with VideoAlign it simply samples
``--num_samples_per_prompt`` videos per prompt and writes them into a single
flat directory using VBench's filename convention::

    {output_dir}/eval_videos/{prompt}-{index}.mp4   index = 0 .. num_samples-1

which is exactly the layout VBench's evaluator expects (see
``VBench/prompts/README.md``). Point VBench at ``{output_dir}/eval_videos``.

Prompts come in as precomputed text embeddings (the ``LatentDataset`` layout
produced by ``data_preprocess/preprocess_hunyuan_embeddings.py``); the companion
``generate_vbench.sh`` runs that preprocessing step first so you only have to
specify a VBench prompt ``.txt`` path.
"""

import argparse
import hashlib
import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm.auto import tqdm
from accelerate.utils import set_seed
from diffusers.utils import check_min_version, export_to_video
from diffusers.video_processor import VideoProcessor

from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
)
from fastvideo.utils.communications import sp_parallel_dataloader_wrapper
from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs
from fastvideo.utils.load import load_transformer, load_vae
from fastvideo.utils.logging_ import main_print
from fastvideo.dataset.latent_rl_datasets import LatentDataset, latent_collate_function

# Reuse the exact rollout sampler from the GRPO eval/training path so generation
# and evaluation share one denoising implementation.
from fastvideo.train_grpo_hunyuan_improved import sd3_time_shift, assert_eq
from fastvideo.eval_hunyuan import run_eval_sample_step

check_min_version("0.31.0")

# VBench filenames are "{prompt}-{index}.mp4". On most filesystems a single path
# component must stay under 255 bytes; clip very long prompts so export never
# fails. VBench matches on the (clipped) prompt text, so be consistent.
_MAX_NAME_BYTES = 200


def _sanitize(prompt):
    """Turn a prompt into a safe single-path-component filename stem.

    VBench uses the raw prompt as the filename, but a stray '/' would create a
    spurious subdirectory, so we only neutralize path separators and clip length.
    """
    stem = prompt.strip().replace("/", "_").replace("\x00", "")
    encoded = stem.encode("utf-8")[:_MAX_NAME_BYTES]
    # Avoid splitting a multibyte char when clipping.
    return encoded.decode("utf-8", "ignore")


def _seed_for(prompt, index, base_seed):
    """Reproducible, prompt- and index-dependent seed.

    VBench asks for a *different* random seed per sampled video (for diversity)
    while keeping sampling reproducible. A stable hash of the prompt gives every
    prompt its own seed family that does not depend on GPU count / shard order.
    """
    h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return (base_seed + int(h[:8], 16) + index) % (2**31)


@torch.no_grad()
def generate_for_prompt(
    args,
    device,
    transformer,
    vae,
    video_processor,
    encoder_hidden_states,  # [1, L, D]
    encoder_attention_mask,  # [1, L]
    prompt,
    video_dir,
):
    """Sample ``args.num_samples_per_prompt`` videos for one prompt and save them
    as ``{prompt}-{index}.mp4`` in ``video_dir``."""
    w, h, t = args.w, args.h, args.t
    sample_steps = args.sampling_steps
    sigma_schedule = sd3_time_shift(args.shift, torch.linspace(1, 0, sample_steps + 1))
    assert_eq(len(sigma_schedule), sample_steps + 1, "sigma_schedule length")

    SPATIAL_DOWNSAMPLE = 8
    TEMPORAL_DOWNSAMPLE = 4
    IN_CHANNELS = 16
    latent_t = ((t - 1) // TEMPORAL_DOWNSAMPLE) + 1
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    rank = int(os.environ["RANK"])
    stem = _sanitize(prompt)

    for index in range(args.num_samples_per_prompt):
        out_path = os.path.join(video_dir, f"{stem}-{index}.mp4")
        if args.skip_existing and os.path.exists(out_path):
            continue

        g = torch.Generator(device=device)
        g.manual_seed(_seed_for(prompt, index, args.eval_seed))
        latents = torch.randn(
            (1, IN_CHANNELS, latent_t, latent_h, latent_w),
            generator=g,
            device=device,
            dtype=torch.bfloat16,
        )

        progress_bar = tqdm(
            range(0, sample_steps), desc="Sampling", leave=False, disable=(rank != 0)
        )
        latents = run_eval_sample_step(
            args,
            latents,
            progress_bar,
            sigma_schedule,
            transformer,
            encoder_hidden_states,
            encoder_attention_mask,
        )

        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                frames = vae.decode(latents, return_dict=False)[0]
                video = video_processor.postprocess_video(frames)[0]
        export_to_video(video, out_path, fps=args.fps)


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    if args.seed is not None:
        set_seed(args.seed)
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    video_dir = os.path.join(args.output_dir, "eval_videos")
    if rank <= 0:
        os.makedirs(video_dir, exist_ok=True)

    main_print(
        f"--> loading transformer ({args.model_type}) from {args.pretrained_model_name_or_path}"
    )
    transformer = load_transformer(
        args.model_type,
        args.dit_model_name_or_path,
        args.pretrained_model_name_or_path,
        torch.float32 if args.master_weight_type == "fp32" else torch.bfloat16,
    )
    fsdp_kwargs, _ = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        False,
        args.use_cpu_offload,
        args.master_weight_type,
    )
    transformer = FSDP(transformer, **fsdp_kwargs)
    transformer.eval()
    main_print("--> model loaded")

    vae, _, _ = load_vae(args.model_type, args.vae_model_path)
    vae.enable_tiling()
    video_processor = VideoProcessor(vae_scale_factor=8)

    eval_dataset = LatentDataset(args.data_json_path, args.t, cfg_rate=0.0)
    # shuffle=False so every prompt is generated deterministically; padding from
    # DistributedSampler just re-generates a few prompts on the last round, which
    # only overwrites identical files (same prompt -> same name/seed).
    sampler = DistributedSampler(
        eval_dataset, rank=rank, num_replicas=world_size, shuffle=False
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        drop_last=False,
    )
    loader = sp_parallel_dataloader_wrapper(
        eval_dataloader,
        device,
        1,
        args.sp_size,
        args.train_sp_batch_size,
    )

    main_print("***** Generating videos for VBench *****")
    main_print(f"  Num prompts = {len(eval_dataset)}")
    main_print(f"  Samples per prompt = {args.num_samples_per_prompt}")
    main_print(
        f"  Sampling steps = {args.sampling_steps}  eta = {args.eta}  shift = {args.shift}"
    )
    main_print(f"  Output dir = {video_dir}")

    num_batches = len(eval_dataloader)
    start_time = time.time()
    for _ in tqdm(range(num_batches), desc="Prompts", disable=(rank != 0)):
        encoder_hidden_states, encoder_attention_mask, caption = next(loader)
        prompt = caption[0] if not isinstance(caption, str) else caption
        generate_for_prompt(
            args,
            device,
            transformer,
            vae,
            video_processor,
            encoder_hidden_states,
            encoder_attention_mask,
            prompt,
            video_dir,
        )

    dist.barrier()
    main_print(
        f"***** Done. {len(eval_dataset)} prompts x {args.num_samples_per_prompt} "
        f"samples written to {video_dir} in {time.time() - start_time:.0f}s *****"
    )
    main_print(f"Run VBench with --videos_path {video_dir}")
    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="hunyuan_hf")
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument("--output_dir", type=str, default="outputs/vbench_gen")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--train_sp_batch_size", type=int, default=1)
    parser.add_argument("--fsdp_sharding_startegy", default="full")
    parser.add_argument("--use_cpu_offload", action="store_true")
    parser.add_argument("--master_weight_type", type=str, default="bf16")

    # generation / sampling
    parser.add_argument("--h", type=int, default=480)
    parser.add_argument("--w", type=int, default=480)
    parser.add_argument("--t", type=int, default=53)
    parser.add_argument("--sampling_steps", type=int, default=28)
    parser.add_argument("--eta", type=float, default=0.0,
                        help="0 = deterministic ODE sampling; >0 = stochastic SDE rollout.")
    parser.add_argument("--sde_type", type=str, default="sde", choices=["sde", "cps"])
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--eval_seed", type=int, default=0,
                        help="base seed; per-video seed = base + hash(prompt) + index.")

    # VBench sampling convention
    parser.add_argument("--num_samples_per_prompt", type=int, default=5,
                        help="VBench samples 5 videos per prompt (25 for temporal_flickering).")
    parser.add_argument("--skip_existing", action="store_true",
                        help="skip a (prompt, index) whose .mp4 already exists (resume).")

    args = parser.parse_args()
    main(args)
