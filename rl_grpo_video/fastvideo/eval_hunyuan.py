"""Evaluation script for HunyuanVideo.

This is the *rollout half* of the GRPO training loop in
``fastvideo/train_grpo_hunyuan_improved.py`` with all the training machinery
removed (no advantages, no PPO loss, no optimizer / lr scheduler, no backward).
It walks the prompt set once, samples a video per prompt (optionally several per
prompt via ``--num_generations``), decodes, scores each video with the
VideoAlign reward model, and reports aggregate VQ / MQ statistics across all
ranks.

The denoising path is the same code as the RL rollout: we reuse
``flux_step`` / ``sd3_time_shift`` from the training module. By default eval is
deterministic (``--eta 0`` -> pure ODE step, the noise-free special case of the
SDE rollout); pass ``--eta`` > 0 with ``--sde_type`` to evaluate the exact
stochastic distribution the policy is trained on.
"""

import argparse
import csv
import json
import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
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

# Reuse the exact rollout sampler helpers from the GRPO trainer so eval and
# training share one denoising implementation.
from fastvideo.train_grpo_hunyuan_improved import sd3_time_shift, flux_step, assert_eq

check_min_version("0.31.0")


class _ShardSampler(Sampler):
    """Yield a fixed list of dataset indices (this rank's shard), no padding."""

    def __init__(self, indices):
        self._indices = list(indices)

    def __iter__(self):
        return iter(self._indices)

    def __len__(self):
        return len(self._indices)


def run_eval_sample_step(
    args,
    z,
    progress_bar,
    sigma_schedule,
    transformer,
    encoder_hidden_states,
    encoder_attention_mask,
):
    """Denoise ``z`` to a clean latent. Same loop as the RL rollout's
    ``run_sample_step`` but we do not collect log-probs (eval needs none).

    ``--eta 0`` uses the deterministic ODE step (``grpo=False, sde_solver=False``);
    ``--eta > 0`` uses the stochastic SDE step the policy is trained with.
    """
    pred_original = None
    for i in progress_bar:
        sigma = sigma_schedule[i]
        timestep_value = int(sigma * 1000)
        timesteps = torch.full(
            [encoder_hidden_states.shape[0]],
            timestep_value,
            device=z.device,
            dtype=torch.long,
        )
        transformer.eval()
        with torch.autocast("cuda", torch.bfloat16):
            model_pred = transformer(
                hidden_states=z,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timesteps,
                guidance=torch.tensor([6018.0], device=z.device, dtype=torch.bfloat16),
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
        if args.eta > 0:
            z, pred_original, _ = flux_step(
                model_pred,
                z.to(torch.float32),
                args.eta,
                sigmas=sigma_schedule,
                index=i,
                prev_sample=None,
                grpo=True,
                sde_solver=True,
                sde_type=args.sde_type,
            )
        else:
            # Deterministic ODE step (eta=0): the noise-free special case of the
            # rollout. We inline it rather than calling flux_step because flux_step
            # unconditionally computes log(std_dev_t), which is log(0) when eta=0.
            latents_f = z.to(torch.float32)
            sigma_cur = sigma_schedule[i]
            dsigma = sigma_schedule[i + 1] - sigma_cur
            z = latents_f + dsigma * model_pred
            pred_original = latents_f - sigma_cur * model_pred
        z = z.to(torch.bfloat16)
    latents = pred_original.to(torch.float32) / 0.476986
    return latents


@torch.no_grad()
def sample_and_score(
    args,
    device,
    transformer,
    vae,
    inferencer,
    encoder_hidden_states,
    encoder_attention_mask,
    caption,
    sample_ids,
    results,
    is_padding=False,
):
    """Generate + decode + score one dataloader batch (after group expansion).

    ``sample_ids[k]`` is the globally-unique id of the k-th video in the batch
    (``prompt_idx * num_gen + gen``); it seeds the init noise and names the
    output file so eval is reproducible and prompts are never double-counted.

    ``is_padding=True`` marks a batch that exists only to keep this rank's
    FSDP forward count in lockstep with the other ranks (the prompt set does not
    divide evenly across ranks). We still run the transformer rollout — that is
    the collective that must match across ranks — but skip the VAE decode,
    export and scoring, and record nothing, so padding never inflates the stats.

    Mirrors ``sample_reference_model`` in the trainer: rollout in micro-batches,
    VAE-decode in chunks, export mp4, score with VideoAlign. Appends per-video
    rows to ``results`` and returns the VQ / MQ / TA reward tensors for this batch.
    """
    w, h, t = args.w, args.h, args.t
    sample_steps = args.sampling_steps
    sigma_schedule = sd3_time_shift(args.shift, torch.linspace(1, 0, sample_steps + 1))
    assert_eq(len(sigma_schedule), sample_steps + 1, "sigma_schedule length")

    B = encoder_hidden_states.shape[0]
    SPATIAL_DOWNSAMPLE = 8
    TEMPORAL_DOWNSAMPLE = 4
    IN_CHANNELS = 16
    latent_t = ((t - 1) // TEMPORAL_DOWNSAMPLE) + 1
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    sample_batch_size = args.sample_batch_size
    assert B % sample_batch_size == 0, (
        f"B ({B}) must be divisible by sample_batch_size ({sample_batch_size})"
    )
    batch_indices = torch.chunk(torch.arange(B), B // sample_batch_size)

    rank = int(os.environ["RANK"])
    video_dir = os.path.join(args.output_dir, "eval_videos")
    os.makedirs(video_dir, exist_ok=True)
    video_processor = VideoProcessor(vae_scale_factor=8)
    vae.enable_tiling()

    vq_rewards, mq_rewards, ta_rewards = [], [], []
    rollout_bar = tqdm(total=B, desc="Eval rollout", disable=(rank != 0), leave=False)
    for index, batch_idx in enumerate(batch_indices):
        cur_bs = len(batch_idx)
        batch_caption = [caption[i] for i in batch_idx]

        # Fixed seed per global sample id keeps eval reproducible across runs.
        cur_latents = []
        for b in batch_idx.tolist():
            g = torch.Generator(device=device)
            g.manual_seed(args.eval_seed + sample_ids[b])
            cur_latents.append(
                torch.randn(
                    (1, IN_CHANNELS, latent_t, latent_h, latent_w),
                    generator=g,
                    device=device,
                    dtype=torch.bfloat16,
                )
            )
        cur_latents = torch.cat(cur_latents, dim=0)

        progress_bar = tqdm(
            range(0, sample_steps), desc="Sampling", leave=False, disable=(rank != 0)
        )
        latents = run_eval_sample_step(
            args,
            cur_latents,
            progress_bar,
            sigma_schedule,
            transformer,
            encoder_hidden_states[batch_idx],
            encoder_attention_mask[batch_idx],
        )

        # Padding batch: the transformer forward above (the FSDP collective that
        # must stay in lockstep across ranks) is done; everything below is local
        # and unscored, so skip it and record nothing.
        if is_padding:
            rollout_bar.update(cur_bs)
            continue

        videos = []
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for c in range(0, latents.shape[0], args.decode_batch_size):
                    chunk = vae.decode(
                        latents[c : c + args.decode_batch_size], return_dict=False
                    )[0]
                    videos.extend(video_processor.postprocess_video(chunk))

        for b in range(cur_bs):
            global_idx = sample_ids[index * sample_batch_size + b]
            video_path = os.path.join(video_dir, f"hunyuan_rank{rank}_{global_idx}.mp4")
            export_to_video(videos[b], video_path, fps=args.fps)
            vq, mq, ta = -1.0, -1.0, -1.0
            if args.use_videoalign:
                try:
                    reward = inferencer.reward(
                        [os.path.abspath(video_path)],
                        [batch_caption[b]],
                        use_norm=True,
                    )
                    vq = float(reward[0]["VQ"])
                    mq = float(reward[0]["MQ"])
                    ta = float(reward[0]["TA"])
                except Exception as e:
                    main_print(f"[warn] reward failed for {video_path}: {e}")
            vq_rewards.append(vq)
            mq_rewards.append(mq)
            ta_rewards.append(ta)
            results.append(
                {
                    "rank": rank,
                    "global_idx": global_idx,
                    "caption": batch_caption[b],
                    "video_path": video_path,
                    "vq": vq,
                    "mq": mq,
                    "ta": ta,
                }
            )
        rollout_bar.update(cur_bs)
    rollout_bar.close()

    return (
        torch.tensor(vq_rewards, device=device, dtype=torch.float32),
        torch.tensor(mq_rewards, device=device, dtype=torch.float32),
        torch.tensor(ta_rewards, device=device, dtype=torch.float32),
    )


def gather_tensor(tensor):
    """All-gather a 1-D tensor whose length may differ across ranks.

    We shard the prompt set without padding (see ``_ShardSampler``), so ranks
    can hold different numbers of videos (e.g. 4 vs 3). ``all_gather`` requires
    equal-sized buffers, so we exchange per-rank counts first, pad every rank up
    to the max, gather, then trim each slice back to its true length.
    """
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    local_size = torch.tensor([tensor.numel()], device=tensor.device, dtype=torch.long)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    sizes = [int(s.item()) for s in sizes]
    max_size = max(sizes) if sizes else 0
    if max_size == 0:
        return tensor
    padded = torch.zeros(max_size, device=tensor.device, dtype=tensor.dtype)
    padded[: tensor.numel()] = tensor
    gathered = [torch.zeros(max_size, device=tensor.device, dtype=tensor.dtype)
                for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    return torch.cat([g[:s] for g, s in zip(gathered, sizes)], dim=0)


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

    inferencer = None
    if args.use_videoalign:
        from fastvideo.models.videoalign.inference import VideoVLMRewardInference

        inferencer = VideoVLMRewardInference(
            args.videoalign_ckpt, device=f"cuda:{device}", dtype=torch.bfloat16
        )

    main_print(f"--> loading transformer ({args.model_type}) from {args.pretrained_model_name_or_path}")
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

    eval_dataset = LatentDataset(args.data_json_path, args.t, cfg_rate=0.0)
    # Shard the prompt set across ranks. Each prompt is assigned to exactly one
    # rank (range(rank, N, world_size)), so it is evaluated once and never
    # double-counted -- unlike torch's DistributedSampler, which repeats prompts
    # to pad the global count (e.g. 31 prompts on 8 ranks -> prompt 0 twice).
    #
    # The prompt set rarely divides evenly across ranks (31 / 8 -> 4,4,4,4,4,4,4,3).
    # FSDP issues a parameter all-gather on every transformer forward, so all ranks
    # MUST run the same number of forwards or the collectives deadlock. We therefore
    # pad each short rank up to the max batch count with a repeated prompt, and flag
    # those batches as padding: they run the forward (to stay in lockstep) but are
    # not decoded, scored, or counted. The size-aware gather_tensor above then
    # tolerates the resulting per-rank reward-count differences.
    my_real_indices = list(range(rank, len(eval_dataset), world_size))
    per_rank = (len(eval_dataset) + world_size - 1) // world_size
    pad_value = my_real_indices[-1] if my_real_indices else 0
    n_pad = per_rank - len(my_real_indices)
    my_prompt_indices = my_real_indices + [pad_value] * n_pad
    is_padding_batch = [False] * len(my_real_indices) + [True] * n_pad
    sampler = _ShardSampler(my_prompt_indices)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=False,
    )
    loader = sp_parallel_dataloader_wrapper(
        eval_dataloader,
        device,
        args.train_batch_size,
        args.sp_size,
        args.train_sp_batch_size,
    )

    main_print("***** Running evaluation *****")
    main_print(f"  Num prompts = {len(eval_dataset)}")
    main_print(f"  Generations per prompt = {args.num_generations if args.use_group else 1}")
    main_print(f"  Sampling steps = {args.sampling_steps}  eta = {args.eta}  shift = {args.shift}")

    num_gen = args.num_generations if args.use_group else 1
    num_batches = len(eval_dataloader)
    results = []
    all_vq, all_mq, all_ta = [], [], []
    start_time = time.time()
    for bi in tqdm(range(num_batches), desc="Eval batches", disable=(rank != 0)):
        encoder_hidden_states, encoder_attention_mask, caption = next(loader)
        # Prompts in this batch, in loader order (matches my_prompt_indices).
        tb = encoder_hidden_states.shape[0]
        prompt_idx_batch = my_prompt_indices[bi * tb : bi * tb + tb]
        if num_gen > 1:
            encoder_hidden_states = torch.repeat_interleave(
                encoder_hidden_states, num_gen, dim=0
            )
            encoder_attention_mask = torch.repeat_interleave(
                encoder_attention_mask, num_gen, dim=0
            )
            if isinstance(caption, str):
                caption = [caption] * num_gen
            else:
                caption = [c for c in caption for _ in range(num_gen)]
        # Stable, globally-unique id per video: prompt_idx * num_gen + gen.
        # Used for the init-noise seed and the output filename so results are
        # reproducible and never collide across ranks. Expansion is prompt-major,
        # gen-minor, matching the repeat_interleave / caption expansion above.
        sample_ids = [p * num_gen + g for p in prompt_idx_batch for g in range(num_gen)]

        vq, mq, ta = sample_and_score(
            args,
            device,
            transformer,
            vae,
            inferencer,
            encoder_hidden_states,
            encoder_attention_mask,
            caption,
            sample_ids,
            results,
            is_padding=is_padding_batch[bi],
        )
        all_vq.append(vq)
        all_mq.append(mq)
        all_ta.append(ta)

    all_vq = torch.cat(all_vq) if all_vq else torch.zeros(0, device=device)
    all_mq = torch.cat(all_mq) if all_mq else torch.zeros(0, device=device)
    all_ta = torch.cat(all_ta) if all_ta else torch.zeros(0, device=device)
    gathered_vq = gather_tensor(all_vq)
    gathered_mq = gather_tensor(all_mq)
    gathered_ta = gather_tensor(all_ta)

    # Write per-video CSV (one file per rank) so nothing is lost to gather padding.
    csv_path = os.path.join(args.output_dir, f"eval_results_rank{rank}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "global_idx", "caption", "video_path", "vq", "mq", "ta"],
        )
        writer.writeheader()
        writer.writerows(results)

    if rank == 0:
        def _stats(gathered):
            valid = gathered[gathered > -1.0] if args.use_videoalign else gathered
            if valid.numel() == 0:
                return float("nan"), float("nan")
            return valid.mean().item(), valid.std().item()

        vq_mean, vq_std = _stats(gathered_vq)
        mq_mean, mq_std = _stats(gathered_mq)
        ta_mean, ta_std = _stats(gathered_ta)
        summary = {
            "num_videos": int(gathered_vq.numel()),
            "vq_mean": vq_mean,
            "vq_std": vq_std,
            "mq_mean": mq_mean,
            "mq_std": mq_std,
            "ta_mean": ta_mean,
            "ta_std": ta_std,
            "elapsed_sec": time.time() - start_time,
            "sampling_steps": args.sampling_steps,
            "eta": args.eta,
            "shift": args.shift,
        }
        with open(os.path.join(args.output_dir, "eval_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        main_print("***** Evaluation summary *****")
        main_print(json.dumps(summary, indent=2))

    dist.barrier()
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
    parser.add_argument("--output_dir", type=str, default="outputs/eval")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--train_sp_batch_size", type=int, default=1)
    parser.add_argument("--fsdp_sharding_startegy", default="full")
    parser.add_argument("--use_cpu_offload", action="store_true")
    parser.add_argument("--master_weight_type", type=str, default="bf16")

    # generation / sampling
    parser.add_argument("--h", type=int, default=480)
    parser.add_argument("--w", type=int, default=480)
    parser.add_argument("--t", type=int, default=53)
    parser.add_argument("--sampling_steps", type=int, default=16)
    parser.add_argument("--eta", type=float, default=0.0,
                        help="0 = deterministic ODE eval; >0 = stochastic SDE rollout.")
    parser.add_argument("--sde_type", type=str, default="sde", choices=["sde", "cps"])
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--eval_seed", type=int, default=0,
                        help="base seed for per-sample init noise (reproducible eval).")
    parser.add_argument("--sample_batch_size", type=int, default=1)
    parser.add_argument("--decode_batch_size", type=int, default=1)

    # how many videos per prompt (set --use_group for >1)
    parser.add_argument("--use_group", action="store_true", default=False)
    parser.add_argument("--num_generations", type=int, default=1)

    # reward model
    parser.add_argument("--use_videoalign", action="store_true", default=False)
    parser.add_argument("--videoalign_ckpt", type=str, default="./videoalign_ckpt")

    args = parser.parse_args()
    main(args)
