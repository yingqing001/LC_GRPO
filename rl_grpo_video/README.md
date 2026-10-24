<h1 align="center">LC-GRPO:<br>Bridging Train-Inference Gap for Flow-Based GRPO with Langevin Correction</h1>

Official implementation for the text-to-video experiments in
[LC-GRPO](https://arxiv.org/abs/2608.05600).

This repository contains the HunyuanVideo experiments. It is built on
[DanceGRPO](https://github.com/XueZeyue/DanceGRPO) and
[FastVideo](https://github.com/hao-ai-lab/FastVideo). The text-to-image
experiments are maintained separately in the LC-GRPO image-generation codebase.

## 🚀 Quick Start

### 1. Installation

Create and activate the environment, then run the provided setup script:

```bash
conda create -n dance-grpo python=3.10.16
conda activate dance-grpo
./env_setup.sh
```

The setup script installs PyTorch 2.5.0 with CUDA 12.1, FlashAttention 2,
the FastVideo package, and the remaining project dependencies.

### 2. Models and Checkpoints

Download the following checkpoints into the expected local directories:

- [HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo) → `data/HunyuanVideo`
- [VideoAlign / VideoReward](https://huggingface.co/KwaiVGI/VideoReward) → `videoalign_ckpt`

For example, using the Hugging Face CLI:

```bash
mkdir -p data/HunyuanVideo videoalign_ckpt
huggingface-cli download hunyuanvideo-community/HunyuanVideo \
  --local-dir data/HunyuanVideo
huggingface-cli download KwaiVGI/VideoReward \
  --local-dir videoalign_ckpt
```

### 3. Training Data

The provided prompt set is `assets/video_prompts.txt`. Precompute its
HunyuanVideo text embeddings with:

```bash
bash scripts/preprocess/preprocess_hunyuan_rl_embeddings.sh
```

This script uses eight GPUs by default and writes the processed dataset to
`data/rl_embeddings`, including the manifest expected by the training launcher:

```text
data/rl_embeddings/videos2caption.json
```

To train on another prompt set, replace `assets/video_prompts.txt` or update
`--prompt_dir` in the preprocessing script.

### 4. LC-GRPO Training

The main entrypoint is:

```bash
bash hunyuan_grpo_lc_grpo.sh
```

The launcher is configured for one node with eight GPUs. Its main experiment
settings are collected near the top of the file:

- `batch_size`: prompt groups processed per GPU
- `grpo_group_size`: number of generations per prompt
- `sampling_steps`: flow-sampling steps
- `noise_level`: Langevin correction strength
- `coeff_type`: correction coefficient (`base_const` or `ddim`)
- `reward_distibute`: enabled VideoAlign reward channels (`V`, `M`, and/or `T`)

`V`, `M`, and `T` denote visual quality, motion quality, and text alignment.
When multiple channels are selected, the launcher assigns them equal weights.
Adjust the `torchrun` arguments in the launcher for a different GPU count or
distributed setup.

Training checkpoints, sampled videos, reward traces, and logs are written under
`outputs_res/<run-name>`. Weights & Biases logging is enabled by default; set
`WANDB_MODE=offline` in the launcher if online logging is not desired.

## Acknowledgements

This codebase is built upon
[DanceGRPO](https://github.com/XueZeyue/DanceGRPO),
[FastVideo](https://github.com/hao-ai-lab/FastVideo),
[diffusers](https://github.com/huggingface/diffusers), and
[DDPO-PyTorch](https://github.com/kvablack/ddpo-pytorch). We thank the authors
for releasing their work.

## Citation

If LC-GRPO helps your work, please cite:

```bibtex
@article{guo2026lc,
  title={LC-GRPO: Bridging Train-Inference Gap for Flow-Based GRPO with Langevin Correction},
  author={Guo, Yingqing and Yuan, Hui and He, Zijian and Wang, Mengdi and Ding, Zheng},
  journal={arXiv preprint arXiv:2608.05600},
  year={2026}
}
```
