<h1 align="center"> LC-GRPO:<br>Bridging Train-Inference Gap for Flow-Based GRPO with Langevin Correction </h1>

Official implementation for the text-to-image experiments in [LC-GRPO](https://arxiv.org/abs/2608.05600).

## 🚀 Quick Start

This repository is based on [Flow-GRPO](https://github.com/yifan123/flow_grpo).

### 1. Installation

Clone this repository and install packages.

```bash
conda create -n lc_grpo python=3.10.16
conda activate lc_grpo
pip install -e .
```

### 2. Models and Checkpoints

The launchers use the Hugging Face model IDs below. The models are downloaded
automatically to the Hugging Face cache; set `MODEL` to a local checkpoint path
if you have downloaded one in advance.

**Models**

* **SD3.5-Medium**: `stabilityai/stable-diffusion-3.5-medium`
* **FLUX.1-Dev**: `black-forest-labs/FLUX.1-dev`

**Reward Models**

* **HPS-v2.1**:
  * `xswu/HPSv2` (`HPS_v2.1_compressed.pt`)
  * `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` (`open_clip_pytorch_model.bin`)
* **CLIPScore**: `openai/clip-vit-large-patch14`
* **OCR**: PaddleOCR English model

### 3. Reward Setup

Install the dependencies for the reward setting you plan to use. HPS+CLIP uses
the HPS-v2.1 setup below; CLIPScore is already supported by the packages installed
with this repository.

#### HPS-v2.1

The HPS reward implementation requires both checkpoints at the exact paths below:

```bash
pip install hpsv2
mkdir -p pretrained/hps_ckpt
huggingface-cli download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir pretrained/hps_ckpt
huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin --local-dir pretrained/hps_ckpt
```

#### OCR

Please install paddle-ocr:

```bash
pip install paddlepaddle-gpu==2.6.2
pip install paddleocr==2.9.1
pip install python-Levenshtein
```

Then, pre-download the model using the Python command line:

```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)
```

### 4. Training

Supported model and reward settings:

- SD3.5-Medium: `ocr`, `hps`, or `hps_and_clip`
- FLUX.1-Dev: `hps` or `hps_and_clip`

All hyperparameters in these launchers follow the settings reported in the paper.
Set `TASK` to the desired reward setting.

```bash
# LC-GRPO
TASK=hps bash scripts/single_node/sd3_lc_grpo.sh
TASK=hps bash scripts/single_node/flux_lc_grpo.sh

# Flow-GRPO baseline
TASK=hps bash scripts/single_node/sd3_flow_grpo.sh
TASK=hps bash scripts/single_node/flux_flow_grpo.sh
```

For multi-node training, the scripts default to two nodes with four GPUs each.
Run the matching script on each machine, using the same rank-0 address and that
machine's rank:

```bash
export MASTER_ADDR=10.0.0.1
TASK=hps bash scripts/multi_node/sd3_lc_grpo.sh 0
TASK=hps bash scripts/multi_node/sd3_lc_grpo.sh 1
```

## 🙏 Acknowledgement

This repository is built upon [Flow-GRPO](https://github.com/yifan123/flow_grpo), which is based on [ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch) and [diffusers](https://github.com/huggingface/diffusers). We thank the Flow-GRPO authors for open-sourcing their excellent work and all the authors of these projects for their valuable contributions.

## 📚 Citation

If LC-GRPO helps your work, we would be grateful if you cited our paper:

```bibtex
@article{guo2026lc,
  title={LC-GRPO: Bridging Train-Inference Gap for Flow-Based GRPO with Langevin Correction},
  author={Guo, Yingqing and Yuan, Hui and He, Zijian and Wang, Mengdi and Ding, Zheng},
  journal={arXiv preprint arXiv:2608.05600},
  year={2026}
}
```
