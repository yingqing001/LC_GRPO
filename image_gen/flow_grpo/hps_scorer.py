import torch
from PIL import Image


class HPSv2Scorer(torch.nn.Module):
    def __init__(
            self,
            device,
            ckpt_path="pretrained/hps_ckpt/HPS_v2.1_compressed.pt",
            clip_path="pretrained/hps_ckpt/open_clip_pytorch_model.bin",
        ):
        super().__init__()
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        model, _, preprocess_val = create_model_and_transforms(
            "ViT-H-14", clip_path, precision="amp", device=device, jit=False,
            force_quick_gelu=False, force_custom_text=False, force_patch_dropout=False,
            force_image_size=None, pretrained_image=False, image_mean=None, image_std=None,
            light_augmentation=True, aug_cfg={}, output_dict=True,
            with_score_predictor=False, with_region_predictor=False,
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        self.model = model.to(device).eval()
        self.preprocess = preprocess_val
        self.tokenizer = get_tokenizer("ViT-H-14")
        self.device = device

    @torch.no_grad()
    def __call__(self, prompts, images):
        x = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        t = self.tokenizer(list(prompts)).to(self.device)
        with torch.amp.autocast("cuda"):
            o = self.model(x, t)
            image_features, text_features = o["image_features"], o["text_features"]
            scores = (image_features * text_features).sum(dim=-1)
        return scores.float()

# Usage example
def main():
    scorer = HPSv2Scorer(device="cuda")
    images = [
        "flow_grpo/assets/test.jpg",
    ]
    pil_images = [Image.open(img).convert("RGB") for img in images]
    prompts = [
        'A cat',
    ]
    print(scorer(prompts, pil_images))

if __name__ == "__main__":
    main()