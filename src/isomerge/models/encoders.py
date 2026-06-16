"""Unified encoder wrappers (plan section 1.1 backbones + Arm B ViT-S).

Every wrapper is an nn.Module with:
  .embed_dim        -- pooled embedding dimension
  .img_size         -- expected square input resolution
  .norm_mean/std    -- input normalization
  .forward(images)  -- (B, 3, H, W) -> (B, D) pooled embedding
  .lora_targets()   -- module names of the attention qkv + output-projection
                       linears in every block (the LoRA target set)

Registry keys: mae_l, dinov2_l, vjepa2_l, ijepa_h, lejepa_s, toy.
toy is a scratch ViT-tiny at 64 px for CPU tests; lejepa_s loads an Arm B
checkpoint produced by isomerge.pretrain.
"""

import re

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TimmEncoder(nn.Module):
    """timm ViT with mean-pooled patch tokens."""

    def __init__(self, model_name: str, pretrained: bool = True,
                 img_size: int = 224, checkpoint: str = None):
        super().__init__()
        import timm
        self.net = timm.create_model(model_name, pretrained=pretrained,
                                     num_classes=0, img_size=img_size)
        if checkpoint:
            sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
            sd = sd.get("encoder", sd)
            missing, unexpected = self.net.load_state_dict(sd, strict=False)
            assert not missing, f"missing keys loading {checkpoint}: {missing[:5]}"
        self.embed_dim = self.net.num_features
        self.img_size = img_size
        cfg = self.net.pretrained_cfg or {}
        self.norm_mean = cfg.get("mean", IMAGENET_MEAN)
        self.norm_std = cfg.get("std", IMAGENET_STD)

    def forward(self, x):
        feats = self.net.forward_features(x)        # (B, T, D)
        n_prefix = getattr(self.net, "num_prefix_tokens", 1)
        return feats[:, n_prefix:].mean(dim=1)      # mean over patch tokens

    def lora_targets(self):
        return _match_targets(self, r"net\.blocks\.\d+\.attn\.(qkv|proj)$")


class HFEncoder(nn.Module):
    """HuggingFace vision encoder (I-JEPA; V-JEPA 2 via frame replication)."""

    def __init__(self, hf_id: str, img_size: int, video: bool = False,
                 n_frames: int = 2):
        super().__init__()
        from transformers import AutoModel
        self.net = AutoModel.from_pretrained(hf_id, trust_remote_code=False)
        self.video = video
        self.n_frames = n_frames
        self.embed_dim = self.net.config.hidden_size
        self.img_size = img_size
        self.norm_mean = IMAGENET_MEAN
        self.norm_std = IMAGENET_STD

    def forward(self, x):
        if self.video:
            # single-frame replication: image -> T identical frames; the
            # JEPA predictor is not part of the representation -- skip it
            vid = x.unsqueeze(1).repeat(1, self.n_frames, 1, 1, 1)  # (B,T,3,H,W)
            out = self.net(pixel_values_videos=vid, skip_predictor=True)
        else:
            out = self.net(pixel_values=x)
        return out.last_hidden_state.mean(dim=1)

    def lora_targets(self):
        # HF attention linears: query/key/value (+ output projection) or
        # fused qkv. Encoder only -- never the V-JEPA 2 predictor stack.
        pat = (r"\.(qkv|query|key|value|q_proj|k_proj|v_proj|out_proj)$"
               r"|attention\.output\.dense$|attention\.proj$|attn\.proj$")
        return [n for n in _match_targets(self, pat) if "predictor" not in n]


def _match_targets(root: nn.Module, pattern: str):
    rx = re.compile(pattern)
    names = [n for n, m in root.named_modules()
             if isinstance(m, nn.Linear) and rx.search(n)]
    assert names, f"no LoRA targets matched {pattern!r}"
    return names


def build_encoder(key: str, lejepa_checkpoint: str = None) -> nn.Module:
    if key == "mae_l":
        return TimmEncoder("vit_large_patch16_224.mae")
    if key == "dinov2_l":
        return TimmEncoder("vit_large_patch14_dinov2.lvd142m", img_size=224)
    if key == "vjepa2_l":
        return HFEncoder("facebook/vjepa2-vitl-fpc64-256", img_size=256,
                         video=True)
    if key == "ijepa_h":
        return HFEncoder("facebook/ijepa_vith14_1k", img_size=224)
    if key == "lejepa_s":
        assert lejepa_checkpoint, "lejepa_s needs --lejepa-checkpoint"
        return TimmEncoder("vit_small_patch16_224", pretrained=False,
                           checkpoint=lejepa_checkpoint)
    if key == "toy":
        return TimmEncoder("vit_tiny_patch16_224", pretrained=False,
                           img_size=64)
    raise KeyError(f"unknown encoder {key!r}")
