"""Models for MediTrust: ResNet-18 and DeiT-Tiny behind one wrapper.

`MediTrustModel` takes images of shape [B, 3, 64, 64] with values in [0, 1], resizes them
to 224x224 (bilinear), applies ImageNet normalisation and returns 9 logits. Doing the
resize and normalisation inside the model keeps the same preprocessing for training,
gradient-based attacks (gradients reach the 64x64 input), ONNX export and the app.
See docs/CAHIER_DES_CHARGES.md, sections 7.2, 12.1 and D2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

BACKBONES = {
    "R18": "resnet18.tv_in1k",
    "DEIT": "deit_tiny_patch16_224.fb_in1k",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224


class MediTrustModel(nn.Module):
    """[B, 3, H, W] in [0, 1] -> resize to 224 -> ImageNet normalisation -> backbone -> logits."""

    def __init__(self, backbone: nn.Module, name: str):
        super().__init__()
        self.backbone = backbone
        self.name = name
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.preprocess(x))


def build_model(name: str, num_classes: int = 9, pretrained: bool = True) -> MediTrustModel:
    """Create `R18` or `DEIT` with a new `num_classes` head (timm, ImageNet-1k weights)."""
    import timm

    if name not in BACKBONES:
        raise ValueError(f"unknown model {name!r}, expected one of {list(BACKBONES)}")
    backbone = timm.create_model(BACKBONES[name], pretrained=pretrained, num_classes=num_classes)
    cfg = backbone.pretrained_cfg
    if tuple(cfg["mean"]) != IMAGENET_MEAN or tuple(cfg["std"]) != IMAGENET_STD:
        raise ValueError(f"{BACKBONES[name]} expects other normalisation constants: {cfg['mean']}, {cfg['std']}")
    return MediTrustModel(backbone, name)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# Attention (T1). timm uses fused attention by default, which does not expose
# the attention weights. It must be disabled for analysis and ONNX export.
# --------------------------------------------------------------------------- #

def attention_modules(model: nn.Module) -> list[nn.Module]:
    """The self-attention modules of the ViT blocks (empty list for ResNet-18)."""
    blocks = getattr(model.backbone, "blocks", None)
    return [] if blocks is None else [b.attn for b in blocks]


def set_fused_attention(model: nn.Module, enabled: bool) -> None:
    for attn in attention_modules(model):
        attn.fused_attn = enabled


def manual_attention(attn: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute one timm attention block explicitly.

    Returns (output, weights), with weights = softmax(Q K^T / sqrt(d_h)) of shape
    [B, heads, N, N]. `output` must equal `attn(x)` (checked in tests).
    """
    B, N, _ = x.shape
    qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = attn.q_norm(q), attn.k_norm(k)
    weights = ((q @ k.transpose(-2, -1)) / attn.head_dim ** 0.5).softmax(dim=-1)
    out = (weights @ v).transpose(1, 2).reshape(B, N, -1)
    out = attn.norm(out)
    if getattr(attn, "gate", None) is not None:
        out = out * attn.gate(x).sigmoid()
    return attn.proj(out), weights


# --------------------------------------------------------------------------- #
# ONNX export (used for the early check in Phase 2 and for E5 in Phase 6)
# --------------------------------------------------------------------------- #

def export_onnx(model: MediTrustModel, path: str, opset: int = 17, image_size: int = 64) -> None:
    """Export to ONNX with a dynamic batch dimension. The resize and normalisation are in the graph."""
    model = model.eval().cpu()
    set_fused_attention(model, False)
    dummy = torch.rand(1, 3, image_size, image_size)
    torch.onnx.export(
        model, (dummy,), path,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset, dynamo=False,
    )
