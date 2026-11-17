"""EfficientNet backbone + BiFPN neck with an explicit 3x3/s2/p1 pooling operator (YOLOv8-EPB-Lite family).

Reconstructed 2026-10-08 from the project documentation (the original files were lost):
  - EfficientNet-B0 (timm, ImageNet weights), features_only, out_indices (1, 2, 3, 4) -> P2..P5, channels 24/40/112/320;
  - BiFPN on 4 levels P2..P5, width 96 (Lite) / 128 (EPB), 2 repeats;
  - fusion node: fast normalised weighted sum -> SiLU -> depthwise 3x3 -> pointwise 1x1 -> BatchNorm (no activation after BN);
  - bottom-up downsampling = WindowPool(kind) 3x3/s2/p1 (3 calls per repeat: P2->P3, P3->P4, P4->P5),
    kind 'max' = model Lite, 'sigma1_svd' = model Lite-SVD (article), 'sigma1_fast' = Lite-SVD-F.
If the co-author's original custom.py / custom_svd.py are available, they take precedence: compare the parameter
count (Lite = Lite-SVD = 4 556 695 in the article) and the layer structure with this file before training.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .custom_pool import WindowPool

_BACKBONES = {"b0": ("efficientnet_b0", (24, 40, 112, 320)), "b4": ("efficientnet_b4", (32, 56, 160, 448))}


class EfficientNetBackbone(nn.Module):
    """timm EfficientNet, returns [P2, P3, P4, P5] (strides 4, 8, 16, 32)."""

    def __init__(self, variant="b0", pretrained=True):
        super().__init__()
        name, self.channels = _BACKBONES[variant]
        self.body = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        assert tuple(self.body.feature_info.channels()) == self.channels, self.body.feature_info.channels()

    def forward(self, x):
        return self.body(x)


class WeightedFeatureFusion(nn.Module):
    """BiFPN node: sum(w_i * x_i) / (sum(w) + eps), w = relu(param) -> SiLU -> DW 3x3 -> PW 1x1 -> BN."""

    def __init__(self, c, n_inputs, eps=1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n_inputs))
        self.eps = eps
        self.act = nn.SiLU()
        self.dw = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)

    def forward(self, xs):
        w = F.relu(self.w)
        x = sum(wi * xi for wi, xi in zip(w, xs)) / (w.sum() + self.eps)
        return self.bn(self.pw(self.dw(self.act(x))))


class BiFPNLayer(nn.Module):
    """One BiFPN repeat on [P2, P3, P4, P5] (all with c channels)."""

    def __init__(self, c, pool):
        super().__init__()
        self.pool = pool                                   # shared, parameter-free
        self.td = nn.ModuleList(WeightedFeatureFusion(c, 2) for _ in range(3))   # P4_td, P3_td, P2_out
        self.bu = nn.ModuleList([WeightedFeatureFusion(c, 3), WeightedFeatureFusion(c, 3), WeightedFeatureFusion(c, 2)])

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="nearest")

    def forward(self, feats):
        p2, p3, p4, p5 = feats
        p4_td = self.td[0]([p4, self._up(p5, p4)])
        p3_td = self.td[1]([p3, self._up(p4_td, p3)])
        p2_out = self.td[2]([p2, self._up(p3_td, p2)])
        p3_out = self.bu[0]([p3, p3_td, self.pool(p2_out)])      # pooling call 1: P2 -> P3
        p4_out = self.bu[1]([p4, p4_td, self.pool(p3_out)])      # pooling call 2: P3 -> P4
        p5_out = self.bu[2]([p5, self.pool(p4_out)])             # pooling call 3: P4 -> P5
        return [p2_out, p3_out, p4_out, p5_out]


class PoolBiFPN(nn.Module):
    """Lateral 1x1 conv (with bias) to c channels + `repeats` BiFPN layers; bottom-up downsampling = WindowPool(kind).

    YAML args: [c, repeats, kind, backbone_variant, fallback]
    """

    def __init__(self, c=96, repeats=2, kind="max", backbone="b0", fallback="strict"):
        super().__init__()
        in_ch = _BACKBONES[backbone][1]
        self.lateral = nn.ModuleList(nn.Conv2d(ci, c, 1) for ci in in_ch)   # 1x1 conv with bias, no BN
        self.pool = WindowPool(kind, fallback)
        self.layers = nn.ModuleList(BiFPNLayer(c, self.pool) for _ in range(repeats))

    def forward(self, feats):
        x = [lat(f) for lat, f in zip(self.lateral, feats)]
        for layer in self.layers:
            x = layer(x)
        return x
