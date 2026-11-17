"""Compare co-author's original custom_svd.py with the reconstructed WindowPool/PoolBiFPN."""
import importlib.util
import sys

import torch

from ultralytics.nn.modules.custom import EfficientNetBackbone, PoolBiFPN
from ultralytics.nn.modules.custom_pool import WindowPool

spec = importlib.util.spec_from_file_location("orig", sys.argv[1])
orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orig)
torch.manual_seed(0)
n = lambda m: sum(p.numel() for p in m.parameters())

# 1. pooling: same tensor, several shapes incl. odd sizes and padding windows
o, w = orig.SVDPooling(3, 2, 1), WindowPool("sigma1_svd", "strict")
for shape in [(2, 8, 20, 20), (1, 96, 40, 40), (2, 4, 7, 9)]:
    x = torch.randn(shape)
    a, b = o(x), w(x)
    print("pool", shape, tuple(a.shape) == tuple(b.shape), "max|d| =", (a - b).abs().max().item())
xh = torch.randn(1, 8, 20, 20).half()
print("orig half -> maxpool fallback:", torch.equal(o(xh), torch.nn.functional.max_pool2d(xh, 3, 2, 1)))

# 2. parameter counts
bo, bn = orig.EfficientNetB0Backbone(), EfficientNetBackbone("b0")
fo, fn = orig.BiFPN([24, 40, 112, 320], 96, 2), PoolBiFPN(96, 2, "sigma1_svd", "b0", "strict")
print("params backbone orig/ours:", n(bo), n(bn), "| bifpn orig/ours:", n(fo), n(fn))

# 3. BiFPN forward with identical weights
sd = {}
for i, lat in enumerate(fn.lateral):
    sd[f"in_convs.{i}.weight"], sd[f"in_convs.{i}.bias"] = lat.weight, lat.bias
for r, layer in enumerate(fn.layers):
    for grp, src in (("td_fusions", layer.td), ("out_fusions", layer.bu)):
        for j, f in enumerate(src):
            with torch.no_grad():                         # random non-trivial weights
                f.w.uniform_(0.2, 2.0)
                f.bn.running_mean.normal_(); f.bn.running_var.uniform_(0.5, 2); f.bn.weight.normal_(); f.bn.bias.normal_()
            p = f"{grp}.{r}.{j}."
            sd[p + "weights"] = f.w
            sd[p + "conv.0.weight"], sd[p + "conv.1.weight"] = f.dw.weight, f.pw.weight
            for k in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                sd[p + "conv.2." + k] = getattr(f.bn, k)
missing = fo.load_state_dict({k: v.detach().clone() for k, v in sd.items()}, strict=True)
fo.eval(); fn.eval()
feats = [torch.randn(1, c, s, s) for c, s in zip((24, 40, 112, 320), (160, 80, 40, 20))]
with torch.no_grad():
    yo, yn = fo(feats), fn(feats)
for i, (a, b) in enumerate(zip(yo, yn)):
    print(f"bifpn P{i+2}", tuple(a.shape), "max|d| =", (a - b).abs().max().item(), "rel =", ((a - b).abs().max() / a.abs().max()).item())
# backbone: same timm model and weights?
print("backbone same features:", bo.model.feature_info.channels(), bn.body.feature_info.channels(),
      all(torch.equal(p, q) for p, q in zip(bo.model.state_dict().values(), bn.body.state_dict().values())))
