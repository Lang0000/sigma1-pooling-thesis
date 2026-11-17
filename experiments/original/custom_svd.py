# ultralytics/nn/modules/custom_svd.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List


class SVDPooling(nn.Module):

    def __init__(self, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # x: [Batch, Channel, Height, Width]
        b, c, h, w = x.shape

        # 1. Застосовуємо Unfold з Padding
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        # 2. Розраховуємо вихідні розміри (формула з урахуванням padding)
        h_out = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (w + 2 * self.padding - self.kernel_size) // self.stride + 1

        patches = patches.view(b, c, self.kernel_size * self.kernel_size, -1)

        # Транспонуємо для SVD: [B*C*L, k, k]
        matrix_patches = patches.permute(0, 1, 3, 2).reshape(-1, self.kernel_size, self.kernel_size)

        # 3. Виконуємо SVD розклад
        try:
            # torch.linalg.svd повертає U, S, Vh. Нас цікавить S (Singular values)
            _, S, _ = torch.linalg.svd(matrix_patches)

            # 4. Беремо найбільше сингулярне число (S[0])
            largest_singular_value = S[:, 0]

            # 5. Відновлюємо форму [B, C, H_out, W_out]
            out = largest_singular_value.view(b, c, h_out, w_out)
            return out

        except RuntimeError:
            return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)


class Index(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]


class EfficientNetB0Backbone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, features_only=True, in_chans=in_channels,
                                       out_indices=(1, 2, 3, 4))
        self.out_channels = self.model.feature_info.channels()

    def forward(self, x):
        return self.model(x)


class WeightedFeatureFusion(nn.Module):
    def __init__(self, num_inputs, out_channels, epsilon=0.0001):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)
        self.epsilon = epsilon
        self.swish = nn.SiLU()
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels, bias=False),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, inputs: List[torch.Tensor]):
        w = torch.relu(self.weights)
        normalized_weights = w / (torch.sum(w) + self.epsilon)
        weighted_sum = sum(w_i * inp for w_i, inp in zip(normalized_weights, inputs))
        return self.conv(self.swish(weighted_sum))


# --- BiFPN with SVD Pooling ---
class BiFPN(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int, num_repeats: int):
        super().__init__()
        self.num_repeats = num_repeats
        self.in_convs = nn.ModuleList([nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list])
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # !!! CHANGE: Додано padding=1, щоб розміри збігалися з YOLO !!!
        self.downsample = SVDPooling(kernel_size=3, stride=2, padding=1)

        self.td_fusions, self.out_fusions = nn.ModuleList(), nn.ModuleList()
        n = len(in_channels_list)
        num_fusions = n - 1
        for _ in range(num_repeats):
            self.td_fusions.append(nn.ModuleList([WeightedFeatureFusion(2, out_channels) for _ in range(num_fusions)]))
            out_fusion_list = []
            for i in range(num_fusions):
                if i == num_fusions - 1:
                    out_fusion_list.append(WeightedFeatureFusion(2, out_channels))
                else:
                    out_fusion_list.append(WeightedFeatureFusion(3, out_channels))
            self.out_fusions.append(nn.ModuleList(out_fusion_list))

    def forward(self, features: List[torch.Tensor]):
        if len(features) == 4:
            p2, p3, p4, p5 = (conv(f) for conv, f in zip(self.in_convs, features))
            initial_features = [p2, p3, p4, p5]
            for i in range(self.num_repeats):
                td, out = self.td_fusions[i], self.out_fusions[i]
                p2, p3, p4, p5 = initial_features
                p4_td = td[0]([p4, self.upsample(p5)])
                p3_td = td[1]([p3, self.upsample(p4_td)])
                p2_out = td[2]([p2, self.upsample(p3_td)])
                p3_out = out[0]([p3, p3_td, self.downsample(p2_out)])
                p4_out = out[1]([p4, p4_td, self.downsample(p3_out)])
                p5_out = out[2]([p5, self.downsample(p4_out)])
                initial_features = [p2_out, p3_out, p4_out, p5_out]
            return initial_features
        elif len(features) == 3:
            p3, p4, p5 = (conv(f) for conv, f in zip(self.in_convs, features))
            initial_features = [p3, p4, p5]
            for i in range(self.num_repeats):
                td, out = self.td_fusions[i], self.out_fusions[i]
                p3, p4, p5 = initial_features
                p4_td = td[0]([p4, self.upsample(p5)])
                p3_out = td[1]([p3, self.upsample(p4_td)])
                p4_out = out[0]([p4, p4_td, self.downsample(p3_out)])
                p5_out = out[1]([p5, self.downsample(p4_out)])
                initial_features = [p3_out, p4_out, p5_out]
            return initial_features