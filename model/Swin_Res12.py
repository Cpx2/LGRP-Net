import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.res12 import Res12
from model.swin_transformer import swin_tiny


class Swin_Res12(nn.Module):
    """
    无参数训练的加权拼接融合骨干。

    核心思想：
    1. CNN与Swin分别在自身特征空间中进行归一化；
    2. 不进行跨骨干坐标级正交投影；
    3. 使用平方根权重控制两个分支对余弦度量的贡献；
    4. 保持输出维度为640+768=1408。
    """

    def __init__(
        self,
        dataset: str = "MiniImageNet",
        swin_weight: float | None = None,
    ):
        super().__init__()

        self.dataset = dataset

        self.cnn = Res12(
            avg_pool=True,
            drop_block="ImageNet" in dataset,
        )
        self.swin = swin_tiny()

        if swin_weight is not None:
            self.swin_weight = float(swin_weight)

        elif dataset == "TieredImageNet":
            # TieredImageNet上Fusion单独低于Swin时，
            # 应以Swin为主，仅引入少量CNN局部信息。
            self.swin_weight = 0.8

        elif dataset == "MiniImageNet":
            self.swin_weight = 0.6


        elif dataset == "FC100":

            self.swin_weight = 0.9


        elif dataset == "CIFAR-FS":

            self.swin_weight = 0.7

        else:
            self.swin_weight = 0.50

        if not 0.0 <= self.swin_weight <= 1.0:
            raise ValueError(
                "swin_weight必须位于[0, 1]范围内。"
            )

        self.cnn_weight = 1.0 - self.swin_weight

        print(
            "==> [Swin_Res12] "
            f"Dataset: {dataset}, "
            f"Swin weight: {self.swin_weight:.2f}, "
            f"CNN weight: {self.cnn_weight:.2f}"
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # ResNet-12使用84×84输入。
        x_cnn = F.interpolate(
            x,
            size=(84, 84),
            mode="area",
        )

        # 提取两个冻结骨干的原始特征。
        feat_cnn = self.cnn(
            x_cnn
        ).flatten(1)

        feat_swin = self.swin(
            x
        ).flatten(1)

        # CNN分支进行中心化和Signed Power Normalization。
        # 该处理主要用于降低CNN局部特征中的大响应偏置。
        feat_cnn = (
            feat_cnn
            - feat_cnn.mean(
                dim=-1,
                keepdim=True,
            )
        )

        feat_cnn = (
            torch.sign(feat_cnn)
            * torch.sqrt(
                torch.abs(feat_cnn) + 1e-6
            )
        )

        # 两个分支分别归一化。
        feat_cnn = F.normalize(
            feat_cnn,
            p=2,
            dim=-1,
        )

        # Swin保持原始特征几何，仅做L2归一化。
        feat_swin = F.normalize(
            feat_swin,
            p=2,
            dim=-1,
        )

        # 使用平方根权重，使融合余弦相似度等价于
        # 两个分支余弦相似度的加权和。
        cnn_scale = math.sqrt(
            self.cnn_weight
        )
        swin_scale = math.sqrt(
            self.swin_weight
        )

        fused_feat = torch.cat(
            [
                cnn_scale * feat_cnn,
                swin_scale * feat_swin,
            ],
            dim=-1,
        )

        # 理论上范数已经接近1，保留归一化以增强数值稳定性。
        fused_feat = F.normalize(
            fused_feat,
            p=2,
            dim=-1,
        )

        return fused_feat