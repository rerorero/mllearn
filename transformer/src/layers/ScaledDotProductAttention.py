import numpy as np
import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    # d_kは単語分散表現の次元数, 512など
    def __init__(self, d_k: int) -> None:
        super().__init__()
        self.d_k = d_k

    def forward(
        self,
        q: torch.Tensor,  # target: "I" in "I am a cat."
        k: torch.Tensor,  # source: [吾輩,は,猫,で,ある]
        v: torch.Tensor,  # source: [吾輩,は,猫,で,ある]
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # なぜルートdkで割るかと言うと、内積計算で大き過ぎる値があるとSoftmaxを掛けたとき、それ以外の値が0になり勾配消失してしまうため、それを防止するためです。
        # dkは単語分散表現の次元数で論文では512です。
        scalar = np.sqrt(self.d_k)
        attention_weight = torch.matmul(q, torch.transpose(k, 1, 2)) / scalar
        if mask is not None:
            if mask.dim() != attention_weight.dim():
                raise ValueError(
                    "mask.dim != attention_weight.dim, mask.dim={}, attention_weight.dim={}".format(
                        mask.dim(), attention_weight.dim()
                    )
                )
            attention_weight = attention_weight.data.masked_fill_(
                mask, -torch.finfo(torch.float).max
            )
        attention_weight = nn.functional.softmax(attention_weight, dim=2)
        return torch.matmul(attention_weight, v)

