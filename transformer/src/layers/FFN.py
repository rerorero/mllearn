
import torch
from torch import nn
from torch.nn.functional import relu


# Positio-wise Feed-Forward Network
# 2つの全結合層からなる
class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(relu(self.linear1(x)))
