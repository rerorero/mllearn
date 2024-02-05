import torch
from torch import nn
from torch.nn import LayerNorm

from layers.Embedding import Embedding
from layers.FFN import FFN
from layers.MultiHeadAttention import MultiHeadAttention
from layers.PositionalEncoding import AddPositionalEncoding


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        heads_num: int,
        dropout_rate: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(d_model, heads_num)
        # DropOut: 学習時ランダムにニューロンを無効化することで過学習を防ぐ
        self.dropout_self_attention = nn.Dropout(dropout_rate)
        # LayerNormは平均が0、分散が1になるように正規化する。各レイヤーの入力を正規化することで学習が安定する
        self.layer_norm_self_attention = LayerNorm(d_model, eps=layer_norm_eps)

        self.ffn = FFN(d_model, d_ff)
        self.dropout_ffn = nn.Dropout(dropout_rate)
        self.layer_norm_ffn = LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.layer_norm_self_attention(self.__self_attention_block(x, mask) + x)
        x = self.layer_norm_ffn(self.__feed_forward_block(x) + x)
        return x

    def __self_attention_block(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        self attention block
        """
        x = self.multi_head_attention(x, x, x, mask)
        return self.dropout_self_attention(x)

    def __feed_forward_block(self, x: torch.Tensor) -> torch.Tensor:
        """
        feed forward block
        """
        return self.dropout_ffn(self.ffn(x))


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        pad_idx: int,
        d_model: int,
        N: int,
        d_ff: int,
        heads_num: int,
        dropout_rate: float,
        layer_norm_eps: float,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, pad_idx)

        self.positional_encoding = AddPositionalEncoding(d_model, max_len, device)

        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model, d_ff, heads_num, dropout_rate, layer_norm_eps
                )
                for _ in range(N)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embedding(x)
        x = self.positional_encoding(x)
        # MultiHeadAttentionをＮ層繰り返す
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        return x
