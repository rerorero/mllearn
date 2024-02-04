import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, pad_idx: int = 0) -> None:
        super().__init__()
        # https://gotutiyan.hatenablog.com/entry/2020/09/02/200144
        # https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=pad_idx
        )

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        return self.embedding_layer(input_batch)
