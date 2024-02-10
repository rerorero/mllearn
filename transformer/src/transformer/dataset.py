from typing import Callable

from torch import Tensor
from torch.utils.data import Dataset

# https://qiita.com/tetsuro731/items/d64b9bbb8de6874b7064
class KfttDataset(Dataset):
    def __init__(
        self,
        path_to_src_corpus: str,
        path_to_tgt_corpus: str,
        max_len: int,
        src_text_to_id: Callable[[str, int], Tensor],
        tgt_text_to_id: Callable[[str, int], Tensor],
    ) -> None:
        self.src: list[str] = []
        self.tgt: list[str] = []

        with open(path_to_src_corpus, "r", encoding="utf-8") as f:
            for line in f:
                self.src.append(line)

        with open(path_to_tgt_corpus, "r", encoding="utf-8") as f:
            for line in f:
                self.tgt.append(line)

        assert len(self.src) == len(self.tgt)

        self.max_len = max_len

        self.src_text_to_id = src_text_to_id
        self.tgt_text_to_id = tgt_text_to_id

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.src_text_to_id(self.src[index], self.max_len), self.tgt_text_to_id(
            self.tgt[index], self.max_len
        )
