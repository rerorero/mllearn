import torch
from os.path import join
from transformer.path import (
    KFTT_TOK_CORPUS_PATH,
    NN_MODEL_PICKLES_PATH,
)
from transformer.text import tensor_to_text, text_to_tensor, get_vocab

TRAIN_SRC_CORPUS_PATH = join(KFTT_TOK_CORPUS_PATH, "kyoto-train.en")
TRAIN_TGT_CORPUS_PATH = join(KFTT_TOK_CORPUS_PATH, "kyoto-train.ja")
src_vocab = get_vocab(TRAIN_SRC_CORPUS_PATH, vocab_size=20000)
tgt_vocab = get_vocab(TRAIN_TGT_CORPUS_PATH, vocab_size=20000)

def src_text_to_tensor(text: str, max_len: int) -> torch.Tensor:
    return text_to_tensor(text, src_vocab, max_len, eos=False, bos=False)

def tgt_text_to_tensor(text: str, max_len: int) -> torch.Tensor:
    return text_to_tensor(text, tgt_vocab, max_len)

def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    word = "Senshu started"
    tgt = ""
    model = torch.load(join(NN_MODEL_PICKLES_PATH, f"epoch_9.pt"))
    model.eval()
    src_tensor = src_text_to_tensor(word, 24).view(1, -1).to(device)
    tgt_tensor = tgt_text_to_tensor(tgt, 24).view(1, -1).to(device)
    output = model(src_tensor, tgt_tensor)
    print(output.shape)

    values, indices = torch.max(output, dim=-1)
    print(tensor_to_text(indices.view(-1), tgt_vocab))
