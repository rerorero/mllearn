from trainer.np import *
from nn2.zlayer import *
from nn2.zutil import *
from nn2.zcbow import SimpleCBOW
from .zoptimizer import Adam
from trainer.trainer import Trainer


def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    contexts = []
    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size+1):
            if t == 0:
                continue
            cs.append(corpus[idx+t])
        contexts.append(cs)
    return np.array(contexts), np.array(target)

def main():
    window_size = 1
    hidden_size = 5
    batch_size = 3
    max_epoch = 1000

    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)

    vocab_size = len(word_to_id)
    contexts, target = create_contexts_target(corpus, window_size)
    target = convert_one_hot(target, vocab_size)
    contexts = convert_one_hot(contexts, vocab_size)

    model = SimpleCBOW(vocab_size, hidden_size)
    adam = Adam()
    trainer = Trainer(model, adam)

    trainer.fit(contexts, target, max_epoch, batch_size)
    trainer.plot()
