from trainer.np import *

# EmbeddingはMatMul(one-hot表現, 重み行列)を効率化するためのレイヤ
class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    # one-hot x W は重み行列Wからone-hot表現の行を抜き出してるだけなので
    # W[idx]するだけでいい
    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        # self.idxに同じ単語が複数回出てくる場合があるので、その分を足し合わせる
        # for i, word_id in enumerate(self.idx):
        #     dW[word_id] += dout[i]
        np.add.at(dW, self.idx, dout)
        return None

# CBOWの出力層の最適化
# one-hot（多値問題）ではなく二値問題に置き換える。
# 入力がyou goodbye だったら出力がsayかどうか？の問題に置き換えることで
# 出力層の計算をSoftmaxからSigmoidに変更することができる。
# また誤った例をサンプリング入力して低い値にするように学習できる（Negative Sampling）
class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    # h: 中間層のニューロンの出力
    # idx: 二値分類の正解ラベル(say)となる単語のインデックス
    def forward(self, h, idx):
        # 正解ラベルの単語の重み行列を抜き出す
        target_W = self.embed.forward(idx)
        # idxであるかどうか重みを適用して出力する
        out = np.sum(target_W * h, axis=1)

        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh
