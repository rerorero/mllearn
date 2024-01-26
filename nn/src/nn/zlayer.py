import numpy as np
from .zutils import *

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0) # [True, False, True, False, ...]
        out = x.copy()
        out[self.mask] = 0 # [0, x2, 0, x4, ...]
        return out

    # 微分=入力の変化が出力にどれだけ影響するか
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Sigmod:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    # 微分=入力の変化が出力にどれだけ影響するか
    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx

# XW + B
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None # 重みの微分
        self.db = None # バイアスの微分

    def forward(self, x):
        self.x = x # 入力を保持
        out = np.dot(x, self.W) + self.b
        return out

    # 微分=入力の変化が出力にどれだけ影響するか
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout) # 重みの微分, 入力の転置と出力の積
        self.db = np.sum(dout, axis=0) # バイアスの微分
        return dx

# 学習の出力レイヤ。推論時は正規化する必要ないので不要
# 推論時の出力をスコアともいう
# Softmaxで出力を正規化し、損失を計算する。
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 損失
        self.y = None # softmaxの出力
        self.t = None # 教師データ(one-hot vector)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t) # 損失を計算
        return self.loss

    # cross entropy が対数なので微分はy-tつまり、出力と教師データの差分
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

