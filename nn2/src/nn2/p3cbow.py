import numpy as np
from nn2.zlayer import *

# Continuous Bag of Words
def main():
    c0 = np.array([1, 0, 0, 0, 0, 0, 0])
    c1 = np.array([0, 0, 1, 0, 0, 0, 0])

    W_in = np.random.randn(7, 3)
    W_out = np.random.randn(3, 7)

    # 入力層はコンテキストのサイズと同じ
    # ここでは入力は2つだけ
    in_layer0 = MatMul(W_in)
    in_layer1 = MatMul(W_in)
    out_layer = MatMul(W_out)

    # 順伝播, 重みかけるだけ
    h0 = in_layer0.forward(c0)
    h1 = in_layer1.forward(c1)
    # 平均をとる
    h = 0.5 * (h0 + h1)
    s = out_layer.forward(h)

    print(s)

