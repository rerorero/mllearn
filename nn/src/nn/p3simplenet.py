import numpy as np
from .zutils import *

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # ガウス分布で初期化

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

def main():
    net = simpleNet()
    t = np.array([0, 0, 1]) # 正解ラベル
    print(net.W)

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)
    print("max index:", np.argmax(p)) # 最大値のインデックスを取得
    print("loss: ", net.loss(x, t))

