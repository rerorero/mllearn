
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr # learning rate 学習率

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum # 過去の勾配をどれだけ重視するか
        self.v = None # 速度

    def update(self, params, grads):
        if self.v is None:
            # 初回は速度を0で初期化
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        # 速度を計算
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key] # 位置を更新
