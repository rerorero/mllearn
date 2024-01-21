import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x

# softmaxは入力の大小関係を変えるわけではないので、
# 推論時NNのクラス分類での出力層では一番大きい値だけ分かればいいので省略されるのが一般的
# ただ学習時はsoftmax必要
def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size) # 1次元の場合は2次元に変換
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    delta = 1e-7 # log(0)は-infになるので微小な値を足しておく
    # one-hot encoding の場合t=0（偽）だと交差エントロピー誤差は0になるので、無視できる。
    # そのため、正解ラベルのインデックスのみを取り出すことで、交差エントロピー誤差を計算できる
    # batch_size=5のときラベル表現の場合、ｔは[2,7,0,9,4]のように正解ラベルのインデックスのみが入っている
    # y[np.arange(batch_size), t]はy[0,2], y[1,7], y[2,0], y[3,9], y[4,4]のように
    # 正解ラベルのインデックスに対応する出力を抽出することができる
    return -np.sum(t * np.log(y[np.arange(batch_size), t] + delta)) / batch_size

# 微分
def numrical_diff(f, x):
    h = 1e-4
    # f(x)とｆ(x+h)の差分を計算するのではなくx-hとの差を計算する（中心差分）
    # hは無限小ではないので、誤差が生じるが、hを小さくすることで誤差を減らすことができる
    # 10^-4がいいかんじらしい
    return (f(x+h) - f(x-h)) / (2*h)

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) # xと同じ形状で0の配列を生成

    # xの各要素に対して一点ずつ微分を計算する(偏微分)
    # nditer は多次元配列の要素を順番に取り出すイテレータを生成する
    # https://www.aipacommander.com/entry/2017/05/14/172220
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()

    return grad

# lr: learning rate 学習率, 1stepごとにどれだけ動かすか
#     一回前の微分の係数で次ステップのxを決める
#     このような重みやバイアスとは別に、学習率のような人の手によって設定する必要があるパラメータをハイパーパラメータと呼ぶ
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numrical_diff(f, x)
        x -= lr * grad
    return x
