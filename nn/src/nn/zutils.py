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
