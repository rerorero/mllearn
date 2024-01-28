import numpy as np

def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word

# コサイン類似度は自然言語処理においてベクトルの類似度に用いられる
def cos_similarity(x, y, eps=1e-8):
    nx = x / np.sqrt(np.sum(x**2) + eps) # xの正規化
    ny = y / np.sqrt(np.sum(y**2) + eps) # yの正規化
    return np.dot(nx, ny)

# Positive Pointwise Mutual Information
# 相互情報量
# the carのtheのように、theが出現するときにcarも出現することが多い。しかし
# カウントベースの手法ではcarはdriveよりもtheとの共起回数が多いため、theの
# 方が共起してしまう。そこで、corpusの中でtheとcarがどれくらい出現するかの
# 比率で正規化することで、carとdriveの共起回数を正しく評価することができる。
# P(a) = コーパス内でaが起こる確率
# PMI(x, y) = log2(P(x, y) / P(x)P(y))
def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C) # 共起行列の全要素の総和
    S = np.sum(C, axis=0) # 各単語の出現回数
    total = C.shape[0] * C.shape[1] # 共起行列の要素数
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100 + 1) == 0:
                    print(f'{100*cnt/total:.1f} done')

    return M
