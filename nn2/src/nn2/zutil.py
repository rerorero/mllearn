from trainer.np import *
import collections

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
# 方が関連性が強いと判断してしまう。そこでthe,car,driveがどれくらい出現するかの
# 比率で正規化することで、carとdriveの関連性をを正しく評価することができる。
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

def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            left_idx = idx - i
            right_idx = idx + i
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix

def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print(f'{query} is not found')
        return

    print(f'\n[query] {query}')
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vacab_size = len(id_to_word)
    similarity = np.zeros(vacab_size)
    for i in range(vacab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(f'{id_to_word[i]}: {similarity[i]}')

        count += 1
        if count >= top:
            return


def convert_one_hot(corpus, vocab_size):
    N = corpus.shape[0]
    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_1, word_ids in enumerate(corpus):
            for idx_2, word_id in enumerate(word_ids):
                one_hot[idx_1, idx_2, word_id] = 1

    return one_hot

def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x

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
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1
        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]
        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]
        # 速度優先、targetが含まれる場合がある
        return np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
