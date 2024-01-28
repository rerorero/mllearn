from .zutil import *
import numpy as np

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

def main():
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    vacab_size = len(word_to_id)
    C = create_co_matrix(corpus, vacab_size)
    # most_similar('you', word_to_id, id_to_word, C, top=5)

    W = ppmi(C)
    np.set_printoptions(precision=3)
    print('covariance matrix')
    print(C)
    print('-'*50)
    print('PPMI')
    print(W)
