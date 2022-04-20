import numpy as np

def W2C_SkipGram(embedding_dim, window_size, input_arr, w_IL_arr=None, w_LO_arr=None):
    doc_vec_num = np.shape(input_arr)[0]
    label_arr = np.zeros(doc_vec_num, window_size*2, embedding_dim)
    if w_IL_arr is None or w_LO_arr is None:
        if not (w_IL_arr is None and w_LO_arr is None):
            raise TypeError("가중치 입력시 모두 입력해주세요.")

        v = np.shape(input_arr)[1]  # (doc_word_num, one_hot_vec_dim)

        w_IL_arr = np.random.random((doc_vec_num, v, embedding_dim))  # x(input)((v).T) * W_IL(v, m(emb_dim)) = (m). | .T(?)
        w_LO_arr = np.random.random((doc_vec_num, embedding_dim, v))  # (m).T           * W_LO(m, v)          = (v).T

    score_vec_all = []  # (doc_word_num, 1, m) > (doc_word_num, 1, m) > softmax((doc_word_num, 1, v))
    for i, input_vec in enumerate(input_arr):
        cnt = 0  # make label
        for j, close_input_vec in enumerate(input_arr[i-window_size:i+window_size+1]):
            if i-window_size < 0 or i+window_size > len(input_arr) or j == i:
                continue
            label_arr[i][cnt] = close_input_vec
            cnt += 1

        temp_IL = np.dot(input_vec, w_IL_arr[i])            # append dot(central vector, w)

        temp_LO = np.dot(temp_IL, w_LO_arr[i])              # (Input <--> Lookup) * w_LO

        softmax_LO = np.exp(temp_LO) / np.sum(np.exp(temp_LO))  # softmax(Lookup <--> Output)

        score_vec_all.append(softmax_LO)  # append every close vec's average * w_LO

    # loss, backpropagation(W update) will be updated in out of function

    return score_vec_all, label_arr, w_IL_arr, w_LO_arr

