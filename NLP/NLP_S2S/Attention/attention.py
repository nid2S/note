import numpy as np

# Q = np.random.random()      # 현시점 디코더 은닉상태
# K = np.random.random()      # 모든 시점 인코더 은닉상태
# V = K.copy()
#
# Wc = np.random.random()     # [어텐션값;Q(연결, 병합)]의 가중치 행렬
# bc = np.random.random()     # [어텐션값;Q(연결, 병합)]의 편향


def softmax(matrix):
    return np.exp(matrix) / np.sum(np.exp(matrix))

def attention(decoder_t_hidden, encoder_all_hidden, wc, bc):
    Q = decoder_t_hidden    # 셀프 어텐션의 경우 encoder_all_hidden
    K = encoder_all_hidden
    V = K.copy()
    
    at_score = np.dot(Q.T, K)

    at = np.sum(np.dot(softmax(at_score), V))
    vt = np.tanh(np.dot(wc, (at + Q)) + bc)  # s(t-1)과 함께 메모리셀(RNN, LSTM)의 입력이 되는 벡터. K와의 유사도 정보가 담김.
    
    # 이후 가중치와 편향을 역전파 과정을 통해 갱신
    return vt, wc, bc


