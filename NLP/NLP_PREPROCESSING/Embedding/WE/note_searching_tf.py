# 검색 문장과 전체 노트 제목/내용을 입력받음
# 각 문서를 토큰화(임베딩 벡터에 맞게 형태소 단위로 토큰화)
# 사전 훈련된 임베딩 벡터(FastText)들을 이용해 각 문장속 단어들을 임베딩 벡터로 변환
# 검색 문장과 각 노트의 제목/내용 속 단어 벡터들을 평균내어 문서벡터를 구함
# 검색 문장의 문서벡터와 각 노트 제목/내용 간 코사인 유사도(sklearn.metrics.pairwise.cosine_similarity|직접)를 구함
# 검색 문장과 비슷한 순서대로 노트 제목(아이디)을 출력

# pre trained vector > https://github.com/Kyubyong/wordvectors

import re
import numpy as np

embedding_dict = dict()
word_index = dict()

with open('ko.vec', 'r+', encoding='utf-8') as f:
    vector_list = f.read().split('\n')[1:]

    for i, vector in enumerate(vector_list):
        if len(vector) == 0:
            continue
        vector = vector.split()
        word = vector[0]
        embedding_vec = vector[1:]

        embedding_dict[word] = embedding_vec
        # embedding_dict[i] = embedding_vec
        # word_index[word] = i

words = list(embedding_dict.keys())
w = []
for i, word in enumerate(words):
    if re.sub(r'[^ㄱ-ㅎㅏ-ㅣ]', r"", word) != '':
        w.append(words[i])

print(w)
