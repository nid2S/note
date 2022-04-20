import re

# 간단한 토크나이저
def toknize(sent: str):
    # 한국어는 koNlpy 속 OKT, kkma등 이미 나와있는 형태소 분석기를 사용하는게 낫다. norm(정제), stem(표제어추출)등도 지원한다.

    # 모델의 이용 용도나 데이터의 성질에 따라 해야 하는 전처리가 다르기에 정규식을 잘 알아두면 쓸 일이 많다.
    sent = re.sub(r"(n't)|('s)|('re)|('m)|\s+", r" \1\2\3\4", sent.lower()).split()

    word_index = dict()  # 단어:빈도수 형태의 딕셔너리.
    for word in sent:
        if word not in word_index:
            word_index[word] = 0
        word_index[word] += 1
    word_index = sorted(word_index.items(), key=lambda item: item[1], reverse=True)
    # 분리된 문자열과 "단어": 인덱스 형태의 딕셔너리를 반환.
    return sent, dict([(word, i+1) for i, (word, freq) in enumerate(word_index)])

