# Attention
***
- 어텐션 메커니즘 : 신경망(RNN, S2S)의 성능증가(기울기소실 보정)을 위한 매커니즘. 트랜스포머의 기반이 됨.
  디코더에서 매 시점마다 인코더의 모든 은닉상태를 활용해 컨텍스트 벡터를 다이나믹하게(각 상태별로)만들어 고정된 사이즈의 문맥벡터에서 오는 정보 소실과 중요한 정보에만 집중할 수 있게 함.
- 어텐션값 : 주어진 Q(쿼리)에 대해 모든 K(키)와의 유사도를 각각 구한 뒤 각각의 V(값)에 반영하고, V를 모두 더한 값.
- 표현 : Attention(Q, K, V) = Attention Value(a(t)) | Q = 현시점의 디코더 은닉상태 | K = V = 모든시점 인코더 은닉상태 | e_ij = w^T tanh(W_s_(i-1) + Vh_j + b)

- Key, Value : 인코더의 은닉상태. 이때 은닉상태는 인코더의 마지막 벡터가 아닌, 모든 시점에서의 벡터가 모인 행렬이 됨.
- Query : 디코더의 전시점 은닉상태. 초기 은닉상태는 인코더의 마지막 은닉상태가 됨.

- 연산과정 : 인코더의 은닉상태들을 스텝별로 구한 뒤(Key), 각 step의 은닉상태들과 이전step디코더의 은닉상태(Query)간의 스코어(내적 등, Attention Score)을 구함.
  이 벡터를 softmax연산을 통해 정규화(어텐션 분포)하고, 이걸 인코더의 은닉상태와 곱해(Value) 가중치가 적용된 벡터행렬로 만들고, 이를 합해 컨텍스트 벡터(어텐션값)으로 만듦.

- a(t) = sum(softmax(\[score(s(t), h(0)) . . .  score(s(t), h(max))]) * h(0 . . . max), dim=1)  | 어텐션 스코어(score) > 어텐션 분포(softmax) > 어텐션값(가중합)
- v(t) = \[a(t);s(t)](결합) | v(t)와 s(t-1)을 메모리셀(LSTM, RNN)의 입력으로 사용해 s(t)를 얻고, 이는 출력층으로 전달되 현시점의 예측값을 구하게 됨.
``` attention
v(t) = [a(t);Q(t-1)]  # 어텐션 레이어 결과
a(t) = sum(softmax([score(Q(t), K(0)) . . .  score(Q(t), K(max))]) * V)  # 어텐션 분포 -> 어텐션 값
score(Q(t), K(i)) = Q(t).T * K(i)  # 어텐션 스코어
```

# order
- 인코더와 디코더의 은닉상태 크기가 같다고 가정.
- s(t) : 현시점 디코더 은닉상태(Q) | h(i) : i번째 인코더 은닉상태(K(i)) | score(s(t), h(i)) : 현시점과 i번째 인코더 은닉상태간 어텐션스코어(유사도) |
### Dot-Product
- Dot-Product Attention(dot, Luong) : score(s(t), h(i)) = s(t).T * h(i).
### Scaled dot
- scaled dot Attention(Vaswani) : score(s(t), h(i)) = s(t).T * h(i) / √(n)
- 주로 Q = s(t), K = h. V = K+어텐션(Normalized Weights). 초기값은 V와 K가 동일.
### General
- general Attention(Luong) : score(s(t), h(i)) = s(t).T * W(a) * h(i)
- W(a): 학습가능한 가중치 행렬
### Concat
- concat Attention(Bahadanau) : score(s(t-1), h(i)) = W(a).T * tanh(W(b)*s(t-1) + W(c)*h(i))
- W(b,c): 학습가능한 가중치 행렬. 병렬화를 위해 h(i)대신 H(h 모두 모음)을 사용.
### lacation-base
- location-base Attention(Luong) : score(s(t), h(i)) = X
- a(t) = softmax(W(a) * s(t))
- 어텐션값 산출시 s(t)만 사용하는 방법
### Additive
- Additive Attention : c_t = sum(a_(t,i) * h_i) = a_t * h.
- a_(t,i) = exp(s_(t,i))/sum(exp(s_(t,i)))        | s_(t,i) = w^T * tanh(W\*d_(t-1) + V*h_i + b)
- W, V = 학습가능 가중치 | w, v = 학습 가능 벡터 가중치 | h_i, d_i = 인코더/디코더 i번째 feature | 
- s_(t,i) = t 시점에서 h_i에 대한 attention score    | a_(t,i) = t 시점에서 h_i에 대한 alignment(0~1)
- c_t = t시점에서 Attention모듈로부터 추출한 context vector | Additive = 첨가된.
### Location Sensitive
- Location Sensitive Attention : s_(t, i) = w^T * tanh(W*d_(t-1) + v\*h_i + U\*f_(t,i)+b)
- f_i = F * a_(i-1) | F = Convolution Filter | a_(i-1) = 지난 시점 Additive Attention alignment score. 
- 이전 시점에서 생성된 attention alignment를 이용해 다음 시점을 구할때 추가로 고려.


# self attention
- 셀프 어텐션 : 쿼리, 키, 벨류가 입력문장의 모든 단어벡터들로 동일. 인코더(Encoder) 혹은 디코더에서(Masked decoder) 이뤄짐. 
  입력문장내 단어들끼리 유사도를 구해 단어의 의미(it 등이 무엇을 뜻하는지)를 찾아냄.
- 셀프 어텐션 실행 : 각 단어벡터들에 가중치행렬(단어벡터차원*(단어벡터차원/num_heads)의 크기를 지님)을 곱해 일정크기(단어벡터차원/num_heads)의 쿼리, 키, 벨류 벡터를 얻음.
  이를 이용해 스코어(q*k/√(k벡터 차원), 트랜스포머는 Scaled dot-product Attention을 사용)를 구한 뒤 softmax를 지나 어텐션 분포를 구하고, 이를 가중합해 어텐션값을 구함.
- 셀프 어텐션 행렬연산 : 위 과정은 벡터 연산이 아닌 행렬연산을 사용하면 일괄계산이 가능해 행렬연산으로 구현됨. 헤드의 수만큼 병렬을 수행함.
- 셀프 어텐션 행렬연산 과정 : 문장행렬에 가중치행렬을 곱해 Q,K,V 행렬을 구하고, Q와 K를 내적곱(Q*K^t)하고 (q\*k/√(k벡터 차원))로 나눠 어텐션스코어를 얻으며,
  여기에 softmax를 지나게 하고(어텐션 분포) V행렬을 곱해 어텐션값 행렬을 만들 수 있음.
 
