# Paper
- [Transfer Learning Framework for Low-Resource Text-to-Speech using a Large-Scale Unlabeled Speech Corpus](https://arxiv.org/pdf/2203.15447.pdf)
- 이름 그대로 트랜스퍼 러닝을 위한 라이브러리를 소개한 논문. 
## Model
- 사용모델 : 약간의 수정을 거친 VITS. 전사되지 않은 거대 음성 corpus에서 사전훈련한 뒤, 조금의 text-labeled 데이터셋으로 파인튜닝함.
- 사전훈련 : 유사음소(pseudo phoneme)라고 명명된 새로음 음성토큰(phonetic token)이 음소 대용으로 사전훈련에 사용됨. 

- 사전훈련 절차 : (?)
- 파인튜닝 절차 : (?)
- 추론 절차 : (?)

## 유사음소(pseudo phoneme)
- 유사음소 : 음성 정보를 포함하는 토큰. 사전훈련에서 음소를 대체해 사용됨. 효과적인 사전훈련을 위해 음소와 비슷한 특성을 가져야 하며, supervision을 쓰지 않기 위해 음성만 있는 corpus에서 획득되어야 함.

