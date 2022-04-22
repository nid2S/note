
# BART
- BART(Bidirectional and Auto-Regressive Transformers) : 넓은 분야에 적용할 수 있도록 S2S구조로 만들어진 denoising auto-encoder. 손상된 text를 복구하도록 모델이 사전학습됨. 
  분류에서 BERT와 비슷한 성능을 내며 generation task(특히 summarization)에서도 SOTA를 달성함. 
  BERT와 GPT의 특성을 모두 적용해, 손상된 text를 bidirectional모델로 인코딩하고, 정답에 대한 likehood를 autoregressive디코더로 계산함.  
- BART 구조 : S2S트랜스포머 구조를 사용, GeLU 활성화 함수 사용. base모델은 6layer, large모델은 12layer를 사용함.
  손상된 text로 학습하며, 디코더의 출력과 원본텍스트의 loss를 줄이도록 함. 다른 오토인코더모델과는 다르게 모든 종류의 노이즈를 적용할 수 있음.


## BART 노이즈 기법
- LM : GPT와 비슷함. left-to-right 트랜스포머를 학습시킴. 어텐션이 빠진 BART 디코더와 동일.
- Permuted LM : XL-Net을 기반으로 함. 1/6의 토큰을 샘플링한 후 랜덤한 순서로 auto-regressive하게 생성. 
- MLM(TokenMasking) : BERT와 같이 15%의 토큰을 mask토큰으로 바꾸고, 독립적으로 이 토큰을 예측하게 함.  
- Multitask MLM : UniLM처럼 셀프어텐션마스크를 추가해 MLM을 학습함. 
- Masked S2S : MASS와 비슷함. 토큰의 50%를 포함하는 span에 mask를 한 뒤 그를 예측하는 S2S를 학습시킴.

- TokenDeletion : 랜덤토큰삭제 후 복구, 삭제된 위치를 알 수 없음. 
- TextInfilling : 포아송분포를 따르는 길이의 text span을 생성해 하나로 마스킹(여러토큰이 하나로 마스킹/없던자리에 추가로 마스킹 됨). 빠진 토큰의 양을 예측.
- SentencePremutaion : 문서를 문장단위로 나눠서 섞음.
- DocumentRotation : 토큰하나를 정해 문장이 해당 토큰부터 시작하게 하고, 문서의 시작을 구분하게 함.


## BART 파인튜닝
- 시퀀스 분류 : 같은 입력이 인코더와 디코더에 주어지고, 디코더의 마지막 은닉상태가 새 선형분류기로 전달됨. BERT와 비슷하나 마지막토큰까지 입력.
- 토큰 분류 : 전체 문서를 인코더와 디코더에 입력. 디코더의 top은닉상태를 각 단어의 표현으로 사용, 이걸 토큰분류에 이용.
- 시퀀스 생성 : autoregressive적 디코더를 가지고있어 바로 파인튜닝가능. 인코더에 입력이 주어지면 디코더에서 출력을 auto-regressive하게 만듦.
- 기계 번역 : 전체모델을 기계번역을 위한 디코더로 사용하고, 새 인코더(외국어를 BART가 학습한 언어로 denoising 할 수 있는 입력으로 맵핑)를 추가해 인코더-디코더를 파인튜닝.
  새 인코더는 BART와 다른 vocab을 사용할 수 있으며, 두단계로 학습하는데, 둘 다 cross-entropy로 역전파를 수행함.

## 타 모델과 비교
- GPT : leftward context만 다뤄 몇몇 task에서 문제 발생.
- GPT2 : 아주 큰 LM이 비지도, multitask모델처럼 동작.
- ELMo : left-only, right-only표현을 연결하는데 두 표현간의 상관관계는 학습하지 않음.
- BERT : 다양한 기법을 사용해 좋은 성능을 보이나 예측이 auto-regressive하지 않아 생성task에는 약함.
 
- UniLM : undirectional LM, Bidirectional LM, sequence LM을 앙상블한 모델. 각 LM사이의 파라미터와 모델 구조를 통일함으로써 여러 LM을 만들어야 헀던 필요성을 완화시킴.
  BART와 달리 예측이 조건부 독립적. BART는 항상 완전한 입력이 디코더에 주어져 사전훈련과 생성의 차이가 적음.
- MASS : BART와 가장 유사. 연속된 span이 마스킹된 문장을 인코더입력으로 주고, 디코더에서 마스킹되었던 토큰들을 예측.
- XL-Net : 마스크된 토큰을 섞인순서로 auto-regressive하게 예측하도록 BERT를 확장함.


## BART의 변형(S2S모델)
- BART : 인코더와 디코더가 있는 S2S모델. 인코더엔 손상된 토큰이, 디코더엔 원래 토큰이 제공됨(단, 원래 트랜스포머처럼 미래단어를 숨기는 마스크 포함).
  인코더의 사전학습에는 무작위 토큰 마스크(MLM같이), 무작위 토큰 삭제, 단일마스크 토큰으로 k범위를 마스킹, 순열문장, 문서를 회전해 특정토큰에서 시작하게 하는 변환이 적용됨. 
- Pegasus : BART와 동일하나 MLM과 GSG(Gap Sentence Generation, 전체 인코더 입력문장이 두번째 마스크토큰으로 대체되어 디코더에 제공, 인과마스크 포함)에 대해 공동으로 사전학습됨.
- MarianMT : BART와 동일한 모델을 사용하는 번역모델용 프레임 워크.
- T5 : 각 레이어에서 학습되는 위치 임베딩이 약간 변경되는 전통적 트랜스포머 모델을 사용. 모든 NLP태스크에서 작동할 수 있도록 특정접두사를 사용해 다른 task를 텍스트를 텍스트로 변환하는 문제로 변환.
  사전교육에는 지도 및 자체지도교육(토큰의 15%를 무작위 제거 후 개별 센티넬 토큰으로 교체해 손상된 토큰을 사용함)이 모두 포함. 
- MT5 : 모델아키텍쳐는 T5와 동일하나 T5의 지도교육을 포함하지 않고, 101개의 언어로 훈련됨.
- MBart : 모델아키텍쳐는 BART와 동일하나 25개의 언어로 학습되며, 감독 및 비감독 기계번역을 위함. 다언어로 된 전체 텍스트를 노이즈제거해 완전한 S2S모델을 사전훈련하는 방법.
- ProphetNet : 미래n-gram예측이라는 새 S2S사전훈련목표를 도입함. 모델아키텍처는 트랜스포머를 기반으로 하나 디코더의 표준 셀프어텐션을 메인셀프어텐션+셀프/n-스트림(예측)셀프어텐션 매커니즘으로 대채함.
  모델은 단일 다음토큰 대신 각 스텝에서 이전 컨텍스트토큰을 기반으로 다음 n개의 토큰을 동시예측함. 미래 n-gram예측은 모델이 미래토큰을 계획하고 강력한 로컬상관관계에 대한 과적합을 방지하도록 명시적으로 궘장. 
- XML-ProphetNet : 모델 아키텍처 및 사전학습목표는 ProphetNet과 동일하나 교차언어 데이터셋 XGLUE에 대해 사전학습됨. 