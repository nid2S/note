# BERT
- 모델 종류 : 모델의 크기에 따라 base모델과 large모델을 제공. L - 트랜스포머 블록 층수(개수), H - hidden size, A - self-attention해드개수. 성능은 Large > base (데이터셋 크기 무관).
- BERT_base : L=12, H=768, A=12, Total Parameters = 110M, feed-forward/filter size = 4H
- BERT_large : L=24, H=1024, A=16, Total Parameters = 340M, feed-forward/filter size = 4H
- 사용 Task : Span Prediction, Zero-shot Learning, 그 외 언어 이해능력이 중요한 Task.
- Span Prediction : 문장 전체의 벡터를 생성, 각 Slot이 없는지/어떤 값이든 상관없는지/존재하는지 예측한 후 각 토큰의 벡터들을 이용해 Slot의 시작/종료점의 확률을 계산, 이를 바탕으로 Slot을 추측.  
- Zero-shot Learning : 아예 처음보는 서비스나 Slot도 처리 가능. Slot의 설명이 주어진다면 언어처리능력으로 기존의 Slot과 비슷함을 인지, 처리.  

## 인코더
### 인코더 입력
- BERT입력층 : 트랜스포머를 기반으로(그중에서도 인코더만 사용)함. 포지셔널 인코딩 대신 포지션 임베딩과 Segment Embeddings를 추가해, 총 세가지 임베딩의 합산 결과를 취함.
- BERT임베딩 : 사용되는 임베딩은 WordPiece(Token)임베딩, Segment임베딩, Position임베딩. 이 세가지 임베딩을 얻어와 합산한 뒤 층정규화 & Dropout을 거친 결과를 인코더의 입력으로 함.
  코드로는[e = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)]으로, pos는 위치에 따른 값(range(0,maxlen)), seg는 토큰타입(입력문장 종류에 따라 각각 다른 값).
- 임베딩 특징 : 다이나믹 임베딩. 문장 형태와 위치에 따라 동일한 단어도 다른 임베딩을 갖게 되어, 중의성을 해소 할 수 있음.
  ELMo를 포함한 BERT의 가장 큰 특징으로, 기존 W2V, GloVe와의 가장 뚜렷한 차이점.
![embedidng image](https://user-images.githubusercontent.com/1250095/50039788-8e4e8a00-007b-11e9-9747-8e29fbbea0b3.png)

### 인코더 블록
- BERT_인코더블록 개수: N개의 인코더블록을 가지고 있음. Base모델은 12개, Large모델은 24개로 구성됨.  
- 인코더블록 : 이전 출력을 현재 입력으로 하는 RNN과 유사한  특성을 가지고 있어, 총 N번 Recursive(재귀적)반복처리됨.
  과정에서 비션형 활성화함수를 거치지 않고 네트워크를 직접 흐르게 해, 기울기 소실이나 폭주(Vanishing/Exploding Gradients)문제를 최소화함.

#### Multi-Head Attention
- Multi-Head Attention : 헤드가 여러개인 어텐션. 인코더블록의 가장 핵심적 부분. 서로 다른 가중치 행렬로 어텐션(Scaled Dot-Product)을 h번 계산 후 이를 서로 연결한 결과를 가짐. 
- in BERT-Base Model : 각각의 토큰 벡터 차원(768차원)을 헤드수(12)만큼 등분해, 64개씩 12조각으로 차례대로 분리 한 후 Scaled Dot-Product Attention을 적용 후 다시 본래차원으로 합침.
  결과 벡터는 부위별로 12번 Attention받은 결과가 되며, softmax는 변동폭이 매우 크고 작은 차이에도 쏠림이 두드러져, 값이 큰 스칼라는 살고 작은 쪽은 0에 가까운 값이 곱해져 배제된다.

- Scaled Dot-Product Attention : Q(임베딩의 fully-connected), K(이전 블록의 출력), V(이전 블록의 출력)를 입력으로 받음. 
  셋의 초깃값은 모두 동일하나 다른 초기화를 거쳐, 다른 값에서 출발하고 구성만 동일함. 동일한 토큰이 문장내의 다른 토큰에 대한 Self-Attention효과를 가짐. 
- Scaled Dot-Product Attention 수식 : [Attention(Q,K,V) = softmax(Q(K.T)/√d_k)*V]. 
- Masked Attention : BERT는 제로패딩으로 입력된 토큰은 항상 마스킹처리를 하며, 패널티를 부과해 어텐션 점수를 받지 못하게 함. 패딩토큰과 실제토큰을 분형하게 구분함.
  어떤 토큰이 패딩이며 어떤토큰이 아닌지를 나타내는 0과 1의 배열. BERT 공식구현 - [adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0]

- Scaled Dot-Product Attention 도식화
![Scaled Dot-Product Attention](https://cdn-images-1.medium.com/max/1600/1*nCznYOY-QtWIm8Y4jyk2Kw.png)

#### Position-wise Feed-Forward Network
- Position-wise Feed-Forward Network : 인코더 블록의 다른 한부분. 어텐션의 결과를 통과시킴. 초기에는 커널사이즈 1의 CNN 둘을 연결하는 방식을 쓰기도 했으나 지금은 쓰이지 않음.
  두개의 LinearTransformations로 구성되어 있으며, 사이엔 ReLU보다 조금 부드러운 형태라 음수에서도 미분이 가능해 약간의 기울기를 전할 수 있는 GELU를 적용함. 
  이 후 정규화와 임베딩(to vocab), softmax를 거쳐 학습(MLM)의 입력으로 사용됨.
- FFN층 수식 : [FFN(x) = max(0, x * W1 + b1) * W2 + b2]

## 학습
- 특징 : Bidirectional하게 학습 함. 원래 Transfomar는 Bidirectional하나, 이후 단어의 예측 확률을 계산해야 하는 
  Statistical Language Model은 Bidirectional 할 수 없었고, BERT는 이를 다른 형태의 문제로 전환해 가능하게 함.
- Bidirectional : Masked Language Model과 Next Sentence Prediction을 사용해 가능하게 함. 특별토큰을 fine-tuning은 물론 pre-trianing시에도(NSP가 존재하여 가능)사용. 
- [CLS\] : 클래스. 문장의 첫번째 토큰. 단일/연속된 문장의 분류문제와 문장 유사도 판별 문제에 사용됨. 토큰시퀀스의 결합된 의미를 가지게됨. 
- [SEP\] : 문장구분. 문장의 마지막 토큰. QA문제(문장 구분)와 사전훈련시의 NSP문제에 사용됨.

- 입력 : NSP를 위해 문장을 뽑아 segment임베딩(embedding A,B를 먹여줌)을 시행. 50%는 진짜, 나머지는 랜덤문장을 사용하며, 
  모든 토큰이 합쳐진 길이는 배치당 512개 이하(OOM)여야함. 이후 마스킹작업을 실시함.
- 사전훈련 하이퍼파라미터 : batch_size = 256, 256*512 = 128000개의 토큰 사용, 이를 백만step -> 3.3billion(33억)word corpus를 40epoch로 학습. adam(lr=0.001), gelu, dr=0.1 사용.

- Masked Language Model : 문장의 다음단어가 아닌(평범한 LM)랜덤한 토큰을 마스킹 한 뒤 이를 주변 단어의 context만 보고 예측하도록 하는 방식. 
  마스킹은 전체 단어의 15%정도만 진행하며, 그중 80%는 [MASK\]로, 10%는 랜덤한 단어, 10%는 정상적인 단어로 진행함. WordPiece토큰화 이용.
  CBOW와 유사하나, 토큰을 학습 한 뒤 Weights를 벡터로 갖는 CBOW와 달리, 학습한 결과를 직접 벡터로 가짐. [MASK\]토큰은 사전훈련시만 사용되고 fine-tuning시에는 사용되지 않으며, 
  이 토큰을 맞추는 task를 수행하며 문맥을 파악하게 됨. 전체 학습이 끝나는(수렴)데에는 LM보다 많은 step이 필요하나, 전체적으로(emperical)는 LM보다 빠르게, 좋은 성능을 냄.
- 마스킹 이유 : 진행되는 마스킹중 80%만 [MASK\]로 하고 다른건 랜덤한 단어와 정상적인 단어를 쓰는 이유는, 학습되는 인코더가 모든 토큰에 대해 분포적 맥락 표현을 학습하도록 하기 위해서임.
  만약 토큰에 대해서만 학습한다면 Fine-tunning시 토큰을 보지 못해 아무것도 예측할 필요가 없다고 생각할 것이며, 어떤 토큰이 바뀐건지 랜덤인지 몰라 문장의 모든 단어에 대한 문맥을 표현하기 위해서임.
  랜덤한 단어로 바꾸는 것은 전체의 1.5%에 불과하여 언어 이해에 해를 끼치지 않음.
- MLM output : 연산비용때문에 Hierachical Softmax나 네거티브 샘플링을 쓰는 W2V와 달리, 전체 Vocab에 대한 Softmax를 모두 계산함.
  구글에서 공개한 영문 학습 모델의 VocabSize는 30522개이며, 한글 모델의 경우는 형태소 분석 결과가 10만개를 넘어가는 경우가 흔해 학습에 더 오랜 시간이 걸림.

- Next Sentence Prediction : 두 문장을 주고, 두번째 문장이 코퍼스 내에서 첫 문장의 바로 뒤에 오는지 여부를 예측하도록 하는 방식. 자연어 추론, QA등의 task에서 도움됨. 
  BERT는 TransferLearning으로 사용되고, QA와 NLI(자연어 추론)등의 Task에선 MLM으로 학습하는 것 만으로는 불충분했기에(두 문장간 관계를 이해하는데 부족(LM에서 캡쳐되지 않음))사용됨.
  실제로 이어지는지 여부는 50:50으로, 참인 문장과 랜덤하게 추출되어 거짓인 문장으로 구성되며 다음문장인지 아닌지를 예측하는 이진분류작업을 수행함. 모든 문장은 마스킹 과정을 거친 믄장임.

## Fine-tuning
- 하이퍼 파라미터 : 사전훈련시와 대부분 동일하나 batchsize, lr, epoch가 약간 달라짐. task마다 달라지나, batch_size=16/32, lr(Adam)=5/3/2e-5, epoch=3/4에서 대부분 잘 학습됨.
- 참고사항 : 데이터셋의 크기가 클수록 하이퍼파라미터에 영향을 덜 받고, 잘 training되나, 데이터크기가 작어도 파인튜닝때는 좋은 성능을 낼 수 있음.
  pre-training에 비해 굉장히 빠르게 학습되며, 따라서 최적의 하이퍼파라미터를 exhaustive search로 찾아내도 괜찮음. 사전훈련 단계에서 step이 많을수록 정확도가 늘어남.
- 파인튜닝시 하이퍼 파라미터 : batch_size=16/32, lr(Adam)=5/3/2e-5, epoch=3/4 사이에서 잘 학습된다고 함. 

- 전처리 : 각 문장의 시작과 끝에 특수토큰([CLS\], [SEP\])을 넣어주고, 모든 문장의 길이를 맞춰줘야(pad, truncate)함.

- sequence-level 분류작업 : 직접적임. 입력 시퀀스에 대해 일정 차원수의 representation결과를 얻기위해, [CLS\]토큰의 트랜스포머 output값을 사용함.
  [CLS\]토큰의 벡터는 H차원을 가지며, 여기에 분류하고 싶은 개수(K)에 따라 classification layer를 붙이고, label probabilities는 softmax로 계산. 모든 파라미터가 같이 파인튜닝됨.
- 근거있는 상식적 추론 : 앞문장이 주어졌을 때, 가장 잘 이어지는 문장을 찾는 task. 
  가능한 입력 시퀀스를 구현(주어진 문장과 가능한 문장들을 각각 concat)하고, 벡터를 학습시킨 뒤, 값들을 합산해 dotProduct > softmax를 거침. 

- span/token-level 예측작업 : 위에서 약간 변형. QA task와 NER(Named Entity Recognition)/형태소분석 task등으로 나눌 수 있음.
- QA : 질문에 정답이되는 단락의 substring을 뽑아내는 것이니, [SEP\]토큰 이후의 토큰들에서 Start, End Span을 찾아내는 task를 수행함.
  질문과 지문이 주어지고 그중에서 substring인 정답을 맞추는 task에서, 질문을 A임베딩, 지문을 B임베딩으로 처리한 뒤 지문에서 정답이 되는 substring의 처음과 끝을 찾는 task로 바꿈.
- NER(이름 개체 인식, 태깅)/형태소분석 : 단일 문장에서 각 토큰이 어떤 class인지를 모두 classifier를 적용해 정답을 알아냄. 각 예측은 주변의 예측에 영향을 받지 않음(CRF, autoregressice사용X).

- Feature-based Approach : 모든 task를 표현하지는 못함으로, 특정 task를 수행가능한 네트워크를 부착해 쓸 수 있으며, 전산적 이점을 얻을 수 있다는 이점이 있음. 
  마지막 레이어에 Bi-LSTM등을 부착해 해당 레이어만 학습시키는 등의 방법이 있으며, 가장 효과적인(마지막 4레이어)경우 파인튜닝과 큰 차이가 없음.

## BERT의 변형(인코더/오토인코딩 모델)
- BERT : MLM, NSP사용. 기본 BERT모델.
- ALBERT : BERT와 동일하나 약간 다름. 은닉크기 >> 임베딩크기(임베딩-컨텍스트독립적, 은닉-컨텍스트종속적 | 매개변수가 더 적음), 레이어가 매개변수를 공유하는 그룹으로 분할(메모리를 위함), 
  NSP > 문장순서예측(Sentence Ordering Prediction)연속되는 두 문장을 준 뒤 순서가 올바른지 예측)의 차이가 있음.
- RoBERTa : 더 나은 사전훈련트릭이 있는 BERT와 동일. 동적마스킹(각 epoch에서 다르게 마스킹, BERT는 한번만 수행), 
  NSP손실이 없음(인접한 텍스트 덩어리를 함께 넣어 더 순서대로 배열됨), 더 큰 배치로 훈련, BPE를 문자가 아닌 하위 단위로 사용(유니코드 문자로 인해)의 차이점이 있음.
- DistillBERT : BERT와 동일하나 더 작음(작고/빠르고/저렴하고/가벼움). BERT의 증류로 훈련됨. 
  교사모델과 동일한 확룰을 찾는것과, 마스킹된 토큰을 올바르게 예측하는것과, 은닉과 교사의 코사인 유사도의 조합을 실제 objective로 함.  
- ConvBERT : 모든 어텐션헤드가 글로벌 self어텐션 블록을 생성하여, 메모리 공간과 계산비용이 많이 들고, 일부 헤드는 로컬종속성만 학습하면 된다는 데서 나오는 계산중복성의 문제를 해결하기 위해
  self어텐션 헤드를 대체하는 span-based dynamic convolution을 제안. 새 헤드는 나머지 self어텐션 헤드와 함께 글로벌/로컬컨텍스트 학습 모두에서 더 효율적인 혼합어텐션 블록을 형성.
- XLM : 여러 언어로 훈련된 트랜스포머 모델. 세가지 다른 유형의 교육이 있음. 전통적 auto-regressive적 훈련인 CLM(인과 언어 모델링, 각 샘플에 언어중 1개가 선택, 
  그것의 span(토큰)256개로 된 문장이 입력), 로베르타와 같은 MLM(각 샘플마다 언어중 하나가 선택, 토큰의 동적마스킹과 함께 256개의 토큰문장이 모델의 입력이 됨),
  MLM과 TLM(번역 언어 모델링)의 조합(무작위 마스킹을 사용해 두개의 다른 언어로 된 문장을 연결. [MASK\]예측을 위해 언어1의 주변 컨텍스트와 언어2의 컨텍스트 사용가능/)이 있음.
- XLM-RoBERTa : XML접근방식에서 로베르타 트릭을 사용하지만 TML은 사용하지 않음(한 언어에서 나오는 문장에만 MLM을 사용). 
  그러나 더 많은 언어(100개)에 대해 학습되었고, 임베딩을 사용하지 않아 언어를 자체적으로 감지 가능.
- FlauBERT : 프랑스어를 위한 비지도 언어 모델. 로베르타와 마찬가지로 문장순서 예측이 없음(MLM에 의해 훈련).
- ELECTRA : 다른(작은) MLM으로 사전훈련된 트랜스포머 모델. 무작위 마스킹된 입력을 사용해 원본과 바뀐 토큰을 예측해야 함. 
  GAN훈련과 같이(단, 모델을 속이지 않기 위해 원본텍스트를 객관적으로 사용)작은 언어모델은 몇단계동안 훈련되고, 그 후 엘렉트라 모델로 몇 스텝동안 훈련됨.
- Funnel Transformer : 레이어는 블록으로 그룹화되고, 각 블록의 시작부분에서 은닉상태는 시퀀스차원에서 풀링되며, 전체 시퀀스길의의 절반이 됨. 3개의 블록이 있어, 원래 길이의 1/4의 길이.
  분류의 경우는 문제없으나, MLM/토큰분휴 같은 작업은 원래 입력과 시퀀스 길이가 동일한 은닉상태가 필요하며, 최종은닉상태는 입력길이로 업샘플링되고 두개의 추가 레이어를 거침.
  따라서 두가지의 버전이 존재하는데, -base접미사가 붙은 버전은 세개의 블록만, 없으면 3개의 블록+추가레이어가 있는 업샘플링 헤드가 포함됨.
- Longformer : 어텐션행렬을 희소행렬로 대체해 더 빨리 진행하는 트랜스포머모델. 









#
***
# 참고
- [1](https://docs.likejazz.com/bert/)
- [2](https://inhovation97.tistory.com/31)
- [3](https://mino-park7.github.io/nlp/2018/12/12/bert-%EB%85%BC%EB%AC%B8%EC%A0%95%EB%A6%AC/?fbclid=IwAR3S-8iLWEVG6FGUVxoYdwQyA-zG0GpOUzVEsFBd0ARFg4eFXqCyGLznu7w)
- [4](https://arxiv.org/abs/1810.04805)
- [5](https://huggingface.co/transformers/model_summary.html)
- [6](https://dladustn95.github.io/nlp/BART_paper_review/)
