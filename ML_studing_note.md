#AI
- 인공지능 > 지능적인 인간의 행동을 모방하는 기계의 능력.
- 심볼릭 AI(symbolic AI) > 프로그래머들이 명시적인 규칙을 충분히 많이 만들면 일반지능(인간수준 인공지능)을 만들 수 있다는 접근 방법. 전문가 시스템과도 연관되어 있음.
- 머신러닝(Machine Learning) > 인공지능의 한 분야, 알고리즘을 이용해 데이터를 분석하고, 그를 통해 학습하며, 그 내용을 기반으로 판단이나 예측을 함. 지도학습과 비지도 학습이 포함되어 있음.
- 딥러닝(Deep Learning) > 인공 신경망에서 발전. 데이터 군집화나 추상화를 시도함(여러 비선형 변환기법의 조합을 통함). 머신러닝과 달리 데이터 학습량이 늘어도 정확도 향상에 한계가 오는 지점이 없음.

- 머신러닝은 최적화와 일반화를 잘 조절해야 함. 100%에 가까운 모델을 만들고, 그 후 일반화를 진행. 데이터 사이즈가 같을때 정확도를 높이려면 층을 추가하고, 크기를 키우고, epoch 를 높이면 됨. 

- 모델을 학습할 때에는 데이터 형태확인(data Engineering 으로 데이터를 다듬거나 새 특성을 찾아내기도 함) > 전처리(normalization, one-hot-encoding) > 모델 생성 및 학습으로 이뤄진다.

##용어
***
- 일반화 > 모델이 처음보는 데이터를 정확하게 예측할 수 있다면, 그것을 훈련세트에서 테스트 세트로 일반화 되었다고 함.
- 과대적합 > 모델이 훈련세트에 너무 가깝게 밎게되어 일반화 되기 어려울때 일어나는 것. 이를 막기위해 모든 특성이 출력에 주는 영향을 최소한으로 만들게(규제)한다.
- 과소적합 > 모델이 너무 간단해 데이터의 면면과 다양성을 잡아내지도 못하고, 훈련세트에도 맞지 않는것. 테스트 세트의 성능이 좋았더라도 테스트 세트와 점수가 비슷하다면 과소적합일 가능성이 있어, 모델의 성능을 더욱 올릴 수 있다.
- 편미분 > 변수가 여러개 있을 때, 어떤 변수를 미분힐지 명시하는 것.

##딥러닝 활성화 함수(activation function)
***
- swish : x*sigmoid(x). 매우 깊은 신경망에서 ReLU 보다 높은 정확도를 가지며, 모든 배치 크기에 대해 ReLU 를 능가함.
- Relu(Rectified Linear Unit) : (X > 0)? X : 0
- Sigmoid : 입력을 전부 0~1의 미분가능한 수로 변환.  |  s(z) = 1/1+e^-z
- tanh(Hyperbolic Tangent) : 입력을 -1~1의 미분 가능한 수로 변환. 시그모이드의 대체제. 시그모이드와 함께 Vanishing gradient problem 을 가지고 있음. | (2/1+e^-2x) - 1
- softmax : 입력을 전부 0~1사이로 정규화. 출력의 총합이 1.  |  np.exp(x  - np.max(x)) / np.exp(x  - np.max(x)).sum()

##활성화 도구(Optimizer)
***
- GD(Gradient descent) > 가장 기본 알고리즘. 경사를 따라 내려가면서 W 업데이트. 
- SGD(Stochastic gradient decent) > full-batch 가 아닌 mini-batch 로 학습.
- Momentum > SGD + Momentum(이전 batch 학습결과 반영, 보통 이전:현재 = 9:1 정도)
- AdaGrad > SGD + notation. 큰 변동 가중치 = 학습률 감소, 저변동 가중치 = 학습률 증가. 무한히 학습시 학습이 아예 안될 수 있음.
- RMSProp > AdaGrad 보완. 가중치보다 기울기가 크게 반영되도록 하고, hyper parameter p를 추가해 h가 무한히 커지지 않게 함.
- Adam > RMSProp + Momentum. 각각 v와 h가 0으로 초기화 되면 학습 초반 W가 0으로 biased 되었는데, 이를 해결.
- lbfgs >  Limited BFGS. 준-뉴턴 방식 (quasi-Newton methods)의 알고리즘 중 가장 흔히 쓰이는 방법. 많은 변수를 가진 최적화 문제에 적합.  제한된 메모리 내 에서 f(x)(스칼라 함수, 비선형이며 미분가능함수.)를 제한 조건이 없는 실수 벡터 x에 대해서 최소화 시키는 것

##손실함수(loss)
***
- 텐서 계산 > y값 산출 > 손실함수에 이용 > 손실 산출    의 구조로 이어진다.
- MSE(평균제곱오차) > 개별 예의 모든 제곱 손실을 합한 뒤 예의 수로 나눔. 선형 회귀 모델.
- categorical_crossentropy > 다중 분류 모델의 손실 함수. one-hot-encoding 된 결과로 입력을 해 주어야 하며, 3개의 클래스 별로 확률값이 나오게 된다. 
- sparse_categorical_crossentropy > 분류 모델의 손실 함수. one-hot-encoding 을 할 필요 없이 정수형태(클래스 번호)로 결과값을 입력해주면 된다,


##정확도 계산(metrics)
***
- Accuracy(정확도) : 모든 예측이 실제로 맞은 비율
- Precision(정밀도) : True 라고 예측한 것중 실제로 True 인 비율.
- Recall(재현율) : 모든 true 중 true 로 예측한 것의 비율. 이것을 높이기 위해 진실을 남발하면 정확도가 떨어지고, 정확도를 높이기 위해 진실을 자제하면 이것이 낮아진다는 관계를 가지고 있다.
- F1-score : 정밀도와 재현율의 조화평균 


##ML
***
### 지도학습
입출력 데이터기반 예측

#### 모델
- 최근접 이웃 : 작은 데이터셋일 경우 기본 모델로 좋고, 설명도 편함.
- 선형 모델 : 대용량, 고차원 데이터셋에 사용 가능.
- 나이브 베이즈 : 분류 전용. 선형 모델에 비해 훨씬 빠름. 대용량, 고차우너 데이터셋에 사용가능.
- 결정 트리 : 매우 빠르며 데이터 스케일 조정이 필요 없음. 시각화와 설명하기 좋음.
- 랜덤 포레스트 : 결정트리 하나보다 좋은 성능을 냄. 안정적이고 강력하며 데이터 스케일 조정이 필요 없지만 고차원 희소 데이터와는 안 맞음.
- 그래디언트 부스팅 결정 트리 : 랜덤 포레스트보다 성능이 좋고 예측리 빠르며 메모리를 덜 사용하지만 힉습이 느리고 매개변수 튜닝이 많이 필요함.
- 서포트 벡터 머신 : 비슷한 의미의 특성으로 이뤄진 중간규모 데이터 셋에 잘 맞음. 데이터 스케일 조정이 필요하고 매개변수 튜닝이 맣이 필요함.
- 신경망 : 대용량 데이터 세트에서 복잡한 모델을 만들 수 있음. 매개변수 선택과 대이터 스케일에 민감. 큰 모델은 학습이 오래 걸림.

>####분류
>- 나올 수 있는 응답이 개별적(국가나 언어등 둘 사이에 무언가가 없음). 레이블이 이산형 범주. 
>
>- 이진 분류 - 범주 두개(y/n). 대부분의 선형 분류가 속함(로지스틱 제외).
>- 다중분류 - 범주 새개 이상. 이진분류를 다중분류로 확장하기 위해서는 일 대 다 라는 방법을 사용함.
>- 일 대 다 - 각 클래스를 다른 모든것과 비교하도록 훈련시킴.
> 
>- F1-Measure - 데이터가 불균형할 때, 정확도 만으로는 성능 측정이 어려워 통계적으로 보정해주는 방법. 다수보다 소수집단의 정답 여부를 더 크게 반영.
>- K-fold Test - 전체 데이터를 다양한 방법으로 쪼개 훈련,테스트,검증의 과정을 여러번(K번) 반복하며 테스트가 편향되어 있지 않고 설명력을 가지게 하려 시행. 
>
>- AutoEncoder - 입력데이터와 출력 데이터를 같게 하고 중간에 레이어를 넣어 원복하게 만드는 구조. 입력 데이터에 대한 일종의 패턴을 찾아냄. 이진 분류시, 정상만 훈련시킨 뒤 Logistic Regression 모델에 이것에 넣었을 때의 값을 넣어 분류할 수 있도록 구성하는데 사용된다. 훈련 데이터가 적을 때 사용된다.
> ```python  # auto encoder clone
> import tensorflow as tf
> import tensorflow.keras.layers as layers
> import tensorflow.keras.models as models
> 
> n_inputs = x_train.shape[1]
> n_outputs = 2
> n_latent = 50
> 
> inputs = tf.keras.layers.Input(shape=(n_inputs, ))
> x = tf.keras.layers.Dense(100, activation='tanh')(inputs)
> latent = tf.keras.layers.Dense(n_latent, activation='tanh')(x)
> 
> # Encoder
> encoder = tf.keras.models.Model(inputs, latent, name='encoder')
> encoder.summary()
> 
> latent_inputs = tf.keras.layers.Input(shape=(n_latent, ))
> x = tf.keras.layers.Dense(100, activation='tanh')(latent_inputs)
> outputs = tf.keras.layers.Dense(n_inputs, activation='sigmoid')(x)
> 
> # Decoder
> decoder = tf.keras.models.Model(latent_inputs, outputs, name='decoder')
> decoder.summary()
> ```
>
> ###### 분류 모델 종류
>- Naive Bayes(NB) - 선형 분류기보다 훈련 속도가 빠르지만 일반화 성능이 조금 뒤짐.
>- GaussianNB - 연속적인 어떤 데이터에도 적용가능. 각 특성의 표준편차와 평균을 저장. 고차원의 데이터셋에 사용.
>- BernoulliNB - 이진데이터에 적용. 각 클래스의 특성 중 0이 아닌것을 셈. 커질수록 모델이 단순해지는 alpha 가 있음.
>- MultinomialNB - 카운트(count) 데이터에 적용. 클래스별 특성의 평균을 계산. alpha.
>
>- 


> ####회귀
>- 나올 수 있는 응답이 연속적. 레이블이 연속형인 숫자.
> 
> ######회귀 모델 종류


> ##### 양쪽 전부 사용가능
>- 결정트리 :  예/아니오를 반복하며 학습. 각 분열된 영역(리프)가 하나의 타깃값을 가질때 까지 반복. 이때의 리프노드를 순수노드라고 함.  
>- 과대적합을 막기 위해 가지치기(사전 - 최대 깊이, 개수 제한, 최소 포인트 개수 제한 | 사후 - 데이터가 적은 노드 삭제)를 해줘야 함.
> 
>- 앙상블 : 여러 모델을 연결해 더 강력한 모델을 만드는 기법. 랜덤 포레스트와 그래디언트 부스팅등이 있다.
>>- 랜덤 포레스트 : 조금씩 다른 여러 결정 트리의 묶음. 다른 방향으로 과대적합된 트리들을 평균냄. 여러개의 데이터중에서 무작위로 만들어낸 데이터의 부트스트랩 샘플을 생성한다. 모든 츠리에 대한 예측을 만든 후, 그 예측을 평균하거나(회귀) 예측한 확률을 평균내어(분류) 예측값을 나타낸다. 트리가 많을수록 랜덤값에 영향을 덜 받는다. 많은 트리는 메모리와 긴 훈련시간을 부른다. 차원이 높고 희소한 데이터에는 잘 작동하지 않는다.
>>- 그래디언트 부스팅 회귀트리 : 약한 학습기 사용. 이전 트리의 오차를 보완하는 방법으로 순차적으로 만듦. 무작위성이 없고, 적은 메모리와 예측도 빠름. 랜덤포레스트 보다 매개변수의 영향을 더 많이 받는다. 커질수록 보정을 많이해 복잡한 모델을 만드는 러닝 레이트 매개변수를 가지고 있다. 랜덤 포레스트보다 조금 더 불안정하다. 
>>- 배깅 : Bootstrap aggregating 의 줄임.  랜덤 샘플링으로 훈련세트를 각기 달리 훈련시킨 뒤 확률값을 평균하거나 빈도가 가장 높은 예측결과 예측값이 된다.
>>- 엑스트라 트리 : 후보 특성을 무작위 분할 후 최적의 분할을 민듦. 랜덤 포래스트와 다른 방식으로 모델에 무작위성을 주입. 
>>- 에이다 부스트 : 약한 학습기 사용. adaptive Boosting. 이전 모델이 잘못 분류한 샘플의 가중치를 높임. 각 모델은 성능에 따라 가중치가 부여. 깊이 1의 트리 사용.
> 
>- 커널 기법 : 선형 모델(분류기)을 새로운 특성ㅇ르 많이 만들지 않고도 학습시키기 위한 수학적 기교.
>- 커널 서포트 벡터 머신 : 커널 기법을 이용한 SVM. 데이터의 특성이 몇 개 안 되도 복잡한 결정 결계를 만들 수 있으나 샘플이 많으면 잘 맞지 않는다.


***

## DeepLearning
***
### 인공신경망 
- 학습이 오래 걸리고 데이터 전처리에 주의해야 한다는 문제점이 있음. 같은 의미를 가진 동질의 데이터에서 잘 작동함.
- 입력층 : 데이터벡터화 
- 은닉층 : 수많은 뉴런 조합, 가중치에 따라 미분계산
- 출력층 : 결과로의 판단

- 학습 진행시 오차의 합이 최소화 하는 방향으로 모델 생성 > 이때 가장 일반적으로 경사하강법이 사용됨. 
- 경사 하강법 : 접선의 기울기가 최소가 되는 지점을 찾음. 이때 미분 사용.
- 이 과정에서 Learn Rate 를 적절한 값으로 설정하지 않으면 Local minima 에 빠지거나 연산이 너무 늦어질 수 있음. 

- 손실함수 : 출력값이 기대값보다 얼마나 벗어나는지 측정. 
- 옵티마이저 : 손실함수로 산출된 점수에 의해 기중치 값을 조금씩 수정히는 괴정을 담당. 이 괴정을 역전파라고 함.
- 역전파 : 옵티마이저의 수정 과정. 최종 손실값에서 부터 각 파라미터가 기여한 정도를 계산하기 위해 미적분의 연쇄법칙을 사용해 최상위 층에서부터 거꾸로 걔산됨.

###### 전처리
- 데이터의 스케일에 매우 민감한 알고리즘들은 그에 맞게 데이터의 특성값을 조정해야 하며, 특성마다 스케일을 조정해 데이터를 변경함.
- 데이터 벡터화 : 입력 데이터를 텐서로 변경하는 것.
- 원 핫 인코딩 : 해당 정보는 1, 니머지는 0으로 표기하는 방법. 영양없는 정보는 0을 대입해 행렬 곱셈 연산등에서 빠른 연산속도를 얻는다.  
  
- 정규화 : 각 입력데이터의 범위나 크기가 다를경우, 네트워크 학습이 어려워 (일반적으로) 평균이 0이고 표준 편차가 1이 되는 0과 1 사이의 값으로 변환하는 것. 
- 정규 분포 : 모든 데이터를 (데이터-평균)/표준편차 로 정규화 해 평균이 0이고 표준 편차가 1이 되는 0과 1 사이의 값으로 변환한 분포.
- 균등 분포 : 모든 데이터를 같은 확률(비율)로 모아 0~1 사이에 분포시킴.
 
- StandardScaler : 각 특성의 평균을 0, 분산을 1로 변경해 모든 특성이 값은 크기를 가지게 함. 최솟값솨 최댓값의 크기를 제한하지는 않음. (데이터-평균)/표준편차.  
- RobustScaler : 특성들이 같은 스케일을 갖게 하지만 평균과 분산 대신 중간값과 사분위 값을 사용해 이상치(전체 데이터와 아주 동떨어진 데이터)에 영향을 받지 않게 함.
- MinMaxScaler : 모든 특성이 정확하게 0과 1 사이에 위치하도록 데이터 변경. 
- QuantileTransformer : 여러개의 분위(Quantile)를 이용해 데이터 균등 분포. 이상치에 민감하지 않으며 젠체 데이터를 0과 1 사이에 위치시킴.`` 
- Normalizer : 특성 벡터의 유클리디안 길이가 1이 되도록 데이터를 조정. 각 데이터가 다른 비율로 스케일 조정. 길이는 상관 없이 데이터의 방향(각도)만 중요할 떄 많이 사용됨.

- 주성분 분석 (PCA): 특성들이 통계적으로 상관관계가 없도록 데이터셋을 회전시키는 기술. 분산이 가장 크고 수직인 방향(주된 분산의 방향, 주성분)을 찾음. 주성분의 일부만 남기는 차원 축소 용도나 특성 추출에도 사용. 데이터 포인트를 일정 개수의 성분을 사용해 가중치 합으로 분해.
- 비음수 행렬 분해 (NMF) : 유용한 특성을 뽑아내기 위한 또다른 비지도 알고리즘. PCA 와 달리 음수가 아닌 성분과 계수값을 찾음. 음수가 아닌 특성을 가진 데이터에만 적용 가능. 패턴을 추출해 섞여있는 데이터에서 원본 성분을 구분할 수 있음(소리, 유전자 표현, 텍스트 데이터등에 적합함). PCA 에 비해 해석하기 쉬움.
- 매니 폴드 학습 : 위의 둘보다 월씬 복잡한 매핑을 민들어 더 나은 시각화를 제공함. 탐색적 데이터 분석에는 유용하나 지도학습용으로는 사용하지 않음. 
- t-SNE : 매니폴드 학습의 알고리즘. 훈련 데이터는 다른 데이터로 바꿀 수 있지만, 다른 새로운 데이터는 적용할 수 없음. 데이터 사이를 가장 잘 보존하는 2차원 표현을 찾음. 각 데이터 포인트를 무작위로 2차원에 배열한 후 원래와 가까운건 더 가깝게, 먼건 더 멀게 만듦.


###### ANN 성능 튜닝
- 미니 배치 : 모든 경우의 수를 계산하는(Full batch) 방법이 아닌 작은 양의 데이터를 분절해 최적값을 찾아나가는 방식.
- 가중치 규제 : 가중치의 절댓값에 비용을 추가하는 L1과 제곱에 비례하는 비용을 추가시키는  L2가 있음. L2는 가중치 감쇠라고도 함.
- 드롭아웃 추가 : 훈련 중 무작위로 층의 일부 출력특성을 제외시킴. 신경망 모델에 융통성을 부여.
- 네트워크 축소 : 모델의 학습 파라미터 수를 줄임. 또 층을 추가하거나 제거해 다른 구조를 시도.
- 하이퍼 파라미터 튜닝 : 하이퍼 파라미터(층의 유닛수, 옵티마이저 학습률, 배치 등)를 바꿔 훈련을 함.


### 신경망 용어
***
- ANN(Artificial Neural Networks) : 인공신경망. 딥러닝의 기초가 되고 있음. 파라미터의 최적값을 찾기 어렵다는 것과 Overfitting(과대적합, 훈련데이터보다 새 데이터에서 성능이 낮아짐) 문제가 있음.
- DNN(Deep Neural Networks) : 은닉층을 2개이상 지닌 학습 방법. 여기서 딥러닝이란 말이 파생.
- CNN(Convolutional Neural Networks, 합성곱신경망) : 데이터의 특징을 추출하여 특징들의 패턴을 파악. Convolution filter 를 사용하여 인식률을 높임. MaxPooling(해당 conv-행렬- 안의 숫자중 가장 큰 숫자만 남김). 딥러닝에서 이미지, 영상데이터등의 처리, 정보 추출, 문장분류, 얼굴인식 등에 사용.  
- RNN(Recurrent Neural Networks, 순환신경망) : 반복적이고 순차적인 데이터 학습에 특화. 내부의 순환구조가 있음. 시계열 데이터에 과거의 학습 구조를 저장해 현재 학습에 이용해 예측률을 높임. LSTM 과 GRU 가 있음. 음성, 텍스트 성분파악등 에 이용.
- GAN(Generative Adversarial Network,생성적 적대 신경망) : 비지도 학습, 제로섬 게임 틀 안에서 서로 경쟁하는 두개의 신경 네트워크 시스템에 의해 구현. fake 신경암을 추가해 서로 경쟁하여 더 좋은 성능을 내개 함. 
- AE(AutoEncoder) : 음성 합성등에 특화된 딥러닝 네트워크.

> #### GAN
>- DCGAN(Deep Convolution) : Convolution 필터만 사용하고 Max Pooling 은 사용하지 않음. 안정성 분제를 조금이나마 해결할 수 있다.
>- LSGAN(Least Squares) : 결정 상자에서 멀리 떨어진 데이터는 페널티를 줌.
>- SGAN(Semi-Supervised) : 데이터를 구분할 때 fake 클래스도 구분한다. 총 10개의 클래스가 있다면 fake 까지 총 11개의 클래스가 생긴다.
>- ACGAN(Auxiliary Classifier) : SGAN 에 Generator 가 학습을 진행할수록 좋은 이미지를 만들어내고 어느 순간부터 데이터가 augmentation 기능을 할 수 있다는 특징이 있다. 먼저 R/F 를 구분한 뒤 어떤 클래스인지 구분한다는 특징이 있다. 
>- cGAN : 기존 noise Z 만 가지고 무작위로 이미지를 생성했던 GAN 과 달리 특정 레이블 y 가 추가되며 특정 이미지만 고정적으로 생산가능하다. 실제 현실의 이미지는 너무 많은 변수가 있다는 문제점이 있었고, 이로 인해 이 개념을 이용해 복잡한 이미지나 영상까지 변경 가능하게 한 pix2pix 가 탄생했다.
>- pix2pix : 데이터 형태와 무관하게 범용적으로 사용 가능, 다른 종류의 손실함수(L1, L2, 유클라디안 > GAN 기반 Loss 학습) 사용 이라는 특징을 가지고 있다.
> ###### code (clone)
> ```python
> import tensorflow as tf
> import numpy as np
> import matplotlib.pyplot as plt
> 
> from tensorflow.examples.tutorials.mnist import input_data
> mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)
> print(mnist.train.images, mnist.train.labels)
> 
> # parameter
> total_epochs = 100
> batch_size = 100
> learning_rate = 0.0002
> n_hidden = 256
> n_input = 28 * 28
> n_noise = 128 
> 
> X = tf.placeholder(tf.float32, [None, n_input])
> Z = tf.placeholder(tf.float32, [None, n_noise])
> 
> # make generator
> G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
> G_b1 = tf.Variable(tf.zeros([n_hidden]))
> G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
> G_b2 = tf.Variable(tf.zeros([n_input]))
>
> def generator(noise_z):
>    hidden = tf.nn.relu(
>                    tf.matmul(noise_z, G_W1) + G_b1)
>    output = tf.nn.sigmoid(
>                    tf.matmul(hidden, G_W2) + G_b2)
>
>    return output
> 
> # make discriminator
> D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
> D_b1 = tf.Variable(tf.zeros([n_hidden]))
> D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
> D_b2 = tf.Variable(tf.zeros([1]))
> 
> def discriminator(inputs):
>    hidden = tf.nn.relu(
>                    tf.matmul(inputs, D_W1) + D_b1)
>    output = tf.nn.sigmoid(
>                    tf.matmul(hidden, D_W2) + D_b2)
>
>    return output
> 
> # make random noise
> def get_noise(batch_size, n_noise):
>    return np.random.normal(size=(batch_size, n_noise))
>
> G = generator(Z)  # make random image with noise
> D_gene = discriminator(G)  # get number of classification image's reality 
> D_real = discriminator(X)  # with real image
> 
> loss_D = -tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))
> loss_G = -tf.reduce_mean(tf.log(D_gene)
> 
> D_var_list = [D_W1, D_b1, D_W2, D_b2]
> G_var_list = [G_W1, G_b1, G_W2, G_b2]
> 
> train_D = tf.train.AdamOptimizer(learning_rate).minimize(loss_D, var_list=D_var_list)
> train_G = tf.train.AdamOptimizer(learning_rate).minimize(loss_G, var_list=G_var_list)
> 
> 
> sess = tf.Session()  # Launch session.
> sess.run(tf.global_variables_initializer())  # clear variable 
> 
> total_batch = int(mnist.train.num_examples/batch_size) 
> loss_val_D, loss_val_G = 0, 0  # set default loss
> 
> for epoch in range(total_epoch):
>     for i in range(total_batch):
>         batch_xs, batch_ys = mnist.train.next_batch(batch_size)
>         noise = get_noise(batch_size, n_noise)
> 
>         # train each NN (Generator and Discriminator)
>         _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise})
>         _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: noise})
> 
>     print('Epoch:', '%04d' % epoch,
>           'D loss: {:.4}'.format(loss_val_D),
>           'G loss: {:.4}'.format(loss_val_G))
> 
>     if epoch % 10 == 0:
>         sample_size = 10
>         noise = get_noise(sample_size, n_noise)
>         samples = sess.run(G, feed_dict={Z: noise})
> 
>         fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))
> 
>         for i in range(sample_size):
>             ax[i].set_axis_off()
>             ax[i].imshow(np.reshape(samples[i], (28, 28)))
> 
>         plt.savefig('./result/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
>         plt.close(fig)
> 
> print('최적화 완료')
> 
> ```

### 비지도학습
출력 없이 오직 입력만 입력된 데이터를 그룹화,분석. 스케일 조정등도 비지도.
레이블이 없기에 뭔가 유용한 것을 학습했는지 평가해야 하며, 그 결과를 확인하기 위해서는 직접 확인하는 것이 유일한 방법일 때가 많다는 과제가 있음.

#### 비지도 변환
- 비지도 변환 : 데이터를 새롭게 표현해 사람이나 다른 머신러닝 알고리즘이 원래 데이터보다 쉽게 해석할 수 있도록 하는 알고리즘. 차원축소( 특성이 많은 고차원 데이터 셋을 특성의 수를 줄이며 꼭 필요한 특징을 포함한 데이터로 표현 )분야에서 널리 사용. 데이터를 구성하는 단위나 성분을 찾기도 함. 

#### 군집 (clustering)
- 클러스터링 : 데이터를 비슷한 것끼리 그룹으로 묶음. 
>- 자동 군집 탐지. 데이터 마이닝 기법의 일환. 목표 변수 없이 패턴을 찾아냄. 비지도 기법이지만. 비즈니스의 목적에 띠라 지도활동이 추기되는 경우기 있음. (마케팅 분야, CRM, 고객 segmentation 에 활용)
>- 하드 클러스터링 : 각 레코드를 하나의 클러스터에 연관시킴. 대부분 이걸로 사용.
>- 소프트 클러스터링 : 각 레코드를 여러개의 클러스터에 연결시킴.  
>- K 평균 클러스터링 알고리즘 : 임의로 K개의 레코드 선택, 각 레코드를 가장 가까운 시드에 배정(군집간 경계 찾음), 군집들의 중심점을 찾음, 군집 생성 완료


## 강화학습(Reinforcement Learning)
***
- 비지도학습. 주어진 환경에서 에이전트의 행동을 통해 결과로 보상을 받아 모델을 학습해 나가는 과정.
- multi armed bandit : 여러개의 슬롯머신중 가장 많은 보상을 주는 기계를 찾아가는 강화학습 모델. 중간 중간 기계의 보상 설정 값이 변화하는 상황이 발생하며, 강화학습의 이해에 가장 좋은 방법.

