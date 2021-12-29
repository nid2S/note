# pytorch

## device
- torch.cuda.is_available() : 현 환경에서 GPU 사용가능 여부를 반환.
- torch.device("cuda") : GPU연산 사용. ("cuda" if USE_CUDA(위의 결과) else "cpu")식으로, GPU 사용이 가능할 때만 사용하게 사용. 
- 모델(함수).to(device(위의 결과)) : 연산을 수행할 위치를 지정.  

- 텐서.cpu() : cpu 메모리에 올려진 텐서 반환.

## tensor
- 텐서 : pytorch의 행렬(데이터)를 저장하는 자료형. numpy의 ndarray와 비슷함. 인덱스접근, 슬라이싱 등이 전부 가능함.
- 브로드 캐스팅 : 크기가 다른 행렬(텐서)들의 크기를 자동으로 맞춰 연산을 가능하게 해주는 기능. 연산시 더 큰 차원에 맞춰짐(요소 복제).

- torch.tensor(i) : 텐서 생성. .item()으로 값을 받아올 수 있음. 
- torch.자료형Tensor(array) : array로 지정된 자료형의 텐서 생성(ex-Float:32bit 부동소수점). 
- torch.zeros(shape) : 0으로 초기화된 shape의 텐서 생성.
- torch.ones(shape) : 1으로 초기화된 shape의 텐서 생성.
- torch.range(start, end, step) : start~end까지 step의 간격으로 채워진 텐서 생성. python내장함수 range와 동일하게 작동.  
- torch.rand(shape) : shape의 랜덤으로 값이 할당된 텐서 생성.
- torch.randn(shape) : shape의, 표준정규분포(평균0, 분산1)내의 범위에서 랜덤으로 값이 할당된 텐서 생성.
- torch.randint(low, high, shape) : shape의, low~high의 범위에서 랜덤으로 값이 할당된 텐서 생성. low는 포함, high는 미포함.
- torch.텐서생성함수_likes(텐서) : 텐서와 동일한 shape의, 텐서 생성함수로 생성할 수 있는 텐서 생성.

- required_grade = bool : 텐서.grad에 텐서에 대한 기울기를 저장. 텐서 생성시 매개변수로 줄 수 있음. 
- 텐서.backword() : 역전파. 해당 수식의 텐서(w)에 대한 기울기를 계산. w가 속한 수식을 w로 미분.

- 텐서에 식 적용 : 텐서 + a , 텐서 > 0.5 등 텐서를 식에 사용하면 텐서내의 모든 데이터에 적용됨(applymap).
- 텐서.shape/dim()/sum()/argmax()/max(-dim=i-)/mean(-dim=i-)/matmul(텐서)/mul(텐서) : 텐서에 대해 사용할 수 있는 연산들. dim 인자는 해당 차원을 제거(해당 차원을 1로 만듦)함.
- 텐서.size(index) : index차원의 차원 수 반환.
- 텐서.view(shape) : 텐서의 크기(차원)변경. numpy의 reshape와 같이 전체 원소수는 동일해야 하고, -1 인자를 사용할 수 있음((out.size(0), -1)(첫 차원 제외 펼침)식).
- 텐서.view_as(텐서) : 텐서의 크기를 입력한 텐서와 동일하게 변경. 마찬가지로 데이터의 개수는 동일해야 함.  
- 텐서.squeeze() : 차원의 크기가 1인 경우 해당차원 제거.
- 텐서.unsqueeze(i) : i 위치(shape의 위치)에 크기가 1인 차원을 추가.
- 텐서.scatter(dim, 텐서, 넣을 인자) : dim차원에서, 텐서의 데이터(내부 데이터를 인덱스로)대로 넣을 인자를 삽입(할당).
- 텐서.detach() : 현재 그래프에서 분리된 새 텐서 반환. 원본과 같은 스토리지를 공유.
- 텐서.numpy() : 텐서를 넘파이배열(ndarray)로 변경.
- 텐서.자료형() : 텐서의 자료형을 변환(TypeCasting).
- 텐서.연산_() : 기존의 값을 저장하며 연산. x.mul(2.)의 경우 x에 다시 저장하지 않으면 x엔 영향이 없으나, x.mul_()은 연산과 동시에 덮어씀.

- 텐서.cpu() : cpu 메모리에 올려진 텐서 반환.

- 텐서.eq(텐서) : 텐서와 입력된 텐서의 데이터가 동일한지 반환.
- 텐서.data.sub_(1) : 값을 0과 1로 변환.  
- torch.log(텐서) : 텐서의 모든 요소에 로그를 적용.
- torch.exp(텐서) : 텐서의 모든 요소에 ln(log_e)를 적용.
- torch.max(텐서) : 텐서 내부의 요소중 최댓값을 텐서로 반환. 이 외에도 텐서.연산()으로 사용가능한 모든 연산은 torch.연산(텐서)으로 사용가능.  
- torch.argmax(텐서) : 텐서 내부의 요소중 최댓값의 인덱스를 반환. dim=i 매개변수를 사용해 특정 차원을 기준으로 볼 수 있음(없으면 전체 요소).
- torch.cat(\[텐서1, 텐서2], dim=i) : i 번째 차원을 늘리며 두 텐서를 연결. 기존 차원을 유지한채 지정 차원의 크기만 커짐.
- torch.stack(\[텐서1, 텐서2, 텐서3], -dim=i-) : 텐서(벡터)들을 순차적으로 쌓음. 차원이 하나 늘어남. i번 차원이 늘어나게 함.
###### tensor expression
- 2D Tensor : (batch size, dim)
- 3D Tensor : (batch size, length(time step), dim)


## model
- 가설 선언 후 비용함수, 옵티마이저를 이용해 가중치, 편향등을 갱신해 올바른 값을 찾음(비용함수를 미분해 grandient(기울기)계산). 

- torch.save(모델.state_dict(), 경로) : 모델의 현재 가중치를 경로에 저장.
- 모델.load_state_dict(torch.load(경로)) : 경로의 가중치를 로드해 모델의 가중치로 사용. 
- 모델.embedding.weight.data.copy_(임베딩벡터들) : 사전훈련된 임베딩벡터값을 모델의 임베딩층에 연결. 
  임베딩벡터는[(필드에 저장)필드.vocab.vectors]로 확인. data까지만 쓰면 임베딩벡터 확인 가능. 

- 모델.eval() : 모델을 추론모드로 전환. 모델 test시 사용.
- torch.no_grad() : 미분을 하지 않음. 파라미터를 갱신하지 않는 test시 사용.  
- torch.nn.init.xavier_uniform_(self.층.weight) : 특정 층 한정으로 가중치 초기화. 형태 변경을 위한 전결합층 등 파라미터 갱신을 원하지 않는 층에 사용.

- torch.manual_seed(i) : 랜덤시드 고정.
- torch.cuda.manual_seed_all(i) : GPU 사용시 랜덤시드 고정.
### class
- 파이토치의 대부분의 구현체(모델)는 모델 생성시 클래스를 사용.
- torch.nn.Module상속 클래스 구현 > __init__에서 super().__init__을 호출, 사용할 모델(층)정의 > forward(self,x)(자동실행, 모델 사용 후 값 반환).
- self.레이어명 = 층 : 사용할 모델의 층을 정의. 층의 경우 torch.nn의 모델도, 시퀀셜 모델이 될 수 도 있음.
```python
# 파이토치 모델 클래스 구현.
import torch
class LinearRegressionModel(torch.nn.Module):
    def __init__(self): #
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):  # 모델을 데이터와 함께 호출하면 자동실행.
        return self.linear(x)
model = LinearRegressionModel()
```

### data
- torch.utils.data.TensorDataset(x, y) : 데이터들을 TensorDataset(PyTorch기본 데이터셋)을 이용해 데이터셋에 저장.
- torch.utils.data.DataLoader(dataset, batch_size=i) : 데이터셋을 i개의 미니배치로 학습시킴. shuffle=bool(Epoch마다 데이터 학습순서를 섞음)매개변수와
  drop_last=bool(batch_size로 배치를 나눈 뒤, 남은(batch_size 미만의)데이터셋을 버릴지)매개변수 사용가능. .dataset으로 내부의 데이터셋 확인가능.
  반환되는건 iterable객체로, enumerate를 이용해 batch_idx와 sample(x, y)을 꺼낼 수 있음. 반복문이 하나 추가되는걸 제외하고는 배치학습법과 동일.  

- 모델.parameters() : 모델의 파라미터들을 꺼냄. [p.requires_grad]로 해당 파라미터가 학습이 되는지, [p.numel()]로 해당 파라미터의 값을 볼 수 있음.   

- 커스텀 데이터셋 구현 : torch.utils.data.Dataset을 상속받는 클래스 제작 후 __init\__(전처리), __len\__(길이, 총 샘플 수), __getitem\__(특정샘플, 인덱스)을 구현해 제작.
```python 
# 커스텀 데이터셋
class CustomDataset(Dataset): 
  def __init__(self):
    self.x_data = [[73, 80, 75],
                   [93, 88, 93],
                   [89, 91, 90],
                   [96, 98, 100],
                   [73, 66, 70]]
    self.y_data = [[152], [185], [180], [196], [142]]

  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.x_data)

  # 인덱스를 입력받아 그에 맞는 입출력 데이터를 Tensor로 리턴
  def __getitem__(self, idx): 
    x = torch.FloatTensor(self.x_data[idx])
    y = torch.FloatTensor(self.y_data[idx])
    return x, y
```


### activation function
- torch.nn.functional.relu(텐서) : 렐루(ReLU)사용. F.relu(층) 처럼 사용. 모델 제작시 활성화 함수를 꼭 사용해 주어야 함.
- torch.nn.functional.sigmoid(텐서) : 시그모이드 사용. torch.sigmoid(텐서(식))로도 사용가능. 
- torch.nn.functional.softmax(텐서) : 소프트맥스 사용. dim=i매개변수(적용될 차원 선택)사용가능. 손실함수에 포함되어있어 잘 쓰이지 않음.
- torch.nn.functional.log_softmax(텐서) : 로그 소프트맥스 사용. torch.log(F.softmax())와 동일.

### loss
- torch.nn.functional.mse_loss(prediction, label) : MSE(평균제곱오차) 손실함수 사용.
- torch.nn.functional.binary_cross_entropy(prediction, label) : 이진분류(로지스틱 회귀)의 손실함수 사용. reduction인자에 'sum'등을 넣어 출력에 적용할 축소를 지정할 수 있음.
- torch.nn.functional.cross_entropy(prediction, label) : cross-entropy 손실함수 사용. F.nll_loss(F.log_softmax(z, dim=1), y)와 동일함.

### optimizer
- 옵티마이저.zero_grad() : gradient 0으로 초기화.
- 옵티마이저.step() : 주어진 학습대상들을 업데이트.
- 옵티마이저 매개변수 : 학습시킬 매개변수들, lr(learning rate), weight_decay(가중치감쇠(L2규제)의 강도)등의 매개변수 사용가능.
```python
# 사용 예
import torch
x1 = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2 = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3 = torch.FloatTensor([[75], [93], [90], [100], [70]])
y = torch.FloatTensor([[152], [185], [180], [196], [142]])

w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD([w1, w2, w3, b], lr=1e-5)
epoch = 1000
for i in range(epoch):
    # 선형회귀 H(x) 계산
    # hypothesis = X.matmul(W) + B      # 행렬 연산으로 식을 간단히 함.
    hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b
    # cost 계산(손실함수 : MSE)
    cost = torch.mean((hypothesis - y) ** 2)
    # cost로 H(x) 개선
    optimizer.zero_grad()   # 옵티마이저 초기화
    cost.backward()         # 기울기(식을 w로 미분한 값)계산
    optimizer.step()        # 옵티마이저를 이용해 주어진 값들을 업데이트
    print(f'Epoch: {i}/{epoch} w1: {w1.item()} w2: {w2.item()} w3: {w3.item()} b: {b.item()} Cost: {cost.item()}')
```
- torch.optim.SGD(\[가중치(학습대상1), 편향(학습대상2)], lr = learning_rate) : SGD(확률적 경사하강법)사용. 모델.parameters()를 넣을 수 있음.
- torch.optim.Adam(모델 파라미터, lr) : 아담 옵티마이저 사용.

### module(layers)
- 모델.parameters() : 모델의 파라미터 출력. w와 b가 순서대로 출력됨. 
  
- torch.nn.Linear(input_dim, output_dim) : 선형회귀모델/전결합층 사용. 이대로 모델로 쓸 수도, 모델에 층으로 넣을수도 있음. bias=bool 로 편향 존재여부 지정가능.
- torch.nn.Conv2d(input_dim, output_dim, kernel_size) : (2차원)CNN층 사용. i의 커널사이즈를 가짐. padding, stride등도 설정해줄 수 있음. 
- torch.nn.MaxPool2d(kernel_size, stride) : (2차원)맥스풀링층 사용. 하나의 정수만 넣으면 커널사이즈와 스트라이드 둘 다 해당값으로 지정됨.
- torch.nn.RNN(input_size, hidden_size) : RNN층 사용. batch_first(bool, 입력텐서의 첫번째 차원은 배치크기), 
  num_layer(int, 은닉층개수(깊은RNN으로 만듦)), bidirectional(bool, 양방향순환신경망으로 만듦)인자 사용가능.  
- torch.nn.LSTM(input_size, hidden_size) : LSTM층 사용. RNN과 동일한 인자 사용가능. RNN계열은 사용시(x, h_0)를 입력으로 해야 하며, h_0는 처음에 초기화가 필요함. 
- torch.nn.GRU(input_size, hidden_size) : GRU층 사용. RNN과 동일한 인자 사용 가능. 

- torch.nn.Embedding(num_embedding, embedding_dim) : 학습가능한 임베딩 테이블 생성. .weight 로 벡터 확인 가능. 이후 층에서 input_size를 embed_dim으로 변경해주어야 함.
  num_embedidng(단어집합 크기(임베딩할 단어개수)), embedding_dim(임베딩벡터의 차원)와 선택적으로 padding_idx(패딩을 위한 토큰의 인덱스)인자 사용가능.
- torch.nn.Embedding.from_pretrained(임베딩 벡터(필드.vocab.vectors), freeze=False) : 사전휸련된 임베딩벡터를 사용해 임베딩층 생성.
- torch.nn.Dropout(f) : f의 비율로 드롭아웃을 시행하는 층 사용. 

- torch.nn.Sigmoid() : 활성화함수 시그모이드 층을 쌓음. Linear() > Sigmoid() 로 로지스틱 회귀 구현 가능.
- torch.nn.ReLU() : 활성화함수 ReLU(렐루)층을 쌓음.
  
- torch.nn.CrossEntropyLoss() : cross-entropy 손실함수 층 사용. softmax함수가 포함되어있음. 
  ignore_index인자에 무시할 토큰의 인덱스를 전달해(pad 등)손실함수가 연산에 포함시키지 않게 할 수 있음.
  CrossentropyLoss는 input(pred)는 float, target(y)는 long이여야 하며, loss가 스칼라가 아니라면 loss_.backward(gradient=loss_)로 backward를 써야 한다.
- torch.nn.BCELoss() : Binary-cross-entropy 손실함수 층 사용.

### model
- torch.nn.Sequential(module) : 시퀀셜 모델 생성. 클래스 형태로 구현되는 모델에서 층의 역할을 함. 아주아주 간단한 모델의 경우엔 모델 그 자체로 이용되기도 함.
- 시퀀셜모델.add_model("레이어명", 레이어) : 모델에 층 추가. 모델생성시 레이어를 넣어 생성하는것과 동일하나, 층의 이름을 지정할 수 있음.

## train/test
### train
- 옵티마이저 지정 : torch.optim.SGD([파라미터(모델.parameters())\], lr=1e-5)식으로, torch.optim의 함수들에 파라미터들과 하이퍼파라미터를 지정해 옵티마이저 생성가능.
- 모델 학습 과정 : optimizer.zero_grad()    : 가중치 초기화
                > model(X)                : 정의한 모델(가설)로 데이터를 예측, 예측값을 얻음 
                > loss_func(Y_pre, Y)     : 지정한 손실함수를 이용해 예측값과 레이블간의 손실(비용)계산 
                > loss.backward()         : 손실을 미분 
                > optimizer.step()        : 미분한 결과와 옵티마이저를 이용해 지정된 파라미터 갱신
                과정을 거쳐 optimizer에 인자로 준 텐서(가중치, 편향)를 갱신함.
- 미니배치 모델 학습과정 : 
      > torch.utils.data.TensorDataset(X, y)   : 데이터와 레이블로 데이터셋 생성. TensorDataset 대신 다른 함수를 사용할 수도 있음.
      > torch.utils.data.DataLoader(데이터셋, batch_size, shuffle=bool, drop_last=bool) : 데이터셋을 실어 미니배치를 생성함.
      > for data, targets in 데이터로더          : 데이터로더에서 지정한 배치사이즈만큼 데이터와 레이블을 가져옴.
      > 위의 full-batch모델 학습과정과 동일.
### test
- 모델 테스트 과정:
      model.eval() : 모델 추론모드로 전환
      > for data, target in 데이터셋 : 로더에서 미니배치를 하나씩 꺼내 추론을 수행
      > model(data) : 데이터를 이용해 출력 계산
      > [_, predicted = torch.max(outputs.data, 1)] : 확률이 가장 높은 레이블 계산(다중분류).
      > [count += predicted.eq(targets.data.view_as(predicted)).sum()] : 정답과 일치한 경우 카운트 증가(다중분류).
      > [count += (targets == (output > 0.5).float()).float().sum()] : 예측값 1/0으로 변환 후, 정답과 일치한 경우 카운트 증가(이진분류).
      > [count/len(데이터로더.dataset)] : 정확도(accuracy)계산.


# torch_%
- torchvision : 비전분야의 유명 데이터셋, 모델, 전처리도구가 포함된 패키지.
- torchtext : 자연어처리 분야의 유명 데이터셋, 모델, 전처리도구(텍스트에 대한 추상화기능)가 포함된 패키지.
- torchaudio : 오디오 분야의 torch_%. 
## vision
- torchvision.datasets.MNIST(경로, train=bool, transform=트랜스폼, download=bool) : MNIST 다운로드. 
  train=false면 test데이터 다운로드, download는 경로에 데이터가 없으면 다운로드받음.
- torchvision.transforms.ToTensor() : 받은 데이터셋을 어떻게 변환할지 선택, 텐서로 변환. 다운로드중 transform매개변수에 넣어 사용.
- 데이터.test_data : 테스트 데이터를 가져옴.
- 데이터.test_labels : 테스트 레이블을 가져옴.
## text
- 제공기능 : 파일로드(다양한 코퍼스 로드), 토큰화(단어단위), 단어집합, 정수인코딩(단어들을 고유한 정수로 맵핑), 단어벡터(단어들에 고유 임베딩벡터 제작), 
  패딩/배치화(훈련샘플의 배치화). 데이터의 분리와 단어-벡터간 맵핑(룩업테이블)은 별도로 해주어야 함.
- 데이터셋 생성 : 필드 정의(토크나이저, 데이터크기 등 전처리방법 정의) > 데이터셋 제작(필드에 따라 데이터 로드, 토큰화) > 단어집합 제작(데이터셋 이용, 정수화) > 
  데이터로더 제작(Iterator, 배치사이즈 정의)의 순서로 이뤄짐.

- torchtext.data.Field() : 필드(앞으로 할 전처리를 정의, 텍스트/레이블 등을 정의)지정. 
- Field인자 : sequential(bool, 시퀀스데이터 여부), use_vocab(bool, 단어집합생성 여부), tokenize(함수, 사용할 토큰화함수), lower(bool. 소문자화 여부),
  batch_first(bool, 미니배치 크기(fix_lenX배치크기)->(배치크기Xfix_len)), is_target(bool, 레이블데이터 여부), fix_length(int, 최대허용길이/패딩길이) 인자를 사용할 수 있음.
- 필드.build_vocab(데이터셋) : 단어집합 생성. vectors(사전훈련된 임베딩벡터 사용), min_freq(int, 단어의 최소등장빈도 조건 추가), 
  max_size(int, (특별토큰제외)단어집합 최대크기)인자 사용가능. 
  [필드.vocab]을 통해 단어집합에 접근할 수 있고, [필드.vocab.stoi]를 통해 생성된 단어집합 내의 단어를, [필드.vocab.vectors]로 벡터값을, 
  [필드.vocab.freqs.most_common(i)]으로 상위빈도수의 단어를(i가 없으면 전부)확인가능.

- torchtext.data.TabularDataset.splits() : 데이터셋을 만들며(데이터를 불러오며)필드에서 정의했던 토큰화방법으로 토큰화를 수행.
- TabularDataset.splits인자 : path(파일 경로), train/test(train,test파일명), format(데이터 포맷(csv 등)), 
  fields(위에서 정의한 필드. [("필드를 호칭할 이름", 필드)\]형식), skip_header(bool, 데이터 첫줄 무시 여부)인자 사용가능.
- torchtext.datasets.데이터셋이름.splits(TEXT필드, LABEL필드) : 데이터 셋을 필드에 가져온 후 train/test 데이터를 나눔. vars(나눠진데이터셋[0\])등으로 데이터 확인 가능.
- 데이터셋.split(split_ratio) : 데이터셋을 나눔. train데이터를 나눠 검증 데이터를 만드는 등에 사용. 지정한 비율이 첫번째 데이터셋의 데이터 비율.

- torchtext.data.Iterator(데이터셋, batch_size=i) : 데이터셋을 이용해 i의 배치크기 만큼 데이터를 로드하게 하는 데이터로더 생성. 배치.필드명 으로 실제 데이터텐서에 접근가능.
- torchtext.data.BucketIterator(데이터셋, batch_size, shuffle=bool, repeat=bool) : 모든 데이터를 배치처리 후 단어를 인덱스 번호로 대체하는 데이터로더 생성. 
  (데이터셋1,데이터셋2)처럼 넣어 여러 데이터셋에도 적용 가능.
- 데이터로더에서 반복문으로 각 배치를 꺼낼 수 있으며, batch.text 로 해당 배치의 실제 데이터에 접근가능.

- torchtext.vocap.Vectors(name=W2V파일명) : 사전훈련된 Word2Vec모델 사용.
- torchtext.vocab.Glove(name, dim) : 토치텍스트 제공 사전훈련된 임베딩벡터(영어)사용. (6B, 50/100/200/300)등이 있음.필드.build_vocap()의 vectors 인자의 입력으로 사용.
## audio
- audioform, sample_rate = torchaudio.load(path) : 오디오파일 로드. 다양한 파일 형식을 지원함. 
- torchaudio.transforms.Resample(sample_rate, new_sample_rate)(audioform) : 리샘플링.

- torchaudio.transforms.Spectrogram()(audioform) : 스펙토그램 확인.
- torchaudio.transforms.MelSpectrogram()(audioform) : Mel스펙토그햄 확인.

- torchaudio.transforms.MFCC()(audioform) : 파형을 MFCC(스피커 식별, 사운드 분류/수정/분석 등에 적합)로 바꿈.
- torchaudio.transforms.MuLawEncoding(audioform) : MuLaw인코딩 적용. 이전에 정규화(-1 ~ 1로)를 거쳐야 함.
- torchaudio.transforms.Fade(fade_in_len= 30000,fade_out_len=30000)(audioform) : 사운드에 페이딩효과를 일으킴. 음파의 시작/끝에서 적용.
- torchaudio.transforms.SlidingWindowCmn()(audioform) : 이퀄라이즈(고음-고주파/저음-저주파 조절).

- torchaudio.functional.gain(audioform, gain_db=5.0) : 오디오 증폭. 파동 전체의 볼륨을 증가시키는 효과도 있음.
- torchaudio.functional.dither(audioform) : 오디오 디더링(Dithering, 높은 bit 해상도 > 낮은 bit 해상도).
- torchaudio.functional.lowpass_biquad(audioform, sample_rate, cutoff_freq=3000) : 특정 주파수 미만의 오디오만 허용. 한계를 벗어나면 감쇠.
- torchaudio.functional.highpass_biquad(audioform, sample_rate, cutoff_freq=2000) : 특정 주파수 초과의 오디오만 허용. 한계를 벗어나면 감쇠.


# Lightning/Ignite
- 텐서플로우의 keras와 같이, High-level인터페이스를 제공하는 오픈소스 라이브러리들. ML사용자들에세 좋음.
## pytorch_lightning
- LightningModule : 모델 내부의 구조를 설계하는 research/science클래스. 모델구조/데이터전처리/손실함수 설정 등 모델 초기화/정의. 
  모든 모듈이 따라야 하는 9가지 필수메서드의 표준 인터페이스를 가지고 있음.
- LightningModule사용 : pl.LightningModule을 상속받는 클래스 생성 후 구조 정의. 많은 코드가 함수(추상화)형태로 LightningModule안에 포함되어있음. init/forward/손실함수/optimizer,
  학습루프(training_step - validation_step - validation_epoch_end), 데이터준비(prepare_data(), train_dataloader, val_dataloader, test_dataloader)등이 포함.
- Lifecycle : LightningModule클래스가 함수를 실행하는 순서. 
  [__init\_\_ > prepare_data > configure_optimizers > train_dataloader > val_dataloader > test_dataloader(.test()호출시)]의 순서.
  여기에 각 배치/epoch마다 루프메소드는 validation_step(배치마다 실행), validation_epoch_end(에폭마다 실행)를 실행함.

- pytorch_lightning.Trainer() : 트레이너 객체 생성. 모델의 학습에 관여되는 engineering(학습epoch/batch/모델의 저장/로그생성까지 전반적으로)을 담당.
- 트레이너.fit(LightningModule모델) : 모델 학습. sklearn의 fit메서드와 비슷함.
- 트레이너.test() : fit한 LightningModule모델 테스트. 
- LightningModule.load_from_checkpoint(모델경로) : 사전훈련된(저장된)모델 로드.

## Ignite
- Ignite : PyTorch지원 라이브러리. Lightning과 달리 표준 인터페이스를 가지고있지 않음.

