# pytorch
- pytorch : 과거 Torch 및 카페2 프레임워크를 기반으로, 파이썬을 스크립팅 언어로 사용하며 진화된 토치 C/CUDA백엔드를 사용하는 딥러닝 프레임워크. 
  GPU사용에 큰 장점을 가짐. 강력한 GPU가속과 파이썬으로 된 텐서, 동적 신경망을 지원함. 각 반복단계에서 즉석으로 그래프를 재 생성할 수 있음.
  torch와 numpy가 상호 호환되기 때문에, ndarray와 같이 인덱스를 조작할 수 있으며 생성된 텐서는 torch 내부적으로도, numpy와 호황되는 라이브러리에도 사용할 수 있음.
- 동적신경망 : 반복할때마다 변경가능한 신경망. 학습 중 숨은 계층을 추가하거나 제거해 정확성과 일반성의 개선이 가능. 
- 브로드 캐스팅 : 크기가 다른 행렬(텐서)들의 크기를 자동으로 맞춰 연산을 가능하게 해주는 기능. 연산시 더 큰 차원에 맞춰짐(요소 복제).
- TorchScript : Pytorch의 JIT(Just-In-Time)컴파일러. 기존의 명령적인 실행방식 대신, 모델이나 함수의 소스코드를 TorchScript컴파일러를 통해 TorchScript로 컴파일하는 기능을 제곰함.
  이를 통해 TF의 symbolic graph execution방식과 같이 여러 optimization을 적용할 수 있고, serialized된 모델을 PythonDependency가 없는 다른환경에서도 활용할 수 있는 이점이 있음.

- 구성요소 : torch - main namespace, Tensor등 다양한 수학 함수가 포함 | .autograd - 자동미분을 위한 함수가 포함. 자동미분여부를 제어하고, 자체 미분가능함수를 정의할 때 쓰는 기반클래스(Function)가 포함
  | .nn - 신경망 구축을 위한 데이터구조나 레이어 정의(모델 층, 활성화함수, 손실함수 등이 정의) | .optim - 파라미터 최적화 알고리즘(옵티마이저)구현. 
  | .utils.data - GD계열의 반복연산시 사용하는 미니배치용 유틸리티함수 포함. | .onnx - ONNX(Open Neural Network eXchange, 서로다른 딥러닝 프레임워크간 모델공유시 사용하는 새 포맷)포맷으로 모델을 export할때 사용함.

## device
- torch.cuda.is_available() : 현 환경에서 GPU 사용가능 여부를 반환.
- torch.device("cuda") : GPU연산 사용. ("cuda" if USE_CUDA(위의 결과) else "cpu")식으로, GPU 사용이 가능할 때만 사용하게 사용. 
- 모델(텐서).to(device) : 연산을 수행할 위치를 지정.
- 텐서.cpu() : cpu 메모리에 올려진 텐서 반환.
- 텐서.cuda() : gpu 메모리에 올려진 텐서 반환. 

## tensor
- 텐서 : pytorch의 행렬(데이터)를 저장하는/다차원 배열을 처리하기 위한 자료형/데이터구조. numpy의 ndarray와 비슷하고, 튜플타입을 가짐. 인덱스접근, 슬라이싱 등이 전부 가능함.
  GPU를 사용하는 계산도 지원하며, 이때는 torch.cuda.FloatTensor를 사용함. GPU텐서 간에만 연산이 가능하며, GPU텐서에서 ndarray로의 변환은 CPU로 이동한 후에 변환가능함.
- 텐서속성 : 텐서를 생성할 때는 모두 dtype=long 식으로 데이터 타입을 지정하고, requires_grad=True 를 사용해 그 텐서에 대해 미분이 가능하도록 연산들을 모두 추적할 수 있게 할 수 있음.
  이 경우, 그 텐서를 사용한 식은 전부 grad_fn이라는 속성을 가지게 됨(기울기를 담고있음).
- 텐서[0, 0\] : 인덱스로 텐서 내 요소에 접근.
- 텐서[:, :2\] : 슬라이스로 텐서 내 요소에 접근.
- 텐서 > 2 : 마스크 배열을 이용해 True값인 요소만 추출.

- torch.tensor(i(iterator객체)) : 텐서 생성. .item()으로 값을 받아올 수 있음. 
- torch.자료형Tensor(array) : array로 지정된 자료형의 텐서 생성(ex-Float:32bit 부동소수점). 
- torch.텐서생성함수_likes(텐서) : 텐서와 동일한 shape의, 텐서 생성함수로 생성할 수 있는 텐서 생성.
- torch.from_numpy(ndarray) : 넘파이 배열을 텐서로 변환.
- torch.zeros(shape) : 0으로 초기화된 shape의 텐서 생성.
- torch.ones(shape) : 1으로 초기화된 shape의 텐서 생성.
- torch.full(size, fill_value) : 특정 값(fill_value)로 채워진 size의 텐서 생성.
- torch.range(start, end, step) : start~end까지 step의 간격으로 채워진 텐서 생성. python내장함수 range와 동일하게 작동.  
- torch.arange(start, end, step) : start~end까지 step의 간격으로 채워진 텐서 생성.  
- torch.rand(shape) : shape의 랜덤으로 값이 할당된 텐서 생성.
- torch.randn(shape) : shape의, 표준정규분포(평균0, 분산1)내의 범위에서 랜덤으로 값이 할당된 텐서 생성.
- torch.randint(low, high, shape) : shape의, low~high의 범위에서 랜덤으로 값이 할당된 텐서 생성. low는 포함, high는 미포함.
- torch.empty(y, x) : 초기화 되지 않은 y*x의 행렬 생성. 그 시점에 할당된 메모리에 존재하던 값이 초기값으로 나타남(쓰지 않는것을 권함).
- torch.eye(n, m) : 희소행렬(대각만 1, 나머진 0인 행렬)생성. (n, m)의 텐서가 생성되며, n만 지정시 m은 자동으로 n이 됨.
- torch.linespace(start, end, step) : 시작과 끝을 포함하고 step의 갯수만큼 원소를 가진 등차수열을 만듦.
- torch.squeeze(텐서) : 텐서의 차원중 차원이 1인 차원을 압축(삭제)함. dim 인자를 입력하면 지정한 차원만 압축할 수 있음.
- torch.unsqeeze(텐서, dim) : 지정한 차원을 추가함(차원은 1).
- torch.transpose(텐서, dim1, dim2) : 주어진 텐서에서 dim1과 dim2를 바꿈.

- torch.nn.init.uniform(텐서, a, b) : 주어진 텐서를 uniform분포로 초기화함.
- torch.nn.init.normal(텐서, a, b) : 주어진 텐서를 normal분포로 초기화함.
- torch.nn.init.constant(텐서, a, b) : 주어진 텐서를 상수로 만듦.

- 텐서.eq(텐서) : 텐서와 입력된 텐서의 데이터가 동일한지 반환.
- 텐서.data.sub_(1) : 값을 0과 1로 변환.  
- torch.add/mul/div(텐서1, 텐서2) : 두 텐서간의 연산을 함. 텐서1 +|*|/ 텐서2 로도 가능함. 연산은 같은 타입의 텐서간/텐서-스칼라 간 에서만 가능함.
- torch.max(텐서) : 텐서 내부의 요소중 최댓값을 텐서로 반환. 이 외에도 텐서.연산()으로 사용가능한 모든 연산은 torch.연산(텐서)으로 사용가능.  
- torch.pow(텐서, 지수) : 텐서를 주어진 지수만큼 제곱.
- torch.log(텐서) : 텐서의 모든 요소에 log(log_10, natural logarithm)를 적용.
- torch.exp(텐서) : 텐서의 모든 요소에 exponential 연산을 적용.
- torch.eig(텐서) : 텐서의 고유값 반환. eigenvectors=True매개변수로 고유벡터를 반환하게 할 수 있음.
- torch.argmax(텐서) : 텐서 내부의 요소중 최댓값의 인덱스를 반환. dim=i 매개변수를 사용해 특정 차원을 기준으로 볼 수 있음(없으면 전체 요소).
- torch.bmm(batch1, batch2) : 두 행렬간의 곱을 배치단위로 처리. 단일 행렬로 계산하는 mm보다 좀 더 효율적(배치단위로 한번에 처리하니)임. 
- torch.matmul(텐서1, 텐서2) : 두 텐서의 종류에 따라 dot(백터 내적), mv(행렬과 벡터의 곱), mm(행렬과 행렬의 곱)중 하나를 선택해 연산함.
- torch.where(condition, x, y) : 조건에 따라 x 또는 y에서 선택한 요소의 텐서 반환.

- torch.cat(\[텐서1, 텐서2], dim=i) : i 번째 차원을 늘리며 두 텐서를 연결. 기존 차원을 유지한채 지정 차원의 크기만 커짐.
- torch.stack(\[텐서1, 텐서2, 텐서3], -dim=i-) : 텐서(벡터)들을 순차적으로 쌓음. 차원이 하나 늘어남. i번 차원이 늘어나게 함.
- torch.split(텐서, split_size, dim) : 텐서를 몇개의 부분으로 나눔.
- torch.chunk(텐서, chunks, dim) : 텐서를 몇개의 부분으로 나눔.

- 텐서에 식 적용 : 텐서 + a , 텐서 > 0.5 등 텐서를 식에 사용하면 텐서내의 모든 데이터에 적용됨(applymap).
- 텐서.shape : 텐서의 shape를 출력.
- 텐서.dim() : 텐서의 차원 출력
- 텐서.sum() : 텐서의 요소합 출력.
- 텐서.max() : 탠서의 요소중 가장 큰 요소 반환.
- 텐서.mean() : 텐서의 요소들의 평균 반환
- 텐서.argmax() : 텐서에서 가장 큰 요소의 인덱스를 반환.
- 텐서.matmul(텐서) : 두 텐서의 행렬곱 반환.

- 텐서.grad : 텐서의 미분값 확인. 미분을 해준 뒤 여야 함.
- 텐서.item() : 값이 하나만 있는 텐서의 값을 반환함(텐서를 스칼라 값으로 만듦).
- 텐서.size(index) : index차원의 차원 수 반환.
- 텐서.reshape(shape) : 텐서의 크기 변경.
- 텐서.reshape_as(텐서) : 텐서의 크기를 주어진 텐서의 크기와 동일하게 변경. 
- 텐서.view(shape) : 텐서의 크기(차원)변경. numpy의 reshape와 같이 전체 원소수는 동일해야 하고, -1 인자를 사용할 수 있음((out.size(0), -1)(첫 차원 제외 펼침)식).
- 텐서.view_as(텐서) : 텐서의 크기를 입력한 텐서와 동일하게 변경. 마찬가지로 데이터의 개수는 동일해야 함.  
- 텐서.permute(shape number - 0,3,1,2 식으로) : 텐서 차원의 순서를 변환.
- 텐서.squeeze() : 차원의 크기가 1인 경우 해당차원 제거.
- 텐서.unsqueeze(i) : i 위치(shape의 위치)에 크기가 1인 차원을 추가.
- 텐서.unbind() : 다차원의 텐서를 1차원의 텐서로 분리(unbind)함.
- 텐서.detach() : 현재 그래프에서 분리된 새 텐서 반환. 원본과 같은 스토리지를 공유.
- 텐서.scatter(dim, 텐서, 넣을 인자) : dim차원에서, 텐서의 데이터(내부 데이터를 인덱스로)대로 넣을 인자를 삽입(할당).
- 텐서.자료형() : 텐서의 자료형을 변환(TypeCasting).
- 텐서.연산_() : 기존의 값을 저장하며 연산. x.mul(2.)의 경우 x에 다시 저장하지 않으면 x엔 영향이 없으나, x.mul_()은 연산과 동시에 덮어씀.
- 텐서.numpy() : 텐서를 넘파이배열(ndarray)로 변경.

- torch.save(model(.state_dict()), path) : 모델 저장. .state_dict()를 붙이면 가중치만 저장하는 것으로, 모델이 코드상으로 구현되어 있어야 함. 기본적으로 .pt확장자.
- model = torch.load(path) : 모델 로드.
- model.load_state_dict(torch.load(path)) : 모델 가중치 로드.
###### tensor expression
- 2D Tensor : (batch size, dim)
- 3D Tensor : (batch size, length(time step), dim)


## model
- 가설 선언 후 비용함수, 옵티마이저를 이용해 가중치, 편향등을 갱신해 올바른 값을 찾음(비용함수를 미분해 grandient(기울기)계산). 

- torch.save(모델.state_dict(), 경로) : 모델의 현재 가중치를 경로에 저장.
- 모델.load_state_dict(torch.load(경로)) : 경로의 가중치를 로드해 모델의 가중치로 사용. 
- 모델.embedding.weight.data.copy_(임베딩벡터들) : 사전훈련된 임베딩벡터값을 모델의 임베딩층에 연결. 
  임베딩벡터는[(필드에 저장)필드.vocab.vectors]로 확인. data까지만 쓰면 임베딩벡터 확인 가능. 

- 텐서.backword() : 역전파. 해당 수식의 텐서(w)에 대한 기울기를 계산. w가 속한 수식을 w로 미분(주로 loss에 수행). 해당 텐서 기준 연쇄법칙 적용. 
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
- optimizer.param_groups : 파라미터 그룹 확인 가능.
```python 
# 사용 예
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
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


### LearningRate Scheduler
- 사용방법 : optimizer와 schduler를 먼저 정의한 뒤 학습 시 batch마다 optimizer.step(), epoch마다 scheduler.step()을 해주면 됨.
- 파라미터 : 공통적으로 optimizer를 넣어주고, last_epoch로 모델 저장 후 시작시 설정을, verbose=bool로 lr갱신 시 마다 메세지를 출력하게 할 수 있음.

- torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func(lambda)) : lambda표현식으로 작성한 함수를 통해 lr을 조절함. 초기 lr에 lambda(epoch)를 곱해 lr을 계산함.
- torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=func(lambda)) : lambda표현식으로 작성한 함수를 통해 lr을 조절. lr에 lambda(epoch)를 누적곱해 lr을 계산함.
- torch.optim.lr_scheduler.StepLR(optimizer, step_size=s, gamma=g) : step사이즈마다 gamma비율로 lr을 감소시킴(일정 step마다 gamma를 곱함). 
- torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[m1,m2\], gamma=g) : 지정 epoch마다 gamma비율로 lr을 감소시킴(지정 epoch마다 gamma를 곱함). 
- torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=g) : learing rate decay가 exponential함수를 따름(매 epoch마다 gamma를 곱함). 
- torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t, eta_min=min) : lr이 cos함수를 따라 eta_min까지 떨어졌다가 다시 초기 lr로 돌아오기를 반복함(최대 T_max번).
- torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode) : 입력한 성능평가지표가 patience만큼 향상되지 않으면 lr을 줄임. optim에 momentum을 설정해야 하고, scheduler.step(평가지표)로 사용함.
- torch.optim.lr_scheduler.CyclicLR(optimizer, step_size_up, base_lr, max_lr, gamma, mode) : base_lr부터 max_lr까지 step_size_up동안 증가하고 감소하기를 반복함. step_size_down, scale_fn등 사용가능.
- torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, steps_per_epoch, epochs, pct_start, anneal_strategy) : 초기 lr에서 1cycle annealing함(초기 lr에서 max_lr까지 anneal_strategy(linear/cos)에 따라 증가 후 감소).
- torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult, eta_min) : 초기 lr에서 cosine annealing 함수에 따라 eta_min까지 떨어졌다가(T_0에 걸쳐) T_mult에 걸쳐 다시 되돌아옴. 이 후 증감을 반복. 
  

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
- 모델 학습 과정 : 가중치 초기화 > 정의한 모델(가설)로 데이터를 예측/예측값을 얻음 > 
  지정한 손실함수를 이용해 예측값과 레이블간의 손실(비용)계산 > 손실을 미분 > 미분한 결과와 옵티마이저를 이용해 지정된 파라미터 갱신 과정을 거쳐 optimizer에 인자로 준 텐서(가중치, 편향)를 갱신함.
```pyhton example
dataset = torch.utils.data.TensorDataset(X, y) 
dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=bool, drop_last=bool)

for epoch in range(num_epochs):
    for batch in dataloader:
        x, y = batch

        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        print('train loss: ', loss.item())
    
        loss.backward()
    
        optimizer.step()
        optimizer.zero_grad() 
```

### validation
``` python example
for epoch in range(num_epochs):
    with torch.no_grad():
        val_loss = []
        for val_batch in val_dataloader:
          x, y = val_batch
          logits = model(x)
          val_loss.append(cross_entropy_loss(logits, y).item())
    
        val_loss = torch.mean(torch.tensor(val_loss))
        print('val_loss: ', val_loss.item())
```

### test
- 모델 테스트 과정: 모델 추론모드로 전환 > 데이터 로더에서 배치를 꺼내 추론수행 > 모델의 예측생성 > argmax(다중분류) > metric 계산.
```pythonr example
model.eval()
count = 0
for data, target in dataset :
    model(data)
    _, predicted = torch.max(outputs.data, 1)
    count += predicted.eq(targets.data.view_as(predicted)).sum()
    # count += (targets == (output > 0.5).float()).float().sum()  # binary
accuracy = count/len(dataset.dataset)
```


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


# pytorch_lightning
- PyTorch Lightning : TF의 keras와 같은 PyTorch에 대한 High-level 인터페이스를 제공하는 오픈소스 Python 라이브러리. 코드의 추상화를 통해 프레임워크를 넘어 하나의 코드 스타일로 자리 잡기 위해 탄생한 프로젝트.
- 장점 : 효율적(정돈된 코드스타일/추상화), 유연함(torch에 적합한 구조, trainer 다양하게 override가능, callback), 구조화(딥러닝시 다뤄야 할 부분들), 다양한 학습 방법에 적용가능(GPU/TPU/16-bit precision),
  PytorchEcosystem(어먀격한 testing과정, Pytorch친화적), logging과 통합/시각화 프레임워크 등등의 장점을 가지고 있음.

## Model use
- model(x) : 훈련된 모델(순전파)사용.

- model = LightningModule.load_from_checkpoint(path) : 사전훈련된(저장된)모델 로드.
- model.freeze() : 모델의 파라미터들을 동결. 모델의 예측시 사용해줘야 함.

## Module
- LightningModule : 모델 내부의 구조를 설계하는 research/science클래스. 모델구조/데이터전처리/손실함수 설정 등을 통해 모델 초기화/정의. 
  모든 모듈이 따라야 하는 9가지 필수메서드의 표준 인터페이스를 가지고 있음.
- Lifecycle : LightningModule클래스가 함수를 실행하는 순서. 아래의 순서에 더해 각 batch와 epoch마다 함수 이름에 맞게 정해진 순서대로 호출됨.
  [__init\_\_ > prepare_data > configure_optimizers > train_dataloader > val_dataloader > test_dataloader(.test()호출시)]의 순서.
### method
- __init\__() : 모델 초기화. 기존 모델과 동일하게 변수/층등의 정의/선언과 초기화(super(CLASS_NAME, self).__init\__())가 진행됨.
- forward() : 모델 실행시 실행될 순전파 메서드. 기존모델과 동일하게 사용할 수 있음. 입력 x를 받아 예측 pred를 반환하는 구조. 
- 손실함수() : 손실함수. 클래스 내부에 정의해 사용하는게 구조화되어 좋음. logits과 labels를 받아 계산된 loss를 반환하는 구조.
- configure_optimizers() : 옵티마이저 설정. self.parameters()와 lr을 인자로 받는 옵티마이저를 반환하는 형태.

- 모델 학습루프 : 복잡하던 훈련과정을 추상화. 3가지의 루프 패턴마다 3개지의 메소드를 가지고 있음. 일반적으로는 training_step -> validation_step -> validation_epoch_end의 구조를 사용함.
- training_step() : 모델 훈련시 진행될 훈련 메서드. train_batch(+batch_idx)를 받아 self.forward -> self.loss -> {'loss': loss, 'log': logs({'train_loss': loss})}반환 형태로 이뤄짐.
- training_step_end() : 모델 훈련시 한 step의 끝마다 수행될 메서드.
- training_epoch_end() : 모델 훈련시 한 epoch의 끝마다 수행될 메서드.
- validation_step() : 모델 훈련시 검증과정에서 진행될 validation메서드. val_batch(+batch_idx)를 받아 self.foward -> self.loss -> {'val_loss': loss}반환 의 형태로 이뤄짐.
- validation_step_end() : 모델 훈련시 검증과정에서 한 step의 끝마다 수행될 메서드.
- validation_epoch_end() : 모델 훈련시 검증과정에서 한 epoch의 끝마다 수행될 메서드. 주로 outputs를 입력받아 평균 val_loss와 log(텐서보드용)를 반환하는 구조. 반환시킬 것(dict)들의 리스트를 반환.
- test_loop_step() : 모델 테스트(.test())시 테스트 과정에서 진행될 test_loop메서드.
- test_loop_step_end() : 모델 테스트시 테스트 과정에서 한 step의 끝마다 수행될 test_loop메서드.
- test_loop_epoch_end() : 모델 테스트시 테스트 과정에서 한 epoch의 끝마다 수행될 test_loop메서드.

- 데이터 준비 : Pytorch의 데이터 준비 과정을 크게 5가지로 구조화. 다운로드, 데이터정리/메모리저장, 데이터셋 로드, 데이터전처리(transforms), dataloader형태로 wrapping
- prepare_data() : 데이터 다운로드/로드 후 데이터 전처리, 분리까지 진행해 self.train_data등의 elements에 정의/선언.
- train_dataloader() : train_Dataloader 반환. self.train_data와 batch_size등을 인자로 해 train dataloader를 생성(wrapping)해 반환함.
- val_dataloader() : val_Dataloader 반환. self.val_data와 batch_size등을 인자로 해 val dataloader를 생성(wrapping)해 반환함.
- test_dataloader() : test_Dataloader 반환. self.test_data와 batch_size등을 인자로 해 test dataloader를 생성(wrapping)해 반환함.

## Trainer
- Trainer : 모델의 학습을 담당하는 클래스. 모델의 학습에 관여되는 engineering(학습epoch/batch/모델의 저장/로그생성까지 전반적으로)을 담당.
- pytorch_lightning.Trainer() : 트레이너 객체 생성. 다양한 args를 통해 트레이너 설정(gpu등)가능.

- 트레이너.fit(모델) : 모델 학습. sklearn의 fit메서드와 비슷함.
- 트레이너.test() : fit한 LightningModule모델 테스트. 


# Ignite
- Ignite : 텐서플로우의 keras와 같이, High-level인터페이스를 제공하는 오픈소스 라이브러리. Lightning과 달리 표준 인터페이스를 가지고있지 않음.


# Geometric
- (?)









# REFERENCE
- [1](https://pytorch.org/)
- [2](https://www.pytorchlightning.ai/)
- [3](https://koreapy.tistory.com/788)
- [4](https://baeseongsu.github.io/posts/pytorch-lightning-introduction/)
