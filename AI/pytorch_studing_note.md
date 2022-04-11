# pytorch
- pytorch : 과거 Torch 및 카페2 프레임워크를 기반으로, 파이썬을 스크립팅 언어로 사용하며 진화된 토치 C/CUDA백엔드를 사용하는 딥러닝 프레임워크. 
  GPU사용에 큰 장점을 가짐. 강력한 GPU가속과 파이썬으로 된 텐서, 동적 신경망을 지원함. 각 반복단계에서 즉석으로 그래프를 재 생성할 수 있음.
  torch와 numpy가 상호 호환되기 때문에, ndarray와 같이 인덱스를 조작할 수 있으며 생성된 텐서는 torch 내부적으로도, numpy와 호황되는 라이브러리에도 사용할 수 있음.
- 동적신경망 : 반복할때마다 변경가능한 신경망. 학습 중 숨은 계층을 추가하거나 제거해 정확성과 일반성의 개선이 가능. 
- 브로드 캐스팅 : 크기가 다른 행렬(텐서)들의 크기를 자동으로 맞춰 연산을 가능하게 해주는 기능. 연산시 더 큰 차원에 맞춰짐(요소 복제).
- TorchScript : Pytorch의 JIT(Just-In-Time)컴파일러. 기존의 명령적인 실행방식 대신, 모델이나 함수의 소스코드를 TorchScript컴파일러를 통해 TorchScript로 컴파일하는 기능을 제곰함.
  이를 통해 TF의 symbolic graph execution방식과 같이 여러 optimization을 적용할 수 있고, serialized된 모델을 PythonDependency가 없는 다른환경에서도 활용할 수 있는 이점이 있음.
- Distributed model : 둘 이상의 GPU를 사용하는 모델을 명명하는 단어. 

- 모델구성 : 파이토치의 대부분의 구현체(모델)는 모델 생성시 클래스를 사용. torch.nn.Module상속 클래스 구현 > __init__에서 super().__init__을 호출, 사용할 모델(층)정의 > forward(self,x)(자동실행, 모델 사용 후 값 반환)의 과정을 거침.
- 구성요소 : torch - main namespace, Tensor등 다양한 수학 함수가 포함 | .autograd - 자동미분을 위한 함수가 포함. 자동미분여부를 제어하고, 자체 미분가능함수를 정의할 때 쓰는 기반클래스(Function)가 포함
  | .nn - 신경망 구축을 위한 데이터구조나 레이어 정의(모델 층, 활성화함수, 손실함수 등이 정의) | .optim - 파라미터 최적화 알고리즘(옵티마이저)구현. 
  | .utils.data - GD계열의 반복연산시 사용하는 미니배치용 유틸리티함수 포함. | .onnx - ONNX(Open Neural Network eXchange, 서로다른 딥러닝 프레임워크간 모델공유시 사용하는 새 포맷)포맷으로 모델을 export할때 사용함.

## device
- torch.cuda.is_available() : 현 환경에서 GPU 사용가능 여부를 반환.
- torch.cuda.device_count() : 사용가능한 GPU개수를 반환.
- torch.device("cuda") : GPU연산 사용. ("cuda" if USE_CUDA(위의 결과) else "cpu")식으로, GPU 사용이 가능할 때만 사용하게 사용. 
- 모델(텐서).to(device) : 연산을 수행할 위치를 지정.
- 텐서.cpu() : cpu 메모리에 올려진 텐서 반환.
- 텐서.cuda() : gpu 메모리에 올려진 텐서 반환. 

### XLA
- torch_xla : Pytorch를 TPU등의 XLA(Accelerated Linear Algebra)장치에서 실행시키기 위한 라이브러리. 새로운 xla장치유형을 추가시키며, 이는 다른 장치유형처럼 작동함. bfloat16 데이터형을 사용가능(XLA_USE_BF16환경변수로 제어).
  `!pip install cloud-tpu-client==0.10 torch==1.9.0 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl` 으로 리눅스 환경(colab)에서 설치할 수 있음.
- XLA_tensor : 다른 텐서와 달리 결과가 필요할때까지 그래프에 작업을 기록. 그래프를 자동으로 구성해 XLA장치로 보내고, 장치와 CPU간 데이터를 복사할때 동기화함. 베리어를 넣으면 명시적으로 동기화됨.

- torch_xla.core.xla_mode.xla_device() : xla장치 유형을 반환. 텐서생성시 device매개변수에 넣거나 to메서드로 유형을 변경할 수 있음.
- torch_xla.core.xla_mode.get_xla_supported_devices() : 지원하는(사용가능한)XLA디바이스들을 반환.
- torch_xla.core.xla_mode.optimizer_step(optimizer, barrier=bool) : XLA에서 옵티마이저 실행. XLA장치를 획득하고 옵티마이저에 베리어를 설정함(barrier매개변수, ParallelLoader사용시 생략(자동생성)).

- torch_xla.distribute.xla_multiprocessing.spawn(main_func, args) : 멀티프로세싱으로 여러 XLA장치를 실행할때 사용하는, 각 XLA장치를 실행하는 프로세스를 생성하는 메서드. main_func에 학습코드가 들어가있으면 됨. 
- torch_xla.distribute.parallel_loader.ParallelLoader(train_loader, [devices\]) : 훈련데이터를 각 장치에 로드하는 로더 생성. .per_device_loader(device)메서드로 일반 데이터로더와 동일하게 사용가능.
- torch_xla.distributed.data_parallel.DataParallel(model, device_ids) : 멀티 스레딩을 사용한 다중 XLA장치 사용시 사용하는 객체 생성. epoch만큼 train_func(model, loader, device, context를 매개변수로)과 train_loader를 입력해 사용. 

## tensor
- 텐서 : pytorch의 행렬(데이터)를 저장하는/다차원 배열을 처리하기 위한 자료형/데이터구조. numpy의 ndarray와 비슷하고, 튜플타입을 가짐. 인덱스접근, 슬라이싱 등이 전부 가능함.
  GPU를 사용하는 계산도 지원하며, 이때는 torch.cuda.FloatTensor를 사용함. GPU텐서 간에만 연산이 가능하며, GPU텐서에서 ndarray로의 변환은 CPU로 이동한 후에 변환가능함.
- 텐서속성 : 텐서를 생성할 때는 모두 dtype=long 식으로 데이터 타입을 지정하고, requires_grad=True 를 사용해 그 텐서에 대해 미분이 가능하도록 연산들을 모두 추적할 수 있게 할 수 있음.
  이 경우, 그 텐서를 사용한 식은 전부 grad_fn이라는 속성을 가지게 됨(기울기를 담고있음).
- 텐서[0, 0\] : 인덱스로 텐서 내 요소에 접근.
- 텐서[:, :2\] : 슬라이스로 텐서 내 요소에 접근.
- 텐서 > 2 : 마스크 배열을 이용해 True값인 요소만 추출.
- torch.set_printoptions(precision=i, sci_mode=bool) : (실수의)표시설정을 조정. 정밀도 i 만큼 되는 소수점을 표시하고, sci_mode가 True면 Scientific Notation(과학적 기수법)을 사용함.

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
- torch.scalar_tensor(number) : 스칼라 텐서 생성.
- torch.empty(y, x) : 초기화 되지 않은 y*x의 행렬 생성. 그 시점에 할당된 메모리에 존재하던 값이 초기값으로 나타남(쓰지 않는것을 권함).
- torch.eye(n, m) : 희소행렬(대각만 1, 나머진 0인 행렬)생성. (n, m)의 텐서가 생성되며, n만 지정시 m은 자동으로 n이 됨.
- torch.linespace(start, end, step) : 시작과 끝을 포함하고 step의 갯수만큼 원소를 가진 등차수열을 만듦.

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
- 텐서.numpy() : 텐서를 넘파이배열(ndarray)로 변경.

- torch.squeeze(텐서) : 텐서의 차원중 차원이 1인 차원을 압축(삭제)함. dim 인자를 입력하면 지정한 차원만 압축할 수 있음.
- torch.unsqeeze(텐서, dim) : 지정한 차원을 추가함(차원은 1).
- torch.transpose(텐서, dim1, dim2) : 주어진 텐서에서 dim1과 dim2를 바꿈.
- torch.where(condition, x, y) : 조건에 따라 x 또는 y에서 선택한 요소의 텐서 반환.

- torch.cat(\[텐서1, 텐서2], dim=i) : i 번째 차원을 늘리며 두 텐서를 연결. 기존 차원을 유지한채 지정 차원의 크기만 커짐.
- torch.stack(\[텐서1, 텐서2, 텐서3], -dim=i-) : 텐서(벡터)들을 순차적으로 쌓음. 차원이 하나 늘어남. i번 차원이 늘어나게 함.
- torch.vstack(\[텐서1, 텐서2]) : 텐서들을 dim 1에 맞게 쌓음(연결함). torch.stack(dim=1)과 동일함.

- torch.split(텐서, split_size, dim) : 텐서를 몇개의 부분으로 나눔.
- torch.chunk(텐서, chunks, dim) : 텐서를 몇개의 부분으로 나눔.
- torch.utils.dat a.random_split(dataset, [ratios\]) : 전달한 데이터셋을 지정한 비율들대로 나눔. [0.9, 0.1\]식으로 비율을 지정하면 됨.

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

- 텐서에 식 적용 : 텐서 + a , 텐서 > 0.5 등 텐서를 식에 사용하면 텐서내의 모든 데이터에 적용됨(applymap).
- 텐서.shape : 텐서의 shape를 출력.
- 텐서.dim() : 텐서의 차원 출력
- 텐서.sum() : 텐서의 요소합 출력.
- 텐서.max() : 탠서의 요소중 가장 큰 요소 반환.
- 텐서.mean() : 텐서의 요소들의 평균 반환
- 텐서.argmax() : 텐서에서 가장 큰 요소의 인덱스를 반환.
- 텐서.matmul(텐서) : 두 텐서의 행렬곱 반환.
- 텐서.연산_() : 기존의 값을 저장하며 연산. x.mul(2.)의 경우 x에 다시 저장하지 않으면 x엔 영향이 없으나, x.mul_()은 연산과 동시에 덮어씀.

###### tensor expression
- 2D Tensor : (batch size, dim)
- 3D Tensor : (batch size, length(time step), dim)


## model
- 가설 선언 후 비용함수, 옵티마이저를 이용해 가중치, 편향등을 갱신해 올바른 값을 찾음(비용함수를 미분해 grandient(기울기)계산). 
- torch.nn.functional : torch.nn이 클래스로 정의되어있는 것과 NN관련 기능들이 함수로 정의되어있는 클래스. `import torch.nn.functional`로 import하지 않으면 사용할 수 없음. 주로 as F로 사용됨.
  nn으로 구현한 클래스의 경우는 attribure를 활용해 state를 저장하고 활용할 수 있고, torch.nn.functional로 구현한 함수는 인스턴스화 시킬 필요 없이 사용이 가능하다는 장점이 있음. 두 방식 모두 결과는 동일하게 제공됨.
- Loss : input(pred)는 float, target(y)는 long이여야 하며, loss가 스칼라가 아니라면 loss.backward(gradient=loss)로 backward를 써야 함.

- 모델.parameters() : 모델의 파라미터 출력. w와 b가 순서대로 출력됨.
- 텐서.backward() : 역전파. 해당 수식의 텐서(w)에 대한 기울기를 계산. w가 속한 수식을 w로 미분(주로 loss에 수행). 해당 텐서 기준 연쇄법칙 적용. 
- 모델.eval() : 모델을 추론모드로 전환. 모델 test시 사용.
- torch.no_grad() : 미분을 하지 않음. 파라미터를 갱신하지 않는 test시 사용. `with torch.no_grad` 식으로 사용하며, 내부에서 사용된 텐서들은 전부 require_grad=False.
- torch.nn.init.xavier_uniform_(self.층.weight) : 특정 층 한정으로 가중치 초기화. 형태 변경을 위한 전결합층 등 파라미터 갱신을 원하지 않는 층에 사용.

- torch.manual_seed(i) : 랜덤시드 고정.
- torch.cuda.manual_seed_all(i) : GPU 사용시 랜덤시드 고정.

### save/load
- torch.save(model(.state_dict()), path) : 모델 저장. .state_dict()를 붙이면 가중치만 저장하는 것으로, 모델이 코드상으로 구현되어 있어야 함. 기본적으로 .pt확장자. dir이 존재하지 않으면 오류 발생.
- torch.jit.save(model, path) : 모델을 Pytorch의 JIT컴파일러를 사용해 제공함. 최종배포를 위해 사용.

- model = torch.load(path) : 모델 로드.
- model.load_state_dict(torch.load(path)) : 모델 가중치 로드.
- model.embedding.weight.data.copy_(임베딩벡터들) : 사전훈련된 임베딩벡터값을 모델의 임베딩층에 연결. 
  임베딩벡터는`(필드에 저장)필드.vocab.vectors`로 확인. data까지만 쓰면 임베딩벡터 확인 가능. 

### data
- torch.utils.data.TensorDataset(x, y) : 데이터들을 TensorDataset(PyTorch기본 데이터셋)을 이용해 데이터셋에 저장.
- torch.utils.data.DataLoader(dataset, batch_size=i) : 데이터셋을 i개의 미니배치로 학습시킴. shuffle=bool(Epoch마다 데이터 학습순서를 섞음)매개변수와
  drop_last=bool(batch_size로 배치를 나눈 뒤, 남은(batch_size 미만의)데이터셋을 버릴지)매개변수 사용가능. .dataset으로 내부의 데이터셋 확인가능. 나올떈 각 텐서가 batch_size만큼 concatenate되어 나옴
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
- torch.nn.functional.F.relu(텐서) : 렐루(ReLU)사용. F.relu(층) 처럼 사용. 모델 제작시 활성화 함수를 꼭 사용해 주어야 함.
- torch.nn.functional.F.sigmoid(텐서) : 시그모이드 사용. torch.sigmoid(텐서(식))로도 사용가능. 
- torch.nn.functional.F.softmax(텐서) : 소프트맥스 사용. dim=i매개변수(적용될 차원 선택)사용가능. 손실함수에 포함되어있어 잘 쓰이지 않음.
- torch.nn.functional.F.log_softmax(텐서) : 로그 소프트맥스 사용. torch.log(F.softmax())와 동일.
- torch.nn.functional.F.log_softmax(input) : logarithm을 따르는 softmax를 적용함. dim 인자로 계산될 차원을 정할 수 있음. 수학적으로는 log(softmax(x))와 동일하나 둘을 따로 하는것은 느리고 숫자적으로 불안정하므로 사용하는 대안공식.

### loss
- torch.nn.functional.F.mse_loss(prediction, label) : MSE(평균제곱오차) 손실함수 사용.
- torch.nn.functional.F.binary_cross_entropy(prediction, label) : 이진분류(로지스틱 회귀)의 손실함수 사용. reduction인자에 'sum'등을 넣어 출력에 적용할 축소를 지정할 수 있음. nn에서는 BCELoss임.
- torch.nn.functional.F.cross_entropy(prediction, label) : cross-entropy 손실함수 사용. F.nll_loss(F.log_softmax(z, dim=1), y)와 동일함. weight 매개변수에 텐서를 전달해 레이블들에 가중치(특정 레이블이 더 큰 영향을 끼치는)를 설정할 수 있음.
- torch.nn.functional.F.nll_loss(input, target) : negative log likehood loss. C클래스 분류 task에서 사용. 

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

- torch.optim.lr_scheduler.StepLR(optimizer, step_size=s, gamma=g) : step사이즈마다 gamma비율로 lr을 감소시킴(일정 step마다 gamma를 곱함). 
- torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func(lambda)) : lambda표현식으로 작성한 함수를 통해 lr을 조절함. 초기 lr에 lambda(epoch)를 곱해 lr을 계산함.
- torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[m1,m2\], gamma=g) : 지정 epoch마다 gamma비율로 lr을 감소시킴(지정 epoch마다 gamma를 곱함). 
- torch.optim.lr_scheduler.LinearLR(optimizer, start_factor, total_iters) : 곱셈인자를 선형적으로 total_iters(0부터 시작)에 걸쳐 변형시켜 학습 속도를 늦춤. 시작 lr * start_factor 에서 * 1/total_iters만큼을 계속 변동시켜 결국은 시작 lr * 0.1로 만드는 듯.
- torch.optim.lr_scheduler.CyclicLR(optimizer, step_size_up, base_lr, max_lr, gamma, mode) : base_lr부터 max_lr까지 step_size_up동안 증가하고 감소하기를 반복함. step_size_down, scale_fn등 사용가능.
- torch.optim.lr_scheduler.ConstantLR(optimizer, factor, total_iters) : epochs가 total_iters에 도달할 때 까지 시작 lr * factor를 lr으로 사용함.
- torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t, eta_min=min) : lr이 cos함수를 따라 eta_min까지 떨어졌다가 다시 초기 lr로 돌아오기를 반복함(최대 T_max번).
- torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=g) : learing rate decay가 exponential함수를 따름(매 epoch마다 gamma를 곱함). 
- torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=func(lambda)) : lambda표현식으로 작성한 함수를 통해 lr을 조절. lr에 lambda(epoch)를 누적곱해 lr을 계산함.
- torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers, milestones) : milestones의 간격대로 주어진 스케줄러를 순차적으로 적용함.
- torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode) : 입력한 성능평가지표가 patience만큼 향상되지 않으면 lr을 줄임. optim에 momentum을 설정해야 하고, scheduler.step(평가지표)로 사용함.

### layers
- torch.nn.Linear(input_dim, output_dim) : 선형회귀모델/전결합층 사용. 이대로 모델로 쓸 수도, 모델에 층으로 넣을수도 있음. bias=bool 로 편향 존재여부 지정가능.

- torch.nn.Conv2d(input_dim, output_dim, kernel_size) : (2차원)CNN층 사용. i의 커널사이즈를 가짐. padding, stride등도 설정해줄 수 있음. 
- torch.nn.MaxPool2d(kernel_size, stride) : (2차원)맥스풀링층 사용. 하나의 정수만 넣으면 커널사이즈와 스트라이드 둘 다 해당값으로 지정됨.

- torch.nn.RNN(input_size, hidden_size) : RNN층 사용. batch_first(bool, 입력텐서의 첫번째 차원은 배치크기), 
  num_layer(int, 은닉층개수(깊은RNN으로 만듦)), bidirectional(bool, 양방향순환신경망으로 만듦), batch_first(bool, 사용시(batch, input_dim, hidden_size)가 됨), dropout(0~1, 드롭아웃 비율)인자 사용가능.  
- torch.nn.LSTM(input_size(vocab_size), hidden_size) : LSTM층 사용. RNN계열 모델은 사용시 h_0(LSTM은 c_0도)을 같이 입력해야 하며, 각 변수는 zeros(D*num_layers, batch_size, hidden_size, requires_grad=False)로 초기화되야 함(D = 2 if bidirectional else 1).
  출력은 [output, (h_0, c_0)]가 되며, output은 (batch_size, input_dim, D\*hidden_size)(batch_first=True)를 가지고 각 t에 따른 마지막 h_t를 담으며, h_n/c_n은 (D\*num_layers, batch, hidden)의 차원을 가지고 마지막 step의 state들이 담김.
- torch.nn.GRU(input_size, hidden_size) : GRU층 사용. RNN과 동일하게 h_0를 입력으로 넣어야 하며, x와 h_c를 반환함. RNN과 동일한 인자 사용 가능. 
- torch.nn.TransformerEncoderLayer(d_model(입력차원), nhead(멀티헤드어텐션 헤드수)) : 트랜스포머의 인코더 층을 생성. 셀프어텐션과 FF층으로 이뤄져 있음. 이외에도 dim_feedforward, dropout, activation, batch_first등의 인자를 사용가능함.
- torch.nn.TransformerEncoder(encoder_layer, num_layers) : encoder_layer를 num_layers개 쌓은 트랜스포머 인코더 생성.
- torch.nn.TransformerDecocerLayer(d_model, nhead) : 트랜스포머 디코더 층 생성. 셀프어텐션과 멀티헤드어텐션, FF층으로 구성. target과 memory(인코더 은닉상태)를 입력해야 함.
- torch.nn.TransformerDecocer(decoder_layer, num_layers) : decoder_layer를 num_layers개 쌓은 트랜스포머 디코더 생성.
- torch.nn.Transformer(nhead, num_encoder_layers) : 트랜스포머 생성. 필요한 모든 애트리뷰트를 수정할 수 있음. 

- torch.nn.Embedding(num_embedding, embedding_dim) : 학습가능한 임베딩 테이블 생성. .weight 로 벡터 확인 가능. 이후 층에서 input_size를 embed_dim으로 변경해주어야 함. 임베딩 층을 사용할 것이라면, X를 int/Long Tensor로 해야 함.
  num_embedidng(단어집합 크기(임베딩할 단어개수)), embedding_dim(임베딩벡터의 차원)와 선택적으로 padding_idx(패딩을 위한 토큰의 인덱스)인자 사용가능.
- torch.nn.Embedding.from_pretrained(임베딩 벡터(필드.vocab.vectors), freeze=False) : 사전휸련된 임베딩벡터를 사용해 임베딩층 생성.
- torch.nn.Dropout(f) : f의 비율로 드롭아웃을 시행하는 층 사용. 포지셔널 임베딩과 마스크드 임베딩은 지원하지 않기때문에, 따로 해줘야만 함.

### Sequential
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


## PyTorch_Mobile
- PytorchMobile : 지연시간을 줄이고, 개인정보를 보호하며 새로운 대화형 사용사례 지원을 위해 에지 장치에서 ML모델을 실행하기 위한 엔드 투 엔드 워크플로우.
  IOS/Android/Linux에서 사용가능(모바일 애플리케이션 및 ML통합에 필요한 전처리/통합작업을 다루는 API제공, 효율적인 모바일 인터프리터 제공). 
- 기타 특징 : 추적 및 스크립팅 지원(TorchScript IR), 부동소수점 커널라이브러리(Arm CPU용 XNNPACK), QNNPACK통합(8bit양자화커널, 채널별 양자화, 동적 양자화),
  빌드수준 최적화 및 선택적 컴파일(최종 바이너리 크기는 앱이 필요로 하는 실제 연산자에 의해 결정), 모델최적화간소화(optimize_for_mobile), GPU/DSP/NPU등의 하드웨어 백엔드 지원(예정)등의 특징이 있음.
- 배포 워크플로우 : 파이토치에서 모델 작성 -(모델 양자화(QUANTIZE, optional)) -> script/trace model -(모델 optimization(optional)) -> SAVE MODEL -> 안드로이드(MAVEN)/IOS(COCOAPODS)로 사용.

- model = torch.quantization.convert(model) : 모델 양자화(아날로그 -> 디지털 | 연속적인 값 -> 띄엄띄엄한 값으로 근사). 선택적으로, 생략가능.
- model = torch.jit.trace(model, example_input) : 모델 trace. PyTorch JIT컴파일러 이용.
- model = torch.jit.script(model) : 모델 script. PyTorch JIT컴파일러 이용.
- opt_model = torch.utils.mobile_optimizer.optimize_for_mobile(model) : 모델 optimize. 선택적으로, 생략가능.
- opt_model._save_for_lite_interpreter(path) : 모델 저장. 저장되는 PyTorchLite모델은 .ptl 확장자를 가지고 있음.

- `implementation 'org.pytorch:pytorch_android_lite:1.9.0'` : 안드로이드(MAVEN)에서 PytorchLite모델을 사용하기 위한 implementation.
- `pod 'LibTorch_Lite','~>1.9.0'` : IOS(COCOAPODS)에서 PytorchLite모델을 사용하기 위한 pod.

## TorchScript
- TorchScript : Python종속적이지 않은 다른 고성능 환경에서 실행할 수 있는 PyTorch 모델(의 하위 클래스)의 중간 표현. 모든 프로그램은 Python프로세스에서 저장하고 Python종속성이 잆는 프로세스에서 로드할 수 있음.
  독립실행형 C++프로그램 등 Python 독립적으로 실행할 수 있는 TorchScript 프로그램으로 순수 Python 프로그램에서 모델을 점진적으로 전환하는 도구를 제공하며, 
  이를 통해 Python에서 모델을 훈련하고 TorchScript를 통해 모델을 Python 프로그램이 성능 및 다중 스레딩 이유로 불리할 수 있는 프로덕션 환경으로 내보낼 수 있음.
- 사용이유 : TorchScript코드는 자체 인터프리터(제한된 Python인터프리터인)에서 호출할 수 있고, Global Interpreter Lock를 획득하지 않아 동일한 인스턴스에서 동시에 많은 요청을 처리할 수 있음.
  이 형식으로 전체 모델을 디스크에 저장하고 Python이외의 언어로 작성된 서버등의 다른 환경에 로드할 수 있으며, 보다 효율적인 실행제공을 위해 코드에서 컴파일러 최적화의 수행이 가능하며, 더 넓은 관점이 요구되는 백엔드/장치 런타임과 인터페이스 할 수 있음.

- torch.jit.trace(model, exam_input) : PyTorch모델을 Trace 컴파일러를 사용해 TorchScript로 변환. 코드 실행해 발생 작업을 기록하여, 정확히 수행하는 ScriptModule을 구성함. 제어흐름(if_else등)은 지워짐. .code로 추적된 그래프를 코드형태로 볼 수 있음. 
- torch.jit.script(model) : PyTorch모델을 Script 컴파일러를 사용해 TorchScript로 변환. Trace와 달리 제어 흐름이 그대로 남지만, 더 많은 비용이 소모됨. 일부 상황에선 Trace를, 일부 상황에선 Script를 사용하는게 좋음.
- `@torch.jit.ignore` : nn.Module은 TorchScript가 아직 지원하지 않는 Python기능을 사용하기에, 일부 메소드를 제외해야 할 때 사용하는 데코레이터.

- model.save(path) : TorchScript모듈을 아카이브 형식으로 디스크에 저장. 코드, 매개변수, 속성, 디버그 정보가 포함됨(아카이브는 완전 별도의 프로세스에서 로드할 수 있는 독립된 표현임).
- torch.jit.load(path) : 저장된 TorchScript모듈을 디스크에 로드.


# torch_%
- torchmetrics : torch를 다루고 torch tensor를 반환하는 pytorch의 metrics들이 모인 패키지.
- torchvision : 비전분야의 유명 데이터셋, 모델, 전처리도구가 포함된 패키지.
- torchtext : 자연어처리 분야의 유명 데이터셋, 모델, 전처리도구(텍스트에 대한 추상화기능)가 포함된 패키지.
- torchaudio : 오디오 분야의 torch_%. 
- torchnlp : (?)
## metrics
- torchmetrics.Accuracy() : Accuracy객체 생성. acc(pred, y)로 사용할 수 있으며, 반환값은 float의 scalar텐서임.
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

#
***

# pytorch_lightning
- PyTorch Lightning : TF의 keras와 같은 PyTorch에 대한 High-level 인터페이스를 제공하는 오픈소스 Python 라이브러리. 코드의 추상화를 통해 프레임워크를 넘어 하나의 코드 스타일로 자리 잡기 위해 탄생한 프로젝트.
- 장점 : 효율적(정돈된 코드스타일/추상화), 유연함(torch에 적합한 구조, trainer 다양하게 override가능, callback), 구조화(딥러닝시 다뤄야 할 부분들), 다양한 학습 방법에 적용가능(GPU/TPU/16-bit precision),
  PytorchEcosystem(어먀격한 testing과정, Pytorch친화적), logging과 통합/시각화 프레임워크 등등의 장점을 가지고 있음. .to(device)를 쓰지않고도 간단하게 다른 하드웨어를 쓸 수 있음.
- pytorch_lightning.seed_everything(seed) : 랜덤시드 고정.

## Model
- model(x) : 훈련된 모델(순전파)사용. forward함수가 구현되어있어야 사용할 수 있음.

- model.to_torchscript() : PL모델을 PyTorch로 내보냄. torch.jit.save 메서드를 사용해 저장할 수 있으며, 바닐라Pytorch보다 더 나은 성능을 발휘할 수 있으나 모든 모델이 깔끔하게 내보내지는건 아니라서 주의가 필요함.
- model.eval() : 모델을 추론모드로 전환. 모델의 예측시 사용해줘야 함.
- model.freeze() : 모델의 파라미터들을 동결. 모델의 예측시 사용해줘야 함.
- trainer.test(test_dataloader) : LightningModule모델 테스트. 따로 테스트할 모델을 지정하지 않으면 val_dataset을 통해 구한 best모델로 test를 진행함.

### callbacks
- 커스텀 콜백 : pytorch_lightning.Callback를 상속받는 클래스를 제작해 LightningModule과 같은(on_train_batch_start등) 함수를 정의해 콜백을 생성할 수 있음.
- pytorch_lightning.callbacks.ModelCheckpoint() : 체크포인트 저장을 커스텀하기 위해 사용하는 콜백. 설정하지 않아도 각 버전마다 체크포인트를 저장함. %_step()함수의 log로 판별함.
  dir_path, file_name, verbose(저장결과 출력여부), save_last(마지막 체크포인트 저장여부), save_top_k(save_last제외 저장할 체크포인트 개수), monitor, mode등의 매개변수 사용.
- pytorch_lightning.callbacks.EarlyStopping() : EarlyStopping사용. monitor, patience, verbose, mode등의 매개변수 사용가능. 스네이크 표기법으로 된 함수도 존재함. %_step()함수의 log로 판별함.

### logger
- 기본 로그저장경로 : `lightning_logs/`
- Trainer(logger=Logger) : Logger 지정. 리스트형태로 Logger를 넣어 여러 버전의 로그를 저장하게 할 수 도 있음.
- Trainer(log_every_n_steps=k, flush_logs_every_n_steps=n) : k step 마다 로그를 기록하고, n step마다 로거에 쓰도록 함. 훈련속도와 비용의 조정을 위함. 기본은 50, 100번. 

- 지원하는 Logger : Comet(Comet.ml), CSV(yaml/csv(로컬파일)), MLFlow, Neptune, WandB, TensorBoard, TestTube(TensorBoard지만 더 나은 폴더구조를 써 로컬파일시스템에 로그인).
- pytorch_lightning.loggers.TensorBoardLogger(path) : TensorBoard의 Log를 저장하는 Logger 생성. `tensorboard --logdir=path`명령어로 로그를 확인할 수 있음.
- pytorch_lightning.loggers.WandBLogger(path) : Weights & Biases 를 사용해 기록하는 Logger 생성.
- pytorch_lightning.loggers.CSVLogger(path) : yaml 및 CSV 형식으로 로컬파일시스템에 로깅하는 로거 생성.

- self.log(변수이름="", 값) : 하나의 값(스칼라)을 수동으로 로깅. 확인하고자 하는 metric(loss등)을 기록. batch_start가 포함된 함수를 제외한 모든 위치에서 기록함.
  on_step(bool, step마다 기록), on_epoch(bool, Epoch마다 기록), prog_bar(bool, 진행률표시줄에 메트릭 기록), logger(bool, 로거)의 옵션 사용가능.
- self.log_dict(metrics) : 여러개의 변수(스칼라)를 수동로깅. metrics는 {변수이름: 값} 형태의 dict.
- logger = self.logger.experiment : 히스토그램, 텍스트, 이미지등과 같이 스칼라가 아닌 모든것을 기록하기 위해 로거객체를 직접 사용.
- logger.add_image() : 로거객체에 이미지 기록.
- logger.add_hitogram(...) : 로거객체에 히스토그램 기록.
- logger.add_figure(...) : 로거객체에 수치(figure) 기록.

### Lightning Module
- LightningModule : 모델 내부의 구조를 설계하는 research/science클래스. 모델구조/데이터전처리/손실함수 설정 등을 통해 모델 초기화/정의. 
  모든 모듈이 따라야 하는 9가지 필수메서드의 표준 인터페이스를 가지고 있음.
- Lifecycle : LightningModule클래스가 함수를 실행하는 순서. 아래의 순서에 더해 각 batch와 epoch마다 함수 이름에 맞게 정해진 순서대로 호출됨.
  [__init\_\_ > prepare_data > configure_optimizers > train_dataloader > val_dataloader > test_dataloader(.test()호출시)]의 순서.

#### method
- __init\__() : 모델 초기화. 기존 모델과 동일하게 변수/층등의 정의/선언과 초기화(super(CLASS_NAME, self).__init\__())가 진행됨.
- forward() : 모델 실행시 실행될 순전파 메서드. 기존모델과 동일하게 사용할 수 있음. 입력 x를 받아 예측 pred를 반환하는 구조. 
- 손실함수() : 손실함수. 클래스 내부에 정의해 사용하는게 구조화되어 좋음. logits과 labels를 받아 계산된 loss를 반환하는 구조.
- configure_optimizers() : 옵티마이저 설정. self.parameters()와 lr을 인자로 받는 옵티마이저를 반환하는 형태. 
  여러 옵티마이저를 사용한다면 리스트 형태로 리턴, training_step에서 optimizer의 인덱스를 추가로 받아 여러 모델을 번갈아 학습하게 됨. 
  스케줄러를 설정해 `return [optimizer], [scheduler]`의 형식으로 반환해 스케줄러를 사용할 수 도 있음.
- configure_callbacks() : 콜백 설정. 사용될 콜백 정의 후 리스트 형태로 반환하게 하면, fit()등을 호출시 호출되어 사용되게 됨.

- 모델 학습루프 : 복잡하던 훈련과정을 추상화. 3가지의 루프 패턴마다 3개지의 메소드를 가지고 있음. 일반적으로는 training_step -> validation_step -> validation_epoch_end의 구조를 사용함.
  %_step()인 함수들의 반환값은 %_epoch_end()함수의 output으로 들어가게 되며, ModelCheckpoint함수의 metric은 %_step()함수의 log, EalryStopping함수는 %_epoch_end()함수의 log여야 함.
- training_step() : 모델 훈련시 진행될 훈련 메서드. train_batch(+batch_idx)를 받아 self.forward -> self.loss -> {'loss': loss}반환 형태로 이뤄짐. 반드시 loss키를 반환해야 함.
- training_step_end() : 모델 훈련시 한 step의 끝마다 수행될 메서드.
- training_epoch_end() : 모델 훈련시 한 epoch의 끝마다 수행될 메서드.
- validation_step() : 모델 훈련시 검증과정에서 진행될 validation메서드. val_batch(+batch_idx)를 받아 self.foward -> self.loss -> {'val_loss': loss}반환 의 형태로 이뤄짐.
- validation_step_end() : 모델 훈련시 검증과정에서 한 step의 끝마다 수행될 메서드.
- validation_epoch_end() : 모델 훈련시 검증과정에서 한 epoch의 끝마다 수행될 메서드. 주로 outputs를 입력받아 평균 val_loss와 log(텐서보드용)를 반환하는 구조. 반환시킬 것(dict)들의 리스트를 반환.
- test_loop_step() : 모델 테스트(.test())시 테스트 과정에서 진행될 test_loop메서드.
- test_loop_step_end() : 모델 테스트시 테스트 과정에서 한 step의 끝마다 수행될 test_loop메서드.
- test_loop_epoch_end() : 모델 테스트시 테스트 과정에서 한 epoch의 끝마다 수행될 test_loop메서드.

- 데이터 준비 : Pytorch의 데이터 준비 과정을 크게 5가지로 구조화. 다운로드, 데이터정리/메모리저장, 데이터셋 로드, 데이터전처리(transforms), dataloader형태로 wrapping. 데이터로더는 리스트의 형태로 반환해 둘 이상 지정할 수 있음.
- prepare_data() : 데이터 다운로드/로드 후 데이터 전처리, 분리까지 진행해 self.train_data등의 elements에 정의/선언.
- train_dataloader() : train_Dataloader 반환. 모델 학습에 사용될 데이터로더 반환.
- val_dataloader() : val_Dataloader 반환. 모델 검증에 사용될 데이터로더 반환.
- test_dataloader() : test_Dataloader 반환. 모델 테스트에 사용될 데이터로더 반환.
- predict_dataloader() : predict_dataloader 반환. 모델 예측에 사용될 데이터로더 반환.

### model save/load
- trainer.save_checkpoint(path) : path에 모델 저장. 저장된 모델은 일반 torch check_point모델로도 사용가능(PL이 Pytorch의 래퍼이니)함.
- model.to_onnx(path, input_sample, export_params=bool) : PytorchLightning 모델을 onnx모델로 내보냄.

- pytorch_lightning.Trainer('resume_from_checkpoint' = path) : 기존의 체크포인트로 저장된 모델과 모델정보를 로드. 학습을 이어서 할 수 있음.
- model = pytorch_lightning.LightningModule.load_from_checkpoint(path) : 사전훈련된(저장된)모델 로드. 

### optimization
#### Automatic optimization
- 자동최적화 : Lightning은 내부적으로 다음을 수행함. 
  각 epoch의 각 batch(+각 optim)마다 training_step을 수행해 loss를 구한 뒤 optimizer.zero_grad() -> loss.backward() -> optimizer.step(loss) -> lr_scheduler.step()
```python 
for epoch in epochs:
  for batch in data:

      def closure():
          loss = model.training_step(batch, batch_idx, ...)
          optimizer.zero_grad()
          loss.backward()
          return loss

      optimizer.step(closure)

  for lr_scheduler in lr_schedulers:
      lr_scheduler.step()
```
- 임의의 간격으로 단계최적화 : 학습룰워밍업, 홀수스케줄링 등 옵티마이저에서의 작업수행을 위해 optimizer_step()을 재정의해줄 수 있음. optimizer_closure매개변수를 받아야 하며, optim과 lr_scheduler의 step을 해주면 됨.
#### Manual optimization
- self.automatic_optimization = False : init에서 설정시 수동 최적화를 할 수 있음. 수행시 pl은 정밀도 및 가속기 논리만 처리하며, 사용자가 가중치 갱신, 누적, 모델 토글등을 해줘야 함.
- opt = self.optimizers() : 옵티마이저 호출.
- opt.zero_grad() : 옵티마이저 초기화. 항상 manual_backward의 앞에 호출되어야 함.
- loss = self.compute_loss(batch) : loss 도출. 
- self.manual_backward(loss) : 수동 역전파 수행
- opt.step() : 가중치 갱신).

- sch = self.lr_schedulers() : 스케줄러 호출. 스케줄러를 임의의 간격으로 호출할 수 있게 함. training_step에서 optim과 함께 이뤄져야 함.
- sch.step() : lr 조정. 

## Trainer
- Trainer : 모델의 학습을 담당하는 클래스. 모델의 학습에 관여되는 engineering(학습epoch/batch/모델의 저장/로그생성까지 전반적으로)을 담당.
- pytorch_lightning.Trainer() : 트레이너 객체 생성. 다양한 args를 통해 트레이너 설정(gpus(GPU개수), callbacks(콜백리스트), max_epochs(epochs)등)가능.
  - accelerator : "dp"전달시 입력한 개수의 GPU에서 분산학습을 진행하겠다는 뜻(Single-Node)이며, "ddp"는 다양한 분산컴퓨터시스템에서 다양한 GPU를 사용하겠다는 뜻(Multi-Node)임.
  - resume_from_checkpoint : 저장된 체크포인트의 경로를 넣으면 자동으로 모델과 학습정보를 로딩해 기존의 학습을 이어가게 됨.
  - num_processes : 멀티 cpu를 사용하게 함.
  - gpus : gpu를 사용하게 함(0 전달시 None과 동일하게 미사용).
  - tpu_cores : TPU로 모델학습을 할 수 있게 함.
  - precision : bit수(16)를 전달하면 16bit-precision이 가능하게 됨.
  - progress_bar_refresh_rate : 진행률 표시 줄의 갱신 비율. 0으로 하면 진행률 바가 나타나지 않게 됨.
  - weights_summary : (학습 초반에 나오는)가중치 요약 창에 대한 설정. None으로 하면 가중치 요약 창이 꺼지게 됨.

- 트레이너.fit(모델) : 모델 학습. sklearn의 fit메서드와 비슷함.
- 트레이너.test(test_dataloader) : fit한 LightningModule모델 테스트. 
- 트레이너.save_checkpoint(path) : path에 모델 저장.


# Ignite
- Ignite : 텐서플로우의 keras와 같이, High-level인터페이스를 제공하는 오픈소스 라이브러리. Lightning과 달리 표준 인터페이스를 가지고있지 않음.


# Geometric
- (?)

# fast.ai
- (?)






#
***
# REFERENCE
- [1](https://pytorch.org/)
- [2](https://www.pytorchlightning.ai/)
- [3](https://koreapy.tistory.com/788)
- [4](https://baeseongsu.github.io/posts/pytorch-lightning-introduction/)
- [5](https://i-am-eden.tistory.com/16)
- [6](https://ichi.pro/ko/pytorch-lightning-model-eul-peulodeogsyeon-e-baepohaneun-bangbeob-139124684689567)
- [7](https://pytorch.org/mobile/home/)
