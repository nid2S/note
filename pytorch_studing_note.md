# pytorch

## tensor
- 텐서 : pytorch의 행렬(데이터)를 저장하는 자료형. numpy의 ndarray와 비슷함. 인덱스접근, 슬라이싱 등이 전부 가능함.
- 브로드 캐스팅 : 크기가 다른 행렬(텐서)들의 크기를 자동으로 맞춰 연산을 가능하게 해주는 기능. 연산시 더 큰 차원에 맞춰짐(요소 복제).

- torch.tensor(i) : 텐서 생성. .item()으로 값을 받아올 수 있음. 
- torch.zeros(shape) : 0으로 초기화된 shape의 텐서 생성.
- torch.randint(shape) : shape의 랜덤으로 값이 할당된 텐서 생성.

- torch.자료형Tensor(array) : array로 지정된 자료형의 텐서 생성(ex-Float:32bit 부동소수점). 
- torch.zeros_like(array) : array와 동일한 차원의 0으로 채워진 텐서 생성. requires_grad 매개변수로 학습을 통해 값이 변경되는 변수(가중치, 편향)인지 명시해줄 수 있음.
- torch.ones_like(array) : array와 동일한 차원의 1으로 채워진 텐서 생성.

- required_grade = bool : 텐서.grad에 텐서에 대한 기울기를 저장. 텐서 생성시 매개변수로 줄 수 있음. 
- 텐서.backword() : 역전파. 해당 수식의 텐서(w)에 대한 기울기를 계산. w가 속한 수식을 w로 미분.

- 텐서.shape/dim()/size()/sum()/argmax()/max(-dim=i-)/mean(-dim=i-)/matmul(텐서)/mul(텐서) : 텐서에 대해 사용할 수 있는 연산들. dim 인자는 해당 차원을 제거(해당 차원을 1로 만듦)함.
- 텐서.view(array) : 텐서의 크기(차원)변경. numpy의 reshape와 같이 전체 원소수는 동일해야 하고, -1 인자를 사용할 수 있음.
- 텐서.squeeze() : 차원의 크기가 1인 경우 해당차원 제거.
- 텐서.unsqueeze(i) : i 위치(shape의 위치)에 크기가 1인 차원을 추가.
- 텐서.scatter(dim, 텐서, 넣을 인자) : dim차원에서, 텐서의 데이터(내부 데이터를 인덱스로)대로 넣을 인자를 삽입(할당).  
- 텐서.자료형() : 텐서의 자료형을 변환(TypeCasting).
- 텐서.연산_() : 기존의 값을 저장하며 연산. x.mul(2.)의 경우 x에 다시 저장하지 않으면 x엔 영향이 없으나, x.mul_()은 연산과 동시에 덮어씀.

- torch.log(텐서) : 텐서의 모든 요소에 로그를 적용.
- torch.exp(텐서) : 텐서의 모든 요소에 ln(log_e)를 적용.
- torch.cat(\[텐서1, 텐서2], dim=i) : i 번째 차원을 늘리며 두 텐서를 연결. 기존 차원을 유지한채 지정 차원의 크기만 커짐.
- torch.stack(\[텐서1, 텐서2, 텐서3], -dim=i-) : 텐서(벡터)들을 순차적으로 쌓음. 차원이 하나 늘어남. i번 차원이 늘어나게 함.
###### tensor expression
- 2D Tensor : (batch size, dim)
- 3D Tensor : (batch size, length(time step), dim)


## model
- 가설 선언 후 비용함수, 옵티마이저를 이용해 가중치, 편향등을 갱신해 올바른 값을 찾음.
- 비용함수를 미분해 grandient(기울기)계산. 
- optimizer.zero_grad() > cost(loss).backward() > optimizer.step() 과정을 거쳐 optimizer에 인자로 준 텐서(가중치, 편향)를 갱신함.

- torch.manual_seed(i) : 랜덤시드 고정.
### class
- 파이토치의 대부분의 구현체(모델)는 모델 생성시 클래스를 사용.
- torch.nn.Model상속 클래스 구현 > __init__에서 super().__init__을 호출, 사용할 모델(층)정의 > forward(self,x)(자동실행, 모델 사용 후 값 반환).
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
- torch.utils.data.DataLoader(dataset, batch_size=i, shuffle=True) : 데이터셋을 i개의 미니배치로 학습시킴. shuffle(Epoch마다 데이터 학습순서를 섞음)매개변수 사용가능.
  반환되는건 iterable객체로, enumerate를 이용해 batch_idx와 sample(x, y)을 꺼낼 수 있음. 반복문이 하나 추가되는걸 제외하고는 배치학습법과 동일.  
- 커스텀 데이터셋 구현 : torch.utils.data.Dataset을 상속받는 클래스 제작 후 __init\__(전처리), __len\__(길이, 총 샘플 수), __getitem\__(특정샘플, 인덱스)을 구현해 제작.
```python 
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
```python 
# 원-핫 인코딩
y_one_hot = torch.zeros_like(hypothesis) 
y_one_hot.scatter_(1, y.unsqueeze(1), 1)
```
### activation function
- torch.sigmoid(텐서(식)) : 시그모이드 사용.
- torch.nn.functional.softmax(텐서) : 소프트맥스 사용. dim=i매개변수(적용될 차원 선택)사용가능.
- torch.nn.functional.log_softmax(텐서) : 로그 소프트맥스 사용. torch.log(F.softmax())와 동일.

### optimizer
- 옵티마이저.zero_grad() : gradient 0으로 초기화.
- 옵티마이저.step() : 주어진 학습대상들을 업데이트.  
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
- torch.optim.SGD(\[가중치(학습대상1), 편향(학습대상2)], lr= learning_rate) : SGD(확률적 경사하강법)사용. 모델.parameters()를 넣을 수 있음.

### loss
- torch.nn.functional.mse_loss(prediction, label) : MSE(평균제곱오차) 손실함수 사용.
- torch.nn.functional.binary_cross_entropy(prediction, label) : 이진분류(로지스틱 회귀)의 손실함수 사용.
- torch.nn.functional.cross_entropy(prediction, label) : cross-entropy 손실함수 사용. F.nll_loss(F.log_softmax(z, dim=1), y)와 동일함.

### module(layers)
- 모델.parameters() : 모델의 파라미터 출력. w와 b가 순서대로 출력됨. 
- torch.nn.Linear(input_dim, output_dim) : 선형회귀모델 사용. 이대로 모델로 쓸 수도, 모델에 층으로 넣을수도 있음.
- torch.nn.Sigmoid() : 시그모이드 층을 쌓음. Linear() > Sigmoid() 로 로지스틱 회귀 구현 가능.

### model
- torch.nn.Sequential(layers) : 시퀀셜 모델 생성. nn.Module층을 쉽게 쌓을 수 있도록 함. 대부분의 파이토치 모델은 클래스로 구성되나 아주 간단한 모델의 경우 가끔 사용됨.







