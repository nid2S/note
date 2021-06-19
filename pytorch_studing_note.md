# pytorch

## tensor
- 텐서 : pytorch의 행렬(데이터)를 저장하는 자료형. numpy의 ndarray와 비슷함. 인덱스접근, 슬라이싱 등이 전부 가능함.
- 브로드 캐스팅 : 크기가 다른 행렬(텐서)들의 크기를 자동으로 맞춰 연산을 가능하게 해주는 기능. 연산시 더 큰 차원에 맞춰짐(요소 복제).

- torch.FloatTensor(array) : array로 32bit 부동소수점 자료형의 텐서 생성.
- torch.자료형Tensor(array) : array로 지정된 자료형의 텐서 생성.
- torch.zeros_like(array) : array와 동일한 차원의 0으로 채워진 텐서 생성.
- torch.ones_like(array) : array와 동일한 차원의 1으로 채워진 텐서 생성.

- 텐서.shape/dim()/size()/sum()/argmax()/max(-dim=i-)/mean(-dim=i-)/matmul(텐서)/mul(텐서) : 텐서에 대해 사용할 수 있는 연산들.
- 텐서.view(array) : 텐서의 크기(차원)변경. numpy의 reshape와 같이 전체 원소수는 동일해야 하고, -1 인자를 사용할 수 있음.
- 텐서.squeeze() : 차원의 크기가 1인 경우 해당차원 제거.
- 텐서.unsqueeze(i) : i 위치(shape의 위치)에 크기가 1인 차원을 추가.
- 텐서.자료형() : 텐서의 자료형을 변환(TypeCasting).
- 텐서.연산_() : 기존의 값을 저장하며 연산. x.mul(2.)의 경우 x에 다시 저장하지 않으면 x엔 영향이 없으나, x.mul_()은 연산과 동시에 덮어씀.

- torch.cat(\[텐서1, 텐서2], dim=i) : i 번째 차원을 늘리며 두 텐서를 연결. 기존 차원을 유지한채 지정 차원의 크기만 커짐.
- torch.stack(\[텐서1, 텐서2, 텐서3], -dim=i-) : 텐서(벡터)들을 순차적으로 쌓음. 차원이 하나 늘어남. i번 차원이 늘어나게 함.

###### tensor expression
- 2D Tensor : (batch size, dim)
- 3D Tensor : (batch size, length(time step), dim)




