# pytorch
- Tensor > Numpy 의 ndarray 와 유사. GPU 연산가속도 가능. 튜플타입을 가져, 모든 튜플 연산을 지원(+ 등 가능). 인덱싱도 가능.

- torch.empty(y, x) > 초기화 되지 않은 y*x의 행렬 생성. 그 시점에 할당된 메모리에 존재하던 값이 초기값으로 나타남. 
- torch.rand(y, x) > 무작위로 초기화된 행렬 생성.
- torch.ones(크기) > 크기에 맞게 1로 채워진 텐서 반환. 
- torch.zeros(y, x,) > 0으로 채워진 행렬을 생성.
- torch.tensor(iterator 객체) > 데이터로 텐서 생성.
- 텐서를 생성할 때는 모두 dtype=long 식으로 데이터 타입을 지정하고, requires_grad=True 를 사용해 그 텐서에 대해 미분이 가능하도록 연산들을 모두 추적할 수 있게 할 수 있다. 이 경우, 그 텐서를 사용한 식은 전부 grad_fn 이라는 속성을 가지게 된다.

- 텐서.size() > 텐서 사이즈 반환.
- 텐서.item() > 요소가 하나만 있는 텐서에서 그 값을 빼온다.
- 텐서.view(사이즈) > 텐서 크기, 모양 변경. (16), (2, 8)등 여러 인수를 넣을 수도 있다. -1은 하나의 인수가 고정되어 있을때 -1을 넣은 인수를 자동으로 넣는다는 의미이다.

- 텐서.numpy() > 텐서를 넘파이 배열로 변환. charTensor 는 변경불가하며, 텐서와 넘파이는 메모리 공간을 공유해 하나를 변경시 다른 하나도 변경된다.
- torch.from_numpy(numpy) > 넘파이배열을 텐서로 변환.

- 텐서.add_(텐서) > 앞쪽 텐서에 요소 텐서를 더함. _는 앞쪽 요소를 바꾼다는 뜻으로, _를 빼면 앞의 텐서는 바뀌지 않는다.0
- torch.add(텐서1,텐서2,out=텐서) > 텐서 1과 2를 더함. out 의 텐서는 크기가 동일해야 하고, 생략할 수 있음.
- 텐서.mean() > 텐서안의 값들의 평균을 텐서로 반환한다.

- res.backward() > res 기준 연쇄법칙 적용.
- x.grad > x에 대한 미분값 출력(미분을 직접 해준 후여야 함

- torch.utils.data.DataLoader(트레인 세트, batch_size=i, shuffle=T/F, num_worker=i) > 데이터를 배치 사이즈대로 나눠 로드. 마지막은 프로세스를 몇개 사용하냐 라는 의미로, 오류가 난다면 0으로 하면 된다.
- torch.FloatTensor(X_data-ndarray 타입) > float 타입 요소를 가진 텐서로 변환. 다른 타입도 존재.
- 텐서.premute(shape number - 0,3,1,2 식으로) > 텐서의 순서를 변환 (20, 32, 32,3)을 채널수가 사이즈보다 먼저 나오는 텐서에 맞게 (0,3,32,32)로 바꿀 수 있다.


# torchvision
***
- torchvision.datasets.데이터이름(root=다운파일 저장경로, train=T/F 트레이닝 데이터/테스트 데이터, download=True, transform=전처리방법(torchvision.transform.Compose( [ tr.Resize(사이즈 x*x),tf.ToTensor(),그외 전처리 방법들 ] ) 식으로) ) : 데이터 셋 다운로드. PIL 이미지 형식이다. 채널수가 앞에 있어 size 가 3,8,8 처럼 나온다.
- torchvision.datasets.load_files("경로") : 파일 로드. .data 로 데이터, .target 으로 레이블을 받아 올 수 있다.
- torchvision.datasets.ImageFolder(root=경로, transform=전처리방법) : 폴더 안의 이미지 파일을 전부 긁어 와준다. Dataloader 에 넣을 수 있다.
