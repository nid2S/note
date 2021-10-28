# etc_info
***
- 이미지를 저장하기 위한 센서의 모든 픽셀에는 하나의 센서만을 저장함.
- 인간의 눈이 녹색광에 더 민감하여 청,적색 픽셀에 비해 두배의 녹색 픽셀이 있고, 이 패턴이 베이어 패턴이다.
- 누락된 두 색은 인접 픽셀의 색으로 보완하는 demosaicing 방법을 사용한다.
- 2x2만 보는 pixel doubling interpolation 과 3x3까지 보는 bilinear interpolation 이 있다.
- 이런 과정을 거치며 RGB 이미지로 변환되며, RGB 픽셀 배열은 일반적으로 디스크 저장 전에 JPG/PNG 형식으로 압축된다.
- 대부분의 이미지 형식은 image header 와 data 부분으로 나뉨.
- numpy 에 저장되는 색 데이터는 당시 유행했던 BGR 방식을 채용했음.

- 머신러닝 과정은 크게 데이터 수집 > 데이터 점검,탐색(분석) > 전처리,정제 > 모델링,훈련 > 평가 > 배포 총 6단계로 나뉜다.
- 머신러닝에 이용되는 라이브러리 중에서는 lasagna 라는 theano 기반 라이브러리도 있다.


# numpy
***
- numpy : 수치연산에 최적화. 배열,행렬,배열에서 작동하는 다양한 수학함수를 지원, 배열의 모양은 각 차원을 따라 크기를 제공하는 정수형 튜플. 
  다차원 배열 슬라이싱을 하려면 각 차원별로 슬라이스 범위를 지정해 줘야 함.

- np.array(리스트) : 리스트에 해당하는 배열 생성
- np.array([[1,1,1],[2,2,2]]) : 2행 3열짜리 2차원 np배열 생성. shape()로는 (2,3)이 출력되며, x[1,2] 식으로 두번째 열의 세번째 요소를 뽑아낼 수 있다.

- np.array() : 리스트, 튜플, 배열로 부터 ndarray 생성
- np.asarray() : 기존의 array 로  부터 ndarray 생성
- np.arange(start, end, step) : 리스트의 슬라이스와 같이 범위대로 배열을 제작.
- np.linespace(start, end, num) : 시작부터 끝까지 같은 간격으로 num(요소개수/간격)의 배열 생성
- np.logspace(start, end, num) : 시작부터 끝까지 log 스케일 간격으로 num 개 요소의 배열 생성
- np.where(조건) : 조건문(리스트<1 식으로 내부에 리스트 포함)에 사용. 조건에 밎는 인덱스들을 ndarray 형태로 반환. 슬라이싱에 사용 가능. 
  (조건문, 맞으면, 아니면) 식으로 구성해 처리를 할 수도 있음.

- np.zeros(shape) : 모든 값이 0인 배열 생성.
- np.ones(shape) : 모든 값이 1인 배열 생성.
- np.full(shape, num) : 모든 값이 num 인 배열 생성.
- np.diag(1차원 배열) :  대각행렬 생성. k 매개변수에 음수를 넣으면 그 절댓값 만큼 아래 행에서 시작.
- np.eye(i) : 대각선이 1이고 나머지는 0인 i*i의 2차원 배열 생성
- np.sin(x) : 사인 함수를 이용해 배열 x와 대응하는 배열 생성
- np.random.random(shape) : 임의의 값(0~1)으로 채워진 배열 생성.
- np.random.permutation(i) : i 까지 랜덤으로 섞인 배열 반환.
- np.random.normal(avg, std, shape) : avg의 평균과 std의 표준편차를 가지고 있는 shape형태의 배열 생성.
- np.random.seed(i) : random 유사난수화. 랜덤에서 난수 생성시 사용되는 시드를 고정시켜 유사난수로 만든다.
- np.unique(배열) : 배열에 있는 값의 종류를 배열로 반환.
- np.percentile(배열, [분위\]) : 배열에서 분위에 해당하는 샘플을 추출해 반환. [0,25,50,75,100\]식으로 지정하면 된다.

- np.dot(배열1, 배열2) : 내적곱(점곱) 생성. 배열1의 열 개수와 배열 2의 행 개수가 동일해야 함. 1의 행과 2의 열 개수를 가짐. 
  행렬곱의(i,j)는 (1의 i행합 * 2의 j열합)의 요소값을 가짐. 2차원에서는 아래와 같으나 고차원에서는 다른 역할을 수행함.
- np.matmul(배열1, 배열2) : 행렬곱 생성. 2차원 이상의 배열은 2차원 배열을 여러개 가지고 있다 보기에((1,2,3,4) > (3,4)를(1,2)개) 
  행렬 1의 마지막 차원 요소 개수와 행렬2의 뒤에서 두번째 차원 요소 개수가 같아야 한다.
- np.argmax(배열) : 배열중 최대치의 인덱스 반환.
- np.square(배열) : 배열의 제곱 반환.
- np.mean(배열) : 배열의 평균을 출력. (x == y)식으로 하면 두 배열의 동일도를 받아볼 수 있음.
- np.expand_dims(np 배열,index) : np 배열의 index 위치에 데이터를 추가해 차원을 늘림.
- np.allclose(nd1, nd2) : 두 행렬이 완전히 동일한지 반환.
- np.reshape(배열, 차원) : 배열을 같은 크기의 다른 형태로 차원 변형. 요소의 총 개수는 같아야 함. 차원에 -1이 들어갔다면 개수에 맞게 알아서 지정된다는 뜻임.
- np.float32(배열) : ndarray 의 데이터 타입을 변환. 다른타입도 가능, 부동소수점 데이터 유형으로 변환시 작업 중 오버플로우를 방지의 기능이 있음.

- np.concatenate((a1, a2, ...), axis=i) : 넘파이 배열을 합침. 합치려는 배열들이 합칠 axis를 제외하고 일치하는 shape를 보유해야 함.
- np.stack((a1, a2, ...), axis=i) : 넘파이 배열을 합침. 합치려는 배열의 모든 shape가 동일해야 함.
- np.hstack((a1, a2, ...)) : 배열을 가로로 이어붙임. concatenate() axis=1과 동일한 결과.
- np.vstack((a1, a2, ...)) : 배열을 세로로 이어붙임. concatenate() axis=0과 동일한 결과.
- np.dstack((a1, a2, ...)) : 배열을 이어붙임. stack() axis=-1(ndim)과 동일함.
- np.column_stack((a1, a2, ...)) : 행벡터가 주어지면 열방향으로 합침. axis=1로 합침.
- np.split(ndarray, n, axis=i) : 배열을 n개로 나눔. n자리에 정수 배열을 넣으면 해당 인덱스를 끊어서 반환함.

- np.linalg.svd(ndarray) : SVD 사용. SVD 대로 직교행렬 U, 대각 행렬(특이값의 리스트형태)S, 또 다른 직교행렬 VT 를 반환함.

- ndarray.flags : 어레이의 메모리 레이아웃에 대한 정보.
- ndarray.shape : 배열 차원의 튜플.
- ndarray.ndim : 배열의 차원 수.
- ndarray.size : 배열의 요소 수.
- ndarray.itemsize : 한 배열 요소의 길이 (바이트).
- ndarray.dtype : 배열의 데이터 타입.
- ndarray.data : 배열 데이터의 시작을 가리키는 파이썬 버퍼 객체.
- ndarray.nbytes : 배열의 요소가 사용한 총 바이트.
- ndarray.T : 배열의 행과 열의 크기 교체.

- ndarray.round(i) : 배열의 소수의 소수점 이하 i까지 출력.
- ndarray.astype(np.데이터타입) : 배열의 데이터타입 변환.
- ndarray.transpose(변환된 axis) : 배열의 axis를 섞음. 0,1,2로 인덱스가 부여되있는 axis를 (1,0,2)식으로 섞을 수 있음.
- ndarray.reshape(shape) : 배열의 shape를 바꿈.

- np.save(이름,배열) : 1개의 배열을 NumPy format 의 바이너리 파일로 저장.
- np.savez(경로,배열(x=x, y=y 식으로 이름 부여 가능)) : 여러개의 배열을 1개의 압축되지 않은 *.npz 포맷 파일로 저장. 
  이때 불러오면 numpy.lib.npyio.NpzFile 이며, 개별 배열은 인덱싱해서(['x'\]) 사용할 수 있다.
- np.savez_compressed(이름,배열) : 여러개의 배열을 1개의 압축된 *.npz 포맷 파일로 저장. 이때도 똑같이 인덱싱 가능.
- np.load(경로) : 저장된 ndarray 파일을 load. close()를 해주어야 하며, 닫은 후에는 불러온 파일을 사용할 수 없다.

- np.savetext() : 여러개의 배열을 텍스트 파일로 저장. header="", footer="" 로 파일 시작과 끝에 #으로 시작하는 주석을 달아 줄 수 있고, 
  fmt="%.1f" 식으로 들어가는 인수들에 대한 포맷을 지정할 수 있다.
- np.loadtext() : 텍스트 파일을 배열로 불러옴. ndarray로 불려옴.

# pandas
***
- 판다스는 시리즈, 데이터프레임, 패널 총 세개의 데이터 구조를 사용함.
- 시리즈 : 1차원 배열, 딕셔너리. 1차원 배열의 각 값에 해당하는 인덱스를 부여할 수 있는 구조.
- 데이터프레임 : 2차원 배열. 행과 열이 존재하는 배열. 시리즈의 모음.
- 패널 : 3차원 배열. 데이터프레임의 모음.

- pd.Series(1차원 리스트 , index(인덱스가 될 리스트)) : 시리즈생성. 인덱스는 정수뿐만 아니라 문자열등도 가능하며, 배열을 제외한 모든 인수는 생략 가능함.
- pd.DataFrame(리스트, index(행이름), columns(열이름)) : 데이터프레임 생성. index나 columns생략시 0부터 자동할당, 리스트,시리즈,딕셔너리,ndarray 등으로 생성, 인덱스등 가능.
- pd.DataFrame.from_dict(dict) : 딕셔너리를 데이터프레임 객체로 바꿈. Collection의 OrderedDictionary를 사용하면 좋음.
- pd.DataFrame.from_records(리스트, columns=열이름리스트) : 리스트틀로부터 데이터프레임 생성.
- pd.DataFrame.from_items(리스트([열이름, 값리스트\]형태)) : 주어진 리스트대로 데이터프레임 생성.

- pd.read_csv(파일경로) : 파일읽기. csv파일이 아니더라도 읽을 수 있고, sep/delimiter(구분이 될 문자 지정), header(헤더설정, None으로 해주면 첫번째 열이 해더가 되지 않음.),
  names(각 열의 이름 설정), index_col=[열이름/정수\](인덱스가 될 칼럼 선택)등의 인수를 쓸 수 있음.
- pd.read_table/fwf/excel/hdf/sql/json/html/stata/clipboard/pickle/gbq() : 해당 포맷의 파일을 읽음. 
  table(일반구분파일),fwf(고정너비형식 테이블),clipboard(클립보드에서 읽은 후 csv로), pickle(pickled object), gbq(Google BigQuery)

- pd.get_dummies(범주형 데이터) : one hot encoding. 나오는 값들이 그 종류만큼 (이름)_(데이터이름)의 형태로 나뉨.
- pd.set_option('display.max_columns', i) : IPython display설정. 최대로 출력할 열의 개수를 i개로 늘림.
- pd.concat(df1, df2) : 두 데이터프레임을(행으로)합침. ignore_index=True로 기존의 인덱스를 무시하고 이어넣을 수 있음. append와 동일한 기능을 함.
- pd.concat([df1, df2\], axis=1) : 두 데이터프레임을 열로 함침. 인덱스가 동일한 항목들의 열을 합쳐 칼럼이 늘어나게 함. ignore_index를 사용하면 순서대로 합쳐짐. 
  딕셔너리로 넣어 열이름을 주고 합칠수도 있음({"label":label, "pred":pred"} 식).

- df.columns : 칼럼 Index 객체로 반환. 여기에 값을 할당해 헤더를 지정해 줄 수도 있음.
- df.values  : 값들 배열형태로 반환.
- df.index   : 인덱스+타입 Index 객체로 반환.
- df.loc[인덱스\] : 데이터 로드. 조건식(==, !=)을 넣어 줄 수도 있음.
- df.iloc[인덱스\] : 인덱스(행번호)를 이용해 행 선택.

- df["칼럼명"\] : 인덱스 지정, 시리즈로 반환. 차원이 여러개일 경우는 [5, 1:3\]처럼 차원별로 인덱스를 지정해 주어야 함.
- df[["칼럼명"\]\] : 인덱스 지정, 데이터프레임으로 반환.
- df["칼럼1","2","3"\] : 열(칼럼) 다중 선택. [칼럼1:칼럼3\]식으로 선택할 수 도 있음.
- df[dataFrame.Age == 30\] : 데이터를 선택해서(조건을 줘서)표시 가능

- df.head(n) : 위에서부터 n개의 데이터를 가져옴. 생략시 5개.
- df.tail(n) : 끝(아래)에서부터 n개의 데이터를 가져옴. 생략시 5개.
- df.sample(n) : n개의 데이터를 랜덤으로 추출.
- df.info() : 데이터프레임의 정보 볼 수 있음.
- df.mean() : 데이터들의 평균값을 얻을 수 있움.
- df.reset_index() : 인덱스 리셋. 인덱스를 0부터 시작하게 함.
- df.plot() : 데이터를 가지고 그래프를 그림. kind="bar" 등으로 종류를 지정해 줄 수 도 있고, stacked=True로 밑을 덮는 형태로도 가능함.
- df.to_csv(filename) : 데이터 프레임을 csv파일로 저장. index=bool. header=bool 인수로 인덱스,헤더의 존재여부를, na_rep인수로 na를 어떻게 표시할지를 설정 가능.
- df.to_dict/numpy() : 데이터프레임을 딕셔너리/ndarray로 변환. 이 외에도 index,values등에 to_list/tolist()등의 함수가 있기도 함.

- df.items() : 열이름, 시리즈 형태의 제너레이터 반환. iteritems()도 비슷함.
- df.iterrows() : (인덱스, 시리즈(열의 값들))형태의 튜플 반환. df[열\]로 빼울때는 .item()을 써야 했던것과 달리 row[열\]만써도 바로 가져올 수 있음.
- df.itertuples(): rows와 비슷하나 map(이터레이터)으로 반환 후 pandas.core.frame.Pandas객체 반환. 사용은 튜플과 비슷.
  
- df[열이름\].nunique() : 열에서 중복된 샘플을 제외한 샘플의 개수 출력. 중복된 값이 있다면 단 하나의 값만 남게 됨.
- df[열이름\].value_counts() : 그 칼럼에 등장하는 값의 종류를 그 값이 나온 수와 함께 나타냄.
- df.isna()/isnull() : 데이터프레임속 값이 널값인지 여부를 표시. .sum()으로 널값의 총 개수를 확인할 수 있음.
- df.fillna(널값을 대체할 값) : 널값을 다른값으로 대체함. df.열이름.fillna()로 속성별로 다르게 널값을 바꿀수 도 있음. inplace=bool로 데이터프레임에 바로 적용되게 할건지 결정가능.
  df[na열이름\].fillna(df.groupbu(열이름)[na열이름\].transform("median"(바꿀값, max등)), inplace=True)처럼 사용해 na를 그럴듯한(중간값)으로 설정하는것도 가능.
- df.drop(삭제할 행) : 행 삭제. axis=1 을 주면 열을 삭제할 수 있음. inplace매개변수를 사용하면 반환형이 NonType이 되어 체인함수의 사용이 불가능해짐(가장 마지막 함수에만 사용해야함).
- df.duplicated() : 행마다 중복된 데이터인지 반환. 인자로 열을 넣어 중복기준 설정 가능(생략시 모든 내용이 동일해야 함).
- df.drop_duplicates() : 열에서 중복 내용 제거. 인자로 열을 넣어 중복기준 설정가능, keep="lsat"로 중복된 값 중 마지막 값을 유지할 수 있음(first가 기본).
  
- df[열이름\].astype("int/float") : 데이터타입 변경. df[칼럼명\]은 df.칼럼명 과 같음. "category"로 설정시 데이터가 범주형으로 변경, 원핫인코딩이 가능해짐.
- df[열이름\].str.replace(요소1, 요소2) : 그 칼럼의 값 중 요소1(정규표현식 패턴도 가능)과 일치하는 값을 요소2로 바꿈. [x1,x2\],[y1,y2\] 식으로 넣어 여러개의 값을 변환할 수 도 있음.
- df[열이름\].apply(함수) : 데이터프레임 해당 열속 모든 행에 함수를 적용. 함수는 row(이름달라도 가능)를 꼭 인자로 받아야함. 
  함수에 직접 값을 전달하려면 apply(함수, 변수명=값)식으로 가능, 함수 내부에서 row.열이름 으로 행의 특정 값에 접근가능.
- df[열이름\].map(함수/딕셔너리) : 데이터프레임 해당 열(칼럼)속 모든 행에 주어진것을 적용. 함수 전달시 apply와 동일하게 사용,작동. 
  딕셔너리는 {원래값:바뀔값}형식으로 넣어 데이터프레임(해당 행)의 값을 바꿀 수 있음(값이 없으면 nan).
- df.applymap(함수) : 데이터프레임속 모든 값에 함수를 적용.
  
- df.append(df2) : 데이터프레임에 데이터프레임을(행으로)추가. ignore_index=True로 본래의 인덱스를 무시하고 기존값에 인덱스를 이어 넣을 수 있음. 
- df.groupby([열\]) : group by. 해당 열의 데이터가 같은걸 모음. .groups로 생성된 그룹과 인덱스/타입 확인가능. __next__시 그룹이름과 그룹요소 개수가 나옴.
- df.query(조건) : 데이터프레임에서 조건에 맞는 값들을 가져옴. []에 조건을 넣는 것과 동일. 조건은 문자열.
- df.filter(조건, axis=0) : 인덱스 조건에 맞게 데이터를 가져옴. item=[열이름\](해당 열), regex=패턴(정규표현식에 맞으면), like=str(문자열이 포함되면), 
  axis=1(판단기준을 열이름으로 바꿈)사용가능.
- df.sort_index([열이름\]) : 인덱스를 기준으로 데이터프레임 정렬.
- df.sort_values([열이름\]) : 해당 열을 기준으로 데이터프레임 정렬. axis=1(열이름 정렬), ascending=False(내림차순 정렬), inplace=True(df에 바로적용), 
  na_position="first/last"(na위치)등을 사용가능.

- 차트 = df.plot(kind='bar', title='날씨', figsize=(12, 4), legend=True, fontsize=12) : 차트 종류,제목,크기,범례 유무,폰트 크기 설정.
- 차트.set_xlabel('도시', fontsize=12)/set_ylabel('기온/습도', fontsize=12)/legend(['기온', '습도'\], fontsize=12) 등 plt메서드 사용가능.

- pandas_profiling : pip install -U pandas-profiling으로 설치가능. 명령어 한번으로 데이터 탐색을 시켜줌.
- df.profile_report() : 프로파일 결과를 반환. .to_file('경로.html')로 html 으로도 확인할 수 있다.

# Scipy
***
- scipy.sparse.csr_matrix(eye) > 주어진 배열 중 0이 아닌 요소만 위치와 값을 저장(희소행렬).
- scipy.sparse.coo_matrix((ones, (arange, arange)) > 주어진 배열 중 0이 아닌 요소만 위치와 값을 저장(희소행렬).

# matplotlib.pyplot
***
- 구동방식 : PyplotAPI(matplotlib.pyplot 모듈에 함수로 정의되어있는 커맨드방식)/객체지향API(객체지향 라이브러리를 직접 활용하는 방식)두가지의 사용 방법이 있음.
- 객체지향API사용 : Figure객체 생성 > 하나 이상의 Axes객체 생성 > 생성된 axes에 대해 조작(헬퍼함수로 primitives생성)  
- plt.subplots() : Figure객체 생성 후 Figure.subplots()를 호출하여 리턴. fig, ax를 반환. constrained_layout=True로 각 플롯간 간격을 자동조절가능.

- matplotlib.font_manager.FontProperties(fname(폰트경로)).get_name() : 경로의 폰트의 이름을 얻음.
- matplotlib.rc('font', family = 폰트이름) : 폰트변경. 기본폰트가 sans-erif 이기에 한글폰트가 깨질 수 있는데, 이를 방지하기 위해 사용.

- plt.figure(figsize=(n,m)) : 그래프 크기 조절.
- plt.show() : 생성한 plot(그래프)를 보여줌.
- plt.savefig("저장경로") : 그래프 저장.

- plt.plot(정수형 리스트) : 리스트대로 선 또는 마커그래프 생성. 리스트를 한개 넣으면 y값 으로 인식하고 x를 자동생성하고, 두개면 순서대로 x,y 라고 인식한다.
- plt.plot(정수형 리스트, 'ro') : ro - 빨간색 원형 마커. 이런식으로 색과 그래프 마커를 지정해 줄 수 있음.
- plt.plot(x,y,type,x,y,type) : 이런 식으로 매개변수를 넣거나 plot 을 여러번 사용하면 여러개의 그래프를 그릴 수 있다.
- plot - color : r(red),g(green),b(blue),c(cyan),m(magenta),y(yellow),k(black),w(white) , color='css_color_name/#rgb'으로 다양한 색상 지정 가능.
- plot - LineStyle : -(solid),--(dashed),-.(dashed-dot),:(dotted).
- plot - Markers : o(circle),s(square),*(star),p(pentagon),+(plus),x(X),D(diamond),|/_(h/v line),^/v/</>(triangle),1/2/3/4(tri)
- plot - style : plt.style.use(스타일) 로 각종 스타일 사용가능. 종류는 'seaborn-white'등, plt.style.available 에서 확인가능함.
- plt.subplot(nrow, ncol, pos) : 여러개의 그래프를 그림(격자형). nrow, ncol은 고정한 채 해당 그래프의 코드 앞에 pos를 증가시키며 사용. pos는 1부터 시작. 
- plt.subplot_adjust(left, bottom, right, top, wspace, hspace) : 서브플롯들의 위치를 직접 조정. 모든 값은 0~1의 소수(비율). 뒤의 둘은 너비/높이의 비율.

- plt.title(title) : 그래프 제목 설정. loc-타이틀 위치('right','left'), pad-타이틀&그래프간격, 폰트 크기와 두께 설정 가능.
- plt.x/ylabel(text) : x/y축에 레이블(축제목) 설정.
- plt.x/yticks(number 리스트) : x/y축에 눈금 표시. 빈리스트를 넣으면 제거. label 매개변수에 리스트를 넣어 각 눈금의 이름을 지정해 줄 수 있음.
- plt.x/ylim([min,max\]) : x/y축 범위 설정. 축의 시작과 끝 지점을 지정할 수 있다.
- plt.axis([x min, x max, y min, y max\]) : 축의 범위 지정.
- plt.x/yaxis(rotation, fontsize) : x/y축의 폰트 회전 및 크기 설정.
- plt.annotate(str, xy, xytext, arrowprops) : xy((x, y))부터 xytext((x, y))까지 설정에 따라({'color':'green'})화살표를 그린 후 문자열을 표시.

- plt.grid() : 그래프에 격자 표시. axis='y/x' 로 가로/세로 방향의 그래프만 그릴 수 있음. color,alpha,linestyle 등의 매개변수 사용가능.
- plt.legend() : 그래프에 레이블(범례) 표시, plot 에서 label="" 로 준 레이블이 그 그래프의 레이블이 됨. loc=위치("upper right"/2 등)인자로 위치를 지정해줄수도 있음.
- plt.tick_params() : 눈금 스타일 설정. axis-적용축('x','y','both'), direction-눈금위치('in','out','inout'),
  pad-눈금&레이블 거리, length/width/color-눈금 길이/너비/색, labelsize/labelcolor-레이블 크기/색, t,b,l,r - bool&눈금표시 위치.

- plt.text(x, y, "text") : 텍스트 표시.
- plt.axhline(y, x_min(0~1), x_max(0~1)) : y에 min 부터 max 까지 수평선을 그음. color, linestyle, linewidth 등 매개변수 사용가능.
- plt.axvline(x, y_min, y_max) : x에 min 부터 max 까지 수직선을 그음. 왼쪽 아래부터 오른쪽 끝까지 0~1로 표현.
- plt.hlines(y, x_min, x_max) : y에 min 부터 max 까지 수평선을 그음. min,max 가 0~1로 표현되지 않음.
- plt.vlines(x, y_min, y_max) : x에 min 부터 max 까지 수직선을 그음.
- plt.fill_between(x, y, alpha) : 그래프에서 그 범위를 채움. (x,y1,y2)식으로 두 그래프 사이의 영역을 채울 수 도 있음. color 매개 변수로 색 지정 가능.
- plt.fill(x,y,alpha) : x,y 점들로 정의되는 다각형의 영역을 자유롭게 채울 수 있음.
  
- plt.imshow(image) : image를 표시함. cmap="gray"(흑백으로)등 이미지에 대한 조작을 할 수 있음. 
- plt.bar(x, y) : 막대그래프를 그림. width(너비),align(눈금위치. 히스토그램처럼 눈금을 막대 끝으로 이동가능,'edge'),
  color,edgecolor,linewidth(테두리두께),tick_label,log(bool, y를 로그스케일로) 등의 매개변수 사용 가능.
- plt.barh(x, y) : 수평 막대그래프를 그림. height 를 제외하면 매개변수 동일. width/height 를 음수로 지정하면 막대 위쪽에 눈금 표시.
- plt.scatter(x, y) : 산점도(상관관계표현)를 그림. s(마커 면적),c(마커 색),alpha(투명도) 등의 매개변수 사용가능.
- plt.hist(리스트(x, y도 가능)) : 히스토그램(도수분포표 그래프)을 그림. 리스트에 나온 계급과 그 빈도를 분석해 자동으로 히스토그램으로 만들어줌. 
  bins(쪼갤영역수),density(bool,막대사이를 이어 하나로), histtype(막대 내부를 채울지,'step') 등 매개변수 사용가능.
- plt.hist2d(x,y,(가로 셀 개수, 세로 셀 개수)) : 격자로 나눠서 빈도를 확인할 수 있는 2차원 히스토그램을 그림. scatter과 같이 쓰면 좋고, hist의 모든 인자사용가능. 
- plt.errorbar(x, y, yerr) : 에러바(데이터편차표현)를 그림. yerr 는 각 y의 편차로 위아래 대칭인 오차로 표시, 
  [(error), (error)\]식으로 넣으면 아래방향/위방향 편차를 나타내게 됨. uplims/lolims(bool, 상한/하한 기호표시) 매개변수 사용가능.
- plt.pie(ratio(각 영역 비율 리스트), label(각 영역 이름 리스트)) : 파이차트(범주별 구성비율 원형표시)를 그림. 
  autopct(영역안에 표시될 숫자 형식 지정), startangle(시작각도), counterclock(bool,반시계 여부), explode(0~1실수 리스트, 차트중심에서 이탈도), 
  shadow(bool,그림자), colors(리스트,색이름/코드), wedgeprops({'width': 0.7, 'edgecolor': 'w', 'linewidth': 5}, 반지름 비율, 테두리색, 테두리너비) 매개변수 사용가능.
- plt.box(x, y) :  극단치, 최소, 최대, 각 사분위, 중앙값, 사분위범위(상자)등을 알 수 있는(기초통계,편차 확인)그래프를 그림.
- plt.hexbin(x, y) : 각 데이터가 겹치는(밀집된)정도를 볼 수 있는 그래프를 그림.
- plt.area(x, y) : 각 데이터의 영역을 볼 수 있는 그래프를 그림.

# seaborn
- seaborn : matplotlib기반 시각화 라이브러리. 통계그래픽을 그리기 위한 고급 인터페이스 제공. 흔히 sns라는 별칭으로 사용됨. 여러개의 그래프를 그리면 한 그래프에 그려짐.
- sns.histplot() : 변수에 대한 히스토그램을 표시. 하나 혹은 두개에 대한 변수분포를 나타냄. 범위에 포함하는 관측수를 세어 표시함.
- sns.kdeplot() : 하나 혹은 두개의 변수에 대한 분포를 표시. 연속된 곡선의 그래프를 얻음. 밀도추정치를 시각화함.
- sns.ecdfplot() : 누적분포를 시각롸. 실제 관측치의 비율을 시각화(선형).
- sns.rugplot() : 축을 따라 눈금을 그려서 주변 분포도를 표시함. 다른 그래프와 함께 쓰이는 경우가 많음.
- sns.barplot() : 이변량(변수 두개)분석을 위한 plot. x축엔 범주형 변수, y축에는 연속형 변숙를 넣음.
- sns.countplot() : 범주형 변수의 발생횟수를 셈. 일변량 분석.
- sns.boxplot() : box plot을 그림. 최소/1사분위/평균/3사분위/최대 를 보기위한 그래프. 특이치 발견에도 좋음.
- sns.violineplot() : box plot과 비슷하나 분포에 대한 정보도 주어짐(박스 양 옆으로 곡선이 그려짐).
- sns.stripplot() : 연속형 변수와 범주형 변수 사이의 그래프. 산점도로 표시되나, 범주형 변수의 인코딩을 추가로 사용함.
- sns.swarmplot() : strip + violin. 데이터포인트 수와 함께 각 데이터의 분포도 제공함.
- sns.heatmap() : 데이터간 수치에 따라 색상을 입힘. 
- sns.clustermap() : heatmap에서 몇몇 특징을 그룹화(토너먼트처럼 선으로)함.
- sns.facetgrid() : 특정 조건에 따라 그래프를 각각 확인.
- sns.jointplot() : 두 변수의 분포에 대한 분석가능. 두 displot사이에 scatter/hex plot이 추가되어 분포를 추가러 확인가능.
- sns.pairplot() : 숫자형 특성들에 대해 각각의 히스토그램과 두 변수간 scatter plot을 그림. 데이터셋을 통째로 넣어도 됨.
- sns.regplot() : Regression(회귀)결과를 그래프로 보여줌.
- sns.lmplot() : regplot + faceGrid. hue에 들어간 칼럼 속 값들을 따로따로 모델링해 결과(regplot)를 보여줌.

# plotnine(ggplot)
***
- plotnine : 업데이트가 멈춘 시각화패키지 ggplot을 다시 포팅한 패키지. 커스터마이징이 되어있음. 
- plotnine.ggplot(petitions)  : 데이터로 그래프 제작
- plotnine.aes('category')  : 데이터 축 설정. x='' , y='' 식으로 레이블을 지정하지 않고 하나만 지정하면 x로 들어감.
- plotnine.geom_bar(fill='green')) : 데이터 종류 설정. geom_point() 식으로 하면 산점도 타입이다.
- plotnine.ggplot(data=데이터, mapping= plotnine.aes(x=, y=, color=) + plotnine.geom_point(alpha=f)) : 식으로도 가능.

# mglearn
***
- mglearn.discrete_scatter(X[:, 0, X[:, 1], y) > 산점도를 그림.

- mglearn.dataset.make/load_데이터이름() > 데이터셋 로드.
- mglearn.dataset.make_forge() > 인위적인 이진분류 대이터셋 로드. x,y에 각각 특성이 들어간다.

- mglearn.plots.plot_모델이름() > 그 모델의 그래프를 그림.
- mglearn.plots.plot_2d_classification(fit 된 모델, X, fill=bool, alpha=) > 선형 이진 분류 그래프를 결정경계와 함께 그린다.
- mglearn.plots.plot_knn_classification(n_neighbors = k) > knn 분류를 그래프로 그림.
- mglearn.plots.plot_knn_regression(n_neighbors = k) > knn 회귀를 그래프로 그림.
- mglearn.plots.plot_ridge_n_samples() > 리지 회귀를 그래프로 그림.

# folium
- python 지도 시각화 패키지.

# 다른 시각화 라이브러리
- Altair : 주피터등 웹 환경에서 유효. 데이터프레임의 칼럼을 기반으로 데이터추출, 다양하고 예쁜 그래프를 쉽게 그림. 
- Plotly : Altair와 같이 JS로 웹브라우저 상에서 이미지를 렌더링하는 라이브러리. D3.js를 사용해서 시각화.


# dlib
***
- viola & Jones 알고리즘 > Face Detection(얼굴에 Bounding Box) 가능하게 함. Face Landmark Detection 이 더 자세한 개념.
- Head Pose Detection(얼굴 방향)/Face Morphing(두 얼굴의 중간 얼굴을 생성)/Face Averaging(얼굴들의 평균 얼굴을 생성)/
  Face Swap/Blink&Drowsy Driver Detection(눈의 깜빡임 감지)/Face Filter 등이 Face Landmark Detection 을 통해 가능해짐.
- dlib.get_frontal_face_detector() > face Detection 을 기능하게 하는 face detector 생성.
- dlib.shape_predictor(데이터셋(얼굴 랜드마크)이 있는 경로) > Landmark detector 생성.
- faceDetector(이미지,업스케일 횟수(0,1,2)) > 이미지에서 얼굴을 찾음.

# OpenCv (cv2)
***
- 설치 : `pip install opencv-python`로 설치해야 함.
- cv2.imread('이미지 경로',읽는 방법) > 이미지 읽기. 이미지가 작업 디렉토리에 있거나 전체 경로를 주어야 함. data type 을 지정하지 않으면 상수형 int 타입으로 저장.
- 1 = 컬러이미지(default) / 0 = 그레이스케일 이미지 / -1 = unchanged
- cv2.cvtColor(BGR 이미지,cv2.COLOR_BGR2RGB) > OpenCv 로 읽어온 BGR 이미지를 RGB 이미지로 변환함
- cv2.cvtColor(BGR 이미지,cv2.COLOR_BGR2GRAY) > 컬러 BGR 이미지를 흑백으로 전환한다.
- 이미지.copy() > 본래의 사진과 완전히 별개인 이미지복사본을 생성
- 이미지[w1:w2,h1:h2] > 저 부분의 이미지만을 복사.
- 이미지.shape[0] = 행에 대한 정보 / 이미지.shape[1] = 열에 대한 정보 / 이미지.shape[2] = 채널에 대한 정보. 그레이스케일(흑백)이라면 없음.

- cv2.imwrite('경로',이미지) > 경로에 이미지 저장
- cv2.imshow("윈도우( = 뜰 창) 이름)",이미지) > 이미지 띄울 준비
- cv2.waitKey(ms) > 입력한 밀리세크(1000ms = 1s)동안 이미지를 띄움. 0일 경우 어느 키보드 입력시까지 띄움.
- cv2.destroyAllWindows() > 이름 그대로. waitKey 이후에 필수.
- cv2.resize(이미지, 절대크기(상대면 0,0), x 스케일(절대면 패스), y 스케일(절대면 패스), interpolation=method(cv2.INTER_LINEAR,보간법)) > 이미지 resize.

- cv2.line(이미지,시작좌표(x,y),끝좌표(x,y),색(R,G,B),thickness(꽉 채울려면 -1),lineType(기본적으로 윤곽선,CV_LINE_AA=안티엘리어싱)) > 선을 그린다.
- cv2.polylines(img, pts, isClosed, color, thickness=None, lineType=None, shift=None) > 점들을 이어 선을 그림
- cv2.circle(이미지,중심좌표(x,y),radius,선 색,두께/채우기 타입,선 타입) > 원을 그린다.
- cv2.ellipse(이미지,중심,(x길이,y길이),기울기,원 시작 각도, 원 끝 각도(시작부터 끝의 각도까지만 그려짐),색,두께,타입) > 타원을 그린다.
- cv2.rectangle(이미지,좌측상단 좌표,우측하단 좌표,색,두께,타입) > 사각형을 그린다.
- cv2.putText(이미지,문자열,표시될 좌표,폰트,글자 크기,색,두께,타입) > 글자를 입력한다.
- cv2.flip(이미지, 방향) > 이미지 반전. 1 = 좌우, 0 = 상하
- cv2.lotation(이미지, cv2.ROTATE_각도_방향) > 이미지 회전. cv2.ROTATE_90_CLOCKWISE = 시계방향 90도. 반시계 방향은 COUNTERCLOCKWISE, 180도는 방향을 쓰지 않는다.

- cv2.VideoCapture(비디오 파일 경로(카메라=0 -여러개일 경우 1씩 추가-)) > VideoCapture 객체 생성.
- VideoCapture객체.isopened() > 읽을 영상이 남아 있는지 반환.
- VideoCapture객체.read() > 영상의 존재 여부와 이미지를 반환.
- VideoCapture객체.release() > 동영상 종료.
- VideoCapture객체.get(cv2.CAP_PROP_FRAME_WIDTH/HEIGHT) > 프레임 너비와 높이 휙득. 실수형이기에 정수형으로 변환 필요.

- cv2.videoWriter('경로/이름',cv2.VideoWriter_fourcc('M','P','4','V'-MP4/'M','J','P','G'-avi),FPS(33이하),(프레임너비,높이)) > videoWriter 객체 생성.
- videoWriter 객체.write(read 로 얻은 이미지) > 저장

- 이미지 모델 학습시에는 np.ndarray(shape=(image_amount, image_size[1\], image_size[0\]), dtype=np.float32) 식으로 준비된 이미지 파일과
  np.ndarray(shape=(image_amount,), dtype=np.int32)식으로 준비된 레이블에
  이미지 오픈 > fit > asarray > normalized(astype(np.float32)>/255.0 ) > all_images[i\]에 넣은 이미지를 사용해야 한다.
  mnist, PIL > (number, y, x) 로 train. | openCv > (number, y, x ,3) 으로 train

# PIL(Pillow)
- PIL : Python Imaging Library. 
- 설치 : `pip install image`로 설치.
- Pillow : PIL의 프로젝트 포크(Fork).
  
- PIL.Image.fromarray(frame(cv2비디오캡쳐.read())) : cv2에서 읽은 이미지/비디오의 프레임을 Image객체로 변환.  
- PIL.ImageTK.PhotoImage(Image객체) : Image객체를 ImageTk객체로 변환. tkinter에서 사용가능(Label의 image인자에 넣을 수 있음).


# tensorflow
***
##### info
- 구글이 주도적으로 개발한, 가장 널리 쓰이는 딥러닝 프레임워크중 하나. keras중심 고수준 API 통합 지원.
- Python, C++, JS, Java, Go, Swift등 다양한 프로그래밍언어의 API를 지원.
- TPU지원, 일반적으로 32bit의 곱셈연산을 16bit로 줄임 등의 특성이 있음.
- Estimators(객체지향 레벨) > layers,losses,metrics > Python/C++ Tensorflow > CPU/GPU/TPU 순으로 아키텍쳐(API)가 구성되어 있음.
- tensorflow in java : tf에서 libtensorflow.jar 다운로드 > 압축 해제후 jar파일 src에 복사 > properties 에서 add jar > 다운한 파일선택 > apply 
  과정을 거친 후 import org.tensorflow로 사용.


##### divece
- tf.test.is_gpu_available() : gpu가 사용가능한 상태인지 반환.
- tf.test.gpu_device_name() : 사용가능한 gpu의 이름을 반환.

##### tensor
- tf.Variable(수식, name="변수명") : 변수 선언 후 수식으로 정의. x+4 식으로 수식을 지정해 변수에 할당하는 방식.
- tf.constant(상수, name="상수명") : 상수 선언 후 값 지정. y = tf.constant(20, name="y") 식으로 사용.

- tf.zeros/ones(shape) = shape대로 0/1으로 채워진 텐서 생성.
- tf.random.uniform(shape, min, max) : shape형태의, min~max사이의 랜덤 값을 가진 텐서 생성.
- tf.random.normal(shape, mean, stddev) : shape형태의, 평균이 mean이고 표준편차가 stddev인(기본은 0,1) 랜덤 값을 가진 정규분포 텐서 생성.
- tf.convert_to_tensor(array) : array를 텐서로 변환.

- tf.rank(텐서) : 텐서의 랭크(차원), 모양, 데이터 타입 출력. tf.Tensor(rank, shape=(), dytpe=type).
- tf.matmul(텐서a, 텐서b) : 두 텐서간 행렬곱. transpose_a/b = True 로 두 행렬중 하나를 전치 후 곱할 수 있음.
- tf.cast(자료형) : 자료형 변환.
- tf.transpose(텐서) : 텐서의 모양을 반전함. ().T
- tf.convert_to_tensor(배열(ndarray)) : 배열 텐서로 변환
- 텐서.numpy() : 텐서 넘파이 변환.

- @tf.function : 데코레이터 아래의 함수를 텐서플로우 함수로 변환. 자동그래프(빠른 연산)생성, GPU연산가능 등의 특징이 있음.
  파이썬의 함수를 텐서플로우의 그래프(텐서로 자료형,연산 등)형태로 다루고 싶을때 사용. 원본 함수는 (tf.funciton).python_function()으로 받을 수 있음.
- tf.function(함수) : 함수를 텐서플로우 함수로 변환. 내부에서의 연산, 반환값 등이 모두 텐서가 됨.
- tf.autograph.to_code(파이썬함수) : 코드(함수)를 기반으로 그래프함수 제작.

- Autograd(자동미분) : tf.GradientTape API 이용, 일부입력(tf.Variable등)에 대한 기울기 계산. 기본적으로 한번만 사용, 변수가 포함된 연산만 기록함.
- tf.GradientTape().gradient(식, 변수) : 식을 변수로 미분한 값을 반환. 변수는 리스트 형태로 여러개 입력할 수 있고, 결과인 미분값로 리스트 형태로 반환됨.
  presistent=True로 설정하지 않으면 하나에 식에 대해 한번의 호출만 가능.

- session : 일종의 실행창. 텐서의 내용과 연산 결과를 볼 수 있음. 세션 선언, 실행, 종료 문으로 구분됨. 텐서플로우2에선 tf.compat.v1.Session으로 사용가능.
- tf.compat.v1.disable_eager_execution() : tf2에서 세션의 사용이 가능하게 함(disable eager 제공).
- tf.Session() : 세션 선언.
- tf.global_variables_initializer() : 변수 초기화. model에 할당해 초기화를 할 수도 있음.
- sess.run(텐서) : 실행. sess.run(model) > sess.run(변수) 식으로 사용(값을 흐르게)할 수 있음.
- sess.close() : 세션 종료.

- tf.keras.utils.to_categorical(정수 리스트) : 정수 리스트에 따라서 원핫 인코딩. \[1,3]을 넣으면 \[\[0,1,0,0],\[0,0,0,1]]을 반환하는 식.
- tf.lite.TFLiteConverter.from_keras_model(model).converter() | open('파일명.tflite', 'wb') > tf 모델 tflite 바이너리로 변환. 
  이렇게 변환한 것은 안드로이드 스튜디오의 에셋에 복사 > app 모듈의 build.gradle 에 패키지 추가 > Main_Activity 에서 이미지 바이너리 변환 > Classifier 에서 모델 사용 >
  Main_Activity 에서 출력 순으로 사용된다.

##### Dataset
- 텐서플로우 데이터셋 : 텐서플로우의 데이터셋. 효율적인 입력파이프의 작성을 지원함(모델의 입력이 됨). 입력되는 요소는 named 튜플/딕셔너리의 중첩 구조.

- tf.data.TextLineDataset(["file1.txt", "file2.txt"\]) : 파일의 라인처리(텍스트파일로 데이터 생성).
- tf.data.Dataset.list_files(path패턴) : 패턴과 일치하는 모든 파일의 데이터세트 생성.

- tf.data.Dataset.from_generator(generator) : 주어진 제네레이터를 기반으로 데이터셋 생성.
- tf.data.Dataset.from_tensor_slices(tensors) : 주어진 텐서의 슬라이스로 데이터셋을 제작. {데이터명:텐서}형태. (x_train, Y_train)식으로 넣어 즐 수 있음.
- tf.data.Dataset.from_tensors(tensors) : 주어진 텐서로 구성된 단일 요소로 데이터셋 제작.

- 데이터셋.shuffle(buffer_size(len(데이터셋))) : 데이터셋을 섞음(ShuffleDataset으로 바꿈).
- 데이터셋.batch(n) : 데이터셋의 배치를 n개로1 나눔(BatchDataset으로 바꿈).
- 데이터셋.repeat(n) : 데이터셋을 n번 반복함.
- 데이터셋.prefetch(tf.data.experimental.AUTOTUNE(동적조정)) :  데이터셋에서 요소를 사전에 가져오는 데이터셋 생성. 메모리의 소모가 늘지만 대기시간이 짧아지고 처리량이 많아짐.

- 데이터셋.cache(path) : 현 데이터셋의 요소를 캐시함. 첫 반복에 지정된 파일/메모리에 캐시되고, 그 뒤엔 캐시된 데이터셋을 사용함.
- 데이터셋.as_numpy_iterator() : 넘파이 이터레이터로 데이터셋 확인.

##### layers
- tf.keras.layers.Input(shape=(입력 차원)) : 입력차원 만큼 입력레이어 구성.
- tf.keras.layers.Dense(노드수,activation="swish/relu")(X) : 전밀집층(모든 노드가 이전 혹은 다음 노드와 연결, 전결합층)제작. input_dim(입력차원)매개변수 사용가능. 
  ((입력의 마지막차원+1(bias))*노드수)개의 파라미터가 생성, (None, 최초입력의 마지막 제외 차원, 노드수) 형태의 반환값을 반환.

- tf.keras.layers.Embedding(총 단어 개수, 결과 벡터의 크기, 입력 시퀀스 길이) : 단어를 밀집벡터로 만듦(임베딩 층 제작, 단어를 랜덤한 값을 가지는 밀집 벡터로 변환 후 학습과정을 거침). 
  (샘플개수, 입력길이)형태의 정수 인코딩이 된 2차원 정수 배열을 입력받아 워드 임베딩 후 3차원 배열을 반환. mask_zero=True인자로 패딩된 토큰을 마스킹 할 수 있음. 
- tf.keras.layers.Masking() : 패딩 토큰을 학습에 영향이 가지 않도록 배제(마스킹)함. layer._keras_mask로 결과를 확인 할 수 있음(마스킹=False).
- tf.keras.layers.Dropout(rate) : Overfitting 을 방지하기 위해 DropOut(나온(중간) 값의 일부를 누락시킴). rate 는 1 = 100%.
- tf.keras.layers.Bidirectional(layer) : 입력한 층을 양방향으로 만들어 줌. SimpleRNN, LSTM 등이 들어감.
- tf.keras.layers.TimeDistributed(layers) : RNN에서 각 스텝마다 오류를 계산해 하위스텝(앞쪽)으로 전파하게 시킴. 
  return_sequences=True 와 이것을 사용해 RNN이 many-to-many문제를 해결(시퀀스를 입력받아 시퀀스를 출력)할 수 있도록 함. 
  각 스텝에서 손실을 계산해 출력을 낼 수 있도록 하며, 없다면 각 스텝의 출력이 마지막 스텝에 모여 순차적으로 FC에 들어가 출력이 된다.
- tf.keras.layers.LayerNormalization/(layers) : 층 정규화 시행.

- tf.keras.layers.Conv1D(kernel, kernel_size, padding, activation) : 1차원 합성곱신경망 사용.
- tf.keras.layers.Conv2D(컨볼루션 크기(행,렬), 필터 이미지 개수(한 행렬의 크기 x,y), padding(='same' 입출력 사이즈 동일), activation, inputShape) : 
  이미지에 convolution filter 를 사용해 행렬을 만듦.

- tf.keras.layers.GlobalMaxPooling1D() : 1차원 풀링 실행. Conv1D 뒤에 위치.
- tf.keras.layers.GlobalAveragePooling1D() : 입력으로 들어오는 모든 벡터들의 평균을 구함. 흔히 임베딩 층 뒤에 사용됨.
- tf.keras.layers.MaxPooling2D((줄일 행렬의 크기 x, y)) : 이미지를 MaxPooling 해 크기를 줄임.

- tf.keras.layers.SimpleRNN(hidden_size) : RNN 사용. hidden_size 는 은닉상태의 크기. (batch_)input_shape 매개변수에 ((batch_size,) 
  timesteps(입력 시퀀스 길이), input_dim(입력 크기)) 로 넣어 입력을 정의해 줄 수 도 있음. return_sequences(전체 은닉상태 출력)와 
  return_state(마지막 은닉상태 한번 더 출력)매개변수 사용 가능.
- tf.keras.layers.LSTM(hidden_size, input_shape=(time_steps, input_dim)) : RNN 의 일종인 LSTM 사용. RNN 층은 (batch_size(배치 크기, 
  한번에 학습할 데이터 양), timesteps(시점, 문장의 길이), input_dim(단어 벡터 차원)) 크기의 3D 텐서를 입력으로 받음. return state 를 true 로 하면 마지막 셀 상태까지 반환, 
  양방향이면 정방향,역방향 둘 다 은닉상태와 셀상태 반환(fh,fc,bh,bc 순).
- tf.keras.layers.GRU(hidden_size, input_shape=(time_steps, input_dim)) : LSTM 을 개량한 GRU 사용. LSTM 에 비해 구조가 간단하고, 
  데이터 양이 적을떄 LSTM 보다 낫다고 알려져 있음.

- tensorflow.keras.preprocessing.sequence.pad_sequences(data, maxlen) : 데이터(리스트)의 요소 개수를 maxlen으로 고정. 적으면 0을 채우고 많으면 버림.


##### model make
- 케라스는 Sequential API, Functional API, Subclassing API 의 구현 방식을 지원.

- model = keras.Sequential([  : Sequential API(딘순히 층을 쌓아 구성할 수 있고, 여러 층을 공유하거나 다양한 종류의 입출력을 사용할 수 
  있지만 그만큼 복잡한 모델 제작에 한계가 있음)를 이용해 모델 설계.
- keras.layers.Flatten(input_shape=(x,y)),  : x\*y 픽셀의 2차원 이미지 배열을 (x*y)의 1차원 배열로 반환. input shape 매개변수는 
  input layer 를 대채할 수 있게 해주며, 배치 크기를 제외하고 차원을 지정하기에 차원이 하나 추가 될 수 있고, 배치까지 지정하려면 batch_input_shape 를 사용한다.
- keras.layers.Dense(128, activation = 'relu'),  : 밀집연결(densely-connected)층/완전연결층. 128개의 노드(또는 뉴런)을 가짐.
- keras.layers.Dense(10, activation = 'softmax') : 10개의 클래스 각각 그 클래스에 속할 확률을 출력.
- ]) : 모델(분류기반, 이 경우 최대 세개까지 레이어 추가 가능) 제작.
- Sequential() : model.add 로도 층 추가가 가능하고, 전결합층(dense)뿐 아니라 임베딩, LSTM, GRU, Flatten, Convolution2D, Batch Normalization 등 다양한 층 추가 가능.

- functional API : 함수형 API 는 Sequential API 와 달리 각 층을 일종의 함수로 정의.
- input(shape) 에서 시작해 Dense(node, activation)(inputs) > Dense()(h1) > Dense()(h2) 후 tf.keras.models.Model(input,output) 식으로 구성.
- Embedding()(input_layer) 와 Embedding_layer = Embedding() > Embedding_layer(input_layer) 는 둘다 모델의 층을 연결(functional API)함.

- Subclassing API : Subclassing API 는 모델을 클래스 형태로 제작해 사용. tf.keras.Model을 상속시키고, init에서 입력과 사용할 층을 정의, call에서 층을 쌓아(연결해)반환.
###### custom
- 모델 커스텀 : 텐서플로우의 subclassing API를 활용하면 모델의 커스텀이 가능함.

- 사용자 정의 레이어 : tf.keras.layers.Layer상속 후 init과 call을 구현함. 각 파라미터를 trainable = False로 지정해 훈련불가 가중치를 추가할 수 있음. 특정 층의 기능만 정의하고 싶을 때 적합.
- init : 층에서 사용될 하이퍼파라미터를 받고, super의 init을 실행한 뒤, 가중치/편향등의 파라미터를 정의함. 모든 파라미터는 self.add_weight(shape, initializer, trainable)형식으로 선언가능함.
- 가중치 정의 : tf.random_normal_initializer()등의 초기화객체를 생성한 뒤 tf.Variable(initial_value=w_init(shape, dtype), trainable)등의 형식으로 생성함.
- 편향 정의 : tf.zeros_initializer()등의 초기화객체를 생성한 뒤 tf.Variable(initial_value = b_init(shape, dtype), trainable)등의 형식으로 생성됨.
- build : 입력의 크기를 알때까지 가중치의 생성을 유보함. 입력으로 input_shape를 받고, 그걸 이용해 shape=(input_shape[-1], self.units)식으로 가중치를 생성함.
- call : input을 받아 연산(가설, tf.matmul(inputs, w) + b)수행 뒤 값을 반환함.
- 기타 변수 : layer.weight(층의 가중치들), layer.losses(층의 손실들), layer.metrics(정확도들)등을 사용할 수 있음.

- 사용자 정의 모델 : .fit()/evaluate()/save()등의 메서드가 필요하면(모델이 필요하면) tf.keras.Model에서 상속함. 변수 추적 외에 내부레이어도 추적해 검사를 쉽게 만들어줌. model.layer사용가능.
- 기본 구조 : init에서 super(모델명, self).__init\__()과 하이퍼 파라미터 받기, 사용할 층/변수 선언을 진행한 후 call에서 데이터(x)를 받아 모든 층을 거친 결과를 반환함.
- call : training인자로 훈련과 추론시의 동작 변동이 가능하며, mask=None으로 인자를 받거나 self.임베딩층.compute_mask(inputs)로 mask를 뽑아서 뒤의 층에 mask=mask로 마스킹이 가능하고,
  self.add_loss(손실함수)로 손실텐서를, self.add_metric(정확도함수, name)으로 정확도 스칼라 작성이 가능함.
- get_config : 사용자 정의 레이어 직렬화에 이용됨. 모델의 설정등을 {"units": self.units}의 형태로 반환함.
- 훈련루프(fit) : GradientTape(train_step메서드 재정의)로 훈련 커스텀 가능. 모델.evaluate()호출도 커스텀하려면 test_step을 같은 방식으로 재정의함. 둘다 @tf.function를 앞에붙여 속도향상가능. 
- train_step : x와 y가 묶여있는 형태의 data를 입력으로 받음. `self.compiled_metrics.update_state(y, y_pred)`로 정확도계산, `with tf.GradientTape() as tape`로 모델의 훈련부분을 시작,  
  내부에서 self(x, training)으로 예측 후 `self.compiled_loss(y, y_pred, regularization_losses=self.losses) or keras.losses의 손실함수(y, y_pred)`로 손실을 계산함. 
  이후 `tape.gradient(loss, self.trainable_variables) > self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))`로 가중치 계산 후 적용. 
  반환은 `{m.name: m.result() for m in self.metrics}`형태로 반환됨. 혹시 입력에 sample_weight가 있다면 길이 검사 후 3개로 언팩하고, loss계산과 compiled_metrics.update_state에 넣어줌.
- 손실/정확도 수동계산 : 먼저 keras.metrics의 loss_tracker(Mean등)와 매트릭 인스턴스를 정의 후 loss_tracker와 매트릭의 상태를 업데이트, 둘의 결과를 반환해 구현할 수 있음.
  `loss_tracker.update_state(loss) > 매트릭.update_state(y, y_pred) > return {"loss": loss_tracker.result(), "매트릭 명": 매트릭.result()}`의 형태로 구현가능.
  이 후 @property의 `def metrics(self): return [loss_tracker, 매트릭]`형태의 메서드를 정의해줌.
- test_step : x와 y가 묶인 data를 입력으로 받아 GradientTape를 제외한 train_step과 같은 작업(compiled_loss, compiled_metrics.update_state등)을 수행함. evaluate수행시 호출됨.

- 사용자 정의 훈련루프 : 자체 교육/평가 루프를 처음부터 작성함. epochs만큼 반복할 for문 내에, 각 epoch당 batch를 반복처리하는 for문을 열고, 각 배치마다 GradientTape를 열어, 모델,손실,갱신을 함.
- 루프 구조(코드) : `for epoch in range(epochs):`내에 `for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):`로 배치별로 나눈 뒤, 
  `with tf.GradientTape() as tape:`로 GradientTape를 열어, `logits = model(x_batch_train, training=True) > loss_fn(y_batch_train, logits)`로 손실을 계산,
  `grads = tape.gradient(loss, model.trainable_weights) > optimizer.apply_gradients(zip(grads, model.trainable_weights))`로 가중치를 갱신, 적용함. 매 배치/스텝마다 손실등을 출력.
- 매트릭 추가 : 각 스텝의 끝에서 metric.update_state()호출, 매트릭의 현재 값을 표시해야 할 때 metric.result()를 호출, 상태를 지울때(epoch끝) metric.reset.states()호출. 
  train과 val의 매트릭 인스턴스는 나눠놔야 함. updata_state에는 (y, y_pred)가 들어감. 모델 정의시 call에 add_loss를 사용했다면 loss += sum(model.losses)로 사용가능(train_step정의시도 동일).

##### model use
###### train
- model.compile(
- optimizer='adam',  : 데이터와 손실함수를 바탕으로 모델 업데이트 방향 결정.
- loss='sparse_categorical_crossentropy',  : 훈련중 모델 오차 측정.
- metrics=['accuracy']  : 훈련단계와 테스트 단계를 모니터링하기 위한 방법.
- ) : 모델 컴파일.

- model.fit(train_data , train_labels , epochs=1000(반복 횟수)) : 학습된 모델 제작. validation_data=(test_data,test_label)로 검증용 데이터로 계산한 손실/정확도를 함께 출력가능하며,
  callback 매개변수에 callbacks의 함수를 넣어 사용할 수 있음. 여러개면 [one, two\]식으로 입력. loss와 accuracy(metrics)가 담긴 딕셔너리를 반환함.

- tensorflow.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose, patience) : 과정합 방지를 위한 조기 종료 설정. 
  patience회 검증 데이터의 손실이 증가하면 학습을 조기종료함. 모델 fit 과정에서 callback 매개변수에 넣어 사용가능.
- tensorflow.keras.callbacks.ModelCheckpoint(모델명, monitor="val_accuracy", mode="max", verbose=1, save_best_only=True) : 
  검증 데이터의 정확도가 이전보다 좋아지면 모델 저장. 모델 fit 과정에서 callback 매개변수에 넣어 사용가능. 모델의 체크포인트를 저장.
- tensorflow.keras.callbacks.LearningRateScheduler(schedule) : 매 epoch가 시작될 때 업데이트된 학습률 값을 가져와 적용. 
  스케줄러 함수(epoch와 lr을 인수로 받음)를 정의해 인수로 넣고, 이를 callback에 넣어 사용.

###### save
- 모델의 저장/로드는 모두 폴더명을 입력해 줘야 함.
- model.save(path) : 전체 모델 저장. 두가지의 다른 파일 형식(SaveModel, HDF5)으로 저장가능. 확장자없이 path만 넣으면 SaveModel, %.h5면 HDF5. 
- model = tf.keras.models.load_model(모델명) : 저장된 모델 로드.

- model.get_weights() : 각 독립변수에 대한 가중치 반환.
- model.save_weights(path) : 모델의 가중치 저장.
- model.load_weight(path) : 모델의 가중치 복원. 원본 모델과 같은 아키텍쳐를 공유해야 함.

- 체크포인트 : 가중치, 모듈 및 하위 모듈 내부의 변수 세트 값. 데이터자체(변수값과 해당속성 조회경로)와 메타데이터용 인덱스파일(실제 저장된 항목과 체크포인트 번호 추적)로 구성.
- tf.train.Checkpoint(model) : 체크포인트 생성.
- 체크포인트.write(path) : 체크포인트 저장. 전체 변수 모음이 포함된 python객체별로 정렬되어 있음.
- 체크포인트.restore(path) : 체크포인트(python객체 값)를 덮어씀.
- tf.train.list_variables(patj) : 체크포인트 확인.

###### use
- model.summary() 로 모델의 정보(이름/none,출력하는 개수/파라미터(가중치의 개수))를 확인 할 수 있다.
- model.predict(X) : 모델을 사용해 입력에 따른 예측 반환.
- model.evaluate(test_images, test_labels) : 모델 성능 비교. loss, accuracy 순으로 반환. verbose = 0 > silent

##### other_API
###### TenserflowLite
- tfLite : 모바일(안드로이드/iOS), 내장형기기(내장형 리눅스, 마이크로 컨트롤러), IoT기기에서 모델을 실행할 수 있도록 지원함. 자바, Swift, Object-C, C++, Python등 지원.
- 특징 : 지연시간(서버왕복X), 개인정보 보호(기기에 정보를 남기지 않음), 연결성(인터넷X), 크기(모델, 바이너리 크기축소), 전력소비(효율적 추론, 네트워크X)의 제약사항 해결.
- 모델 표현방식 : FlatBuffers(.tflite 확장자)라는 효율적 이동이 가능한 특수형식으로 표현됨. 일반 모델 형식에 비해 축소된 크기, 추론속도개선(추가 파싱/압축해제 X, 직접 액세스)등의 이점이 있음. 
  선택적으로 메타데이터를 포함할 수 있으며, 여기에는 추론중 파이프라인의 전처리와 후처리의 자동생성을 위한 모델설명(인간이 읽을 수 있는)과 데이터(머신이 읽을 수 있는)가 포함됨.
- 생성방법 : 기존 tflite모델 사용, tflite모델 생성, tf모델을 tflite모델로 변환 의 세가지 방법으로 모델의 생성이 가능함.
- 추론실행 : 모델 유형에 따라 메타데이터가 없으면 tflite인터프리터 API를 사용(다양한 플랫폼/언어 지원)하고, 있으면 즉시사용가능한 API를 활용하거나 맞춤 추론 파이프라인을 빌드할 수 있음.

- 기존 tf모델 선택 : 텐서플로우 사이트, tflite예제 페이지에서 몇가지 태스크에 쓸 수 있는 tflite모델을 다운로드하거나, 깃허브에서 사용 예를 볼 수 있음.
- TensorflowLiteModelMaker : `pip install tflite-model-maker`로 다운할 수 있는 이미지/텍스트분류, 질문답변 task에 대한 ML작업을 지원하는 라이브러리. 예제는 tf홈페이지에서 확인가능.
- TensorflowLite 변환기 : `tf.lite.TFLiteConverter.from_saved_model(path) > converter.convert()`등을 이용해 tf모델을 tflite모델로 변환가능. 모델은 파일오픈 > write로 저장함.

- 메타데이터 : 모델 설명에 대한 표준 제공. 모델정보와 입출력 정보에 대한 중요한 지식소스. 모델스키마 아래 TFLITE_METADATA필드(파일)에 저장됨. 일부는 분류라벨/어휘 파일도 같이 제공가능.
- tflite-support : 메타데이터 도구. pip로 설치가능. 모델정보, 입/출력 정보등에 대한 정보객체를 각각 생성해야 함. `tflite_support.metadata_schema_py_generated as _metadata_fb`.
- _metadata_fb.ModelMetadataT() : 모델 정보 객체 생성. .name/description[튜플 할당\]/version/author/license등의 속성에 값 할당가능.
- 모델정보객체.content = _metadata_fb.ContentT() : 입/출력 정보를 위한 객체의 content생성. 
- 입력정보객체.content.contentProperties = _metadata_fb.ImagePropertiesT() : contentProperties객체 생성. .colorSpace, .contentPropertiesType등의 지정이 가능함.
- 입력정보객체.processUnit/stats : 모델의 정규화/stat(max,min등)의 정보 할당. `_metadata_fb.NormalizationOptionsT/StatsT()`로 객체생성, 값 할당, 객체 할당의 과정이 필요함.
- 라벨 : 출력정보객체에 .content.content_properties/contentPropertiesType와 .stats설정 후 .associatedFiles를 설정해 주어야 하는데, 여기 넣을 값은 객체의 리스트로, 각 객체는
  _metadata_fb.AssociatedFileT()객체를 만들어 name(path)와 description 설정 후, `.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS`설정을 해 주어야 한다.
- 모델 정보 결합 : _metadata_fb.SubGraphMetadataT()객체 생성 후 `input/outputTensorMetadata = [입/출력 정보 객체]`할당, `모델정보객체.subgraphMetadata = [subgraph]`.
  tflite_support.flatbuffers.Builder(0)객체 생성 후 `b.Finish(모델정보객체.Pack(b), tflite_support.metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)`,
  `metadata_buf = b.Output()`으로 플랫버퍼를 생성함.
- 모델로 압축 : _metadata.MetadataPopulator.with_model_file(model_file)객체 생성 후 `.load_metadata_buffer(플랫버퍼) > .load_associated_files([레이블파일path])`,
  `populator.populate()`으로 메타데이터/관련파일을 모델로 압축할 수 있음. 
- 메타데이터 읽기 : PYTHON에서 JSON형식으로 읽거나, JAVA에서 메타데이터 추출기 라이브러리를 사용하거나, 일반적 zip파일로 압축을 풀고 볼 수 있음. 예제는 홈페이지에 있음.

  
- tfliteInterpreter : 메타데이터가 없는 모델 실행. 다양한 플랫폼과 언어(안드로이드, iOS, 리눅스등)에서 지원됨. `모델로드 > 데이터변환 > 추론실행 > 출력해석`의 단계를 거침.
- 모델 실행 : `모델 메모리에 로드 > 기존모델 기반 인터프리터 구축 > 입력텐서값 설정 > 추론호출 > 출력텐서값 읽기`의 단계를 거쳐야 함. 

- 안드로이드(자바) : `dependencies {implementation 'org.tensorflow:tensorflow-lite-task-text:0.1.0'}`로 tflite사용 후, 
  인터프리터 객체를 생성한 뒤 Map<String, Object>객체로 데이터를 받아, 인터프리터.run(input, output)으로 사용됨.
  이 외에도 runSignature[서명이 있을때\], runForMultipleInputsOutputs[다중 입/출력일 때\]등을 사용할 수 있음. 
  getInput/OutputIndex(opName)메서드를 사용할 수 있으며(유효하지 않으면 예외), 인터프리터.close()로 리소스 해제를 해 주어야 함.
- iOS(Swift) : `import TensorFlowLite`, 인터프리터 객체 생성 후 `allocateTensors > copy > invoke > output`의 과정을 거쳐 사용.
- iOS(Object-C, C) : `@import TensorFlowLite;` 혹은 `#include "tensorflow/lite/c/c_api.h`, 이후 다양한 과정을 거쳐 사용(예제는 tf홈페이지에서).
- Linux(Python) : `interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH)`로 인터프리터 객체 생성 후 `interpreter.get_signature_runner() > my_signature(x)`혹은
  `allocate_tensors > get_input/output_details[입/출력 정보 획득] > set_tensor > invoke > get_tensor`로 이뤄짐.
- 마이크로컨트롤러/Linux/iOS/안드로이드(C++) : FlatBufferModel를 통해 인터프리터 사용가능. `모델로드 > 인터프리터 빌드 > 인풋 > Invoke(); > 아웃풋`의 순서로 진행. 예제는 tf홈페이지.

# Pytorch ( torch )
***
- [pytorch_studing_note](pytorch_studing_note.md)
- Tensor > Numpy 의 ndarray 와 유사. GPU 연산가속도 가능. 튜플타입을 가져, 모든 튜플 연산을 지원(+ 등 가능). 인덱싱도 가능.

- torch.empty(y, x) > 초기화 되지 않은 y*x의 행렬 생성. 그 시점에 할당된 메모리에 존재하던 값이 초기값으로 나타남.
- torch.rand(y, x) > 무작위로 초기화된 행렬 생성.
- torch.ones(크기) > 크기에 맞게 1로 채워진 텐서 반환.
- torch.zeros(y, x,) > 0으로 채워진 행렬을 생성.
- torch.tensor(iterator 객체) > 데이터로 텐서 생성.
- 텐서를 생성할 때는 모두 dtype=long 식으로 데이터 타입을 지정하고, requires_grad=True 를 사용해 그 텐서에 대해 미분이 가능하도록 연산들을 모두 추적할 수 있게 할 수 있다. 
  이 경우, 그 텐서를 사용한 식은 전부 grad_fn 이라는 속성을 가지게 된다.

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

- torch.utils.data.DataLoader(트레인 세트, batch_size=i, shuffle=T/F, num_worker=i) > 데이터를 배치 사이즈대로 나눠 로드. 
  마지막은 프로세스를 몇개 사용하냐 라는 의미로, 오류가 난다면 0으로 하면 된다.
- torch.FloatTensor(X_data-ndarray 타입) > float 타입 요소를 가진 텐서로 변환. 다른 타입도 존재.
- 텐서.premute(shape number - 0,3,1,2 식으로) > 텐서의 순서를 변환 (20, 32, 32,3)을 채널수가 사이즈보다 먼저 나오는 텐서에 맞게 (0,3,32,32)로 바꿀 수 있다.

# scikit learn
***
#### dataset Load
- sklearn.datasets.load_데이터셋 이름() > 데이터셋 로드.  데이터셋.keys()로 키들을 볼 수 있고, DESCR 에는 데이터셋 설명이, 
  target_names 에는 클래스가, feature_names 에는 각 특성의 이름이. data 에는 샘플별 데이터가, target 에는 샘플의 종류가 클래스 순서대로 0부터 들어있다. 
  .fit(train_data, train_label)로 train, .predict(test_data)로 예측할 수 있다. n_sample, noise, random_state 등의 매개변수가 있다.
- 회귀용 데이터셋인 boston_housing, diabetes. 다중 분류용 데이터 셋인 digit, 두개의 달처럼 생겨 선형으로는 분류가 어려운 two_moons 등 다양한 데이터 셋이 있다. 
  먼저 간단한 모델(선형, 이웃, NB 등)로 성능을 실험해 보며 데이터를 이해한 뒤 다른 모델을 적용시켜 보는게 좋다.
- sklearn.model_selection.train_test_split(x,y,random_state,train_size) > 데이터를 트레인과 테스트로 나눈다. random_state 는 유사 난수 생성으로 꼭 초기화를 해주는 게 좋다.
- sklearn.model_selection.KFold(k(int), shuffles(bool), random_seed(int)) > k번 K-Fold 를 해주는 머신을 생성한다. .split(data)으로 k개의 train / test 데이터셋을 생성한다. 
  data[train\], data[test\] 식으로 사용한다.

#### method
- 불러온 모델들은 .fit(train, labels)로 fit, predict(data)로 사용한다. .score(test_img, test_label) 로 정확도를 측정할 수 있다.

#### Regression
###### k
- sklearn.neighbors.KNeighborsRegression(n_neighbors=k) > k개의 이웃을 찾는 knn 회귀모델 로드.
###### Linear
- sklearn.linear_model.LinearRegression() > OLS(최소제곱법) 선형회귀 모델 로드. 가중치는 .coef_ 에 ndarray 로, 편향은 .intercept_에 저장되어 있다.
###### Ridge, Lasso
- sklearn.linear_model.Ridge(alpha=i) > 리지 선형회귀 모델 로드. 알파 값은 기본 1이며, 높이면 더 단순히(가중치를 0에 가깝게) 만들어 
  훈련세트의 성능은 나빠져도 일반화는 더 잘되게 만들 수 있다.
- sklearn.linear_model.Lasso(alpha=i, max_iter) > 라소 선형회귀 모델 로드. 리지와 비슷하나 어떤 값은 진짜 0이 될수 있다. 
  np.coef_ != 0 의 합계를 구하면 사용한 특성수를 알 수 있고, 과소 적합을 피하려면 알파를 줄이고 max_iter 를 늘려야한다.
###### decision tree
- sklearn.tree.DecisionTreeRegressor() > 결정트리 회귀 모델 로드. mex_depth, max_leaf_nodes, min_samples_leaf 중 하나만 지정해도 과대적합을 막을 수 있다.

#### Classification
.decision_function(test data) > 데이터를 분류하며 그 데이터가 분류한 클래스에 속한다고 생각하는 정도를 기록해 반환. 양수값은 양성 클래스를 의미한다.
.predict_proba(test data) > 각 클래스에 대한 확률. (샘플 수, 클래스 개수) 의 형태를 갖음. 과대적합된 모델은 틀렸어도 확신이 강한 편이고, 복잡도가 낮으면 예측에 불획실성이 더 많다.
###### k
- sklearn.neighbors.KNeighborsClassifier(n_neighbors=k) > k개의 이웃을 찾는 knn 분류모델 로드.
###### linear
- sklearn.linear_model.LogisticRegression() > 로지스틱 회귀 분류 모델 로드. 이진분류에서 로지스틱 손실 함수를, 다중 분류에서 crossentropy 손실 함수를 사용함. 
  penalty='l1'으로 l1규제를 사용할 수 있다.
- sklearn.svm.LinearSVC() > 선형 서포트 벡터 머신 (분류)모델 로드. 로지스틱과 이것은 규제의 강도를 결정하는 매개변수 C를 가지고 있음. C로 낮은 값을 지정하면 가중치를 0에 가깝게 지정함.
- 선형 회귀의 alpha 와 분류의 C는 각각 클수록/작을수록 모델이 단순해진다는 특징이 있고, 보통 log 스케일(10배씩)로 최적치를 정하며, 중요특성이 믾지 않다고 생각하면 L1규제를, 
  아니면 기본 L2를 사용한다.
###### Naive Bayse
- sklearn.naive_bayes.MultinomiaNB() : 나이브 베이즈 분류기 로드.  alpha, class_prior, fit_prior 등의 매개변수 사용이 가능. 입력으로 DTM이나 TF-IDF 행렬을 입력으로 받음.
###### decision tree
- sklearn.tree.DecisionTreeClassifier(max_depth=i, random_state=0) > 결정트리 분류기 로드. 최대 깊이 i 까지 가지를 뻗게 한다. 모델.feature_importances_ 로 
  각 특성들의 중요도를 볼 수 있다.
- sklearn.tree.export_graphviz(트리모델, out_file='파일명.dot', class_names=["첫번째","두번째"\],feature_names=이름들, impurity=bool, filled=bool) > 트리모델 
  시각화 해 저장. graphviz 모델 의 .Source(파일.read()) 을 디스플레이해 표시할 수 있다.
###### ensemble
- sklearn.ensemble.RandomForestClassifier(n_estimators=n, random_state=0) > random forest 분류 모델 로드. n개의 트리를 생성해 예측한다. 
  각 트리는 .estimators_ 에 저장되어있다.
- sklearn.ensemble.GradientBoostingClassifier(learning_rate=r, n_estimators=n, max_depth=m, random_state=0) > 그래디언팅 부스트 분류 모델 로드. 
  n개의 트리를 생성해 r 의 러닝레이트로 예측한다. 각 트리는 .estimators_ 에 저장되어있다. 기본값은 100개, 0.1, 3의 깊이 다.
- sklearn.ensemble.BaggingClassifier(모델, n_estimators=n, oob_score=bool, n_jobs=-1, random_state=0) > 모델을 n개 연결한 배깅 분류기 생성. 
  oob 를 T 로 생성하면 부트스트래핑에 포함되지 않은 매개변수로 모델을 평가함.
- sklearn.ensemble.ExtraTreesClassifier(n_estimators=n, n_jobs=-1, random_state=0) > 엑스트라 트리 분류 모델 로드.
- sklearn.ensemble.AdaBoostClassifier(n_estimators=n, random_state=0) > 에이다 부스트 분류 모델 로드. 
  기본적으로 깊이 1의 결정 트리 모델을 사용하나 base_estimator 매개 변수로 다른 모델 지정 가능.
###### SVM
- sklearn.svm.SVC(kernel='rbf', C=i, gamma=r) > 서포트 벡터 머신 로드. r을 키우면 하나의 훈련샘플이 미치는 영향이 제한되고, i 는 규제가 커진다. 
  커널 SVM 에서는 특성들의 크기차이가 크면 문제가 생겨, 평균을 빼고 표준편차로 나눠 평균을 0, 분산을 1로 만드는 전처리를 해줘야 한다.
###### DL MLP
- sklearn.neural_network.MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[]) > 다중 퍼셉트론 분류 모델 로드. 
  히든 레이어에 넣은 숫자, 넣은 숫자의 개수대로 히든레이어가 생성된다. alpha 매개변수에 값을 넣어 줄 수 도 있다. svc 와 비슷하게 신경망도 일반화(평균 0, 분산 1)를 해주는게 좋다. 
  sgd 라는 옵션으로 다른 여러 매개변수와 함께 튜닝해 최선의 겨로가를 만들 수 있음.


#### preprocessing
- 불러온 프로세서들은  .fit(train_data) 로 스케일러를 훈련시키고, .transform(data)로 변환한다. .fit_transform(train_data)로 둘을 한번에 할 수있다. 
  트레인 데이터와 테스트 데이터 모두 같은 스케일 조정을 해주어야 하기에 테스트 데이터는 0~1의 범위를 벗어날 수 있다.
###### Scaler
- sklearn.preprocessing.MinMaxScaler() : 특성마다 최솟값과 최댓값을 계산해 데이터의 스케일을 0~1로 조정하는 MinMax 스케일러 로드.
- sklearn.preprocessing.StandardScaler() : 모든 특성을 정규 분포로 바꿔준다.
- sklearn.preprocessing.QuantileTransformer(n_quantile = n) : n개의 분위를 사용해 데이터를 균등하게 분포시키는 스케일러 로드. 
  output_distribution='normal'로 균등 분포가 아니라 정규분포로 출력을 바꿀 수 있음.
- sklearn.preprocessing.PowerTransformer() : 데이터의 특성별로 정규분표에 가깝게 변환해주는 스케일러 로드. method='yeo_johnson'(양수 음수 값 둘 다 지원, 기본값) 과 
  method='box-cox'(양수만 지원.)를 지정해 줄 수 있다.
- sklearn.preprocessing.KBinsDiscretizer(n_bins=n, strategy='uniform') : n개로 구간 분할 모델 로드. .bin_edges_ 에 각 특성별 경곗값이 저장되어 있다. 
  transform 메서드는 각 데이터 포인트를 해당 구간으로 인코딩하는 역할을 한다. 기본적으로 구간에 원 핫 인코딩을 적용한다. transform 결과.toarray()로 원핫 인코딩된 결과를 볼 수 있다.
- sklearn.preprocessing.PolynomialFeatures(degree=i, include_bias=bool) : x**i 까지 고차항(다항식)을 추가해 특성을 풍부하게 나타내는(구간 분할과 비슷한 효과) 모델 로드.
  bool 이 T 면 절편에 해당하는 1인 특성을 추가한다. 다항식 특성을 선형 모델과 같이 사용하면 전형적인 다항 회귀 모델(결과가 곡선)이 된다. 
  고차원 다항식은 데이터가 부족한 영역에서는 너무 민감하게 동작한다는 문제점이 있다.
- sklearn.preprocessing.LabelEncoder() : 여러개의 카테고리가 존재하는 데이터를 고유한 정수로 인코딩하는 클래스 로드..
###### decomposition
- sklearn.decomposition.PCA() : 주성분 분석(PCA) 프로세서 로드. 기본값은 데이터를 회전,이동만 시키고 모든 주성분을 유지하지만 
  n_component 매개변수에 값을 넣어 유지시킬 주성분의 개수를 정할 수 있다. fit 시 모델.components_ 속성에 주성분이 저장된다. whiten=T 로 주성분의 스케일이 같아지게 할 수 있다. 
  .inverse_transform 을 사용해 원본 공간으로 되돌릴 수 있다.
- sklearn.decomposition.NMF(random_state = 0) : NMF 프로세서 로드. n_component 매개변수에 값을 넣어 유지시킬 주성분의 개수를 정할 수 있다.
- sklearn.manifold.TSNE(random_state=n) : 매니폴드학습 알고리즘의 t-SNE 알고리즘 모델 로드. 데이터들을 알아서 나눔. 
  훈련시킨 모델만 변경 가능해 transform 메서드 없이 fit_transform() 메서드만 지원한다.
- sklearn.decomposition.LatentDirichletAllocation(n_components=n, learning_method="batch/online", max_iter=i, random_state=0) : 
  n개의 토픽을 생성하는 LDA 로드. 기본 학습 방법인 online 대신 조금 느리지만 성능이 더 나은 batch 방법을 사용할 수 있고, i를 높이면 모델 성능이 좋아진다(기본값은 10). 
  각 단어의 중요도를 저장한 .components_ 의 크기는 (n, n_words)이다.
###### cluster(agglomerative)
- .fit_predict(data) 로 각 데이터 포인트가 속한 클러스터들을 리스트 형태로 받아 볼 수 있다.
- sklearn.cluster.KMeans(n_cluster=n) : n개의 클러스터 중심점을 생성하는 k-평균 알고리즘 모델 로드. .labels_ 에 각 데이터 포인트가 포함된 클러스터들을 리스트 형태로 볼 수 있고, 
  .predict 로 새로운 데이터의 데이터포인트가 어느 클러스터에 속할 지 예측할 수 있다.
- sklearn.cluster.AgglomerativeClustering(n_cluster=n, linkage="ward/average/complete") : 병합 군집 모델 로드. 모든 클러스터 내의 분산을 가장 적게 증가시키는
  (기본, 대부분 알맞음)/평균 거리가 가장 짦은/최대 거리가 가장 짦은 두 클러스터를 합친다.
- sklearn.cluster.DBSCAN(min_sample=i, eps=j) : DBSCAN 군집 모델 로드. j 거리에 i 개 이상 데이터가 있다면 그 데이터 포인트를 핵심 샘플로 지정한다. 
  어느 군집에도 속하지 않는 포인트를 noise 로 지정해, -1의 레이블을 가지며 이를 이용해 이산치 검출을 할 수 있다.
###### one-hot-encoding
- sklearn.processing.OneHotEncoder(spares=bool) : 원 핫 인코딩 모델 로드. sparse 가 False 면 희소행렬이 아니라 ndarray 로 반환된다. 
  .fit_transform(data)로 사용, .get_feature_names() 로 원본 데이터의 변수 이름을 얻을 수 있다.
- sklearn.compose.ColumnTransformer([("scaling",스케일러,['스케일 조정할 연속형 열 이름들']), ("onehot",원핫인코더,['원핫인코딩할 범주형 열 이름들'])]) : 
  각 열마다 다른 전처리(스케일 조정, 원핫인코딩)을 하게 해주는 ct 로드.
- sklearn.compose.make_column_transformer([(['스케일 조정할 연속형 열 이름들'], 스케일러), (['원핫인코딩할 범주형 열 이름들'], 원핫인코더)]) : 
  클래스 이름을 기반으로 각 단계에 이름을 붙여주는  ct 로드.
###### feature-selection
- sklearn.feature_selection.SelectKBest/SelectPercentile(score_func=f, percentile=i) : 일변량 통계 모델 로드. 고정된 K 개의 특성을/지정된 비율만큼 특성을 선택한다. 
  f 는 분류면 feature_selection.f_classif, 회귀면 feature_selection.f_regression 를 사용하고, i 는 백분율로 입력한다.
- sklearn.feature_selection.SelectFromModel(모델, threshold='median/mean') : 모델을 이용한 모델기반 자동선택 모델 로드. 
  중요도가 임계치(threshold)보다 큰 모든 특성을 선택한다. 중간/평균 이며 '1.3*median' 식으로 비율을 지정할 수도 있다.
- sklearn.feature_selection.RFE(모델, n_feature_to_select = n) : 모델을 이용해 n개 까지 재귀적 특성 제거를 하는 반복적 특성 선택 모델 로드.
###### feature-extraction
- sklearn.feature_extraction.text.CountVectorizer() : 빈도수 기반 벡터화(정수인코딩) 머신 로드. DTM(서로 다른 Bow 결합)제작. extraction(추출).
- sklearn.feature_extraction.text.TfidfTransformer() : TF-IDF 생성기 로드. DTM을 자동으로 TF_IDF로 만들어줌.


#### metrics
- sklearn.metrics.classification_report(테스트 결과, 실제) : 정확도를 다양한 방식이 모여있는 표로 반환.
- sklearn.metrics.***_score(테스트 결과, 실제) : 훈련 결과의 정확도를 다양한 방식으로 반환.
- sklearn.metrics.accuracy_score(테스트 결과, 실제) : 둘의 일치도를 그대로 정확도로 반환.
- sklearn.metrics.f1_score(테스트 결과, 실제) : F1 Measure 를 이용하여 정확도를 반환.

- sklearn.model_selection.cross_val_score(모델, data, labels, cv=i) : 모델을 이용해 i번 폴드하는 교차 검증 사용. 총 i개 모델의 정확도를 배열로 반환한다. 
  보통 반환값.mean()을 이용한 평균값으로 간단하게 정확도를 나타낸다. cv 에 KFold 객체 등을 넣어 사용할 수 도 있다. 그리드 서치와 함께 n_jobs 매개변수로 사용할 
  cpu 수를 지정할 수 있다. 데이터셋과 모델이 너무 클때는 여러 코어를 쓰면 메모리 사용량이 너무 커져 메모리 사용 현황을 모니터링해야 한다.
- sklearn.model_selection.cross_validate(모델, data, labels, cv=i, return_train_score=bool) : 위와 같지만 훈련과 테스트에 걸린 시간까지 담아 딕셔너리로 반환한다. 
  bool 이 T 면 테스트 스코어도 같이 포함해 반환된다. 모델을 그리드 서치로 사용하면 중첩 교차 검증의 구현이 가능하다. scoring="accuracy/rou_auc"등으로 모델, 
  최적 매개변수 선택을 위한 평가 방식을 바꿀 수 있다.
- sklearn.model_selection.GridSearchCV(모델, 파라미터 딕셔너리({'변수명':[넣어볼 수들의 리스트\]},딕셔너리의 리스트로 넣으면 조건부로 그리드 탐색), cv=i, return_train_score=bool): 
  교차검증을 사용한 그리드 서치 매개변수 조정 방법 로드. .fit(data, label)로 설정된 매개변수 조합에 대해 교차검증을 수행하고, .cv_results_로 결과 확인 가능.
  가장 성능이 좋은 매개변수로 데이터에 대한 새 모델을 자동으로 만들며(refit=True), 모델엔 .score(test_data, test_label)과 .predict(data)로 접근 가능.
  [.best_params_ : 최고의 매개변수 | .best_score_ : 이 설정으로 얻은 정확도 | .best_estimator_ : 최고의 모델 | .cv_result_ : 각 결과]속성 사용 가능.

- sklearn.model_selection.LeaveOneOut() :  LOOCV 로드. 위의 cv 매개변수로 넣어 사용할 수 있다. 큰 데이터 셋에선 시간이 오래 걸리지만, 작은 데이터 셋에선 종종 더 좋은 효과를 낸다.
- sklearn.model_selection.ShuffleSplit(train_size=i/f, test_siz=i/f, n_split=n) :  i개의 데이터 포인트로/f의 데이터 포인트 비율로 n번 반복분할 하는 임의 분할 교차 검증 로드. 
  cv 매개변수에 넣어 사용할 수 있다.
- sklearn.model_selection.RepeatedStratifiedKFold/RepeatedKFold(random_state = i) : 반복 교차 검증 로드. 
  분류/회귀 이며 model_selection.StratifiedKFold/KFold를 기본으로 사용하기 때문에 import 를 해주어야 한다. cv에 매개변수.

- sklearn.metrics.confusion_matrix(label, pred_logreg(예측결과)) : 오차 행렬 표시. [(음성데이터)[음성으로 예측한 수, 양성으로 예측한 수\] 
- (양성데이터)[음성으로 예측한 수, 양성으로 예측한 수\]\] 식으로 반환된다.
- sklearn.metrics.precision_recall_curve(test_label, 모델.decision_function(test_data)(확신에 대한 측정값)) : 정밀도-재현율 곡선 로드. 
  precision, recall, threshold 총 세가지를 반환한다.가능한 모든 임계값에 대해 정밀도와 재현율의 값을 정렬된 리스트로 반환.
- sklearn.metrics.average_precision_score(test_label, 모델.predict_proba(test data)(확신에 대한 측정값)) : 위 곡선의 아래 면적인 평균 정밀도를 계산해 반환.
- sklearn.metrics.roc_curve(test_label, 확신에 대한 측정값) > ROC 곡선 로드. FPR, TPR, threshold 총 셋을 반환한다.
- sklearn.metrics.roc_auc_score(test_label, 확신에 대한 측정값) > ROC 곡선의 아래 면적인 AUC 를 계산해 반환한다.

#### Pipeline
- sklearn.pipeline.PipeLine([("임의의 단계 1 이름", 추정기 객체), ("단계 2 이름", 추정기 객체)\]) : 여러 처리 단계를 추정기 형태로 묶어주는 pipeline 객체 로드. 
  각 단계를 이름과 추정기(스케일러,모델 등)으로 이뤄진 튜플 형태로 묶어 리스트로 전달한다. .fit(), .score()등이 전부 사용 가능 하다. 
  어떤 추정기와도 연결할 수 있으며, 마지막 모델을 제외하고는 전부 transform 메서드를 가지고 있어야 한다.
- sklearn.pipeline.make_pipeLine(추정기1, 추정기2) : 파이프 라인과 똑같지만 단계의 이름을 자동으로 만들어 준다. .steps 속성에 각 단계의 이름이 들어있다. 
  단계 이름을 키로 가진 .named_steps 딕셔너리도 파이프 라인의 각 단계 속성애 쉽게 접근 할 수 있다. 그리드 서치에 파이프 라인을 사용할 때는 모델에 넣는다.

#### text
- sklearn.feature_extraction.text.CountVectorizer() : BOW 표현을 하게 해주는 변환기 로드. .fit(문자열이 담긴 리스트)로 사용, 
  .vocabulary_ 속성에서 반환된 {단어:등장횟수} 형태의 딕셔너리를 볼 수 있음.  tf-idf 와 함께 ngram_range=(연속 토큰 최소길이, 최대길이) 로 연속된 토큰을 고려할 수 있다. 
  보통은 하나만 하지만 많을 때 바이그램정도로 추가하면 도움이 된다.
- Bow 표현을 만드려면 .transform(list), Scipy 희소 행렬로 저장되어 있으며, .get_feature_names()로 각 특성에 해당하는 단어들을 볼 수 있음. 
  min_df 매개변수로 토큰이 나타날 최소 문서 개수를 지정할 수 있고, max_df 매개변수로 자주 나타나는 단어를 제거할 수 있다. 
  stop_words 매개변수에 "english" 를 넣으면 내장된 불용어를 사용한다.
- sklearn.feature_extraction.text.TfidVectorizer(min_df=i) : 텍스트 데이터를 입력받아 BOW 특성 추출과 tf-idf 를 실행하고 L2정규화(스케일 조정)까지 적용하는 모델로드. 
  훈련데이터의 통걔적 속성을 사용하므로 파이프 라인을 이용한 그리드 서치를 해 주어야 한다. .idf_ 에서 훈련세트의 idf 값을 볼 수 있다. 
  idf 값이 낮으면 자주 나타나 덜 중요하다 생각되는 것이다.

# tensorboard
- 텐서보드 : 머신러닝 실험에 필요한 시각화 및 도구를 제공. 실시간으로 학습과정을 그래프로 확인가능하며, 기존에 학습했던 것과 동시 비교 분석이 가능.
- 제공기능 : 측정항목(손실/정확도등)추적/시각화, 모델그래프(레이어)시각화, 가중치/편향/기타텐서의 경과에 따른 히스토그램,
   저차원공간에 임베딩 투영, 이미지/테스트/오디오 데이터 표시, 텐서플로우 프로그램 프로파일링, 그 외 다양한 도구 제공.
- 사용 : 텐서플로우의 함수들(tf.summary의 scalar/marge_all/FileWriter등)을 이용해 파일을 생성한 후 
  cmd에 [tensorboard --logdir=./logs/ 혹은 python -m tensorboard.main]를 입력해 사용할 수 있음.
##### 텐서플로우 함수들 
###### 저장할 것 설정
- 추후 텐서보드에서의 분석을 위해 데이터(summary)를 작성.
- tf.summary.scalar(name, scalar) : 스칼라 summary를 작성. 텐서보드 내의 SCALARS 메뉴(대쉬보드)에 넣음.
- tf.summary.image(name, image) : 이미지 summary를 작성. 텐서보드 내의 IMAGES 메뉴에 넣음.
- tf.summary.histogram(name, histogram) : 히스토그램 summary를 작성. 텐서보드 내의 HISTOGRAMS 메뉴에 넣음.
###### 기록할 장소 설정
- tf.summary.merge_all() : 앞서 지정한 모든 summary를 통합(marge).
- tf.summary.merge(summaries) : 원하는 summaries를 통합. 
- tf.summary.FileWriter(log_dir, graph) : 텐서보드를 위해 생성된 파일들(marge>sun>add)을 저장. tf.session()을 돌리고 sess.graph로 그래프를 넣으면 됨. 
  넣은 순간 텐서보드에서 그려짐.  
###### 기록
- summary = sess.run(merge) : 원하는 스텝마다 merge를 실행해 summary값을 구함. 스텝은 batch_num(epoch마다 나옴)보단 global step(반복x 0부터)으로 넣어주면 좋음.
- writer.add_summary(summary, global_step) : 나온 summary를 FileWriter에 추가함. 넣을때마다 새로운 event가 저장됨. tf.train.global_step()으로 글로벌 스텝 획득가능.

# wandb
- wandb(Weights & Biases, WandB) : ML을 위한 개발 툴. Tensorboard와 비슷한 기능(Dashboard/Sweeps/Artifacts)을 하나 tf, pytorch등 여러 모듈에서 사용가능함. 기본적으로는 웹사이트에서 그래프를 보여줌.

- wandb.login() : wandb login. 먼저 wandb홈페이지에서 회원가입을 한 후 사용가능. 계정이 없다면 계정을 생성, 그 후 로그인을 진행.
- wandb.init() : wandb 초기설정. project에 project명을, entity에 계정명을, name에 저장하는데 사용하는 이름을 인자로 줄 수 있음. 이 외에도 config등의 인자 사용가능.
- wandb.run.name : 매 실행 이름. 여기에 값을 할당해 실행 이름을 지정할 수 있고, 생략시 임의로 지정됨.  wandb.run.id를 넣어 생성된 runID임을 명시할 수 도 있음.

- wandb.config.updata(args) : wandb에서 갱신할 변수들을 설정. parser.parse_args()(argparse)를 넣거나 {변수명: 값}형태의 딕셔너리를 넣어 파라미터 일부/전체를 업데이트 가능. 
- wandb.config.변수명 : 설정을 wandb에 넣어둠. 변수명은 epochs/batch_size등이며, init에서 config매개변수에 딕셔너리 형태로도 넣을 수 있음. 

- wandb.log(dict) : 이미지, accuracy, test_loss등의 로그를 기록. {저장될 이름: 값}형태이며, 이미지(plt)/히스토그램(wandb.Histogram(numpy_array_or_sequence))등이 전부 가능함. 
- wandb.watch(model) : 모델의 학습을 따라가며 진행과정을 보여줌. 


# os | os(파일, 디렉토리)관련 명령
- os.getcwd() : 현재 작업 폴더 반환.
- os.chdir(경로) : 디렉토리 변경.
- os.makedirs(dir명) : 디렉토리 생성.  
- os.path.abspath(상대 경로) : 절대 경로 반환. 
- os.path.dirname(경로) : 디렉토리명만 반환.
- os.path.basename(경로) : 파일 이름만 반환.
- os.listdir(경로) : 경로 안의 파일 이름을 전부 반환.
- os.path.join(상위, 하위) : 경로를 병합해 새 경로 생성. ('C:\Tmp', 'a', 'b')식으로 넣는다.
- os.path.isdir(경로) : 폴더의 존재 여부를 반환.
- os.path.isfile(경로) : 파일의 존재 여부를 반환.
- os.path.exists(경로) : 파일 혹은 디렉토리의 존재 여부를 반환.
- os.path.getsize(경로) : 파일의 크기 반환.

# selenium
- 웹 크롤링, 웹상 자동화(메일등)에 이용되는 패키지. 사용자가 웹사이트를 이용하는 방법과 동일하게 동작.
- import selenium.webdriver 으로 import해야 webdriver사용가능.
  
- selenium.webdriver.웹브라우저() : 해당 웹브라우저를 이용한 웹드라이버 객체 생성. .close()로 닫을 수 있음.
- 웹드라이버 메서드 : .get(url){url로 진입(창을 띄움)}, .find_element(s)_by_name/tag(찾을 수단){값(들)을 찾음, 객체(들 시퀀스)로 반환}등 사용가능.
- selenium.webdriver.common.keys.Keys : 다양한 키들을 정의. .RETURN{엔터}등이 정의되어있음.
- 타겟속성.send_keys(값) : 값(키보드 입력값)을 해당 부분으로 전송. 클릭은 .click(), 새로고침은 .refresh()로 사용. 클릭이나 엔터 사용시 로딩 시간을 
  기다리려 time.sleep()메서드를 사용.
- 타겟속성.get_attribute(속성) : 해당 부분에서 속성 부분을 가져옴. 여기서 url을 얻었다면 urllib.request.urlretrieve를 이용해 다운로드를 할 수 있음.

- xpath : W3C의 표준. XML문서 구조를 통해 항목을 배치/처리하는 방법을 기술한 언어. XML보다 쉽고, 약어로 되어 있으며 문서 노드정의를 위해 경로식을 사용.
> driver.find_element(s)_by_xpath()로 xpath를 이용해 노드를 찾음.
> 노드선택 : /(루트노드로부터 선택), //(현 노드로부터 문서상의 모든 노드 조회), .(현 노드), ..(부모노드), @(현 노드의 모든 속성), 노드명(노드이름이 노드명인 노드 선택)
> 술부 : []형태로 기술, 특정 값/조건에 해당되는지 반별. *(매칭되는 모든 ElementNode), @\*(매칭되는 모든 속성노드), Node()(현 노드로부터 문서상 모든 노드 조회)등이 주로 사용. 
> 여러 경로 선택시 | 를 이용해 나눔.
> 예시 : //div[@class='language-text highlighter-rouge'\]/pre[@class='highlight'\]/code (현 노드에서 특정 클래스의 div속 특정 클래스의 pre속 code 선택).

- 웹드라이버.execute_script(자바스크립트) : 자바스크립트문 실행.
```python 
# 사용 예 (스크롤 전부 내리기)
last_height = driver.execute_script("return document.body.scrollHeight)")  # 브라우저 높이 구하기
while True:
  driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")  # 스크롤 내리기
  time.sleep(Load_Time)
  new_height = driver.execute_script("return document.body.scrollHeight)")
  if new_height == last_height:
    # 스크롤을 내리면 로딩버튼이 나오는 사이트라면, 그 버튼을 선택 > 클릭하게 하고 더이상 나오지않아 오류가 나게 되면 종료시킬 수 있음. 
    break
  last_height = new_height
```
- 가끔 사이트의 봇을 막는 기능때문에(urllib.error.HTTPError: HTTP Error 403: Forbidden)안된다면 특정 코드를 사용해 브라우저로 위장해야 함.
```python 
# 브라우저인 것처럼 속이는 헤더 추가.
opener=urllib.request.build_opener()            # 오프너 생성.
opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64)\  # 자기 정보(브라우저 헤더에 넣을)입력
 AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
urllib.request.install_opener(opener)           # 오프너 오픈
urllib.request.urlretrieve(imgUrl, "test.jpg")  # 이미지 다운로드
```

# urllib(3)/zipfile | url,zip
- urllib : url 이용 라이브러리. urllib3 은 따로 install, import가 필요함. 

- urllib3.PoolManager() : url poolManager 로드. url이용에 사용가능.
- http(Pool).request('GET', url, preload_content=False) : url 오픈. with 등을 이용해 파일객체로 열 수 있고, 다운로드를 위해 shutil이 필요.
- shutil(shutil모듛).copyfileobj(request객체, 복사받을_파일객체(wb)) : 파일객체를 복사해옴.

- urllib.request.urlretrieve(주소, filename) : 주소의 파일을 파일 이름으로 다운로드.
- urllib.request.build_opener() : 오프너 생성. urllib.request의 함수들은 import urllib.request로 import해야 사용가능.
- 오프너.addheaders=[넣을 헤더\] : 오프너에 브라우저 헤더추가.
- urllib.request.install_opener(opener) : 오프너로 오픈. 이 이후 평범하게 코드 사용.

- zipfile.ZipFile(zipfilepath, 'r') : zip파일객체 오픈. .extractall(path)로 압축을 헤재할 수 있음.

# requests | http요청
- requests : http요청을 보내는 라이브러리. 
- requests.get/post/delete/head/options(url, timeout=n) : 요청. 상황에 맞게 헤더/파일등을 포함해 요청가능. delete엔 data인자를 꼭 넣어주어야 함. 
  요청성공시 응답 상태와 데이터가 전송되어 옴. 
- 인자 구조 :  headers = {'Authorization': 'Mozilla/5.0' } | data = {'key1':val1 'key2':val2'} | files = 
  \[('image', (image.png, open(image.png, 'rb'), 'image/png', {'Expires': '0'}))]의 구조를 가지고 있음. 

# Scrapy
- scrapy : 파이썬 웹 크롤링(스크래핑) 패키지.

# timeit
- timeit.timeit(함수) : 함수의 시작부터 끝까지 걸린 시간을 측정.


# pygame
- pygame : python을 통해 게임을 만들 수 있도록 지원해주는 모듈. [import->init(초기화)->전역변수선언->이벤트/화면/사용자 행위 설정(반복문)]의 구조로 이뤄짐.
- pygame.init() : 모듈 초기화. pygame모듈 사용시 필수로 해줘야 함. 
- pygame.display.set_caption() : 창이 켜질때 창의 이름을 설정.
- pygame.display.set_mode([x, y\]) : pygame으로 생성할 GUI창의 크기를 설정 후 화면을 설정하기 위한 객체 생성. .fill((r,g,b))으로 배경색을 설정하는 등의 조작이 가능.
- pygame.time.Clock() : 화면을 초당 몇번 출력하는지(FPS)설정하기위한 Clock객체 생성. .tick(i)로 FPS설정 가능.
- pygame.event.get() : 게임중간에 발생한 이벤트를 캐치. [event.type == pygame.QUIT]으로 창에서 x버튼이 눌렸는지 등의 이벤트를 검사할 수 있음.
- pygame.display.filp() : draw함수나 screen(set_mode())로 화면에 작성한 모든것을 업데이트. 

- pygame.draw.rect/polygon/circle/eclipse/arc/line/lines/aaline/aalines() : 도형/선을 그림. 여러 매개변수를 주어 도형의 색/크기/위치/그 외 기타등등을 설정가능.
