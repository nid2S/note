# sub
***
- 이미지를 저장하기 위한 센서의 모든 픽셀에는 하나의 센서만을 저장함.
- 인간의 눈이 녹색광에 더 민감하여 청,적색 픽셀에 비해 두배의 녹색 픽셀이 있고, 이 패턴이 베이어 패턴이다.
- 누락된 두 색은 인접 픽셀의 색으로 보완하는 demosaicing 방법을 사용한다.
- 2x2만 보는 pixel doubling interpolation 과 3x3까지 보는 bilinear interpolation 이 있다.
- 이런 과정을 거치며 RGB 이미지로 변환되며, RGB 픽셀 배열은 일반적으로 디스크 저장 전에 JPG/PNG 형식으로 압축된다.
- 대부분의 이미지 형식은 image header 와 data 부분으로 나뉜다.
- numpy 에 저장되는 RGB 데이터 구조는 당시 유행했던 BGR 방식을 채용했음.
- lasagna 라는 theano 기반 라이브러리도 있다.


# numpy
***
- numpy => 수치연산에 최적화. 배열,행렬,배열에서 작동하는 다양한 수학함수를 지원, 배열의 모양은 각 차원을 따라 크기를 제공하는 정수형 튜플.
  
- np.array(리스트) > 리스트에 해당하는 배열 생성
- np.array([[1,1,1],[2,2,2]]) > 2행 3열짜리 2차원 np배열 생성. shape()로는 (2,3)이 출력되며, x[1,2] 식으로 두번째 열의 세번째 요소를 뽑아낼 수 있다. 

- np.array() : 리스트, 튜플, 배열로 부터 ndarray 생성
- np.asarray() : 기존의 array 로  부터 ndarray 생성

- np.eye(i) > 대각선이 1이고 나머지는 0인 i*i의 2차원 배열 생성
- np.arange(start, end, step) > 리스트의 슬라이스와 같이 범위대로 배열을 제작.
- np.linespace(start, end, step) > 시작부터 끝까지 간격(개수)만큼 나눠진 배열 생성
- np.sin(x) > 사인 함수를 이용해 배열 x와 대응하는 배열 생성
- np.where(조건) > 조건문(리스트<1 식으로 내부에 리스트 포함)에 사용. 조건에 밎는 인덱스들을 ndarray 형태로 반환. 슬라이싱에 사용 가능. (조건문, 맞으면, 아니면) 식으로 구성해 처리를 할 수도 있음.
- np.random.permutation(i) > i 까지 랜덤으로 섞인 배열 반환.
- np.percentile(배열, [분위]) > 배열에서 분위에 해당하는 샘플을 추출해 반환. [0,25,50,75,100]식으로 지정하면 된다.
- np.unique(배열) > 배열에 있는 값의 종류를 배열로 반환.  

- np.float32(이미지) > cv2로 읽어온 이미지의 데이터 타입을 변환. 이런식으로 부동소수점 데이터 유형으로 변환시 작업 중 오버플로우를 방지가능.
- np 객체.dtype > 데이터 타입 문자열로 반환
- np 객체.shape > 배열의 마지막 좌표(행,열) 출력([5,10,15]의 경우 (3,)식으로)
  
- np.argmax(배열) > 배열중 최대치의 인덱스 반환
- np.mean(x == y) > x와 y 배열의 동일도를 출력.
- np.expand_dims(np 배열,index) > np 배열의 index 위치에 데이터를 추가해 차원을 늘림. 한개의 이미지만 사용할때도 2차원으로 만들어 주어야 함.

- ndarray.flags > 어레이의 메모리 레이아웃에 대한 정보.
- ndarray.shape > 배열 차원의 튜플.
- ndarray.ndim > 배열의 차원 수.
- ndarray.size > 배열의 요소 수.
- ndarray.itemsize > 한 배열 요소의 길이 (바이트).
- ndarray.dtype > 배열의 데이터 타입.
- ndarray.data > 배열 데이터의 시작을 가리키는 파이썬 버퍼 객체.
- ndarray.nbytes > 배열의 요소가 사용한 총 바이트.
- ndarray.T > 2차원 배열의 경우 행과 열의 크기 변환.  
- ndarray.reshape((shape)) > 같은 크기의 다른 형태로 차원 변형.

- np.save(이름,배열) : 1개의 배열을 NumPy format 의 바이너리 파일로 저장.
- np.savez(경로,배열(x=x, y=y 식으로 이름 부여 가능)) : 여러개의 배열을 1개의 압축되지 않은 *.npz 포맷 파일로 저장. 이때 불러오면 numpy.lib.npyio.NpzFile 이며, 개별 배열은 인덱싱해서( ['x'] ) 사용할 수 있다.
- np.savez_compressed(이름,배열) : 여러개의 배열을 1개의 압축된 *.npz 포맷 파일로 저장. 이때도 똑같이 인덱싱 가능.
- np.load(경로) : 저장된 ndarray 파일을 load. close()를 해주어야 하며, 닫은 후에는 불러온 파일을 사용할 수 없다.

- np.savetext() : 여러개의 배열을 텍스트 파일로 저장. header="", footer="" 로 파일 시작과 끝에 #으로 시작하는 주석을 달아 줄 수 있고, fmt="%.1f" 식으로 들어가는 인수들에 대한 포맷을 지정할 수 있다.
- np.loadtext() : 텍스트 파일을 배열로 불러옴. ndarray로 불려옴. 


#pandas
***
- 판다스는 시리즈, 데이터프레임, 패널 총 세개의 데이터 구조를 사용함.
  
- pd.Series(1차원 리스트 , index(인덱스가 될 리스트)) : 시리즈(1차원 배열의 각 값에 대응하는 인덱스를 부여할 수 있는 구조) 생성. 인덱스는 정수뿐 아닌 문자열등도 가능함.
- pd.DataFrame(2차원 리스트, index(행이름), columns(열이름)) : 데이터프레임(행과 열이 존재)생성. .index(인덱스+타입, Index 객체), .columns(칼럼+타임, Index 객체), .values(값만 리스트로) 사용가능.
- index 나 columns 를 안쓰면 0부터 자동할당되고, 리스트,시리즈,딕셔너리(키가 열이름),ndarray 등으로도 생성할 수 있음.

- pd.read_csv("/경로/파일명.csv") > 파일읽기. 자신과 같은 디렉토리에 있으면 그냥 써도, 그냥 웹상의 주소를 써도 됨. 문자열의 형태로 읽힌다.
- pd.get_dummies(범주형 데이터) > one hot encoding. 범주형 종속변수가 그 종류만큼 (이름)_(데이터이름)의 형태로 나뉜다.

- 데이터.head() > 위쪽 데이터 5개. 
- 데이터.tail() > 끝쪽 데이터 5개. 안에 숫자를 넣으면 그 숫자큼 나옴.
- 데이터.info() > 데이터셋의 정보 볼 수 있음
- 데이터.groupby([열]) > group by
- 데이터.columns  > 칼럼이름. 리스트로 반환된 후 dtype 도 출력한다.

- 데이터["열 이름"] > 그 열과 이름, 데이터 타입 출력.
- 데이터[["열이름"]] > 열 제목과 그 열 출력. 
- 데이터["열 이름1","2","3"] > 열(칼럼) 다중 선택.
- 데이터[:, "k1":"k3"] > k1 부터 k3까지 열(칼럼) 선택
- 데이터.loc[["가져올 칼럼 명 들"]] > 특정 인덱스의 데이터만 가져옴. (데이터[칼럼명] == 1) 식으로 조건식을 넣을 수도 있다.
- 데이터[dataFrame.Age > 30] > 이런식으로 데이터를 선택해서 표시할 수 있다.
  
- 데이터["칼럼명"].astype("int/float") > 데이터 수치형으로 변경
- 데이터["칼럼명"].astype("category") > 데이터 범주형으로 변경. 원핫인코딩이 가능해짐.
- 데이터["칼럼명"].value_counts() > 그 칼럼에 등장하는 값의 종류를 그 값이 나온 수와 함께 나타냄. 
 
- 데이터.isna()(.sum()) 으로 na의 위치(혹은 개수, 결측치)를 확인 할 수 있다. NaN(숫자아닌 자료)를 모델에 그대로 넣으면 오류가 남.
- 데이터.mean() > 으로 데이터들의 평균값을 얻을 수 있다.
- 데이터["칼럼"].fillna(평균값) > na에 평균값을 넣어 오류를 없앨 수 있다.
  
- 차트 = 데이터.plot(kind='bar', title='날씨', figsize=(12, 4), legend=True, fontsize=12) > 차트 종류,제목,크기,범례 유무,폰트 크기 설정
- 차트.set_xlabel('도시', fontsize=12)          # x축 정보 표시
- 차트.set_ylabel('기온/습도', fontsize=12)     # y축 정보 표시
- 차트.legend(['기온', '습도'], fontsize=12)    # 범례 지정


# matplotlib.pyplot
***
- plt.plot(정수형 리스트) : 리스트대로 선 또는 마커그래프 생성. 리스트를 한개 넣으면 y값 으로 인식하고 x를 자동생성하고, 두개면 순서대로 x,y 라고 인식한다. 
- plt.plot(정수형 리스트, 'ro') : ro - 빨간색 원형 마커. 이런식으로 색과 그래프 마커를 지정해 줄 수 있음.
- plt.plot(x,y,type,x,y,type) : 이런 식으로 매개변수를 넣거나 plot 을 여러번 사용하면 여러개의 그래프를 그릴 수 있다. 
- plot - color : r(red),g(green),b(blue),c(cyan),m(magenta),y(yellow),k(black),w(white) , color='css_color_name/#rgb'으로 다양한 색상 지정 가능.
- plot - LineStyle : -(solid),--(dashed),-.(dashed-dot),:(dotted).
- plot - Markers : o(circle),s(square),*(star),p(pentagon),+(plus),x(X),D(diamond),|/_(h/v line),^/v/</>(triangle),1/2/3/4(tri)

- plt.title(title) : 그래프 제목 설정. loc-타이틀 위치('right','left'), pad-타이틀&그래프간격, 폰트 크기와 두께 설정 가능.
- plt.xlabel(text) : x축에 레이블(축제목) 설정.
- plt.ylabel(text) : y축에 레이블(축제목) 설정.
- plt.axis([x min, x max, y min, y max]) : 축의 범위 지정.

- plt.fill_between(x, y, alpha) : 그래프에서 그 범위를 채움. (x,y1,y2)식으로 두 그래프 사이의 영역을 채울 수 도 있음. color 매개 변수로 색 지정 가능.
- plt.fill(x,y,alpha) : x,y 점들로 정의되는 다각형의 영역을 자유롭게 채울 수 있음.

- plt.grid(bool) : 그래프에 격자 표시 여부 결정. axis='y/x' 로 가로/세로 방향의 그래프만 그릴 수 있음. color,alpha,linestyle 등의 매개변수 사용가능.
- plt.legend() : 그래프에 레이블(범례) 표시, plot 에서 label="" 로 준 레이블이 그 그래프의 레이블이 된다.
- plt.xticks(number 리스트) : x축에 눈금 표시. label 매개변수에 리스트를 넣어 각 눈금의 이름을 지정해 줄 수 있음.
- plt.yticks(number 리스트) : y축에 눈금 표시. 
- plt.tick_params() : 눈금 스타일 설정. axis-적용축('x','y','both'), direction-눈금위치('in','out','inout'), pad-눈금&레이블 거리, length/width/color-눈금 길이/너비/색, labelsize/labelcolor-레이블 크기/색, t,b,l,r - bool&눈금표시 위치.

- plt.axhline(y, x_min(0~1), x_max(0~1)) : y에 min 부터 max 까지 수평선을 그음. color, linestyle, linewidth 등 매개변수 사용가능.
- plt.axvline(x, y_min, y_max) : x에 min 부터 max 까지 수직선을 그음. 왼쪽 아래부터 오른쪽 끝까지 0~1로 표현.
- plt.hlines(y, x_min, x_max) : y에 min 부터 max 까지 수평선을 그음. min,max 가 0~1로 표현되지 않음.
- plt.vlines(x, y_min, y_max) : x에 min 부터 max 까지 수직선을 그음.

- plt.bar(x, y) : 막대그래프를 그림. width(너비),align(눈금위치. 히스토그램처럼 눈금을 막대 끝으로 이동가능,'edge'),color,edgecolor,linewidth(테두리두께),tick_label,log(bool, y를 로그스케일로) 등의 매개변수 사용 가능.
- plt.barh(x, y) : 수평 막대그래프를 그림. height 를 제외하면 매개변수 동일. width/height 를 음수로 지정하면 막대 위쪽에 눈금 표시.
- plt.scatter(x, y) : 산점도(상관관계표현)를 그림. s(마커 면적),c(마커 색),alpha(투명도) 등의 매개변수 사용가능. 
- plt.hist(리스트) : 히스토그램(도수분포표 그래프)을 그림. 리스트에 나온 계급과 그 빈도를 분석해 자동으로 히스토그램으로 만들어줌. bins(쪼갤영역수),density(bool,막대사이를 이어 하나로), histtype(막대 내부를 채울지,'step') 등 매개변수 사용가능.
- plt.errorbar(x, y, yerr) : 에러바(데이터편차표현)를 그림. yerr 는 각 y의 편차로 위아래 대칭인 오차로 표시, [(error), (error)]식으로 넣으면 아래방향/위방향 편차를 나타내게 됨. uplims/lolims(bool, 상한/하한 기호표시) 매개변수 사용가능.
- plt.pie(ratio(각 영역 비율 리스트), label(각 영역 이름 리스트)) : 파이차트(범주별 구성비율 원형표시)를 그림. autopct(영역안에 표시될 숫자 형식 지정), startangle(시작각도), counterclock(bool,반시계 여부), explode(0~1실수 리스트, 차트중심에서 이탈도), shadow(bool,그림자), colors(리스트,색이름/코드), wedgeprops({'width': 0.7, 'edgecolor': 'w', 'linewidth': 5}, 반지름 비율, 테두리색, 테두리너비) 매개변수 사용가능.

- plt.show() : 생성한 plot(그래프)를 보여줌.


# plotnine
***
- (plotnine.ggplot(petitions)  : 데이터로 그래프 제작
-  plotnine.aes('category')  : 데이터 축 설정. x='' , y='' 식으로 레이블을 지정하지 않고 하나만 지정하면 x로 들어감.
-  plotnine.geom_bar(fill='green')) : 데이터 종류 설정. geom_point() 식으로 하면 산점도 타입이다.
- plotnine.ggplot(data=데이터, mapping= plotnine.aes(x=, y=, color=) + plotnine.geom_point(alpha=f)) : 식으로도 가능하다.   


# wordcloud
***
> wordcloud = wordcloud.WordCloud(  : 워드 클라우드 제작. plt imshow 로 출력.
>                        font_path = fontpath, 
>                        stopwords = [문자열 리스트], 
>                        background_color = 색('white'), 
>                         width = width, height = height).generate(data)


# Scipy
***
- scipy.sparse.csr_matrix(eye) > 주어진 배열 중 0이 아닌 요소만 위치와 값을 저장(희소행렬).
- scipy.sparse.coo_matrix((ones, (arange, arange)) > 주어진 배열 중 0이 아닌 요소만 위치와 값을 저장(희소행렬).


# mglearn
***
- mglearn.discrete_scatter(X[:, 0, X[:, 1], y) > 산점도를 그림.

- mglearn.dataset.make/load_데이터이름() > 데이터셋 로드.
- mglearn.dataset.make_forge() > 인위적인 이진분류 대이터셋 로드. x,y에 각각 특성이 들어간다.

- mglearn.plots.plot_모델이름() > 그 모델의 그래프를 그리는 듯.
- mglearn.plots.plot_2d_classification(fit 된 모델, X, fill=bool, alpha=) > 선형 이진 분류 그래프를 결정경계와 함께 그린다.
- mglearn.plots.plot_knn_classification(n_neighbors = k) > knn 분류를 그래프로 그림.
- mglearn.plots.plot_knn_regression(n_neighbors = k) > knn 회귀를 그래프로 그림.
- mglearn.plots.plot_ridge_n_samples() > 리지 회귀를 그래프로 그림.


# dlib
***
- viola & Jones 알고리즘 > Face Detection(얼굴에 Bounding Box) 가능하게 함. Face Landmark Detection 이 더 자세한 개념.
- Head Pose Detection(얼굴 방향)/Face Morphing(두 얼굴의 중간 얼굴을 생성)/Face Averaging(얼굴들의 평균 얼굴을 생성)/Face Swap/Blink&Drowsy Driver Detection(눈의 깜빡임 감지)/Face Filter 등이 Face Landmark Detection 을 통해 가능해짐.
- dlib.get_frontal_face_detector() > face Detection 을 기능하게 하는 face detector 생성.
- dlib.shape_predictor(데이터셋(얼굴 랜드마크)이 있는 경로) > Landmark detector 생성.
- faceDetector(이미지,업스케일 횟수(0,1,2)) > 이미지에서 얼굴을 찾음.


# OpenCv (cv2)
***
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
- cv2.resize(이미지, 절대크기(상대면 0,0), x 스케일(절대면 패스), y 스케일(절대면 패스), interpolation=method(cv2.INTER_LINEAR,보간법)) > 이미지 resize. 이미지 crop 의 경우는 이미지 슬라이스로 이미지의 일부만을 잘라낸 후 사용한다.

- cv2.line(이미지,시작좌표(x,y),끝좌표(x,y),색(R,G,B),thickness(꽉 채울려면 -1),lineType(기본적으로 윤곽선,CV_LINE_AA=안티엘리어싱)) > 선을 그린다.
- cv2.polylines(img, pts, isClosed, color, thickness=None, lineType=None, shift=None) > 점들을 이어 선을 그림
- cv2.circle(이미지,중심좌표(x,y),radius,선 색,두께/채우기 타입,선 타입) > 원을 그린다.
- cv2.ellipse(이미지,중심,(x길이,y길이),기울기,원 시작 각도, 원 끝 각도(시작부터 끝의 각도까지만 그려짐),색,두께,타입) > 타원을 그린다.
- cv2.rectangle(이미지,좌측상단 좌표,우측하단 좌표,색,두께,타입) > 사각형을 그린다.
- cv2.putText(이미지,문자열,표시될 좌표,폰트,글자 크기,색,두께,타입) > 글자를 입력한다.
- cv2.flip(이미지, 방향) > 이미지 반전. 1 = 좌우, 0 = 상하
- cv2.lotation(이미지, cv2.ROTATE_각도_방향) > 이미지 회전. cv2.ROTATE_90_CLOCKWISE = 시계방향 90도. 반시계 방향은 COUNTERCLOCKWISE, 180도는 방향을 쓰지 않는다.

- cv2.VideoCapture(비디오 파일 경로(카메라=0 -여러개일 경우 1씩 추가-)) > VideoCapture 객체 생성.
- VideoCapture 객체.isopened() > 읽을 영상이 남아 있는지 반환
- VideoCapture 객체.read() > 영상의 존재 여부와 이미지를 반환.
- VideoCapture 객체.release() > 동영상 종료
- VideoCapture 객체.get(cv2.CAP_PROP_FRAME_WIDTH/HEIGHT) > 프레임 너비와 높이 휙득. 실수형이기에 정수형으로 변환 필요.

- cv2.videoWriter('경로/이름',cv2.VideoWriter_fourcc('M','P','4','V'-MP4/'M','J','P','G'-avi),FPS(33이하),(프레임너비,높이)) > videoWriter 객체 생성.
- videoWriter 객체.write(read 로 얻은 이미지) > 저장


#tensorflow
***
- tf.keras.layers.Input(shape=[열 개수(독립변수 개수)]) > 독립변수의 개수만큼 입력레이어 구성.
- tf.keras.layers.Dense(히든레이어 노드수,activation="swish/relu")(X) > 히든레이어 제작.
- tf.keras.layers.Dense(열 개수(종속변수 개수))(X) > 입력으로 부터 종속변수의 개수만큼의 출력을 내놓는 레이어 구성. 퍼셉트론에서 가중치와 편향을 변경시킴.

- tf.keras.layers.Dense(units, input_dim, activation) > 으로 input 없이 바로 사용할 수도 있다.

- model = tf.keras.models.Model(X,Y)  > 모델 제작 
- model = keras.Sequential([
-   keras.layers.Flatten(input_shape=(x,y)),  > x*y 픽셀의 2차원 이미지 배열을 (x*y)의 1차원 배열로 반환. 앞에 input shape 는 input layer 를 대채할 수 있게해준다.
- 	keras.layers.Dense(128, activation = 'relu'),  > 밀집연결(densely-connected)층/완전연결층. 128개의 노드(또는 뉴런)을 가짐.
- 	keras.layers.Dense(10, activation = 'softmax') > 10개의 클래스 각각 그 클래스에 속할 확률을 출력.
- ]) > 모델(분류기반,최대 세개) 제작.
- Sequential() > .add > .add 로도 모델 제작이 가능함.

- session : 일종의 실행창. 텐서의 내용과 연산 결과를 볼 수 있음. 세션 선언, 실행, 종료 문으로 구분됨.
- tf.Session/InteractiveSession() : 세션 선언 / 자동으로 기본 세션을 지정해주는 세션 선언.
- sess.run(tf.global_variables_initializer()) : 변수 초기화.
- sess.run(텐서) : 실행. 흔히 eval 사용.  |  텐서.eval() : 텐서 객체 데이터 확인.
- sess.close() : 세션 종료. with 로 오픈시 필요 없음.

- model.compile(
-  optimizer='adam',  > 데이터와 손실함수를 바탕으로 모델 업데이트 방향 결정.
-  loss='sparse_categorical_crossentropy',  > 훈련중 모델 오차 측정. 
-  metrics=['accuracy']  > 훈련단계와 테스트 단계를 모니터링하기 위한 방법.
- ) > 모델 컴파일.

- model.fit(train_images , train_labels , epochs=1000(반복 횟수)) > 학습된 모델 제작. verbose=0 으로 떨어지는 모습을 보지 않을 수 있고, validation_data=(test_image,test_label) 로 테스트용 데이터로 계산한 손실과 정확도를 함꺠 출력시킬 수 있다.
- 모델 학습시에는 np.ndarray(shape=(image_amount, image_size[1], image_size[0]), dtype=np.float32) 식으로 준비된 이미지 파일과 np.ndarray(shape=(image_amount,), dtype=np.int32)식으로 준비된 레이블에 이미지 오픈 > fit > asarray > normalized(astype(np.float32)>/255.0 ) > all_images[i]에 넣은 이미지를 사용해야 한다. mnist, PIL > (number, y, x) 로 train. | openCv > (number, y, x ,3) 으로 train
- model.save('파일명.h5') > 모델 저장
- model.evaluate(test_images, test_labels) > 모델 성능 비교. loss, accuracy 순으로 반환. verbose = 0 > silent

- model.predict([[15]]) > 모델을 사용해 입력에 따른 예측 반환. [[숫자]]나 inde[:5]식으로 모델에 넣을 수 있음. 2차원 이미지를 넣어 2차원 배열이 반환됨. 종속변수가 여러개일 경우 경우 [[종1,종2,종3]]식으로 , 2중 for 문으로 하나씩 뽑을 수 있다.
- model.get_weights() > 각 독립변수에 대한 가중치 출력.
- model.summary() 로 모델의 정보(이름/none,출력하는 개수/파라미터(가중치의 개수))를 확인 할 수 있다.

- tf.keras.layers.Conv2D(컨볼루션 크기(행,렬), 필터 이미지 개수(한 행렬의 크기 x,y), padding(='same' 입출력 사이즈 동일), activation, inputShape) > 이미지에 convolution filter 를 사용해 행렬을 만듦.
- tf.keras.layers.MaxPooling2D((줄일 행렬의 크기 x, y)) > 이미지를 MaxPooling 해 크기를 줄임.
- tf.keras.layer.Dropout(rate) > Overfitting 을 방지하기 위해 DropOut. rate 는 1 = 100% 다.

- tf.keras.utils.to_categorical(정수 리스트) > 정수 리스트에 따라서 원핫 인코딩. [1,3]을 넣으면 [[0,1,0,0],[0,0,0,1]]을 반환하는 식.
- tf.lite.TFLiteConverter.from_keras_model(model).converter() | open('파일명.tflite', 'wb') > tf 모델 tflite 바이너리로 변환. 이렇게 변환한 것은 안드로이드 스튜디오의 에셋에 복사 > app 모듈의 build.gradle 에 패키지 추가 > Main_Activity 에서 이미지 바이너리 변환 > Classifier 에서 모델 사용 > Main_Activity 에서 출력 순으로 사용된다.


# Pytorch ( torch )
***
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


# scikit learn
***
#### dataset Load
- sklearn.datasets.load_데이터셋 이름() > 데이터셋 로드.  데이터셋.keys()로 키들을 볼 수 있고, DESCR 에는 데이터셋 설명이, target_names 에는 클래스가, feature_names 에는 각 특성의 이름이. data 에는 샘플별 데이터가, target 에는 샘플의 종류가 클래스 순서대로 0부터 들어있다. .fit(train_data, train_label)로 train, .predict(test_data)로 예측할 수 있다. n_sample, noise, random_state 등의 매개변수가 있다.
- 회귀용 데이터셋인 boston_housing, diabetes. 다중 분류용 데이터 셋인 digit, 두개의 달처럼 생겨 선형으로는 분류가 어려운 two_moons 등 다양한 데이터 셋이 있다. 먼저 간단한 모델(선형, 이웃, NB 등)로 성능을 실험해 보며 데이터를 이해한 뒤 다른 모델을 적용시켜 보는게 좋다.ㄴ   
- sklearn.model_selection.train_test_split(x,y,random_state,train_size) > 데이터를 트레인과 테스트로 나눈다. random_state 는 유사 난수 생성으로 꼭 초기화를 해주는 게 좋다.
- sklearn.model_selection.KFold(k(int), shuffles(bool), random_seed(int)) > k번 K-Fold 를 해주는 머신을 생성한다. .split(data)으로 k개의 train / test 데이터셋을 생성한다. data[train], data[test] 식으로 사용한다.

#### method
불러온 모델들은 .fit(train, labels)로 fit, predict(data)로 사용한다. .score(test_img, test_label) 로 정확도를 측정할 수 있다.

#### Regression
###### k
- sklearn.neighbors.KNeighborsRegression(n_neighbors=k) > k개의 이웃을 찾는 knn 회귀모델 로드.
###### Linear  
- sklearn.linear_model.LinearRegression() > OLS(최소제곱법) 선형회귀 모델 로드. 가중치는 .coef_ 에 ndarray 로, 편향은 .intercept_에 저장되어 있다.
###### Ridge, Lasso
- sklearn.linear_model.Ridge(alpha=i) > 리지 선형회귀 모델 로드. 알파 값은 기본 1이며, 높이면 더 단순히(가중치를 0에 가깝게) 만들어 훈련세트의 성능은 나빠져도 일반화는 더 잘되게 만들 수 있다.
- sklearn.linear_model.Lasso(alpha=i, max_iter) > 라소 선형회귀 모델 로드. 리지와 비슷하나 어떤 값은 진짜 0이 될수 있다. np.coef_ != 0 의 합계를 구하면 사용한 특성수를 알 수 있고, 과소 적합을 피하려면 알파를 줄이고 max_iter 를 늘려야한다.
###### decision tree
- sklearn.tree.DecisionTreeRegressor() > 결정트리 회귀 모델 로드. mex_depth, max_leaf_nodes, min_samples_leaf 중 하나만 지정해도 과대적합을 막을 수 있다. 


#### Classification
.decision_function(test data) > 데이터를 분류하며 그 데이터가 분류한 클래스에 속한다고 생각하는 정도를 기록해 반환. 양수값은 양성 클래스를 의미한다. 
.predict_proba(test data) > 각 클래스에 대한 확률. (샘플 수, 클래스 개수) 의 형태를 갖음. 과대적합된 모델은 틀렸어도 확신이 강한 편이고, 복잡도가 낮으면 예측에 불획실성이 더 많다.
###### k
- sklearn.neighbors.KNeighborsClassifier(n_neighbors=k) > k개의 이웃을 찾는 knn 분류모델 로드.     
###### linear
- sklearn.linear_model.LogisticRegression() > 로지스틱 회귀 분류 모델 로드. 이진분류에서 로지스틱 손실 함수를, 다중 분류에서 crossentropy 손실 함수를 사용함. penalty='l1'으로 l1규제를 사용할 수 있다.
- sklearn.svm.LinearSVC() > 선형 서포트 벡터 머신 (분류)모델 로드. 로지스틱과 이것은 규제의 강도를 결정하는 매개변수 C를 가지고 있음. C로 낮은 값을 지정하면 가중치를 0에 가깝게 지정함.
- 선형 회귀의 alpha 와 분류의 C는 각각 클수록/작을수록 모델이 단순해진다는 특징이 있고, 보통 log 스케일(10배씩)로 최적치를 정하며, 중요특성이 믾지 않다고 생각하면 L1규제를, 아니면 기본 L2를 사용한다.
###### decision tree
- sklearn.tree.DecisionTreeClassifier(max_depth=i, random_state=0) > 결정트리 분류기 로드. 최대 깊이 i 까지 가지를 뻗게 한다. 모델.feature_importances_ 로 각 특성들의 중요도를 볼 수 있다.
- sklearn.tree.export_graphviz(트리모델, out_file='파일명.dot', class_names=["첫번째","두번째"],feature_names=이름들, impurity=bool, filled=bool) > 트리모젤 시각화 해 저장. graphviz 모델 의 .Source(파일.read()) 을 디스플레이해 표시할 수 있다.
###### ensemble
- sklearn.ensemble.RandomForestClassifier(n_estimators=n, random_state=0) > random forest 분류 모델 로드. n개의 트리를 생성해 예측한다. 각 트리는 .estimators_ 에 저장되어있다.  
- sklearn.ensemble.GradientBoostingClassifier(learning_rate=r, n_estimators=n, max_depth=m, random_state=0) > 그래디언팅 부스트 분류 모델 로드. n개의 트리를 생성해 r 의 러닝레이트로 예측한다. 각 트리는 .estimators_ 에 저장되어있다. 기본값은 100개, 0.1, 3의 깊이 다.
- sklearn.ensemble.BaggingClassifier(모델, n_estimators=n, oob_score=bool, n_jobs=-1, random_state=0) > 모델을 n개 연결한 배깅 분류기 생성. oob 를 T 로 생성하면 부트스트래핑에 포함되지 않은 매개변수로 모델을 평가함.   
- sklearn.ensemble.ExtraTreesClassifier(n_estimators=n, n_jobs=-1, random_state=0) > 엑스트라 트리 분류 모델 로드.   
- sklearn.ensemble.AdaBoostClassifier(n_estimators=n, random_state=0) > 에이다 부스트 분류 모델 로드. 기본적으로 깊이 1의 결정 트리 모델을 사용하나 base_estimator 매개 변수로 다른 모델 지정 가능.
###### SVM
- sklearn.svm.SVC(kernel='rbf', C=i, gamma=r) > 서포트 벡터 머신 로드. r을 키우면 하나의 훈련샘플이 미치는 영향이 제한되고, i 는 규제가 커진다. 커널 SVM 에서는 특성들의 크기차이가 크면 문제가 생겨, 평균을 빼고 표준편차로 나눠 평균을 0, 분산을 1로 만드는 전처리를 해줘야 한다.    
###### DL MLP
- sklearn.neural_network.MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[]) > 다중 퍼셉트론 분류 모델 로드. 히든 레이어에 넣은 숫자, 넣은 숫자의 개수대로 히든레이어가 생성된다. alpha 매개변수에 값을 넣어 줄 수 도 있다. svc 와 비슷하게 신경망도 일반화(평균 0, 분산 1)를 해주는게 좋다. sgd 라는 옵션으로 다른 여러 매개변수와 함께 튜닝해 최선의 겨로가를 만들 수 있음. 


#### preprocessing
- 불러온 프로세서들은  .fit(train_data) 로 스케일러를 훈련시키고, .transform(data)로 변환한다. .fit_transform(train_data)로 둘을 한번에 할 수있다. 트레인 데이터와 테스트 데이터 모두 같은 스케일 조정을 해주어야 하기에 테스트 데이터는 0~1의 범위를 벗어날 수 있다.
###### Scaler  
- sklearn.preprocessing.MinMaxScaler() : 특성마다 최솟값과 최댓값을 계산해 데이터의 스케일을 0~1로 조정하는 MinMax 스케일러 로드.
- sklearn.preprocessing.StandardScaler() : 모든 특성을 정규 분포로 바꿔준다.
- sklearn.preprocessing.QuantileTransformer(n_quantile = n) : n개의 분위를 사용해 데이터를 균등하게 분포시키는 스케일러 로드. output_distribution='normal'로 균등 분포가 아니라 정규분포로 출력을 바꿀 수 있음. 
- sklearn.preprocessing.PowerTransformer() : 데이터의 특성별로 정규분표에 가깝게 변환해주는 스케일러 로드. method='yeo_johnson'(양수 음수 값 둘 다 지원, 기본값) 과 method='box-cox'(양수만 지원.)를 지정해 줄 수 있다. 
- sklearn.preprocessing.KBinsDiscretizer(n_bins=n, strategy='uniform') : n개로 구간 분할 모델 로드. .bin_edges_ 에 각 특성별 경곗값이 저장되어 있다. transform 메서드는 각 데이터 포인트를 해당 구간으로 인코딩하는 역할을 한다. 기본적으로 구간에 원 핫 인코딩을 적용한다. transform 결과.toarray()로 원핫 인코딩된 결과를 볼 수 있다.  
- sklearn.preprocessing.PolynomialFeatures(degree=i, include_bias=bool) : x**i 까지 고차항(다항식)을 추가해 특성을 풍부하게 나타내는(구간 분할과 비슷한 효과) 모델 로드. bool 이 T 면 절편에 해당하는 1인 특성을 추가한다. 다항식 특성을 선형 모델과 같이 사용하면 전형적인 다항 회귀 모델(결과가 곡선)이 된다. 고차원 다항식은 데이터가 부족한 영역에서는 너무 민감하게 동작한다는 문제점이 있다.    
###### decomposition
- sklearn.decomposition.PCA() : 주성분 분석(PCA) 프로세서 로드. 기본값은 데이터를 회전,이동만 시키고 모든 주성분을 유지하지만 n_component 매개변수에 값을 넣어 유지시킬 주성분의 개수를 정할 수 있다. fit 시 모델.components_ 속성에 주성분이 저장된다. whiten=T 로 주성분의 스케일이 같아지게 할 수 있다. .inverse_transform 을 사용해 원본 공간으로 되돌릴 수 있다. 
- sklearn.decomposition.NMF(random_state = 0) : NMF 프로세서 로드. n_component 매개변수에 값을 넣어 유지시킬 주성분의 개수를 정할 수 있다. 
- sklearn.manifold.TSNE(random_state=n) : 매니폴드학습 알고리즘의 t-SNE 알고리즘 모델 로드. 데이터들을 알아서 나눔. 훈련시킨 모델만 변경 가능해 transform 메서드 없이 fit_transform() 메서드만 지원한다. 
- sklearn.decomposition.LatentDirichletAllocation(n_components=n, learning_method="batch/online", max_iter=i, random_state=0) : n개의 토픽을 생성하는 LDA 로드. 기본 학습 방법인 online 대신 조금 느리지만 성능이 더 나은 batch 방법을 사용할 수 있고, i를 높이면 모델 성능이 좋아진다(기본값은 10). 각 단어의 중요도를 저장한 .components_ 의 크기는 (n, n_words)이다.  
###### cluster(agglomerative)  
- .fit_predict(data) 로 각 데이터 포인트가 속한 클러스터들을 리스트 형태로 받아 볼 수 있다.
- sklearn.cluster.KMeans(n_cluster=n) : n개의 클러스터 중심점을 생성하는 k-평균 알고리즘 모델 로드. .labels_ 에 각 데이터 포인트가 포함된 클러스터들을 리스트 형태로 볼 수 있고, .predict 로 새로운 데이터의 데이터포인트가 어느 클러스터에 속할 지 예측할 수 있다.
- sklearn.cluster.AgglomerativeClustering(n_cluster=n, linkage="ward/average/complete") : 병합 군집 모델 로드. 모든 클러스터 내의 분산을 가장 적게 증가시키는(기본, 대부분 알맞음)/평균 거리가 가장 짦은/최대 거리가 가장 짦은 두 클러스터를 합친다.   
- sklearn.cluster.DBSCAN(min_sample=i, eps=j) : DBSCAN 군집 모델 로드. j 거리에 i 개 이상 데이터가 있다면 그 데이터 포인트를 핵심 샘플로 지정한다. 어느 군집에도 속하지 않는 포인트를 noise 로 지정해, -1의 레이블을 가지며 이를 이용해 이산치 검출을 할 수 있다.   
###### one-hot-encoding
- sklearn.processing.OneHotEncoder(spares=bool) : 원 핫 인코딩 모델 로드. sparse 가 False 면 희소행렬이 아니라 ndarray 로 반환된다. .fit_transform(data)로 사용, .get_feature_names() 로 원본 데이터의 변수 이름을 얻을 수 있다.
- sklearn.compose.ColumnTransformer([("scaling",스케일러,['스케일 조정할 연속형 열 이름들']), ("onehot",원핫인코더,['원핫인코딩할 범주형 열 이름들'])]) : 각 열마다 다른 전처리(스케일 조정, 원핫인코딩)을 하게 해주는 ct 로드. 
- sklearn.compose.make_column_transformer([(['스케일 조정할 연속형 열 이름들'], 스케일러), (['원핫인코딩할 범주형 열 이름들'], 원핫인코더)]) : 클래스 이름을 기반으로 각 단계에 이름을 붙여주는  ct 로드.
###### feature-selection
- sklearn.feature_selection.SelectKBest/SelectPercentile(score_func=f, percentile=i) : 일변량 통계 모델 로드. 고정된 K 개의 특성을/지정된 비율만큼 특성을 선택한다. f 는 분류면 feature_selection.f_classif, 회귀면 feature_selection.f_regression 를 사용하고, i 는 백분율로 입력한다.
- sklearn.feature_selection.SelectFromModel(모델, threshold='median/mean') : 모델을 이용한 모델기반 자동선택 모델 로드. 중요도가 임계치(threshold)보다 큰 모든 특성을 선택한다. 중간/평균 이며 '1.3*median' 식으로 비율을 지정할 수도 있다. 
- sklearn.feature_selection.RFE(모델, n_feature_to_select = n) : 모델을 이용해 n개 까지 재귀적 특성 제거를 하는 반복적 특성 선택 모델 로드. 


#### metrics
- sklearn.metrics.classification_report(테스트 결과, 실제) : 정확도를 다양한 방식이 모여있는 표로 반환.
- sklearn.metrics.***_score(테스트 결과, 실제) : 훈련 결과의 정확도를 다양한 방식으로 반환.
- sklearn.metrics.accuracy_score(테스트 결과, 실제) : 둘의 일치도를 그대로 정확도로 반환.
- sklearn.metrics.f1_score(테스트 결과, 실제) : F1 Measure 를 이용하여 정확도를 반환.
  
- sklearn.model_selection.cross_val_score(모델, data, labels, cv=i) : 모델을 이용해 i번 폴드하는 교차 검증 사용. 총 i개 모델의 정확도를 배열로 반환한다. 보통 반환값.mean()을 이용한 평균값으로 간단하게 정확도를 나타낸다. cv 에 KFold 객체 등을 넣어 사용할 수 도 있다. 그리드 서치와 함께 n_jobs 매개변수로 사용할 cpu 수를 지정할 수 있다. 데이터셋과 모델이 너무 클때는 여러 코어를 쓰면 메모리 사용량이 너무 커져 메모리 사용 현환을 모니터링해야 한다.
- sklearn.model_selection.cross_validate(모델, data, labels, cv=i, return_train_score=bool) : 위와 같지만 훈련과 테스트에 걸린 시간까지 담아 딕셔너리로 반환한다. bool 이 T 면 테스트 스코어도 같이 포함해 반환된다. 모델을 그리드 서치로 사용하면 중첩 교차 검증의 구현이 가능하다. scoring="accuracy/rou_auc"등으로 모델, 최적 매개변수 선택을 위한 평가 방식을 바꿀 수 있다. 
- sklearn.model_selection.GridSearchCV(모델, 파라미터 딕셔너리({'변수명':[넣어볼 수들의 리스트]}, 딕셔너리의 리스트로 넣으면 조건부로 그리드 탐색), cv=i, return_train_score=bool) : 교차검증을 사용한 그리드 서치 매개변수 조정 방법 로드. .fit(data, label)로 설정된 매개변수 조합에 대해 교차검증을 수행하고, 가장 성능이 좋은 배개변수로 데이터에 대한 새 모델을 자동으로 만듦. 모델엔 .score(test_data, test_label)과 .predict(data)로 접근 가능하다. {.best_params_ : 선택한 매개변수, .best_score_ :  이 설정으로 얻은 정확도 ,  .best_estimator_ :  최고의 모델, .cv_result_ : 각 결과}  
-   
- sklearn.model_selection.LeaveOneOut() :  LOOCV 로드. 위의 cv 매개변수로 넣어 사용할 수 있다. 큰 데이터 셋에선 시간이 오래 걸리지만, 작은 데이터 셋에선 종종 더 좋은 효과를 낸다.
- sklearn.model_selection.ShuffleSplit(train_size=i/f, test_siz=i/f, n_split=n) :  i개의 데이터 포인트로/f의 데이터 포인트 비율로 n번 반복분할 하는 임의 분할 교차 검증 로드. cv 매개변수에 넣어 사용할 수 있다.
- sklearn.model_selection.RepeatedStratifiedKFold/RepeatedKFold(random_state = i) : 반복 교차 검증 로드. 분류/회귀 이며 model_selection.StratifiedKFold/KFold 를 기본으로 사용하기 때문에 import 를 해주어야 한다. cv 에 매개변수. 
 
- sklearn.metrics.confusion_matrix(label, pred_logreg(예측결과)) : 오차 행렬 표시. [(음성데이터)[음성으로 예측한 수, 양성으로 예측한 수] (양성데이터)[음성으로 예측한 수, 양성으로 예측한 수]] 식으로 반환된다.
- sklearn.metrics.precision_recall_curve(test_label, 모델.decision_function(test_data)(확신에 대한 측정값)) > 정밀도-재현율 곡선 로드. precision, recall, threshold 총 세가지를 반환한다.가능한 모든 임계값에 대해 정밀도와 재현율의 값을 정렬된 리스트로 반환.   
- sklearn.metrics.average_precision_score(test_label, 모델.predict_proba(test data)(확신에 대한 측정값)) > 위 곡선의 아래 면적인 평균 정밀도를 계산해 반환.
- sklearn.metrics.roc_curve(test_label, 확신에 대한 측정값) > ROC 곡선 로드. FPR, TPR, threshold 총 셋을 반환한다.
- sklearn.metrics.roc_auc_score(test_label, 확신에 대한 측정값) > ROC 곡선의 아래 면적인 AUC 를 계산해 반환한다.

#### Pipeline
- sklearn.pipeline.PipeLine([("임의의 단계 1 이름", 추정기 객체), ("단계 2 이름", 추정기 객체)]) : 여러 처리 단계를 추정기 형태로 묶어주는 pipeline 객체 로드. 각 단계를 이름과 추정기(스케일러,모델 등)으로 이뤄진 튜플 형태로 묶어 리스트로 전달한다. .fit(), .score()등이 전부 사용 가능 하다. 어떤 추정기와도 연결할 수 있으며, 마지막 모델을 제외하고는 전부 transform 메서드를 가지고 있어야 한다.
- sklearn.pipeline.make_pipeLine(추정기1, 추정기2) : 파이프 라인과 똑같지만 단계의 이름을 자동으로 만들어 준다. .steps 속성에 각 단계의 이름이 들어있다. 단계 이름을 키로 가진 .named_steps 딕셔너리도 파이프 라인의 각 단계 속성애 쉽게 접근 할 수 있다. 그리드 서치에 파이프 라인을 사용할 때는 모델에 넣는다.

#### text
- sklearn.feature_extraction.text.CountVectorizer() : BOW 표현을 하게 해주는 변환기 로드. .fit(문자열이 담긴 리스트)로 사용, .vocabulary_ 속성에서 반환된 {단어:등장횟수} 형태의 딕셔너리를 볼 수 있음.  tf-idf 와 함께 ngram_range=(연속 토큰 최소길이, 최대길이) 로 연속된 토큰을 고려할 수 있다. 보통은 하나만 하지만 많을 때 바이그램정도로 추가하면 도움이 된다.  
- Bow 표현을 만드려면 .transform(list), Scipy 희소 행렬로 저장되어 있으며, .get_feature_names()로 각 특성에 해당하는 단어들을 볼 수 있음. min_df 매개변수로 토큰이 나타날 최소 문서 개수를 지정할 수 있고, max_df 매개변수로 자주 나타나는 단어를 제거할 수 있다. stop_words 매개변수에 "english" 를 넣으면 내장된 불용어를 사용한다.
- sklearn.feature_extraction.text.TfidVectorizer(min_df=i) : 텍스트 데이터를 입력받아 BOW 특성 추출과 tf-idf 를 실행하고 L2정규화(스케일 조정)까지 적용하는 모델로드. 훈련데이터의 통걔적 속성을 사용하므로 파이프 라인을 이용한 그리드 서치를 해 주어야 한다. .idf_ 에서 훈련세트의 idf 값을 볼 수 있다. idf 값이 낮으면 자주 나타나 덜 중요하다 생각되는 것이다.

> ###### spacy
>+ 영어와 독일어를 지원하는 NLP 파이썬 패키지. 표제어 추출 방식이 구현되어 있음. 
>+ python -m spacy download en 으로 언어의 모델을 먼저 다운받아야 함.
>+ spacy.load('en') : spacy 의 영어 모델 로드. 
>+ 모델(document) : 문서 토큰화. 찾은 표제어들 반환.
   
> ###### nltk
>+ 포터 어간 추출기가 구현되어 있는 파이썬 패키지.
>+ nlty.stem.PorterStemmer() : PorterStemmer 객체 생성.
>+ 객체.stem(토큰.norm_.lower()) > 토큰(어간) 찾기.
 
> ###### KoNLpy
>+ 한글 분석을 가능하게 하는 파이썬 패키지.
>+ konlpy.tag.Okt() >  Okt 클래스 객체 생성. .morphs(text)로 형태소 분석이 가능하다.





### RL




# os
- os.getcwd() : 현재 작업 폴더 반환.
- os.chdir(경로) : 디렉토리 변경.
- os.path.abspath(상대 경로) : 절대 경로 반환. 
- os.path.dirname(경로) : 디렉토리명만 반환.
- os.path.basename(경로) : 파일 이름만 반환.
- os.listdir(경로) : 경로 안의 파일 이름을 전부 반환.
- os.path.join(상위, 하위) : 경로를 병합해 새 경로 생성. ('C:\Tmp', 'a', 'b')식으로 넣는다.
- os.path.isdir(경로) : 폴더의 존재 여부를 반환.
- os.path.isfile(경로) : 파일의 존재 여부를 반환.
- os.path.exists(경로) : 파일 혹은 디렉토리의 존재 여부를 반환.
- os.path.getsize(경로) : 파일의 크기 반환.


