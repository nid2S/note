# tensorflow
- Tensorflow : 구글이 주도적으로 개발한, 가장 널리 쓰이는 딥러닝 프레임워크중 하나. C++로 작성됨. keras중심 고수준 API 통합 지원. 다양한 프로그래밍언어의 API를 지원.
  시냅스 웨이트(텐서) 네트워크 모델을 따라 흐른다는데서 이름이 유래, TPU지원, 일반적으로 32bit의 곱셈연산을 16bit로 줄임 등의 특성이 있음.
- 각 단계에서 즉석으로 그래프를 재생성할 수 있는 pytorch와 달리 텐서플로우는 기본적으로 단일 데이터흐름 그래프를 만들고, 그래프코드를 성능에 맞게 최적화한 뒤 모델을 학습시킴. 
- Estimators(객체지향 레벨) > layers,losses,metrics > Python/C++ Tensorflow > CPU/GPU/TPU 순으로 아키텍쳐(API)가 구성되어 있음.
- tensorflow in java : tf에서 libtensorflow.jar 다운로드 > 압축 해제후 jar파일 src에 복사 > properties 에서 add jar > 다운한 파일선택 > apply 
  과정을 거친 후 import org.tensorflow로 사용.
- XLA(Accelerated Linear Algebra) : 구글이 개발한 TF용 컴파일러. 2017/03에 공개됐음. CPU/GPU 및 TPU등과 같은 기기에 대해 JIT컴파일 기법을 사용해, 사용자가 생성한 TF그래프를 분석하고 
  실제 런타임 차원과 유형에 맞게 최적화하며 여러 연산을 합성해 이에 대한 효율적인 네이티브 기계어 코드를 내보냄. softmax등 원시 연산들의 조합을 자동으로 최적화하는데 도움을 줌.
  이외에도 AOT(Ahead-Of-Time) 컴파일에 XLA를 활용해 실행파일의 크기를 줄여 휴대기기등의 환경에서 도움을 줌(전체 그래프가 XLA로 컴파일된 다음 연산을 구현하는 세밀한 기계어 코드를 내보내는 방식, 상당한 크기감소효과를 냄).

## divece
- tf.test.is_gpu_available() : gpu가 사용가능한 상태인지 반환.
- tf.test.gpu_device_name() : 사용가능한 gpu의 이름을 반환.

## tensor
- 모든 텐서는 값이 변경과 추가가 블가능하고, 오직 새로 만드는 것만 가능함. 따라서 연산(값 추가)은 `tf.convert_to_tensor(tensor.numpy(), ndarray)`식으로 해야 함.
- tf.Variable(수식, name="변수명") : 변수 선언 후 수식으로 정의. x+4 식으로 수식을 지정해 변수에 할당하는 방식.
- tf.constant(상수, name="상수명") : 상수 선언 후 값 지정. y = tf.constant(20, name="y") 식으로 사용.

- tf.zeros/ones(shape) = shape대로 0/1으로 채워진 텐서 생성.
- tf.random.uniform(shape, min, max) : shape형태의, min~max사이의 랜덤 값을 가진 텐서 생성.
- tf.random.normal(shape, mean, stddev) : shape형태의, 평균이 mean이고 표준편차가 stddev인(기본은 0,1) 랜덤 값을 가진 정규분포 텐서 생성.
- tf.convert_to_tensor(array) : array를 텐서로 변환.
- tf.make_tensor_proto(텐서) : 텐서를 Protobuf(.pb)텐서로 바꿈. 

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
- tf.lite.TFLiteConverter.from_keras_model(model).convert() | open('파일명.tflite', 'wb') > tf 모델 tflite 바이너리로 변환. 
  이렇게 변환한 것은 안드로이드 스튜디오의 에셋에 복사 > app 모듈의 build.gradle 에 패키지 추가 > Main_Activity 에서 전처리 > 모델사용 > Main_Activity 으로 사용됨.

## dataset
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

## preprocessing
### tokenize
- tf.keras.preprocessing.text.text_to_word_sequence(sentence) : 모든 알파벳을 소문자로 변환, 구두점 제거, 죽약형은 분리하지 않는 단어 토큰화 함수. 
  정제와 단어 토큰화를 동시에 적용.
- tf.keras.preprocessing.text.Tokenizer() : 정수 인코딩을 위한 토크나이저 로드. 단어집합 생성과 토큰화를 병행. .fit_on_texts(단어집합)으로 단어 빈도수가 
  높은 순으로 낮은 정수 인덱스를 부여, texts_to_sequences로 변환. .word_index 로 단어와 인덱스를 확인할 수 있고, .word_counts 로 단어의 개수를 확인 할 수 있다. 
- Tokenizer() 매서드 : .texts_to_matrix(문장배열,mode='count')로 DTM(인덱스 0부터 시작)을 생성할 수 있다. 모드가 'binary' 면 단어의 존재여부만 보여주는 행렬을, 
  tfidf 는 tfidf 행렬을, freq 는 (단어 등장 횟수/문서 단어 총합)의 행렬을 보여준다.
- Tokenizer() 메개변수 : num_words(단어 빈도순으로 num_words개 보존), filters(걸러낼 문자모음. 디폴트 - !"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n),
  lower(입력 문자열 소문자 변환여부. bool), split(단어분리기준. str), char_level(문자를 토큰으로 취급. bool),
  oov_token(값이 지정된 경우, text_to_sequence 호출 과정에서 word_index에 추가되어 out-of-vocabulary words를 대채) 매개변수 사용가능.
### vectorize
- tf.keras.utils.to_categorical(벡터) : 원 핫 인코딩을 해줌. (요소 개수, 요소 종류)의 형태를 가짐.
- tokenizer.texts_to_sequences(단어집합) : 각 단어를 이미 정해진 인덱스로 변환. 만약 토크나이저 로드시 인수로 i+1을 넣었다면 i 까지의 인덱스를 가진 단어만을 사용하고 나머지는 버린다.
- tf.keras.preprocessing.sequence.pad_sequences(인코딩된 단어 집합) : 가장 긴 문장의 길이에 맞게 문장의 앞에 0을 삽이비해 ndarray 로 반환. 
  padding='post' 로 문장 뒤에 0을 삽입할 수 있고, maxlen 매개변수로 길이를 지정할 수 있다.
- tf.keras.layers.experimental.preprocessing.PreprocessingLayer() : 전처리 층을 위한 base층. 커스텀 층을 만들어 상속, 정의하는 방식으로 사용됨. lambda나 타 서브 층을 써도됨. 
- tf.keras.layers.experimental.preprocessing.TextVectorization() : 전처리(소문자, 공백분할, 구두점제거, 정수화) 층을 생성. 층.adapt(문자데이터)로 vocab을 추가해 줘야 하며, 
  혹은 배치를 직접 넣어줄 수 있음(패딩토큰('')과 OOV토큰('[UNK]')이 같이 들어감). 바로 앞층은 shape 1, dtype=tf.string인 인풋이여야 함. .get_vocabulary()로 vocab확인가능.
  레이어를 여러번 조정할 경우, model.compile/내부데이터셋.map(layer)/직접tf.function을 쓰는 경우 모두 layer.adapt()이후에 해야 함. 간단한 전처리에 좋을 듯. 
- TextVectorization로드 : 로드한 모델의 layer는 train당시의 vocab을 가지고 있지 않아 전부 OOV로 변환하게 됨. 커스텀오브젝트로 TextVectorization을 지정해 문제를 해결할 수 있음. 
  `load_model(path, custom_objects={"TextVectorization":TextVectorization})`로 커스텀 오브젝트 지정.
- TextVectirization인자 : max_tokens(vocab size), standardize(입력에 적용될 정규화. None이면 동작X, Callable이면 해당 작용 수행. 기본은 'lower_and_strip_punctuation'),
  split(None/whitespace(기본)/Callable), output_mode("int(seqVec)/tf-idf/binary/count"), ngrams(n그램 벡터화의 n, None), output_sequence_length(max_len), 
  pad_to_max_tokens(bool, max_tokens에 맞춰 패딩. binary/count/tf-idf에서만 작동), vocabulary(vocab(토큰의 배열)).
### embedding
- tf.keras.layers.Embedding(총 단어 개수, 결과 벡터의 크기, 입력 시퀀스 길이) : 단어를 밀집벡터로 만듦(임베딩 층(Dense 같은)제작). 
  모델 내에서 (num of sample, input_length)형태의 정수 인코딩이 완료된 2차원 정수 배열을 입력받아 워드 임베딩 후 3차원 배열을 반환. 
  weights 매개변수에 사전 훈련된 임베딩 벡터의 값들을 넣어 이미 훈련된 임베딩 벡터 사용 가능.
### pooling
- tf.keras.layers.GlobalMaxPooling1D() : 1차원 풀링 실행. Conv1D 뒤에 위치.
### normalization
- tf.keras.layers.LayerNormalization/(layers) : 층 정규화. 텐서의 마지막 차원에 대해 평균과 분산을 구하고, 이를 이용해 값을 정규화 함.

## layers
- tf.keras.layers.Input(shape=(입력 차원)) : 입력차원 만큼 입력레이어 구성.
- tf.keras.layers.Dense(노드수,activation=활성화함수) : 전밀집층(모든 노드가 이전 혹은 다음 노드와 연결, 전결합층)제작. input_dim(입력차원)매개변수 사용가능. 
  ((입력의 마지막차원+1(bias))*노드수)개의 파라미터가 생성, (None, 최초입력의 마지막 제외 차원, 노드수) 형태의 반환값을 반환.

- tf.keras.layers.Embedding(총 단어 개수, 결과 벡터의 크기, 입력 시퀀스 길이) : 단어를 밀집벡터로 만듦(임베딩 층 제작, 단어를 랜덤한 값을 가지는 밀집 벡터로 변환 후 학습과정을 거침). 
  (샘플개수, 입력길이)형태의 정수 인코딩이 된 2차원 정수 배열을 입력받아 워드 임베딩 후 3차원 배열을 반환. mask_zero=True인자로 패딩된 토큰을 마스킹 할 수 있음. input_dim인자(vocab)사용가능.
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
  return_state(마지막 은닉상태 한번 더 출력)매개변수 사용 가능. 임베딩 > RNN > 출력층 만으로도 간단한 분류 task의 구현이 가능.
- tf.keras.layers.LSTM(hidden_size, input_shape=(time_steps, input_dim)) : RNN 의 일종인 LSTM 사용. RNN 층은 (batch_size(배치 크기, 
  한번에 학습할 데이터 양), timesteps(시점, 문장의 길이), input_dim(단어 벡터 차원)) 크기의 3D 텐서를 입력으로 받음. return state 를 true 로 하면 마지막 셀 상태까지 반환, 
  양방향이면 정방향,역방향 둘 다 은닉상태와 셀상태 반환(fh,fc,bh,bc 순).
- tf.keras.layers.GRU(hidden_size, input_shape=(time_steps, input_dim)) : LSTM 을 개량한 GRU 사용. LSTM 에 비해 구조가 간단하고, 
  데이터 양이 적을떄 LSTM 보다 낫다고 알려져 있음.

- tf.keras.layers.experimental.preprocessing.TextVectorization() : 텍스트 데이터를 전처리함. 텍스트 전체를 shape(1)로 받아 처리. 모델로드시 커스텀오브젝트를 사용해야함.
- tf.keras.layers.experimental.preprocessing.PreprocessingLayer() : 프리프로세스 층을 위한 base층. abstract층이기 때문에 직접적으로 사용할 수 없음.
- tf.keras.preprocessing.sequence.pad_sequences(data, maxlen) : 데이터(리스트)의 요소 개수를 maxlen으로 고정. 적으면 0을 채우고 많으면 버림.

## model make
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
### custom
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

## model use
### train
- model.compile(
- optimizer='adam',  : 데이터와 손실함수를 바탕으로 모델 업데이트 방향 결정.
- loss='sparse_categorical_crossentropy',  : 훈련중 모델 오차 측정.
- metrics=['accuracy']  : 훈련단계와 테스트 단계를 모니터링하기 위한 방법.
- ) : 모델 컴파일.

- model.fit(train_data , train_labels , epochs=1000(반복 횟수)) : 학습된 모델 제작. validation_data=(test_data,test_label)로 검증용 데이터로 계산한 손실/정확도를 함께 출력가능하며,
  callbacks 매개변수에 callbacks의 함수를 넣어 사용할 수 있음. 여러개면 [one, two\]식으로 입력. loss와 accuracy(metrics)가 담긴 딕셔너리를 반환함.

### callback
- keras.Model.fit/evaluate/predict() 의 메서드에 callbacks 매개변수로(리스트 형태로)전달할 수 있음.

- tensorflow.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose, patience) : 과적합 방지를 위한 조기 종료 설정. 
  patience회 검증 데이터의 손실이 증가하면 학습을 조기종료함. 모델 fit 과정에서 callback 매개변수에 넣어 사용가능.
- tensorflow.keras.callbacks.ModelCheckpoint(모델명, monitor="val_accuracy", mode="max", save_best_only=True) : 
  검증 데이터의 정확도가 이전보다 좋아지면 모델 저장. 모델 fit 과정에서 callback 매개변수에 넣어 사용가능. 모델의 체크포인트를 저장.
- tensorflow.keras.callbacks.LearningRateScheduler(scheduler) : 매 epoch가 시작될 때 업데이트된 학습률 값을 가져와 적용. 
  스케줄러 함수(epoch와 lr을 인수로 받음)를 정의해 인수로 넣고, 이를 callback에 넣어 사용.
- tensorflow.keras.callbacks.TensorBoard(log_dir, write_graph=True, write_images=True, histogram_freq=1) : log가 저장될 경로와 가중치등의 시각화 여부를 지정해 
  텐서보드로 훈련 진행 상황과 결과를 시각화 할 수 있게 함.
- tensorflow.keras.callbacks.LambdaCallback(메서드명=함수) : 특정 상황에 특정 함수를 실행하는 콜백. 메서드 명의 종류는 하단, 커스텀 콜백에서 오버로딩 가능한 함수의 종류와 동일.

- tf.keras.callbacks.Callback을 상속하는 클래스를 만들어 훈련, 테스트, 예측에서 호출되는 메서드 세트를 재정의 할 수 있음. 각 단계마다 호출되는 함수가 정의되어 있음. 
  각 logs는 dict로, 손실값과 배치/에포크 끝의 모든 메트릭이 포함됨.
- def on_(train|test|predict)_begin(self, logs=None) : 메서드 시작시 호출. 
- def on_(train|test|predict)_end(self, logs=None) : 메서드 종료시 호출. 
- def on_(train|test|predict)_batch_begin(self, logs=None) : 메서드 연산증 배치 처리 직전에 호출. 
- def on_(train|test|predict)_batch_end(self, logs=None) : 메서드 연산중 각 배치(연산)이 끝날때 호출. 이때 logs는 메트릭 결과를 포함하는 dict.
- def on_epoch_begin(self, epoch, logs=None) : 훈련중 epoch가 시작될 때 호출. 훈련만 해당. 
- def on_epoch_end(self, epoch, logs=None) : 훈련중 epoch가 끝날 때 호출. 훈련만 해당.

## TensorBoard
- 로그생성 : 이전의 로그데이터를 모두 지운 뒤, logdir을 설정하고(주로 `os.path.join('./logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))`식), 
  `tf.summary.create_file_writer(logdir)` -> `with file_writer.as_default():` -> tf.summary.xxx() or callback, 로 로그파일을 생성할 수 있음.
- 텐서보드 사용 : `%tensorboard --logdir ./logs`로 확인할 수 있음.
- 주의사항 : colab만의 문제점인지는 모르겠으나, 허깅페이스(kodialoGPT사용)를 사용했을 때 Tensorboard callback을 사용하면 특정 메모리에 너무 많이 할당되었다고 학습이 중단됨. 

### save
- model.save(path) : 전체 모델 저장. 두가지의 다른 파일 형식(SaveModel, HDF5(keras))으로 저장가능. 확장자없이 path만 넣으면 SaveModel(.pb), %.h5면 HDF5. subclassing API로 작성된 모델은 hdf5로 저장할 수 없음. 
- model = tf.keras.models.load_model(path.h5) : 케라스 형식(.h5)으로 저장된 모델 로드.
- model = tf.saved_model.load(path) : 텐서플로우 기본 형식(saved_model.pb 파일이 들어있는 폴더)으로 저장된 모델 로드.

- model.get_weights() : 각 독립변수에 대한 가중치 반환.
- model.save_weights(path) : 모델의 가중치 저장.
- model.load_weight(path) : 모델의 가중치 복원. 원본 모델과 같은 아키텍쳐를 공유해야 함.

- 체크포인트 : 가중치, 모듈 및 하위 모듈 내부의 변수 세트 값. 데이터자체(변수값과 해당속성 조회경로)와 메타데이터용 인덱스파일(실제 저장된 항목과 체크포인트 번호 추적)로 구성.
- tf.train.Checkpoint(model) : 체크포인트 생성.
- 체크포인트.write(path) : 체크포인트 저장. 전체 변수 모음이 포함된 python객체별로 정렬되어 있음.
- 체크포인트.restore(path) : 체크포인트(python객체 값)를 덮어씀.
- tf.train.list_variables(patj) : 체크포인트 확인.

### use
- 커스텀 모델은 model()으로 call함수를 사용할 수 있음(predict 사용불가).
- model.summary() 로 모델의 정보(이름/none,출력하는 개수/파라미터(가중치의 개수))를 확인 할 수 있다.
- model.predict(X) : 모델을 사용해 입력에 따른 예측 반환.
- model.evaluate(test_images, test_labels) : 모델 성능 비교. loss, accuracy 순으로 반환. verbose = 0 > silent

## other_API
### TenserflowLite
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

- 안드로이드 : 모델을 안드로이드 모듈의 assets디렉토리에 복사 후, build.gradle에 tflite라이브러리를 추가함. 이후 tflite서포트 라이브러리의 프로세서를 이용해 전처리를 하고, 
  결과를 저장할 TensorBuffer객체를 만든 후, 모델을 로드하고, 그걸 기반으로 인터프리터를 생성 후 실행(x.getBuffer(), 출력텐서.동일()). 출력텐서.getFloatArray()를 통해 출력 획득가능.
  레이블을 결과와 매핑하려면, txt파일을 assets에 복사 후 FileUtil.loadLabels()로 생성, `TensorLabel(레이블, 출력텐서(후처리)) > .getMapWithFloatValue()`로 가능. 예제 확인 추천.
- 안드로이드 흐름 : 모델 assets에 삽입 > build.gradle > 모델파일 메모리에 > 모델 로드 후 인터프리터 획득 > input/output준비(배열 등) > 인터프리터.run(x, output) > 결과사용. 
- 안드로이드 래퍼코드 : 메타데이터로 강화된 tflite모델은 안드로이드스튜디오 ML 모델바인딩을 사용해 프로젝트를 위한 설정을 자동으로 구성하고, 모델 메타데이터에 기초한 래퍼클래스를 생성가능.
  ByteBuffer와 직접 상호작용할 필요 없이 Bitmap, Rect등의 형식화된 객체로 모델과 상호작용할 수 있음. 코틀린/자바를 위한 샘플코드 섹션도 제공함.
  `New > Other > Tensorflow Lite Model`를 생성하면, 자동으로 필요한 파일이 build.gradle에 삽입되며 선택적으로 GPU옵션도 선택할 수 있음.
  스레드 수를 이용한 코드가속방법도 제공(실행과 생성의 스레드가 같아야함, 모델 생성시 param으로 설정)하며, gpu dependency를 이용해 가능하다면 GPU를 사용하는 코드도 tf홈페이지에서 제공함.
- 코드생성기를 이용한 모델인터페이스 생성 : tflite-support설치 후 `flite_codegen --model=모델path --package_name=org.tensorflow.lite.classify \
  --model_class_name=모델클래스명 --destination=저장될 폴더`로 생성. 이 후 destination을 압축, 다운로드해 안드로이드 스튜디오에서 import Module, 
  `implementation project(":classify_wrapper")`를 종속성 세션에 추가해 사용. 이 후 부턴 평범하게 사용하면 됨. 예제는 마찬가지로 홈페이지에 존재함.
### other_languages
- 지원하는 언어들 : C, JS, C++, Java, C#, Haskell, Julia, MATLAB, R, Ruby, Rust, Scala, Go, Swift. 지원기능이 조금씩 달라, 사용전 해당 언어의 공식문서를 살펴봐야 함.
  C는 타언어와의 바인딩을 빌드, JS, C++, Java는 Python과 비슷한 수준의 API, Go와 Swift는 보관/지원되지 않는 언어바인딩을, 나머지는 바인딩을 지원함. 개발중인 API존재.
- C : 편리함보다 단순성과 통일성을 위해 설계되었으며, tf홈페이지의 url을 이용해 다운로드할 수 있고, CPU와 GPU전용이 구분되어있음. `#include <tensorflow/c/c_api.h>`로 사용.
  Go/Rust등의 요구사항. 자체적으로도 C와 C++에서 사용할 수 있음. `auto* session = TF_LoadSessionFromSavedModel()`로 저장된 모델을 사용가능하며, 다양한 인자를 필요로함.
- JS : 자체적으로 텐서나 시퀀셜/함수형 API를 사용해 모델을 만들 수 도 있고, Graph모델과 LayersModel(평소 쓰는 모델)을 나눠 로드, 브라우저에서 다운하거나 
  모델을 http서버에 보내는 등 다양한 기능을 지원함. 또한 사용시 predict/OnBatch(), fit/Dataset()등 데이터셋/배치의 함수를 따로 지정함.
- Java : 모든 JVM(Java/Scala/Kotlin)에서 실행해 머신러닝 빌드/학습/배포에 사용할 수 있음. Maven이나 Gradle, 소스로 설치할 수 있음. 사용은 tflite와 비슷하게 사용가능.
- C# : nuget으로 설치할 수 있고, 세션과 그래프를 사용하는 형식. `var graph = new TFGraph() > graph.Import(File.ReadAllBytes("MySavedModel"));`로 모델로드.
- Go : tf-1처럼 세션과 그래프를 사용. LoadSavedModel()함수로 다른 언어로 빌드된 모델의 로드가 가능하며, 그래프와 디스크의 체크포인트에서 초기화된 변수로 세션을 초기화하는 형식.
- Rust : 텐서플로우 C API에 의존함. Cargo.toml에 `[dependencies] tensorflow = "0.17.0"`를 추가해 사용시작. 세션과 그래프를 사용하는 방식.
  `SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, export_dir)?;`로 모델의 로드가 가능함.


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


# tensorflow_serving
- Tensorflow Serving : 허깅페이스의 TF모델배포/성능개선을 돕는 도구. HTTP를 사용하는 API와 서버에서 추론실행을 위해 gRPC를 사용하는 API를 제공. 도커나 pip로 설치. 
- 모델 배포/사용 : saveModel생성 -> (도커의 경우)모델을 넣을 도커 컨테이너를 생성 후 실행 -> REST API를 통해 모델쿼리(JSON("instances":sent)으로 만들어 post).

- request = tensorflow_serving.apis.predict_pb2.PredictRequest() : 예측생성을 위한 gRPC요청 생성. 
- request.model_spec.name = 모델명 : 모델명 설정.
- request.model_spec.signature_name = 서명 : gRPC쿼리의 포맷으로 사용되는 서명. 기본은 "serving_default".
- request.inputs["input_ids"\].CopyFrom(tf.make_tensor_proto(토큰 텐서)) : input_ids input을 주어진 텐서(encoded_input["input_ids"\])로 설정.  
- request.inputs["attention_mask"\].CopyFrom(tf.make_tensor_proto(토큰 텐서)) : attention_mask 설정.
- request.inputs["token_type_ids"\].CopyFrom(tf.make_tensor_proto(토큰 텐서)) : token_type_ids 설정.

- import grpc -> grcp.insecure_channel("localhost:"+port) : 컨테이너의 GRPC(google Remote Procedure Call)포트와 연결될 채널 생성.
- tensorflow_serving.apis.predict_service_pb2_grpc.PredictionServiceStub(채널) : 예측 생성을 위한 stub생성. 이 stub은 GRPC요청을 TF서버에 보내는데 사용됨.
- result = stub.Predict(request) : gPRC요청을 TF서버로 전송해 결과를 얻음. 출력은 protobuf이며, 키 logits에 할당된 확률목록 뿐임.
- result.outputs["logits"\].float_val : 확률이 float라면, 리스트를 float형 ndarray로 변환함.
