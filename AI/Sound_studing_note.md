# 음성
- 음성파일 확인
```python 
# 음성 확인
print("녹음된 소리를 출력합니다.")
winsound.PlaySound(WAVE_OUTPUT_FILENAME, winsound.SND_FILENAME)     # 음성 재생

plt.style.use("seaborn-white")
fig = plt.figure(figsize=(14, 4))
wav, rate = librosa.core.load(WAVE_OUTPUT_FILENAME)
librosa.display.waveplot(wav, sr=rate)                              # 음성 그래프화
plt.show()
```

## 소리
- 소리 : 진동으로 인한 공기의 압축. 압축이 된 정도를 파동으로 표시. 대부분의 소리는 복합파(서로다른 정현파의 합으로 이뤄짐).
- 소리에서 얻을 수 있는 물리량 : 진폭(진폭 세기, 소리크기), 주파수(떨림의 빠르기/압축된정도, 음정/높낮이/진동수), 위상(파동모양, 음색/소리감각).

- 음성파일 형식 : \[무압축] .wav(윈도우)/aiff(맥), \[소실압축] .mp3/aac(Higher quality at lower bitrate)/ogg(Vorbis),
  \[무손실 압축] .flac/alac, \[?] .m4a/wma
  
- 샘플링 : 아날로그 신호(소리)를 디지털 신호로 변환하는 과정. 초당 샘플링 횟수를 샘플링 레이트 라고 함.
- 푸리에변환 : 임의의 입력신호를 다양한 주파수를 갖는 주기함수(복수지수함수)의 합으로 분해해 표현. 변환 후 복소수(절댓값-주파수강도|phase-위상)를 결과로 얻음.

- 채널 : 정보가 저장된 방식의 개수. 컬러이미지의 경우(RGB) 세가지, 오디오의 경우 녹음과정에서 나눈 소리의 방향/종류(1~7.1 까지).

## 음성
- 음성 : 사람이 조음기관(목,입,입술 등)을 사용해 뜻을 전달하기 위해 의도적으로 만들어낸 소리. 음소의 합으로 구성되어 있음.
- 음소문자 : 음소 하나하나가 문자의 자음/모음에 대응하는 문자. 로마자/한글 등이 포함.

## 음성처리
- 음성처리 : 음성코딩(음성신호를 효율적으로 전송/저장하기위한 코딩), 음성인식(음성언어를 기계가 인식 및 해석, STT), 
  음성합성(말소리의 음소를 기계가 자동으로 만들어내는 기술, TTS), 음성변환(음성의 스타일을 변환, STS)으로 나뉨. 

- 음성 코딩 알고리즘 : 코덱(아날로그>디지털)의 알고리즘 - LPC, WLPC, A-law, μ-law 등.
- 음성 인식 알고리즘 : STT의 알고리즘 - HMM (Hidden Markov models), 동적시간왜곡(DTW), 신경망(NN) 등.
- 음성 합성 알고리즘 : 프론트엔드(텍스트를 발음 기호로 변환), 백엔드(프론트엔드 결과를 음성으로 만듦)분야가 있음

### 음성인식
- 음성인식 시스템 : 몇개의 서브시스템으로 구성, 각 서브시스템은 입력 신호를 받아 미리 준비된 테이블과 비교해 가장 근사한 아이템을 반환.
- 음성인식 시스템 구현 : 전처리(음성신호로 부터 시간/주파수 영역의 특징을 추출), 패턴인식(문장 구성에 필요한 음소, 음절, 단어를 인식), 
  후처리(음소, 음절, 단어를 재구성해 문장을 복원해 냄)의 단계가 필요함.
- 판정기준 : 화자독립성(화자가 달라져도 기능 수행 가능), 연속단어 처리 가능(단어 사이에 묵음이 없어도 인식 가능), 
  처리 단어수(처리할 수 있는 단어 수) 총 세가지(+인식률,문법,잡음처리)가 판정 기준. 고전 음성인식 알고리즘에선 이 기준들이 상충되기도 함.  
- 어려움 : 변화가 큰(각 음소/음운의 문장 내 위치, 발화기관 상태에 따라 달라짐)데이터 임.

#### 관련 알고리즘
- 퍼지이론 : 어떤 속성을 나타내는 집합 설정 후, 원소들이 이 집합에 속하는 정도를 0~1로 나타냄. 
- 신경망 : 생체의 지능 시스템을 수학적으로 해석.
- 시뮬레이티드 어닐링 : 고온에서 액체인 물질이 식는 속도에 따라 최종결정상태가 바뀌는 현상을 통해, 함수의 최솟값을 결정할 수 있는 알고리즘 발명.
- 유전 알고리즘 : 교배, 돌연변이 등을 수학적으로 표현해 결국 최고 인구를 같게 되는 알고리즘. 

## CNN in Audio
- 2DCNN : 주로 사용. 타임과 주파수 영역 모두를 이용해 Conv층을 쌓음. 시간/주파수 두 영역에서 패턴을 찾음.
- 1DCNN : 시간화 특화된 값을 뽑아낼 때 사용. NLP에서의 CNN과 비슷하게, 커널의 너비를 frequncy로 고정하고 Time에 따라 움직이며 압축을 수행함.
- Sample CNN : 화자인식등 페이즈(음색등의 정보)영역이 필요한 모델에서 사용. CNN사용시 row한 오디오 인풋을 어느정도 패딩간격을 두며 사용.
  Sampling rate(1초당 가져오는 소리정보)가 8000/16000/44100정도로 커지며 샘플레벨에서 CNN을 돌리기엔 필요한 연산 과정이 너무 커져 사용. 

## RNN in Audio
- 입력의 크기(길이)가 어떻든 커버할 수 있고, 모델의 크기가 증가하지 않기에 사용됨.
- 과거의 정보(Historical Infomation)를 잘 활용하고, 시간축에 따름 가중치 공유가 진행됨.

# 음성인식(STT)
- 음성인식 : 음성을 듣고 이해하는 능력. 사람이 내는 소리를 짧은 단위(음소)로 추출하고, 이를 조합해 소리의 의미/의도를 파악하는 과정.
- 사람의 음성 인식 : 정보를 음파형태로 받아 > 공기/액체매질을 통과하는 밀도파로 > 코르티기관에서 전기신호로 변환 > 뇌에 전달되며 정보 처리.
- (?)

# 음성합성(TTS)
- 음성 합성(speech synthesis) : 인위적으로 사람의 소리를 합성하여 만들어 내는 것. 텍스트를 음성으로 변환한다는 데서 TTS(text-to-speech)하고도 함.
- 라이브러리 : gtts(Google Text to Speech API), speech, sound, pyttsx3등의 라이브러리가 있음.
- 어텐션 역할 : 화자가 말 하는 법을 학습. "여기에 집중해라"를 학습하므로 특정 문자의 발음이나 말을 쉬는 시간등을 학습하게 됨. 일반화(학습하지 않은 문장도 합성)를 담당함.
- 푸리에 변환 : 시간이나 공간에 대한 함수를 시간 또는 공간 주파수 성분으로 분해하는 변환. 일종의 적분변환. 함수 x(t)가 복소수 범위에서 정의되어 있고 르베그 적분이 가능할 때, -∞∫∞ x(t)e^(-2πiξt). | t = 시간, ξ = 변환변수, 주파수.  
- 스펙트로그램(Spectrogram) : 소리나 파동을 시각화하여 파악하기 위한 도구로, 파형(시간에 따른 진폭의 변화)과 스펙트럼(주파수에 따른 진폭의 변화)의 특징이 조합되어 있음. 
  시간축과 주파수 축의 변화에 따라 진폭의 차이를 인쇄 농도/표시색상의 차이로 나타냄.
- 멜스펙트로그램(Mel-spectrogram) : 음성의 특징 추출 방법 중 하나. 주파수의 단위를 특정 공식에 따라 멜 단위로 바꾼 스펙트럼. 
  멜 스케일 -> 특정 pitch에서 발견된 사람의 음을 인지하는 기준을 반영한 scale변형함수, Mel(f) = 2595log(1+f/700) 
- STFT(Short Time Fouriter Transform) : 주파수 특성이 시간에 따라 달리지는 사운드를 분석하기 위한 방법. 시계열 일정한 시간구간으로 나눈 뒤, 각 구간에 대해 스펙트럼을 구함. 시간-주파수의 2차원데이터.

## 모델
### Tacotron
- tacotron1 : 딥러닝 기반 음성합성의 대표적 모델. attention + S2S 기반. 인코더, 디코더, 어텐션, 보코더(오디오생성)으로 나눌 수 있음. 이 후 오디오로 복원하기 위해 그리핀-림 알고리즘을 사용함.
  오디오 멜 스펙토그램을 학습해 유사한 음파를 합성해 마치 말하는 것과 같은 음성을 보여 줌. loss는 디코더 출력 멜스펙트로그램에 대한 loss와 postCBHG의 출력 스펙트로그램의 loss를 더한 것이 됨. 한명의 발화자를 완벽히 학습하기 위해선 스무시간 이상의 데이터가 필요함.
- 전처리 : 문자단위로 입력데이터를 나눠 정제(공백제외 비문자 제거)와 정수인코딩 과정을 거친 뒤 문자임베딩 층을 거쳐 임베딩 됨. 
- 인코더 : 1DCNN을 을 거친 데이터와 원래 데이터를 Residual연결해 Highway Layer를 통과시켜 BiRNN으로 특정을 추출. CBHG구조를 거친다고 함(문자단위 표현에 효과적인 구조라고 함).
  CBHG참고 : Character Embedding -> SingleLayer Convolution + Relu -> MaxPooling(Stride 5) -> Segment Embedding -> 4Layer Highway Network -> SingleLayer BiGRU
- 디코더 : 멜스펙토그램을 출력. reduction factor(하이퍼 파라미터)에 따라 몇개의 프레임을 출력할 지 결정. 출력은 다음 시퀀스의 입력이 되며, 초깃값은 <GO>(=0).
- 어텐션 : 디코더는 pre-net을 거쳐 어텐션의 키로 사용됨. 쿼리(인코더 출력)와 키를 구했으니 스코어 계산, value concat의 과정을 거침.
- 보코더 : concat한 벡터를 multi GRU에 입력해 멜-스펙트로그램을 출력하고, 이후 하나 더 있는 CBHG에 멜스펙트로그램을 입력해 스펙트로그램을 출력함. 이를 그리핀-림 알고리즘을 이용해 오디오신호로 복원. 

- [tacotron2](../PAPER/TTS/tacotron2.md)
- tacotron2 : 타코트론1과 가장 큰 차이점은 wavenet보코더의 유무(없어졌음).
- 차이점 : 인코더에서 FC-CBHG 구조가 아닌 1Dconv - Bi_zeroShot_LSTM 구조가 되었고, 어텐션에 location aware를 적용했으며, 디코더의 attentionRNN이 GRU에서 GRU에서 zoneout LSTM으로 변경되었고,
  어텐션 출력이 residual BIGRU -> 멜스펙트로그램 에서 단순히 linear net을 거쳐 멜스펙토그램과 stop토큰(audio가 있는 프레임이면 0, 없으면(패딩)1)을 만드는 것으로 바뀜. 
  멜스펙토그램은 다시 5개의 conv층을 가지고, residual처럼 linear 출력과 conv 출력을 더해서 최종 멜스펙트로그램을 만듦
- zoneout : 타코트론의 모든 LSTM에 적용. 현재 state에 이전 state를 뺌 -> 드롭아웃 -> 이전 state에 더해 새 state(cell/hidden 둘다 적용)제작. RNN에선 드롭아웃보다 효과적/큰 차이 없는데 느림 등의 말이 있음.
- zoneout 식 : (1-W) * dropout({h^t} - {h^t-1}, keep_prob=1-W) + {h^t-1}
- Location Sensitive Attention : 기존 어텐션과 달리 score계산에 이전 state가 들어가고, 일반적으로 softmax를 이용했던 것에 반해 smoothing을 사용. 이 과정에서 교사강요도 사용함. 
- smoothing : sigmoid(e_ij)/^LΣ_(j=1)sigmoid(e_ij). softmax의 지수(exponential)을 sigmoid로 바꾼 것.
- 디코더 : 층 구조만 바뀐 인코더와 달리, DecoderRNN부분이  linear projection(FC 1층)으로 바뀜. 어텐션과 어텐션LSTM을 연결한 벡터를 입력해 하나는 멜스펙트로그램을(activation 없음), 하나는 stop토큰을(sigmoid)만듦.
  stop토큰은 오디오가 있으면 0, 패딩이면 1이 되고(loss에 포함), 멜-스펙은 다시 5개의 conv층을 거치고, (residual처럼)출력과 층에 들어오기 전의 원본을 더해 최종 멜스펙토그램 출력을 만듦. 이후 보코더로 음성을 복원.

#### Multi-Speaker Tacotron
- Multi-Speaker(N-Speaker)Tacotron : DeepVoice2에서 제안된 타코트론1의 변형모델. N명의 목소리를 하나의 모델로 제작. 여러개의 목소리를 만들 때 메모리의 절약이 가능함. 데이터가 적은 목소리라도 Attention을 제대로 배워 좋은 성능을 끌어낼 수 있음.
- Speaker Embedding : 각 스피커의 정보가 담긴 임베딩. 타코트론의 중간중간에 들어감. CBHG의 Residual connection과 BiRNN, Pre-Net, DecoderRNN에 들어가게 됨.

### WaveNet
- WaveNet : 딥마인드에서 공개한 오디오 시그널 모델. TTS를 위한 종전의 방법(parametric TTS, concatenative TTS)들과는 달리 오디오의 waveforms자체를 모델링해 음성생성.
- 장점 : 한번 만든 모델에서 목소리를 바꾸어 오디오를 생성하거나, 음악등 사람의 목소리와는 다른 분야에도 활용 가능.

### WaveGlow
- WaveGlow : 멜스펙트로그램 컨디셔닝을 사용해 가우스 분포에서 오디오 샘플을 생성하는 흐름기반 생성모델. 훈련중 모델은 일련의 흐름을 통해 데이터세트 분포를 구형 가우스 분포로 변환하는 방법을 학습함.
- 구조 : 흐름의 한 단계는 invertible convolution과 attine coupling layer의 역할을 하는 수정된 WaveNet아키텍쳐로 구성됨. 추론중 네트워크가 반전되고, 가우스분포에서 오디오샘플이 생성됨.

## 모델 학습
- 데이터셋 : 일반적으로 공개되어 있는 음성데이터들의 품질은 좋지 않음(대본과 틀리다던가). 학습해 합성은 가능하나 굉장히 음질이 떨어짐. 틀린문장은 없는지 확인과 대본에 대한 검토가 충분히 필요하며,
  데이터들이 음성 톤이나 음량이 다른 부분이 있기에(따라서, 제대로 하려면 전문 성우가 필요), 데이터 생성 후 음성, 음량, 톤 등을 일정하게 맞춰줘야 함(효과 증대 위해).
  음성 데이터는 단순 음성만 있는게 아니라 대본: 음성 데이터 구성으로 만들어지며, 학습시 읽기 편하도록 음성/대본간 매칭되어있는 정보를 볼 수 있는 일종의 인덱스 파일을 만들어 줘야 함.
- 전처리 : 일종의 음성 normalize가 필요. 

#
***

# SpeechRecognition | STT
- SpeechRecognition : 파이썬 음성인식(STT) 라이브러리. WAV, AIFF, AIFF-C, FLAC 파일 형식을 지원.

- speech_recognition.Recognizer() : Recognizer 객체 생성. 여러 기업에서 제공하는 음성인식 기술 사용 가능. 
- speech_recognition.AudioData(byte객체) : byte객체를 오디오데이터로 만듦. sr에 사용가능.
- speech_recognition.AudioFile(파일경로/파일객체) : 오디오파일(소스)오픈. 문자열(경로)이거나 io.ByteIO(소스, 이와 비슷한것도 가능)여야 함.

- sr객체.record(오디오소스) : STT API를 사용할 수 있도록 sr객체에 오디오소스를 등록.   

- SR객체 음성인식 메서드 : sr객체.recognize_/google()/google_cloud()/bing()/houndify()/ibm()/wit()/sphinx().
  차례대로 {Google Web Speech API}, {Google Cloud Speech, google-cloud-speech 설치필요}, {Microsoft Bing Speech}, {SoundHound Houndify},
  {IBM Speech to Text}, {Wit.ai}, {CMU SPhinx, PocketSphinx 설치 필요}. sphinx()제외 모든 함수는 인테넷 연결이 되어야만 사용가능.
```python  
# 음성파일을 텍스트화
with sr.AudioFile(파일명) as source:
    audio = Recognizer객체.record(source)  # offset=i로 가져오기 시작할 초를, duration=i 로 가져올 초를 설정할 수 있음.
txt = recognizer객체.recognize_google(audio_data=audio, language='en-US')  # 구글 웹 API제외 키 등 필요. 언어는 '언어-국가'('ko-KR':한국어).
```

# 음성합성(TTS) | pyttsx3, gtts 
- pyttsx3 : TTS패키지. 기계적인 여성 목소리가 출력됨. 한국어 전용(영어는 발음이 이상함)이나 숫자와 고유명사는 읽을 수 있음. 
- engine = pyttsx3.init() : tts 엔진 생성.
- engine.say('좋은아침.') : 말 할 대사 지정.  
- engine.runAndWait() : 지정한 대사를 말함.

- gtts : 구글에서 제공하는 tts서비스. `pip install gTTS`로 설치 가능.
- tts = gtts.gTTS(text, lang=lang) : gTTS객체 생성 후 변환할 문자열과 언어를 지정함. 언어는 'en', 'ko'등임.
  en으로 지정시 한글이 포함되어 있으면 무시하고, ko로 지정하면 읽기는 하지만 꽤나 이상하다고 함. 영어는 여자 성우, 한글은 남자 성우.
- tts.save(path) : 변환된 음성을 파일로 저장함.
- tts.write_to_fp(f) : binary 파일객체에 오디오소스를 저장함.

# ClovaSpeechSynthesis API
- API : 입력된 텍스트를 RESTful API 방식으로 전달하면 서버에서 인식해 mp3 포맷의 스트리밍 데이터나 파일로 리턴해주는 API. 서비스 이용량 한도를 직접 조정할 수 있고, 1000글자당 4원의 비용이 청구됨. 사이트에서 신청 후 이용가능.
- 신청 사이트 : [네이버 클라우드 클랫폼 - CLOVA Speech Synthesis(CSS)](https://www.ncloud.com/product/aiService/css)
- 라이브러리 : urllib.request 라이브러리를 이용해 요청을 한 뒤 데이터를 받아와야 함.

- urllib.parse.quote(text) : sppech로 바꿀 text지정. 원 문자열은 UTF-8로 인코딩 되어있어야 하며, 반환값은 URL 인코딩으로 변환된 값임.
- data = "speaker=화자&speed=속도&text=" + encText : 화자와 속도, 텍스트 정의. 한국은 영문을 읽을 수 있지만(발음 안 좋음) 영어는 한글 입력시 오류가 나거나 읽지 못함. ','는 조금 쉬었다 말하고, '.\n'은 구분된 문장으로 합성됨.
  화자종류 - 한국 남여(jinho/mijin)|영어 남녀(matt/clara)|일본어 남여(shinji/yuri)|중국어 남여(liangliang/meimei)|스페인어 남여(jose/carmen). 속도는 -5~5로, -5면 1.5배, 5면 0.5배속으로 읽음.

- request = urllib.request.Request(url) : 요청을 전송할 request객체 생성. url은 "https://openapi.naver.com/v1/voice/tts.bin".
- request.add_header("X-Naver-Client-Id",client_id) : 클라이언트 id를 header에 추가함. 네이버 OpenAPI 신청시 주어지는 값으로 대체해야 함.
- request.add_header("X-Naver-Client-Secret",client_secret) : 클라이언트 Secret을 header에 추가함. 네이버 OpenAPI 신청시 주어지는 값으로 대체해야 함.
- response = urllib.request.urlopen(request, data=data.encode('utf-8')) : 요청을 보내고 반환값을 받아옴. 
- response.getcode() : 반환값의 코드를 실행. 이 결과값이 200이면 아래의 코드를 실행하고, 아니면 print("Error Code:" + rescode)식으로 예외코드를 작성하는 게 좋음.
- response.read() : 변환된 음성 데이터를 받아옴. 이것의 반환값은 open('*.mp3', 'wb').write(response_body) 로 음성파일로 저장해 쓰거나, 그대로 사용해도 됨. 

# pyaudio | 음성녹음
- portaudio library를 python을 이용하여 사용할 수 있도록 함.
  
- pyaudio.PyAudio() : PyAudio객체 생성.
  
- 객체.get_device_count() : 디바이스들의 인덱스 리스트 반환. 
- 객체.get_device_info_by_index(인덱스) : 디바이스의 정보를 가져옴. desc["name"\], desc["defaultSampleRate"\]등으로 정보 휙득.

- 객체.open(format, channels, rate, input, frames_per_buffer) : 레코드(스트림)객체 오픈. .read(CHUNK)로 소리를 읽을 수 있음. 
  각 인자들은 FORMAT(pyaudio.paint16, 데이터 저장이 어떤 방식(크기)로 될지 지정), CHANNELS(int, 보통 1), RATE(int, 샘플링레이트), 
  RECORD_SECONDS(int, 녹음할 시간. RATE/CHUNK*SEC 로 사용), INPUT(True, OUTPUT일 경우 재생용 스트림), CHUNK(int, 1024/버퍼당 프레임)를 값으로 받음.
  
- 스트림.read(CHUNK) : CHUNK만큼 소리 녹음. CHUNK개의 샘플(프레임)을 스트림(소리)에서 읽어옴.
```python 
# 소리 녹화
CHUNK = 1024                # 버퍼당 연산할 프레임(샘플)수.
FORMAT = pyaudio.paInt16    # 데이터 저장 포맷(16bit/24bit 등)지정.
CHANNELS = 1                # 녹음과정에서 소리가 녹음된 종류(방향)의 개수. 
RATE = 44100                # 1초당 샘플링할 횟수(가져올 샘플의 개수).
RECORD_SECONDS = 5          # 녹음할 시간.
WAVE_OUTPUT_FILENAME = "output.wav"  # 결과 wav 파일 이름.

p = pyaudio.PyAudio()       # pyAudio객체 생성.
# 사전에 설정해둔 값 대로 객체 설정 후 오픈.
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,  # 이 스트림이 입력 스트림인지(소리를 입력받는 용도인지)설정
                frames_per_buffer=CHUNK)

print("Start to record the audio.")

frames = []   # 오디오가 저장될 배열 선언

for i in range(int(RATE / CHUNK * RECORD_SECONDS)):   # 가져온 샘플의 개수를 한번에 읽을 양으로 나눈 뒤 초를 곱한(지정 초 동안 계산할 수)만큼 반복. 
    data = stream.read(CHUNK)                         # 청크 만큼 음성 데이터를 읽음
    frames.append(data)                               # 읽어온 음성 데이터를 저장
frames = b''.join(frames)

print("Recording is finished.")

stream.stop_stream()  # 녹음 종료(시작은 open/read)
stream.close()
```

# wave
- .wav 파일의 처리를 위한 패키지
  
- wave.open(파일명, 모드) : wave객체 오픈.
- wf(객체).setnchannels(CHANNELS) : 채널(녹음 과정에서 나눈 방향의 개수. 보통 1, 좌우음향 등이 2)설정. 
- wf.setsampwidth(오디오객체.get_sample_size(FORMAT)) : 샘플의 길이(한번 샘플링시 얼마나 샘플링 할지. 포맷으로 미리 정의됨)설정.
- wf.setframerate(RATE) : 프레임당 rate(샘플링 횟수) 설정.
- wf.writeframes(b''.join(녹화된 소리)) : 바이트단위로 바뀐 소리를 파일에 작성(저장).
```python 
# 녹화된 소리를 .wav파일로 저장
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
```

# PyDub | 오디오데이터 전처리
- pydub.AudioSegment.from_파일형식(오디오파일명) : 오디오 파일을 불러옴.
- 오디오파일 속성 : .channels(채널 수), .sample_width(샘플길이), .frame_rate(프레임율), .frame_width(프레임폭)
- 오디오파일.export(out_f=파일경로.파일형식, format=파일형식(wav)) : 불러온 다른 형식으로 변경.
- 오디오파일.split_to_채널(mono)() : 음성을 해당 채널로 쪼개 리스트로 반환. 원 채널은 오디오파일.channels로 볼 수 있음.

# 오디오 재생 | winsound, playsound
- winsound.PlaySound(파일명, winsound.SND_FILENAME) : 소리재생.
- playsound.playsound(path) : 오디오 파일 재생.

# librosa | 음성파일분석 
- librosa : 파이썬 음성파일분석 라이브러리. 사용시 ffmpeg의 설치가 필요함(음성파일로드).
- librosa.load(오디오파일경로) : 오디오파일 로드. 하나의 오디오파일로 반환받거나, y(음원 파형데이터)/sr(샘플링레이트, 주파수분석/파형의 간격)으로 나눠받을 수 있음.

- librosa.stft(y(파형), n_fft, win_length, hop_length) : Time도메인 파형을 Frequency도메인으로 변형시키는 푸리에변환. 
  전체파형을 대상으로 하면 제대로 분석이 불가하기에, 짧은 시간단위로 분리해 각 구간에 대해 변환. 
- librosa.power_to_db(np.abs(stft결과)) : 파워 스펙트로그램(stft)을 dB(decibel)유닛으로 변환. 절댓값을 취하면 역변환시 음원으로 재변형이 불가능해짐.
- librosa.feature.melspectrogrma(np.abs(stft결과), sr, n_mels, win_len, hop_len) : Mel스케일 변환을 통해 사람의 뒤에 맞춰진 스펙트로그램 생성.
- librosa.feature.mfcc(p2db결과, sr, n_mfcc) : Mel스펙트로그램에 DCT를 거쳐 나온 결과값. 압축된 정보를 담고 있으며, 과정에서 노이즈가 제거되는 효과가 있음.
  역변환시 원본 파형데이터 형태로도 연산이 가능해, 음성데이터 분석시 주로 사용.
- librosa.inverse.[mel/mfcc\]_to\_[stft/audio/mel/audio\] : 역변환. mel과 앞의 두개, mfcc와 뒤에 두개를 사용가능. 단계순서에 관계없이도 역변환이 가능.

- librosa.display.specshow(결과) : 스펙트로그램, chromagram, cqt, 그 외 여러가지를 display. plt로 show()해서 인자로 넣은 결과의 그래프를 볼 수 있음.
