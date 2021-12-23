# 음성인식
[음성인식](../../Git_project/NLP/NLP_SR/SpeechRecognition.py)
- 음성인식 : 음성을 듣고 이해하는 능력. 사람이 내는 소리를 짧은 단위(음소)로 추출하고, 이를 조합해 소리의 의미/의도를 파악하는 과정.
- 사람의 음성 인식 : 정보를 음파형태로 받아 > 공기/액체매질을 통과하는 밀도파로 > 코르티기관에서 전기신호로 변환 > 뇌에 전달되며 정보 처리.
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
- TTS 관련 python 패키지 : pyttsx3, gtts, speech, sound

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
- 2DCNN : 주로 사용. 타임과 주파수 영역 모두를 이용해 Conv층을 쌓음. 시감/주파수 두 영역에서 패턴을 찾음.
- 1DCNN : 시간화 특화된 값을 뽑아낼 때 사용. NLP에서의 CNN과 비슷하게, 커널의 너비를 frequncy로 고정하고 Time에 따라 움직이며 압축을 수행함.
- Sample CNN : 화자인식등 페이즈(음색등의 정보)영역이 필요한 모델에서 사용. CNN사용시 row한 오디오 인풋을 어느정도 패딩간격을 두며 사용.
  Sampling rate(1초당 가져오는 소리정보)가 8000/16000/44100정도로 커지며 샘플레벨에서 CNN을 돌리기엔 필요한 연산 과정이 너무 커져 사용. 

## RNN in Audio
- 입력의 크기(길이)가 어떻든 커버할 수 있고, 모델의 크기가 증가하지 않기에 사용됨.
- 과거의 정보(Historical Infomation)를 잘 활용하고, 시간축에 따름 가중치 공유가 진행됨.

# 음성합성(TTS)
- 음성 합성(speech synthesis) : 인위적으로 사람의 소리를 합성하여 만들어 내는 것. 텍스트를 음성으로 변환한다는 데서 TTS(text-to-speech)하고도 함.
- Tacotron : (?). 음성 합성에 쓰이는 모델. 



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

# pyttsx3 | 음성합성(TTS)
- 이 외에도 gtts, speech, sound등의 TTS 패키지가 있음.
```python  
# TTS 패키지 pyttsx3 사용 예
import pyttsx3
engine = pyttsx3.init()
engine.say(str('Good morning.'))
engine.runAndWait()
```

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

# winsound | 오디오 재생(윈도우)
- winsound.PlaySound(파일명, winsound.SND_FILENAME) : 소리재생

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
