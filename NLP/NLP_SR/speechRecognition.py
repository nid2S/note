import pyaudio
import wave
import winsound
import matplotlib.pyplot as plt
import librosa.display
import speech_recognition as sr
import pyttsx3
# 음성 녹음 > 저장 > 확인 > 인식(STT) > 음성합성(TTS)

CHUNK = 1024                    # 버퍼당 프레임(한번에 처리할 샘플)
FORMAT = pyaudio.paInt16        # 데이터 저장 포맷(몇바이트로 저장할지 등의 방식)
CHANNELS = 1                    # 녹음과정에서 소리가 녹음된 방향의 수.
RATE = 44100                    # 샘플링 레이트(원본 소리 데이터에서 초당 샘플링(아날로그 -> 디지털)할 횟수)
RECORD_SECONDS = 3              # 녹음할 초
WAVE_OUTPUT_FILENAME = "D:/workspace/Git_project/NLP/NLP_SR/Audio/output.wav"     # 녹음된 소리가 저장될 파일 명

# 소리 녹음
p = pyaudio.PyAudio()               # PyAudio객체 생성
stream = p.open(format=FORMAT,      # 설정한 정보대로 스트림 오픈
                channels=CHANNELS,
                rate=RATE,
                input=True,         # 입력스트림(소리를 입력 받는 용도인지)설정
                frames_per_buffer=CHUNK)
print("녹음시작")
frames = []
for i in range(int(RATE / CHUNK * RECORD_SECONDS)):     # (RATE/CHUNK) * RECORD_SECONDS 만큼 반복
    data = stream.read(CHUNK)   # 한번에 청크 만큼 소리 저장
    frames.append(data)
print("녹음종료")
stream.stop_stream()
stream.close()
p.terminate()


# 녹음된 소리 저장
with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))


# 음성 확인
print("녹음된 소리를 출력합니다.")
winsound.PlaySound(WAVE_OUTPUT_FILENAME, winsound.SND_FILENAME)     # 음성 재생
plt.style.use("seaborn-white")
fig = plt.figure(figsize=(14, 4))
wav, rate = librosa.core.load(WAVE_OUTPUT_FILENAME)
librosa.display.waveplot(wav, sr=rate)                              # 음성 그래프화
plt.show()


# 음성인식(STT)
txt = ""
recognizer = sr.Recognizer()
with sr.AudioFile(WAVE_OUTPUT_FILENAME) as source:
    audio = recognizer.record(source)                                # 음성파일을 읽어옴.

try:  # 구글 웹 API이용 한국어로 STT.  음성이 없으면 오류.
    txt = recognizer.recognize_google(audio_data=audio, language='ko-KR')
except sr.UnknownValueError:
    print("언어가 인지되지 않았습니다.")
    exit(1)
print(txt)

# 음성합성(TTS)
engine = pyttsx3.init()
engine.say(txt)
engine.runAndWait()
