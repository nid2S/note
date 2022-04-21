# Tacotron2
- 특징 : Attention기반 S2S TTS모델구조. <문장, 음성>쌍으로 이뤄진 데이터만으로도 별도의 작업없이 학습가능한 End-to-End모델. 합성품질이 뛰어남.
- 학습단계 : 텍스트를 받아 음성을 합성. 텍스트로부터 바로 음성을 생성하긴 어려우므로, 텍스트로부터 Mel-spectrogram을 생성하고(Task1) -> Mel-spectrogram으로부터 음성을 합성(Task2)함.
- Task1 : 텍스트로부터 멜스펙트로그램을 생성. S2S 딥러닝 구조의 타코트론2 모델이 담당.
- Task2 : 멜스펙트로그램으로부터 음성을 합성. Vocoder로 불리며, WaveNet 모델을 변형해 사용.
- 학습 : teacher-forcing을 사용. 학습단계에서는 생성된 것이 아닌 ground-truth mel-spectrogram과 ground-truth waveform을 이용. 
- 평가 : 한사람의 음성을 담은 데이터셋을 이용, 피실험자에게 음성을 들려주고 점수를 매개게 하는 MOS(mean opinion score) 테스트를 진행.
  linguastic feature를 이용하여 음성을 생성하는 WaveNet, Concatenative 모델, 타코트론1, Parametric 모델을 학습하여 비교모델로 활용했음.

## Tacotron2 model | S2S
- 입력 : character
- 출력 : mel-spectrogram
- 모델 구조 : 크게 Encoder, Attention, Decoder.

- 전처리 : 텍스트와 음성 쌍으로 되어있는 데이터를 input(문자)과 output(Mel-spectrogram)으로 변환함.
- text : 문자단위로 변경. 알파벳등은 문자단위로 바꾸면 되고, 한글의 경우 자모단위로 분리함. 이때, 초성과 종성이 동일하게 표기될 수 있어 구분해 사용함.
- audio : 음성 데이터로부터 멜스펙트로그램을 추출하기 위해서는 세가지 작업을 거쳐야 함. STFT(Short-Time Fourier Transform) -> Mel scaling -> Log transform

- STFT : 오디오 데이터에 섞여있는 여러개의 오디오(frequency)를 분리해 표시하기 위해 푸리에 변환(Fourier Transform)을 활용함. 
  단, 모든 데이터에 푸리에 변환일 적용하면 시간에 따른 오디오변화를 반영할 수 없으므로 sliding window를 이용해 오디오를 특정 길이로 잘라 푸리에 변환을 적용함. 이 결과물을 spectrogram이라고 함.
- Mel scaling : 스펙트로그램에 mel-filter bank라는 비선형함수를 적용해 저주파(low frequency)영역을 확대하는 작업. 
  사람의 귀가 고주파보다 저주파에 민감하므로, 저주파를 확대하고 고주파를 축소해 사용함. 더 명료한 음성의 생성을 위해 feature를 사람이 쉽게 인지가능한 scale로 변환하는 작업.
- Log transform : log를 취함. 진폭(amplitude)영역에서의 log scaling. 결과값은 mel-spectrogram. 모델의 output으로 활용됨.

- Encoder : character를 일정 길이의 벡터(context vector)로 변환. 캐릭터 임베딩, 3convLayer, BiLSTM으로 구성.
- 인코더 자세히 : character단위의 ont-hot vector를 encoded feature로 변환. 
  ont-vector가 입력되면 512차원의 임베딩 벡터로 변환되고, 임베딩 벡터는 3개의 ConvLayer(1D Conv + batchNorm)를 지나 BiLSTM Layer로 들아가 encoded feature로 변환됨.

- Attention : Encoder에서 생성된 일정 길이의 벡터로부터 시간순서에 맞게 정보를 추출해 Decoder에 전달함. 생성된 featur와 Decoder LSTM의 전 시점에서 생성된 feature를 이용해 Encoder로부터 어떤 정보를 가져올지 정렬하는 과정.
- 어텐션 자세히 : LocalSensitiveAttention을 사용하는데, 이는 Additive(Bandau)Attention에 attention alignment정보를 추가한 형태임. 
  k개의 filter를 가지고있는 1D Conv를 이용해 AttentionAlignment를 확장해 f_i matrix를 생성함. 이후 학습가능한 U와 내적한 뒤 Addictive attention의 구성에 포함해 계산함.

- Decoder : 얻은 정보를 이용해 mel-spectrogram을 생성. Pre-Net, Decoder LSTM, Projection Layer, Post-Net으로 구성. 
- Pre-Net : 2개의 FC-Layer(256dim) + ReLU로 구성. 병목구간으로써, 이전 시점에서 생성된 멜스펙트로그램이 입력으로 들어오면 중요 정보를 거르는 역할을 함.
- DecoderLSTM : AttentionLayer의 정보와 Pre-Net으로부터 생성된 정보를 이용해 특정시점에 해당하는 정보를 생성함. 생성된 매 시점의 벡터는 두개로 분기되어 처리됨.
- 분기 1 : 종료 조건의 확률을 계산하는 경로. 생성된 벡터를 FC Layer를 통과시킨 후 sigmoid함수를 취해 0~1의 확률으로 변환하며, 이게 설정한 threshold를 넘으면 inference단계에서 멜스펙트로그램 생성을 멈추는 역할을 함.
- 분기 2 : 멜스펙트로그램을 생성하는 경로. 생성된 벡터와 Attention에서 생성된 context vector를 합친 뒤 FC Layer를 통과, 생성된 mel-vector는 inference단계에서 다음시점 디코더의 입력이 됨.
- Post-Net : 5개의 1D ConvLayer로 구성. ConvLayer는 512개의 filter와 5*1의 커널사이즈를 가지고 있으며, 전 단계에서 생성된 mel-vector는 Post-Net을 통과한 뒤 다시 mel-vector화되는 구조(Residual Connection)로 되어있음.
  mel-vector를 보정하는 역할을 하며, 최종 결과물인 mel-spectrogram의 품질을 높이는 역할을 함.

- Loss : 생성된 Mel-spectrogram과 실제 melspectrogram의 MSE를 이용해 모델을 학습함. 

## Vocoder | WaveNet
- Vocoder : 멜스펙트로그램으로부터 음성(waveform)을 생성하는 모듈. WaveNet의 구조를 조금 변경한 모델을 Vocoder로 사용. 
- WaveNet 차이점 : 원래 WaveNet모델은 softmax함수를 이용해 매 시점 -2^15 ~ 2^15 + 1 사이의 숫자가 나올 확률을 추출하고 waveform을 생성하나, 
  이를 수정해 MoL(mixture of logistic distribution)을 이용하여 매 시점 -2^15 ~ 2^15 + 1 사이의 숫자가 나올 확률을 생성함. 

- 구조 : 여러개의 Residual block과 FC Layer로 구성. 멜스펙트로그램을 입력받아 MOL parameter와 MOL probability를 출력하고, 여기서 Wave를 생성함.

- Loss : 생성된 waveform과 실제 waveform의 시점 별 Negative log-likelihood Loss를 이용하여 모델을 학습함.
















# REFERRENCE
- [논문](https://arxiv.org/abs/1712.05884v2)
- [논문 리뷰](https://joungheekim.github.io/2020/10/08/paper-review/)
- [구현체1](https://github.com/NVIDIA/tacotron2)
- [구현체2](https://github.com/BogiHsu/Tacotron2-PyTorch/blob/master/model/model.py)