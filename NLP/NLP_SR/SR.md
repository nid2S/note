# 음성인식
[음성인식](/../python%20practice/note/NLP_SR_studing_note.md)

- [채널\]로 녹음된 소리데이터에서 1초당 [샘플링레이트\] 만큼 샘플링(아날로그->디지털)을 진행 >> 
  버퍼당 [청크\]개의 프레임(샘플)을 처리하며 소리데이터를 지정된[포맷\]에 맞게 저장 / 소리 재생. 
  
- 스펙토그램을 입력으로 받아 > Invariant Convolution( / 피라미드 형태의 LSTM으로 인코딩 > 디코딩(다음 층으로 가는)과정에서 어텐션) > 
  Bi RNN/GRU > Fully Connected > CTC에서 디코딩 의 과정을 거침.
  
- (?)



