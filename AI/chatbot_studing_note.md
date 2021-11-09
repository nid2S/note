# ChatBot
- 챗봇 : 음성이나 문자를 통한 인간과의 대화를 통해 특정 작업을 수행하도록 제작된 컴퓨터 프로그램. 이를 구현하기 위한 다양한 API가 존재함. 
- 동작방식 : 자연어처리를 심각하게 적용하는 언어 이해 방식, 입력에서 특정 단어/어구를 검출해 준비된 응담을 출력하는 검색방식, 각본을 미리 만든 뒤 그에 따르는 각본방식이 있음.

- 대화시스템(Dialog Systems) : 챗봇의 시스템. 챗봇과 혼용되기도 함.
- 기능수행 챗봇(Task-Oriented Dialog Systems) : 인공지능 스피커에서 주로 이용되는 시스템. 문장의 의도를 알아내는 Intent Classification과 
  구체적인 요청을 알아내는 Slot Filling으로 구성. 보통 NLU, DM, NLG 3개의 모듈 아키텍쳐로 구성됨. 
- NLU : 사용자의 발화를 이해하기 위한 모듈. intent(Search-Weather)와 Slot(location(Seoul), date(2020-05-17)등)을 분석함.
- Dialogue Manager(DM) : 사용자와 시스템이 주고받는 대화흐름을 관리. 크게 DialogStateTracker(대화추적관리, 대화진행중 데이터유지)와 
  PolicyManager(시스템의 액션결정. NLG나 DB로 연결)로 구성.
- NLG : DM으로부터 데이터를 입력받아 사람이 이해할 수 있는 시스템의 응답을 생성. 각 시스템의 페르소나를 결정할 수 있기에 매우 주의깊게 다뤄져야 함.

## BERT With ChatBot
- Span Prediction : 문장 전체의 벡터를 생성해, 각 Slot이 없는지/어떤 값이든 상관없는지/존재하는지 예측한 후 각 토큰의 벡터들을 이용해 Slot의 시작/종료점의 확률을 계산,
  이를 바탕으로 Slot을 추측.  
- Zero-shot Learning : 아예 처음보는 서비스나 Slot도 처리 가능. Slot의 설명이 주어진다면 언어처리능력으로 기존의 Slot과 비슷함을 인지, 처리.  
