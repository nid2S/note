# Text Summarizarion
- 텍스트요약 : 주어진 텍스트중 중요한 정보만 정제해내는 과정. 상대적으로 큰 원문을 핵심 내용만 간추려 상대적으로 작은 요약문으로 변환하는 것.
- 분류 : 기본적으로는 추출적/추상적 요약으로 나뉘나, 원문의 개수에 따라 single/multi document summarization, 생성해내는 텍스트 형태에 따라 keyword/sentence summarization, 
  요약 과정에서 원문 외 외부 정보의 사용빈도에 따라 knowlege-poor/rich summarization 등 다양한 구분이 있음.


- 추출적 요약(extractive summarization) : 원문에서 핵심문장(단어구)를 뽑아 이들로 구성된 요약문을 만드는 방법. 문서 내의 각 문장이 서머리에 사용될지 여부를 정하는 문제로 생각해 해결.
  - 텍스트랭크 : 페이지랭크 알고리즘을 기반으로한 추출적 텍스트 요약 알고리즘. 문서를 문장으로 나눈 후, 각 문장간 유사도를 구해 점수를 구하고, 상위 n개의 문장을 서머리로 사용함.
- 추상적 요약(abstractive summarization) : 원문에 없던 문장이라도 핵심 문맥을 반영한 새 문장을 생성해서 원문을 요약하는 방법. 대부분 딥러닝(NLG)을 통해 이뤄짐.

- Multi documents summarization(MDS) : 복수개의 문서를 요약하는 작업.
- Long documents summarization : 긴 텍스트를 입력으로 받기 위해 통계적방법으로 추출된 요약을 만든 후, 모델의 입력으로 사용.








# 참고
- [1](https://wikidocs.net/72820)
- [2](https://medium.com/@eyfydsyd97/bert%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%ED%85%8D%EC%8A%A4%ED%8A%B8-%EC%9A%94%EC%95%BD-text-summary-b582b5cc7d)
- [3](https://github.com/uoneway/Text-Summarization-Repo)
