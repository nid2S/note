# have to study
> RNN -> Conv1d -> Attention -> transformer -> transferlearning

> -데이터 전처리
>     - bpe
>     - 형태소 분석기(konlpy, mecab)
>     .... 등

> variation of RNN
>      - RNN
>     -GRU
>     -LSTM

> variation of transformer
> encoder
>     - Bert
> decoder 
>     -gpt
> Bert + gpt -> Bart(generator)

> 그리고 2번째는 mnist-> 영화 감성 분류 (IMDB) -> 캐글 재난 분류(이건 바뀔수도)

#soynlp
***
soynlp.Tokenizer.RegexTokenizer(String) > 문자열을 단어, 문장부호 등으로 일일이 나눠 리스트로 반환해준다. Regex 는 정규 표현식이라는 뜻.
soynlp.noun.LRNounExtractor(verbose(bool)).train(단어리스트).extract() > 사용 빈도대로 워드 클라우드를 만들 수 있는 단어 배열 반환.



