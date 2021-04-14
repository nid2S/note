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

# 자연어 처리
- 자연어와 컴퓨터간 상호작용에 대함. 인공지능과 컴퓨터 언어학의 주요 분야중 하나.
###### 토큰화 
- 텍스트를 토큰이라는 작은 부분으로 분할하는 과정.
- 문장 토큰화와 단어 토큰화로 나뉜다.
- 단어로 매칭하거나 공백으로 매칭하는 두가지 방법의 정규표현식 방식으로도 단어 토큰화를 수행할 수 있다.
###### 정규화
- 문장부호 제거, 전체 대/소문자화, 숫자 단어화, 약어 전개, 텍스트 정규화 등 자연어 텍스트 처리 수행을 위한 과정.
- 문장부호 제거는 반복문이나 기타 클래스/메서드로, 대/소문자화는 String.upper()/lower()로 가능하다.
###### 불용어 처리
- 문장의 전체적 의미에 크게 기여하지 않는 불용어를 검색 공간 줄이기 등의 이유로 제거하는 과정.
###### 텍스트 대체  
- 축약이나 줄임말 등을 본래대로 풀어 처리를 효과적이게 하는 과정.
- 정규 표현식 이용 텍스트 대체:  import re >  [(r'won\'t', 'will not'), (바꿀단어, 바꿜 단어)]식으로 제작 > re.compile(바꿀 단어) > re.subn(컴파일, 바뀔 단어, text)[0] 의 과정을 거쳐 할 수 있음.
###### 반복 문자 처리
- 무의미하고 오류를 일으키는 반복문자를 일반 문자로 변환.
- 반복문자를 포함하는 단어를 역참조 방식을 사용해 제거.
###### 단어 동의어 대체
- 더 훌륭한 성능과 적은 오류를 위해 같은 의미의 단어를 하나로 변환.
###### 유사척도
- 지프의 법칙 : 토큰이 언어로 배포되는 방법을 설명. 토큰 빈도가 정렬된 목록의 순위와 정비례하게 함. 
- 편집 거리 : 두 문자열을 동일하게 하려면 삽입,대체,삭제해야 하는 문자의 수를 계산. 
- 자카드 계수(타니모토 계수) : 두 세트의 곱집합 / 합집합. 동일하면 1, 완전 다르면 0이다.
- 스미스 워터맨 거리 : 편집거리와 유사. 관련된 단백질서열 및 광학정렬을 검출하기 위해 개발됨.
- 이진 거리 : 문자열 유사도 메트릭. 두 라벨이 동일하면 0, 다르면 1을 반환.
- 매시 거리 : 부분 일치에 기초. 1 - (곱집합 길이/합집합 길이)*(두 세트의 길이차에 따른 점수(1, 0.67, 0.33, 0))
- 유클리디안 거리 : 단어간의 거리가 짧으면 더욱 유사. 벡터 공간에서 유사도 측정.
- 코사인 유사도 : 두개의 문자열을 축으로 사용하여, 그 문자열들과의 유사도를 측정. 축과 문장간의 각도를 이용해 유사도를 측정. 둘의 곱/둘의 거리
- 문서의 유사도 구하기 : bag of ward 에 코사인 유사도를 적용해(bow 각 요소들의 곱의 합/bow 합의 제곱근) 
- 문서의 유사도 구하기 : BOW 에 TF-IDF 를 적용하는 방법으로도 구할 수 있음. 구현이 쉽고 불용어를 잘 거른다는 장점이 있지만, 단어단위로 보기에 동음이의어를 잡지 못하고 토픽은 알 수 없다는 단점이 있음.
- LSA(latent Semantic Analysis, 잠재 의미 분석) : 단어를 행, 문장을 열로 나타낸 뒤, SVD 를 사용해 세개의 매트릭스(토픽을 위한 단어, 토픽 강도, 문장)로 나타내어 뒤의 두 매트릭스를 곱해 단어간 유사도를 파악.
- SVD(특이값 분해) : m*n 차원의 행렬 A 를 UΣV^T 로 분해하는 행렬분해 방법 중 하나. U,V 는 orthogonal matrix(직교,UU^T = I, U^-1 = U^T) , Σ는 diagonal matrix(대각제외 모두 0.)이다.  
- Word Embeddings(Word2Vec,Glove) : 단어의 유사도
- ConceptNet : 지식 그래프 사용. 단어간의 관계를 정의해 유사도 측정.
###### 단어 빈도 (ngrams)
- 연어 : 함께하는 경향이 있는 둘 이상의 토큰의 집합.
- n-그램 : 연속적으로 n개의 토큰이 모인것. 한개면 유니그램, 두개면 바이그램등으로 지칭. bag of words 의 단점 보완 가능.
###### 문장 벡터화
- bag of words : 문장을 숫자로 표현하는 방법중 하나. 단어의 출연 빈도로 문장을 나타낸다. 문장의 유사도파악니나 모델의 입력으로 사용된다.
- bag of words 단점 : Sparsity(단어의 개수가 많다보니 0이 많아 계산량이 많아짐), 흔한 단어의 힘이 세짐, 단어의 순서를 완전 무시, 처음 보는 단어는 처리 불가 등의 단점이 있다.
###### 중요도 측정
- TF-IDF(term frequency inverse doc freq) : 문서의 각 단어별 문서 연관성(문서에서 가진 정보)을 파악함. 단어의 출현빈도와 중요도가 비례할 것이라는 가설에 기초함.  
- IDF : 자주 등장하는 단어(불용어)에 패널티를 주어 위의 것을 보완한 방법. log(총 문장 개수/단어 출현 문장 개수(+1, DIV_0를 피하기 위해)) 의 공식을 사용함.




# NLTK
- nltk.download() : NLTK 세트 다운로드. 특정 세트의 이름을 넣으면 그것만 다운로드한다.

### tokenize
- nltk.tokenize.sent_tokenize(text) : 문장 토큰화함수. 문서를 문장단위로 나눠준다. PunktSentenceTokenizer 인스턴스(문장의 시작과 끝을 표시하는 문자나 문장 기호에 기초해 다른 유럽언어로 토큰화를 수행)를 사용한다. 
- 영어가 아닌 언어를 토큰화 하려면 'tokenizer/punkt/언어.pickle' 파일을 로드하고 사용하면 된다.  로드한 언어.pickle 에  .tokenize(text) 매서드를 사용해서도 토큰화를 사용할 수 있다.
  
- nltk.word_tokenize(sentence) : 단어 토큰화 함수. 문장을 단어 단위로 나눠준다. TreebankWordTokenizer 를 사용한다. 펜트리뱅크 코퍼스에 따른 기준을 따른다.
- nltk.tokenize.TreebankWordTokenizer() : 트리뱅크워드 토큰화 함수 로드. .tokenize(Sentence) 로 토큰화를 수행할 수 있다. 분리된 축약형('Do', 'n't')으로 작동된다. 
- nltk.tokenize.WordPunctTokenizer() : 또 다른 단어 토크나이저 로드. 분리된 문장부호(문장부호를 완전 새 토큰으로 분할해 제공)로 작동한다. 각단어가 완전 새 토큰을 생성하는 대신 유지된다. 텍스트를 알파벳과 아닌 문자로 토큰화한다.

- nltk.tokenizer.RegexpTokenizer(정규 표현식) : 정규 표현식을 이용한 단어 토큰화 클래스 로드. .tokenize(String)으로 사용할 수 있다. gaps = True 로 화이트 스페이스를 사용한 토큰화를 할 수 있다. 
- nltk.tokenizer.regexp_tokenize(sentence, patten='정규 표현식') : 정규 표현식 단어 토큰화 함수 로드. 

- nltk.tokenizer.BlanklineTokenizer() : 공백으로 단어 토큰화. 이미 정의된 정규 표현식을 사용하는 것. .split('\n')과 유사하게 사용됨.
- nltk.tokenizer.WhitespaceTokenizer() : 화이트 스페이스-탭 으로 단어 토큰화. 토크나이저.span_tokenize(sent)로 토큰의 오프셋인 튜플의 순서를 받을 수 있다. 
- nltk.tokenizer.SpaceTokenizer() : 스페이스로 토큰화.  .split()과 유사하게 작동됨.  spans_to_relative(토크나이저.span_tokenize(sent))식으로 스판의 순서를 주면 상대 스판의 순서를 반환한다. 
- nltk.tokenizer.util.string_span_tokenize(문자열, "separator(구분자)") : 구분자대로 분할해 전송된 토큰의 오프셋을 반환.

### corpus
- nltk.corpus.stopwords() : 불용어 처리 클래스 로드. .words('언어')로 불용어 목록을 받아올 수 있으며(set 로 묶어주는게 좋고, not in 으로 거를 수 있음) .fileids()로 불용어 목록이 있는 언어를 확인할 수 있다.
- nltk.corpus.wordnet() : 유의어들의 목록 로드. 유사한 단어와 각 단어의 유사도를 파악 가능. synsets(word).definition()으로 단어의 유사어 확인 가능, .path_similarity(synsets)로 단어의 유사도 확인 가능. 
- nltk.corpus.alpino() : 알피노 코퍼스(네덜란드 신문에 나오는 단어 모음) 로드. .word()로 내부의 단어들을 꺼낼 수 있다. 다른 것들도 사용가능.

### metrics
- nltk.metrics.accuracy(sentence1, sentence2) : 두 토큰화된 단어 리스트의 정확도(같은 정도)반환.
- nltk.metrics.precision(sentence1, sentence2) : 두 토큰화된 단어 리스트의 정밀도(TP/(TP+FP))반환.
- nltk.metrics.recall(sentence1, sentence2) : 두 토큰화된 단어 리스트의 재현율(TP/(TP+FN))반환.
- nltk.metrics.f_measure(sentence1, sentence2) : 두 토큰화된 단어 리스트의 f1점수(정밀도와 재현율의 조화 평균(역수의 평균의 역수,곱/합))반환.

- nltk.metrics.edit_distance(word1, word2) : 두 단어간 편집거리 반환.
- nltk.metrics.jaccard_distance(set1, set2) : 두 세트간 자카드 계수 반환.
- nltk.metrics.binary_distance(set1, set2) : 두 세트간 이진거리 계수 반환.
- nltk.metrics.masi_distance(set1, set2) : 두 세트간 매시거리 계수 반환.

### ngrams
- nltk.util.ngrams(단어 리스트, n) : n개의 토큰이 연결되어 있는 n그램 생성. 
- nltk.collocations.BigramCollocationFinder : 바이그램 탐색기. .from_words(tokens)로 토큰을 저장할 수 있다.
- 토큰저장탐색기.nbest(nltk.metrics.BigramAssocMeasures.likelihood_ratio, n) : n개의 바이그램을 찾아 리스트를 받아볼 수 있다.
- 토큰저장탐색기.score_ngrams(nltk.collocations.BigramAssocMeasures().raw_freq) : 바이그램을 찾는 또다른 방법.
- nltk.probability.LidstoneProbDist(fd, gamma=f, bins = n) = 최대 우도 추정 사용. fd(빈도분포)를 기반으로 f(0~1)를 사용해 n개의 샘플을 생성. 샘플들의 총 합은 1. 

# re
- 정규 표현식 사용을 지원.
- 정규표현식
> - [] : []안의 문자들과 매치. [abc]면 a,b,c중 하나와 매치를 뜻함. [a-c],[a-zA-Z]식으로 범위를 지정할 수 도 있음.
> - 패턴 앞에 ^를 붙이면 부정, \d는 숫자, \D는 숫자가 아닌것, \s는 whitespace( \t\n\r\f\v), \w는 문자+숫자, \W는 문자+숫자의 부정이다.
> - . 는 모든 문자를 뜻하고, .와 매칭되게 하고 싶을 때에는 [.]로 쓸 수 있으며, ? 는 있어도 되고 없어도 된다로 사용된다.  
> - *는 0번 포함 반복(앞의 문자가 몇번 나와도 매치), +는 한번 이상 반복. {n,m}은 n번 부터 m이하 반복, {n}은 반드시 n번 반복이다.
- re.compile() : 정규 표현식 컴파일. 결과 객체 반환. re.S(.이 \n을 포함하게 함) 등을 매개변수로 줄 수도 있다.
- re.sub(패턴, 바꿀문자, 바뀔 문장) : 바뀔 문장에서 패턴에 일치하는 부분을 바꿀 문자로 바꾼다.
- re.match(패턴, 문자열) : 컴파일을 거치지 않고 매치 사용. .group()으로 매치된 문자열을 볼 수 있다 
- 컴파일.match(word) : 단어가 표현식에 맞는지 반환. 없으면 None, 있으면 매치 객체를 반환. 문장의 경우 첫 단어만 판단.
- 컴파일.search(sent) : match 와 동일하지만 문장의 경우 문장 전체를 검색한다. .start()로 시작 위치, .end()로 끝 위치, .span()으로 (시작, 끝)을 받을 수 있다.
- 컴파일.findall(sent) : 정규 표현식에 맞는 단어만 리스트로 반환. 
- 컴파일.finditer(sent) : 정규 표현식에 맞는 단어만 각각을 매치 객체의 형태로 하여 반복 가능한 개채의 형태로 돌려준다. 

# replacers
- replacers.RegexpReplacer() : 텍스트 대체 클래스 로드. .replace(text)로 사용, 축약을 해제하고 단어 토큰화까지 진행해 리스트로 반환.
- replacers.RepeatReplacer() : 반복 문자 삭제 클래스 로드. 위와 동일하게 사용할 수 있으며, 반복된 단어를 일반 단어로 바꿔 반환한다. nltk 의 wordnet.synsets(word)에 이미 있다면 처리하지 않도록 하면 일반 단어는 반복을 삭제하지 않는다.
- replacers.WordReplacer({'바꿀단어':'바뀔단어'}) : 단어를 동의어로 변환하는 클래스 로드. 마찬가지로 사용, 목록에 있는 단어는 바꿔서, 아니면 그대로 반환한다.

# sklearn
- sklearn.feature_extraction.text.CountVectorizer() : BOW 표현을 하게 해주는 변환기 로드. .fit(문자열이 담긴 리스트)로 사용, .vocabulary_ 속성에서 반환된 {단어:등장횟수} 형태의 딕셔너리를 볼 수 있음.  tf-idf 와 함께 ngram_range=(연속 토큰 최소길이, 최대길이) 로 연속된 토큰을 고려할 수 있다. 보통은 하나만 하지만 많을 때 바이그램정도로 추가하면 도움이 된다.  
- Bow 표현을 만드려면 .transform(list), Scipy 희소 행렬로 저장되어 있으며, .get_feature_names()로 각 특성에 해당하는 단어들을 볼 수 있음. min_df 매개변수로 토큰이 나타날 최소 문서 개수를 지정할 수 있고, max_df 매개변수로 자주 나타나는 단어를 제거할 수 있다. stop_words 매개변수에 "english" 를 넣으면 내장된 불용어를 사용한다.
- sklearn.feature_extraction.text.TfidVectorizer(min_df=i) : 텍스트 데이터를 입력받아 BOW 특성 추출과 tf-idf 를 실행하고 L2정규화(스케일 조정)까지 적용하는 모델로드. 훈련데이터의 통걔적 속성을 사용하므로 파이프 라인을 이용한 그리드 서치를 해 주어야 한다. .idf_ 에서 훈련세트의 idf 값을 볼 수 있다. idf 값이 낮으면 자주 나타나 덜 중요하다 생각되는 것이다.

# spacy
- 영어와 독일어를 지원하는 NLP 파이썬 패키지. 표제어 추출 방식이 구현되어 있음. 
- python -m spacy download en 으로 언어의 모델을 먼저 다운받아야 함.
- spacy.load('en') : spacy 의 영어 모델 로드. 
- 모델(document) : 문서 토큰화. 찾은 표제어들 반환.

# nltk
- 포터 어간 추출기가 구현되어 있음.
- nlty.stem.PorterStemmer() : PorterStemmer 객체 생성.
- 객체.stem(토큰.norm_.lower()) > 토큰(어간) 찾기.

# KoNLpy
- 한글 분석을 가능하게 함. 자바로 이뤄져 있어 JDK 1.7 이상과 JPype 가 설치되어 있어야 한다.
- konlpy.tag.Okt() >  Okt 클래스 객체 생성. .morphs(text)로 형태소 분석이 가능하다.


