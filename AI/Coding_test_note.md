# Coding Test
- 주의 : 침착, 문제 주의깊게 잘 읽기, 문제 이해(입출력, 조건)완료, 알고리즘 작성.
- 주의 : 최대한 단순하게, 한번해 할 생각 말고 나눠서, 말렸다 싶으면 처음부터 다시 생각하는것도 방법임.
## algorithm
### Search
- 선형탐색 : 그냥 반복문을 돌면서 원하는 값이 나올때까지 찾음.
  - 구현 : 단순 for문
- 이분탐색(이진탐색, 파라메트릭 서치) : 절반씩 나누어 가며 내가 원하는 값과 더 가까운 쪽을 찾아감. 데이터가 정렬되어 있어야 하며 1억개의 요소가 있더라도 27번만에 찾을 수 있다는 장점이 있음.
  - 구현 : while end-start > 1: mid=(start+end)//2, array[mid] == trg: return mid, elif > target: end = mid, else: start = mid
- 완전탐색(브루트포스) : 주어진 범위 내의 모든 경우의 수를 다 탐색해보며 조건에 해당하는지 탐색하는 경우. 
- DFS(깊이우선탐색) : 갈 수 있는 끝까지 탐색해 리프노드 방문 후, 이전 갈림길에서 선택하지 않았던 노드를 방문하는 식. 한가지 정점과 연결된 모든 정점을 탐색해야 하는 경우나, 경로를 찾아야 하는 경우, 사이클이 존재하는 경로를 찾는 경우에 사용된다.  
  - 구현 : while stack등. 돌면서 가능한걸 전부 넣고 하나를 빼서 또 넣고 를 반복하면 결국 DFS가 됨.
- BFS(너비우선탐색) : 루트노드와 같은 거리의 노드를 우선으로 방문. 큐 등의 자료구조 사용으로 구현할 수 있으며(while 큐, while stack등), 리스트.pop(i)는 시간복잡도 O(N)이라 collections의 deque를 사용하면 효율적인 코드를 짤 수 있음. 최단거리를 구하는 경우에 사용됨.
  - 구현 : while queue등. 돌면서 가능한걸 전부 넣는건 동일하나 큐로 먼저 들어간게 먼저 나오니, BFS가 됨.
### else
- DP(동적 프로그래밍) : 기본적인 수학적 사고에 영향을 많이 받음. 완전탐색으로 풀었을 때 시간초과가 나는 경우 주로 이용. 일단 문제에서 구하라고 하는 부분이나 조건을 주로 배열로 선언함.
- 그리디 알고리즘 : 그때그때 상황에서 최적해가 전체 최적해가 된다는 원리를 이용한 방법.
- 투포인터 알고리즘 : 구간을 구하기 위해 사용하는 알고리즘. 완전탐색으로도 가능하나 효율성 측면에서 투포인터를 사용하는것이 좋음. start와 end포인터를 가지고 조건에 맞춰 크기를 키워나가거나 구간을 이동시킴. 대상 배열이 한개인 경우와 두개인 경우가 있음.

## Ques
- [2020 KAKAO 문자열압축](https://programmers.co.kr/learn/courses/30/lessons/60057#)
  - 나 : 1부터 문자열 길이까지(커널사이즈)돌면서 temp_s의 앞에서 i를 분리, 바로 뒤의 문자가 동일하지 않으면 넘기고, 아니면 동일하지 않을때까지 문자열을 제거.
  - 에러사항 : 문제의 정확한 조건을 신경쓰지 않고, 중복 문자의 수가 두자리 이상일 경우를 고려하지 않음. 
  - 타인 : [range(1, len//2+1)]+[len(text)] 의 리스트를 돌며, 문자를 range(0, len, tok_len)으로 돌며 token으로 나눔. words와 words + 1을 zip으로 묶어 돌리며, 
    [(토큰, 토큰이 중복되는 개수)]의 리스트를 제작. 이후 각 단어의 길이와 토큰개수(cnt>1이면 0)의 총합을 반환.
- [Roman to Interger](https://leetcode.com/problems/roman-to-integer/)
  - 나 : 각 로마자와 예외들을 딕셔너리로 만들어 합치고 제거.
  - 타인 : 로마자가 큰수 -> 작은수 임을 이용, 뒤에서부터 한글자씩 돌며 글자들을 치환했고, 
    만약 현재 바꾼 숫자 * 4 < answer라면 뺄셈, 그렇지 않다면 덧셈(원래는 작은거부터라 * 4를 해도 안되지만, IV등은 그렇지 않아 예외 원칙에 속한다는 뜻이니 감산)함.
- [palindrome-linked-list](https://leetcode.com/problems/palindrome-linked-list/submissions/)
  - 나 : linked_list의 각 요소들을 while node is not None 으로 리스트에 담아 list == list[::-1]을 반환.'
- [RansomNote](https://leetcode.com/problems/ransom-note/)
  - 나 : ransomNote를 set로 만들어 각 글자를 빼낸 후, ransomNote의 count가 magezine의 count보다 하나라도 많다면 False반환
  - 에러사항 : constructed를 구성하는이 아닌 포함하는 이라고 인식, in을 사용해 해결을 시도했음.
  - 타인 : 동일하지만 count문을 밖으로 빼 훨씬 빠른 시간을 달성. 아마 파이썬의 동작(호출)방식과 관련있는듯.
- [Fizz-Buzz](https://leetcode.com/problems/fizz-buzz/)
  - 나 : 1부터 n까지 돌며 3으로 나눠떨어지면 Fizz추가, 5면 Buzz추가, 비어있으면 str(i)추가 후 리스트에 결과 추가, 리스트 리턴.
- [Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/)
  - 나 : 각 노드들을 전체 리스트에 담으면서 len(node)//2를 반환.
  - 타인 : 일단 전체 길이를 구한 뒤, 절반으로 나누고, 해당 길이에 도달할때 까지 노드를 타고 들어가 노달하면 반환.
- [The K Weakest Rows in a Matrix](https://leetcode.com/problems/the-k-weakest-rows-in-a-matrix/)
  - 나 : 먼저 주어진 메트릭스를 (열합, idx)로 변환, 해당 리스트를 (열 합 오름 -> idx 오름)으로 정렬 후 idx만 리스트화, 마지막으로 리스트[:k]반환.
  - 타인 : 힙을 이용. (값, i)를 전부 넣고 k개를 빼서 반환. | 힙이란? -> 완전 이진트리의 일종으로 우선순위 큐에 주로 이용(O(logn)으로 가장 빠름). 큰게 상위에 있고 작은게 하위에 있다는 정도| 
- [Number of Steps to Reduce a Number to Zero](https://leetcode.com/problems/number-of-steps-to-reduce-a-number-to-zero/)
  - 나 : 먼저 num이 0이면 반환하게 한 뒤, num이 0이 아닐때까지 cnt += 1 + num % 2를 반환하게 하고 num //= 2를 적용해줌. 이후 cnt-1(필연적으로 1로 끝나게 되고, 이때 2를 추가하기 때문에)를 반환함.
  - 타인 : num & 1 == 0 이면 num >>= 1, 아니면 num ^= 1(not 1은 - 1의 효과가 있음(첫번째 제외 전부 반전이니))을 해줌(비트연산).
- [Richest Customer Wealth](https://leetcode.com/problems/richest-customer-wealth/)
  - 나 : 리스트 표현식을 사용, 각 row의 합을 리스트로 만든 뒤 max 출력.
  - 타인 : 전부 풀어 사용. | 사견 -> 아무래도 풀어쓰는게(리스트 표현식 같은거 X, 함수(함수(값)) X)훨씬 빠른 듯 함. 속도가 필요할때 참고하면 좋을거 같긴 한데 오류가 날 정도면 알고리즘과 시간복잡도의 문제라 코드의 깔끔도가 더 중요할듯.
- [Path With Minimum Effort](https://leetcode.com/problems/path-with-minimum-effort/)
  - 타인 : 이진 탐색을 이용, 전체 값의 중간값을 기준으로 시작해, 해당 중간값보다 적은 cost를 사용해 끝에 도달할 수 있다면 end를 mid로, 그렇지 않으면 start를 mid로 만들고, 다시 중간값을 구하는걸 반복, end와 start의 차이가 1보다 적어지면 end를 반환함.
- [Evaluate Division](https://leetcode.com/problems/evaluate-division/)
  - 타인 : DFS를 이용, 먼저 default_dict를 이용해 a/b와 b/a를 dict형태(d[a][b]=v)로 저장한 뒤, queue를 이용, 만약 a가 있다면 큐에 (a, 1)을 넣고, 
    큐에서 하나씩 뽑아 i가 b가 될 때 까지 i가 a인 모든 변수들을 한번씩 뽑아 큐에 넣고, i가 b와 같아지면 value를 찾아 리스트에 추가한 뒤 반환함.
    -> 결국 (A, X)일때, ?가 나온적 없는(A, ?)들을 찾고, 그걸 기반으로 (?, ??)인 것들을 찾으면서 value*해당value를 저장, 결국 (?, B)인걸 찾게 되면 해당것의 value를 반환하게 하며 만약 못찾으면(혹은 a가 없으면)-1로 하게 함.  
- [k진수에서 소수 개수 구하기](https://programmers.co.kr/learn/courses/30/lessons/92335)
  - 나 : 먼저 n을 pow(k, p)가 n보다 작은동안 p를 증가시킨 후 n//pow -> res += div -> n%pow 로 k진수로 바꾼 뒤, sqrt를 이용하는 방법으로 소수인지를 반별해 개수를 반환함.
  - 에러사항 : 일단 while n > pow(k, p)로 했었고, 진수 변환 단계에서 정확한 알고리즘의 생각 없이 코딩.
  - 타인 : 진수 변환 -> while n: `n%k -> n//k` 이후 s를 뒤집어서 반환. | 소수 판정 -> if n <= 1: False, while i**2 <= n: if n%i==0: False, i++ | 이외에는 동일함.
- [주차 요금 계산](https://programmers.co.kr/learn/courses/30/lessons/92341)
  - 나 : 일단 모든 시간을 기록 후, 해당 시간을 받고 문제에 맞게 주차 시간을 구해(먼저 기본금 + 추가 시간)정렬 후 반환함.
  - 에러사항 : 문제 이해를 성급히 했고(문제 연산과 조건 등), 모든걸 한번에 처리하려고 하였음.
  - 타인 : 따로 클래스를 만들어, defaultdict를 이용해 dict[car]에 계속 업데이트를 하게 하였고, 정렬 후 calc_fee를 이용해 반환하였음.
    update -> self.in_flag변수를 이용(IN이면 T,아니면 F)self.in_time에 시간을 기록하게 하고, 아니면 self.total에 누적 시간을 더함. | calc_fee -> 만약 self.in_flag가 T면 23.59로 update를 한 뒤, 조건에 따라 계산(math.ceil(올림)사용). | 결론 -> 클래스를 잘 사용함.    
- [양궁대회](https://programmers.co.kr/learn/courses/30/lessons/92342)
  - 타인 : 깊이 우선 탐색을 이용. 하나씩 화살을 쏘아 화살이 다 떨어지고 한바퀴 이상을 돌면 종료. 그 전까지는 화살을 한발씩 쏘아 화살이 다 떨어지면 점수를 구하고, 그게 끝나면 화살을 제거. for i in range(left, -1, -1): 라이언[idx]=i, dfs(idx-1, left-1, 라이언), 라이언[idx]=0 식.  
- [회의실 배정](https://www.acmicpc.net/problem/1931)
  - 나 : 회의들을 입력받아 끝나는 시간 -> 시작하는 시간 순으로 정렬 후 회의가 끝나면 카운트하는 식으로 계산.
  - 에러사항 : 시작시간 -> 끝나는 시간 순으로 정렬해 돌아갔음. 문제를 깊게 생각하고, 말렸다 싶으면 틀릴만한 곳을 생각하는게 중요해 보임.
- [숫자야구](https://www.acmicpc.net/problem/2503)
  - 나 : 일단 모든 경우의 수를 입력받은 뒤 123 부터 999까지 돌면서 중복되거나 0이 있으면 continue, 스트라이크 개수 비교 -> 볼 개수 비교 의 과정을 거쳐 가능한 경우의 수를 기록함.
  - 에러사항 : i의 변동을 눈치채지 못함. 코드의 동작을 주의 깊게 살필 필요가 있음. 추가로 중복과 0의 제거 등 문제의 사항을 주의깊게 살필 필요가 있음.
- [오픈채팅방](https://programmers.co.kr/learn/courses/30/lessons/42888)
  - 나 : 일단 레코드를 돌면서 nickname_dict에 각 닉네임들을 저장, 이후 다시 돌면서 act에 맞는 대사와 닉네임을 결합.
  - 타인 : 로직은 동일하나 훨씬 더 간결하게 코드 작성.



