# Coding Test
- 주의 : 침착, 문제 주의깊게 잘 읽기, 문제 이해(입출력, 조건)완료, 알고리즘 작성.
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










