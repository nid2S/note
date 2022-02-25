- [Q](https://www.acmicpc.net/problem/1463)
```python
def resolve(num, cnt = 0, answer = 99999):
    if num == 1:
        if cnt < answer:
            return cnt
        else:
            return answer
    
    cnt += 1
    if cnt >= answer:
        return answer
    
    if num % 3 == 0 :
        answer = resolve(num/3, cnt, answer)
    if num % 2 == 0:
        answer = resolve(num/2, cnt, answer)
    answer = resolve(num-1, cnt, answer)
    
    return answer

answer = resolve(int(input()))
print(answer)
```
