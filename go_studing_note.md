# GO
## Golang
```go
package main
import "fmt"
func main(){ 
    fmt.Println() 
} 식으로 구성 
```

## RUN
- go run 파일명.go : 터미널에서 go 파일을 실행
- go build 파일명.go : 현 폴더에 exe 실행파일 생성 (go파일 빌드)

## Variable
- 변수명 := 값 : var을 사용하지 않고 변수 선언. 묵시적 선언.
- var 변수명 = 값 : 변수 선언. 묵시적 선언. 선언시 값을 할당해줘야 함.
- var 변수명 자료형 : 자료형의 변수 선언. 명시적 선언. 값을 할당하지 않으면 각 자료형의 기본값이 할당됨.  
- 자료형 종류 : uint8~64(그냥은 64), int8~64, float32/64, complex64/128(실수부+허수부), byte, bool, string,
                uintptr(uint와 같은 크기의 포인터), rune(유니코드, int32와 같은 크기)이 있음.


## print
- fmt.Println() : 출력 문장 뒤에 \n를 붙여 출력. fmt 패키지의 import가 필요.
- fmt.Printf() : 포맷이 적용되어 출력. ("%T(변수 자료형)", 변수) 식으로 사용. 
- fmt.Print() : 파이썬의 print와 동일한 출력문.

## if
- if 조건 { } : 조건문 사용. ||등 조건연산자를 사용할 수 있음. else 사용 가능.
- switch 변수 { case 값 : ... } : switch case 사용.

## for
- 





