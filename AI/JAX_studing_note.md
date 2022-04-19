# JAX
- JAX : 구글이 만든 Python과 Numpy만을 결합한 머신러닝 라이브러리. 넘파이를 GPU(TPU도 가능)에서 연산할 수 있게 해 기존 넘파이의 성능을 뛰어넘음.
  자동미분계산과 JIT(Just In Time)컴파일 기법과 XLA컴파일러를 사용해 컴파일 할 수 있음. 수치프로그램의 변환을 표현하고 구성하기 위한 언어. Python함수를 변환할 수 도 있으며, 이땐 Python을 jaxpr이라는 중간언어로 변환해 수행함.
- 장점 : TPU 사용에 장점을 가짐. 파이썬과 넘파이만으로 개발되어 Tensor Array를 고려하지 않고 Numpy Array만을 고려해 코드를 짤 수 있고, 프로그램이 순수 함수여야 하므로 오류등의 제어가 쉬우며,
  jit 컴파일 데코레이터 함수를 적용하면 부분 컴파일이 가능해, 깔끔하게 작성된 코드에서 큰 장점을 가짐. 업데이트된 Autograd를 통해 기본 Python 및 Numpy코드를 자동으로 구별할 수 있음.
- 주요사항 : 숫자 코드 작성시 더 유용한 프로그램 변환. jit(코드 속도 향상), grad(파생상품), vmap(자동 벡터화/batch처리)가 있음. 모든 JAX 연산은 XLA(가속 선형 대수 컴파일러)의 연산 측면에서 구현됨. 
- VS Numpy : JAX는 편의를 위해 Numpy에서 영감을 받은(거의 동일한)인터페이스를 제공하고, 덕파이핑을 통해 JAX배열과 Numpy배열을 교체해 사용할 수 있지만, JAX배열은 생성 후 변경할 수 없음.
- 학습 flow : parmaters와 x를 입력으로 받는 모델을 정의하고, loss_fn에서 그를 사용 -> 손실을 구한 뒤, optimize(update)에서 grad(loss)(params, x, y)를 포함한 optimize과정을 구현해 epochs동안 update실행함.

- jax.local_device_count() : 사용가능한 디바이스 개수를 반환.
- jax.config.config.update("jax_debug_nans", True) : NaN생성시 계산오류를 발생시킴.
- 함수.block_until_ready() : 해당 함수를 비동기 실행으로 바꿔줌.

## Numpy
- jax.numpy : JAX의 넘파이 연산들이 모인 패키지. 넘파이의 함수는 전부 사용할 수 있고, 일반 넘파이 배열도 연산할 수 있음. 인덱스가 오버된다면 항상 배열의 마지막값이 반환됨.
- jaxarr.at[idx\] : JAX배열의 해당 인덱스에 접근. 슬라이싱도 가능.
- jaxarr.at[index\].set(i) : JAX배열의 index가 i로 바뀐 복사본을 반환함.
- jax.numpy.where(condition, t_value, f_value) : 조건이 참일때는 t_value, 거짓일때는 f_value로 바뀐 행렬 반환.

- key : jax.random.PRNGKey(i) : JAX의 난수 생성을 위한 키 생성. split(key, num=i)로 subkey를 한번에 여러개 생성할 수 있음(받을땐 key, *subkeys).
  기존의 numpy는 Mersenne Twister PRNG를 사용해 난수를 생성했는데, 여기에는 보안이나 추론등에 어려움이 있어 분할 가능한 최신 Threefry 카운터 기반 PRNG(pseudo random number generation, 의사 난수 생성)를 사용함. 
  키는 두개의 unsigned-int32로 설명되는데, 동일한 키를 사용하면 동일한 난수가 나오고, 새로운 난수가 필요할 때 마다 `key, subkey = jax.ramdom.split(key)`를 통해 PRNG를 분할해야 함.
- jax.random.normal(key, shape) : normal범주 안의 난수로 채워진 shape형태의 행렬생성.

- jax.numpy.convolve(x, y) : 컨볼루션을 진행.

## lax
- jax.lax : jax.numpy에 비해 더 엄격하고 종종 더 강력한 하위수준 API. 암시적 인수 승격을 통한 혼합 데이터 유형간 작업 허용 등의 기능이 없음.
- jax.lax.conv_general_dilated(x, y, stride, padding) : 훨씬 더 일반적인 컨볼루션 사용. 

- jax.lax.scan(f) : HLO동안 single XLA로 낮춤. 파이썬 루프가 롤링되지 않아 대규모 연산이 이뤄지게 하기떄문에 컴파일 시간을 줄여줌.
- jax.lax.stop_gradient(x) : 기울기계산을 하지 않게 함. 대상의 종속성을 무시. 매개변수에 의존하는 않는 것처럼 취급. 일부 손실의 기울기가 신경망 매개변수의 하위 집합에만 영향을 미치도록 하려는 경우에도 쓸 수 있음.
- jax.lax.cond(bool, true_func, false_func, operand) : 미분 가능한 조건(제어흐름)식 사용.
- jax.lax.while_loop(cond_func, body_func, init_val) : fwd모드로 미분가능한 반복문 사용. while(cond_func(init_val): body_func)로 동작함.
- jax.lax.fori_loop(start, stop, body_func, init_val) : 일반적으로 fwd모드로 미분가능한 for문 사용. 끝점이 정적인 경우 fwd및 rev-mode를 구분할 수 있음. for i in range(start, end): val=body_func(init_val)로 사용됨.

## jit
- jax.jit(func) : 여러 작업(함수)을 XLA을 사용해 한번에 컴파일함. 한번에 한 작업씩 커널을 디스패치 하던걸 한번에 하는걸로 바꿔 속도를 높임. 함수의 일부 부분만 컴파일 할 수 도 있음. 데코레이터 `@jit`으로도 사용가능.
- 조건 : 단, 모든 배열이 정적이여야 하는(실행중 모양을 알 수 없는 배열을 생성하지 않아야 하는)등의 조건이 있음. 실제 데이터를 추적하는 형식으로 진행되며, 기록된 계산 시퀀스는 파이썬 코드의 재실행 없이 모양과 type이 같은 새 입력에 효율적으로 적용될 수 있음.
  정적 작업은 Python컴파일 타임에 평가되고, 추적된 작업은 XLA 런타임시 컴파일 및 평가됨. 따라서, 정적인 작업에는 numpy를, 추적해야 하는 작업에는 jnp를 사용하는게 유용함. static_argnums매개변수를 사용해 제어흐름이 있는 함수에도(하나의 흐름으로 고정해)사용할 수 있음.
- 캐시 : jax.jit은 처음 f를 호출할 때 컴파일 하고, 결과인 XLA코드를 캐시함. 서브시퀀스의 f호출에는 캐시된 코드를 재새용함. 만약 static_argnums를 명시한다면, 캐시된 코드는 오직 정적으로 레이블된 동일한 매개변수 값일때만 사용됨. 만약 하나라도 바꾸면 중복으로 재컴파일됨.
  만약 많은 값들이 있다면, 프로그램이 한번 연산을 실행하는것보다 많은 식나을 컴파일에 소모할수도 있음. 따라서 loop안에서 jax.jit을 부르는것을 피해야 함. 캐시는 함수의 해시에 의존하기에, 다른 해시를 반환하는 partial이나 lambda등에 사용하면 루프에서 매번 불필요한 컴파일이 발생함.
- jax.make_jaxpr(func) : jit 컴파일시 추출되는 작업시퀀스(JAX표현식, jaxpr)를 확인 할 수 있음. 변수가 사용되는 제어문은 추적할 수 없으나, `@functools.partial(jit, static_angnums`로 정적변수로 표시하면 추적할 수 있음.
- jax.device_put(x) : 모든 연산시마다 GPU로 데이터를 전송해 속도가 느렸던 동작으로 필요할 때만 값을 CPU에 복사하도록 변경. jit(lambda x: x)와 동일하나 더 빠름.

## vmap
- jax.vmap(func) : 자동 벡터화. 함수 내에서 자동으로 벡터를 행렬로 승격시키게 함. batch연산시 사용. 두개의 1차원 벡터간의 연산으로 설계된 함수를, 두개의 행렬간의 연산이 가능하게 바꿔줌. 배치차원이 첫번째가 아닌 경우, out_axes인수를 사용해 지정할 수 있음.
  특정 매개변수에는 차원을 추가하고 싶지 않다면 in_axes매개변수를 이용해 지정할 수 있고(추가하지 않을 매개변수의 위치에는 None, 이외에는 0으로 된 튜플), 샘플당 가속된 grad계산을 하고 싶다면, jit(vmap(grad(loss_func)))로 할 수 있음. 
- jax.pmap(func) : SPMD(동일한 계산(순전파 등)이 다른장치에서 병렬로 다른 입력데이터로 실행되는 병렬기술)을 위해 내장. 기본 사용법은 vmap과 동일함. 배치를 사용가능한 장치수와 동일하게 제작한 뒤 변환. 배열의 요소가 병렬처리에 사용되는 모든 장치에 걸쳐 분할되기에, ShardedDeviceArray를 반환함. 
  vmap과 동일하게 in_axes등의 매개변수를 사용할 수 있으며, 과정중 JIT컴파일이 포함되어 있기에 jit()은 필요하지 않음. 

## pytree
- 파이트리 : leaf elements와/또는 pytree의 컨테이너. 일반적으로 모델 매개변수, 데이터세트 항목, 데이터세트로 대량작업, RL에이전트 관챃 등에서 찾을 수 있음. 
  컨테이너는 list, tuple, dict를 포함함. leaf element는 배열 등 pytree가 아닌 모든것이며, 파이트리는 중첩될 수 있는 표준 혹은 유저등록의 파이썬 컨테이너임. 중첩된 경우 컨테이너유형이 일치하지 않아도 됨. 단일 leaf(컨테이너가 아닌 오브젝트)도 파이트리로 간주됨. 
- 주의점 : node를 leaf로 착각하거나(구성요소가 pytree노드인 튜플등 이며 그것의 내용은 leaves인 배열이 있다면, map은 튜플로 불러오는게 아니라 leaf인 각 요소들을 불러오게 됨. 이 경우 튜플을 leaf인 ndarray나 jnparray로 바꾸거나 tree_map코드를 다시 작성할 수 있음),
  None을 pytree의 자식으로 설정하려 한다거나(이경우 자식이 없다고 판단함)등의 주의점이 있음.

- jax.tree_leaves(pytree) : 파이트리 객체를 만듦. pytree는 pytree로 간주되는 것들로 이뤄진 컨테이너형이 될 수 있으며, 2차원 배열또한 될 수 있음. 
- jax.tree_map(func, lists) : 2차원 배열의 각 배열들에 func를 적용시켜 합침. func의 매개변수를 2개 이상 사용하려면, 2차원 배열들 또한 2개 이상 넣으면 됨.  
- jax.tree_util.register_pytree_node(MyContainer, flatten_MyContainer, unflatten_MyContainer) : 커스텀 파이트리 노트를 등록. 등록하지 않아도 pytree로 사용할 수는 있으나 tree_map등의 사용을 위함.
  flatten은 컨테이너의 각 요소를 리스트 형태로 만든것과 컨테이너의 .name을 반환하는 함수이고, unflatten은 aux_data(.name)과 flat_contents를 입력받아 컨테이너로 만들어 반환하는 함수면 됨.
- jax.tree_transpose(outer_treedef, inner_treedef, pytree_to_transpose) : 내부 및 외부 pytree의 구조를 지정해 pytree의 구조를 바꿀 수 있는 함수. 각 구조는 jax.tree_structure([0 for e in episode_steps])처럼 지정하면 됨.

## grad
- jax.grad(func) : 자동 미분. 기울기를 계산. 주로 jax.grad(loss_fn)(params, data_batch)로 사용됨. jax.grad되는 함수는 스칼라를 반환하는 함수만 정의되어 있어 반환값이 튜플이라면 hax_aus=True를 해줘야 함.
- jax.value_and_grad(loss_fn)(x, y) : (value(loss), grad)의 튜플을 반환함.
- jax.jacfwd(func) : autodiff에 해당하는 함수의 Jacobian행렬(야코비 행렬, 다변수 벡터 함수의 도함수 행렬)을 계산. 순방향 모드. 역방향과 답변의 차이는 없으나 특정 상황에서 더 효율적일 수 있음.
- jax.jacrev(func) : autodiff에 해당하는 함수의 Jacobian 행렬을 계산. 역방향 모드. 순방향과 답변의 차이는 없으나 특정 상황에서 더 효율적일 수 있음.

## nn
- jax.nn : jax의 neural network와 관련된 메서드들이 모여있는 패키지.
### initializer
- jax.nn.initializers.glorot_normal() : glorat_normal 초기화 사용. 주로 가중치 초기화에 사용됨. (key, shape)로 사용.
- jax.nn.initializers.normal() : normal초기화 사용. 주로 bias 초기화에 사용됨. (key, shape)로 사용.
### activate function
- jax.nn.sigmoid() : sigmoid사용.

## stax
- jax.experimental.stax : 파이토치나 케라스와 유사하게 더 높은 수준의 추상화 Layer를 제공하는 패키지. 아직까지 RNN계열 모델은 잘 지원되지 않아 직접 만들어야 함.
- stax.serial(layers) : 각 순전파연산(층)들을 체인하는 wrapper. init_fn와 model(net)을 반환하며, `_, params = init_fn(key, shape)`로 init_fn을 사용할 수 있음.
### layers
- stax.Conv(output_dim, kernel, stride) : Convolution Layer 사용.
- stax.Dense(output_dim) : Dense Layer 사용.
- stax.BatchNorm() : Batch Normalization 사용.
- stax.Flatten : Flatten 층 사용.
### activate function
- stax.Relu : Relu 사용.
- stax.LogSoftmax : LogSoftmax 사용.
### optimizer
- jax.experimental.optimizers : optimizer들이 정의되어있는 패키지. `opt_init, opt_update, get_params = optimizers.adam(lr)`식으로 사용되며 init은`opt_state = opt_init(params)`식으로 사용함.
- optimizers.adam(lr) : adam옵티마이저 사용. 
