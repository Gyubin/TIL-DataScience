# PyTorch fundamentals

설치는 http://pytorch.org/ 에 들어가서 자기 사양 선택하면 설치 커맨드가 나온다. 내 맥 환경의 경우 다음과 같다. 

```sh
pip3 install http://download.pytorch.org/whl/torch-0.2.0.post3-cp36-cp36m-macosx_10_7_x86_64.whl 
pip3 install torchvision 
```

## 1. 기본

- 변수 선언
    + `torch.Tensor(i, j, k)` : shape이 (i, j, k)인 데이터를 생성한다. rank는 자유롭게 설정가능
    + `torch.rand(i, j)` : shape이 (i, j)인 0에서 1 사이의 난수 데이터 생성
- add 연산 종류
    + `a + b` : 단순 덧셈 연산으로 element-wise add 연산 가능
    + `torch.add(a, b, out=result)` : 리턴값도 있고, 덧셈 연산 결과를 특정 변수에 저장 가능
    + `a.add_(b)` : `add_` 함수 호출. 호출한 변수의 값 자체가 변한다.
- 인덱싱은 numpy와 같은 방식으로 사용한다. ex) `a[:, 1:4]` , `a[3, 4]`
- `torch.ones(i, j)` , `torch.zeros(i, j)` : (i, j) shape으로 데이터 만드는데 1과 0으로 초기화
- torch to numpy: 모든 torch.tensor에서 `.numpy()` 함수를 호출하면 ndarray로 변환된 값이 리턴된다. 원본 torch tensor의 영향을 받는다.
    + `x1=torch.ones(2, 3) ; x2=x1.numpy()`
    + `x1.add_(1)`
    + `print(x1, x2)` : x1에만 값을 1 더했는데 x2 값도 변했음을 알 수 있다.
- numpy to torch: 위 경우와 반대.
    + `a=np.ones(5) ; b=torch.from_numpy(a)`
    + `np.add(a, 1, out=a)`
    + `print(a, b)` : 역시 b도 영향을 받는다.
- tensor를 GPU로 옮기고싶다면 `.cuda()`

    ```py
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        x + y
    ```

## 2. Variable

```py
import torch
from torch.autograd import Variable

x = Variable(torch.Tensor([3, 4, 5, 6]), requires_grad=True)
w = Variable(torch.Tensor([10]), requires_grad=True)
z = w*x + 2
out = z.mean()

out.backward()
print(x.grad, w.grad)
```

- parameters : function을 실행할 때 넣는 값
    + `data` : Tensor 데이터
    + `requires_grad` : grad를 계산할건지 말건지를 나타내는 bool 값
- variables : `.`으로 호출 가능한 값들
    + `data` : 갖고 있는 데이터 값
    + `grad` : data와 매칭되는 미분값. reassign 안된다.
    + `requires_grad` : grad를 계산한다고 한 값인지 Boolean으로 리턴. leaf node에서만 바뀔 수 있다.
    + `is_leaf` – leaf인지, 즉 user가 1차로 만든 값인지
    + `grad_fn` – 만들어진 미분 공식
- 미분 계산을 쉽게 할 수 있다. `Variable` 안에 데이터를 넣고, gradient를 계산할지 말지를 boolean으로 정해준다.
- 위처럼 다음으로 넘어가는 graph를 그리고, 최종 값인 `out`은 스칼라 값으로 한다.
- 미분을 계산할 기준이 out이라면 out에서 `backward()` 함수를 호출하나.
- 호출한 이후에 `x`와 `w`에서 미분값을 가져올 수 있다.
    + `x.grad` = d(out) / d(x)
    + `w.grad` = d(out) / d(w)
