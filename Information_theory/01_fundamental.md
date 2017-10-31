# 01. Fundamentals of Information theory

## 1. Quantity of information

- 정보량(Information quantity, **I**) : 그 정보를 얻었을 때 느끼는 놀라움의 정도. 즉 희박한 확률로 일어날 사건이 발생하면 놀라움이 크므로 정보량이 크다고 본다.
    + 조건1, 확률이 작으면 정보량이 크다: `P(x1) > P(x2)` => `I(x1) < I(x2)`
    + 조건2, 독립인 사건이라면 정보량 단순 덧셈 성립: `I(x1x2) = I(x1) + I(x2)`
- 수식 유도 : `I(x) = -log2(P(x))`
    + `I(x) = 1 / P(x)` : 조건1에 따라서 정보량을 확률의 역수로 만든다.
    + 하지만 위 경우엔 조건2 가법성이 성립되지 않는다. A, B 사건이 각각 0.5, 0.25의 확률일 때 정보량은 2, 4의 합은 6인데, joint probability인 0.125의 정보량은 8이다. 6과 8은 다르다.
    + 그래서 log를 씌워서 `log(1/P(x))`를 정보량으로 활용한다. 정보이론에선 주로 밑을 2로 사용. 위의 케이스에서 정보량은 1, 2이고 합은 3이며, joint probability의 정보량 역시 3이다.

## 2. Entropy

- Entropy(**H**) : 모든 사건들의 정보량의 평균적인 기대값(expectation), 즉 확률을 가중치로 하는 정보량의 가중평균이다.
- 모든 사건들에 대해서, 정보량에 그 정보량이 일어날 확률을 곱한 값들을 다 더한다.
- 확률 분포의 무작위성(randomness)라고 표현하기도 한다.
- `H(p) = H(X) = Sum( P(x) * log2(1/P(x)) )`

## 3. Cross entropy

기존 entropy가 하나의 probability distribution에서 정보량의 가중평균을 계산했다면 이젠 p, q 두 개에서 계산한다.

### 3.1 기본 식

![CE-default](https://wikimedia.org/api/rest_v1/media/math/render/svg/80bd13c723dce5056a6f3aa1b29e279fb90d40bd)

- `p` : True distribution
- `q` : Unnatural probability distribution(최적화할 것)
- `H(p)` : Entropy of p
- `Dkl(p||q)` : Kullback-Leibler divergence of q from p(= relative entropy of p with respect to q)

### 3.2 p, q가 discrete 할 때

![CE-discrete](https://wikimedia.org/api/rest_v1/media/math/render/svg/0cb6da032ab424eefdca0884cd4113fe578f4293)

- p, q가 discrete하다면 **p 확률분포로 q의 정보량을 가중평균한 값**으로 해석
    + p는 true distribution, q는 optimize 해야할 distribution
- As loss
    + p와 q가 같다면 entropy와 식이 같아지고 가장 최소값이 된다.
    + p, q의 차이가 크면 클수록 cross entropy 값이 커진다.
    + 무조건 양수값이고, q를 p에 가깝게 만드는 것이 목표이므로 loss로 사용 가능
- MSE와 성능은 비슷하면서 학습이 더 빠르다. 미분해보면 식이 더 단순해져서 그런 것 같다. 나눠서 사라지는 항들이 생긴다.

## 4. Kullback-Leibler divergence

![KL-divergence](https://wikimedia.org/api/rest_v1/media/math/render/svg/70e86d9c6ac6b602308b0d55eba981f3eaeb8048)

- cross entropy와 결국 같은 의미다.
- 위 식의 log의 분수 부분을 나누면 `H(P, Q) - H(P)` 꼴이 된다.
- 앞의 항은 P, Q의 cross entropy고 뒷항은 true distribution의 엔트로피
- 즉 뒷항은 줄일 수 없는 고정된 값이고, 앞의 항을 뒷항에 가깝게 optimize해서 값을 줄여야한다.
