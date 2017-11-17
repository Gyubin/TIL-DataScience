# SVM loss

SVM loss는 Hinge loss라고도 불린다. 처음 SVM을 배울 때 constraint와 [lagrange multiplier](https://en.wikipedia.org/wiki/Lagrange_multiplier)로 수식을 유도하는 방식으로 배워서 이 모델과 관련된 loss가 있을거라곤 생각을 못했다. [CS231n Lecture 3 slide](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture3.pdf)에서 "Multiclass SVM loss"로 처음 접했고, 실제로 maximum-margin classifier들이 이 loss를 이용한다고 해서(대표적으로 SVM) 좀 더 명확히 이해할 필요가 생겨 정리해본다.

## 1. 공간에서 classifier 이해

![multinomial-logistic-regression](https://i.imgur.com/NRaq0iA.png)

> source: CS231n Lecture 3 - 7th slide

- Notation
    + 이미지는 feature가 매우 많지만, 설명의 편의를 위해 2개만 있는걸로 생각하겠다. 그래서 좌표평면에 나타낸 것이 위 이미지와 같다. 축은 x1과 x2 데이터 두 축이다.
    + 편의상 car classifier의 parameter를 `w1`, `w2`, `b`라고 나타낼 것이고, 물론 다른 airplane, deer classifier 각각 parameter들이 존재한다.
- car classifier를 나타내는 직선은 `w1x1 + w2x2 + b = 0` 으로 표현할 수 있다. 즉 이 수식에 데이터 (x1, x2)를 집어넣었을 때 0 값이 나오는 데이터들을 쭉 직선으로 이은 것이다.
    + 값이 0보다 크다면, 해당 class로 분류를 하는 것이고, 값이 0보다 작다면, 해당 class가 아니라고 분류를 한다.
    + 값이 0인 이유: logistic regression은 sigmoid 함수를 사용한다. 위 classifier hyperplane의 수식에 sigmoid를 씌우는 꼴로 나타나는데 0을 집어넣으면 0.5가 나오기 때문. 즉 cutoff가 기본적으로 0.5로 잡혀있으니 수식의 값이 0인 점들이 classifier 지점이 되는 것이다.
    + 즉 만약 cutoff가 0.5가 아니라면 수식의 값이 0이 아닌 어떤 지점에 classifier가 그려질 것. 더 아래쪽 3차원 공간에서 다시 설명하겠다.
- 이미지에서 표시된 총 3개의 법선 벡터는 각 classifier들에 수직인 벡터를 말하고, car classifier의 법선벡터는 `[w1, w2].T` 로 나타낼 수 있다.
- 법선 벡터를 위처럼 나타낼 수 있는 이유는 다음처럼 유도된다.
    + car classifier 위의 어떤 두 점을 `p1 = (k, l)`, `p2 = (m, n)`로 잡고, 서로 빼서 car classifier와 평행한 벡터 `[k-m, l-n].T`를 만든다.
    + 각 점을 classifier 수식에 집어넣으면 `w1*k + w2*l + b = 0` , `w1*m + w2*n + b = 0` 가 나온다.
    + 위 두 수식을 빼면 `w1*(k-m) + w2*(l-n) = 0`이 되고, 이것은 `[w1, w2].T` 벡터와 `[k-m, l-n].T` 벡터의 dot product 꼴이다.
    + 처음에 car classifier와 평행한 벡터 `[k-m, l-n].T`를 만들었고, 이것과 `w1, w2].T` 벡터가 수직이란 것은 결국 car classifier와도 수직이란 것을 뜻한다. 즉 법선벡터다.
- 법선벡터는 결국 우리가 최적화해야할 모든 weight 값들을 갖고 있다. 처음 학습을 시작할 때 wegith들을 매우 작은 값으로 랜덤 초기화하는데 이것은 위 그래프에서 아무렇게나 그래프를 랜덤으로 그리는 것이다. 그리고 점점 데이터를 잘 분류하도록 그래프의 위치와 기울기를 변경해나갈 것이고, 이런 과정이 "학습"이다.

![3d-hyperplane](https://i.imgur.com/askA6BZ.png)

- 위에서 classifier 수식 값의 기준이 0인지는 어떻게 정해지는 것일까. logistic regression이라면 위 이미지에서처럼 cutoff(threshold)를 어떤 수치로 정하느냐에 따라 달라진다.
- 0.5로 cutoff를 정했을 때는 위처럼 classifier 직선이 저 교차점에 그려지는 것이고, 저 지점은 수식의 값이 0인, 즉 sigmoid를 씌웠을 때 0.5 값이 나오는 지점이다.
- 즉 cutoff를 몇으로 설정하느냐에 따라 boundary가 다른 위치에 정해지는 것이고, 그 때의 수식의 값이 달라진다. 예를 들어 cutoff를 0.8로 정한다면 수식의 값은 1.4 정도가 될것이고, 즉 수식의 값이 1.4인 지점에 classifier, boundary가 위치할 것이다.
- 즉 그 지점은 `p(y=1|x)`의 값, 다시 말해 `sigmoid(f(x))`의 값이 cutoff인 지점을 뜻한다.

## 2. SVM loss

### 2.1 수식 설명

지금까지의 설명에서 어떤 것을 loss로 할지는 말하지 않았다. loss의 특징에 따라 optimization의 결과도 많이 달라진다. SVM loss의 수식은 다음과 같다.

![hinge_loss](https://wikimedia.org/api/rest_v1/media/math/render/svg/65b2021f4608cc428cbc4f829ddad5c964d5d38c)

- `t` : 분류하고자 하는 class 각각을 의미한다.(ex) dog, cat, rabbit, etc..)
- `y` : `t` 중에서도 True class를 `y`라고 한다.
- `x` : 어떤 class인지 판단해야할 데이터
- `np.dot(w_t, x)` : t classifier의 weight와 x의 dot product의 결과물, 즉 t class로 분류할 score를 말한다. 높을수록 t class로 분류하게 된다. 위 수식에서는 True class를 제외한 t로만 계산한다.
- `np.dot(w_y, x)` : True class의 score 값
- `1` : hyperparameter이고 delta라고도 지칭한다. 아래에서 다시 설명하겠다.
- 수식을 해석하면 False classifier의 score에서 1을 더하고, True classifier score를 빼서 0 이상인 것들만 그 차이를 loss로 합산한다는 의미다.
- 즉 False classifier의 score 중 True classifier score보다 큰 값들만(delta에 따라 delta보다 작은 것들까지 인정) 그 차이를 loss 계산에 사용하겠다는 것.

### 2.2 Delta를 주면 좋은 예제 1

![delta](https://i.imgur.com/SugeCFZ.png)

- 위 그래프는 3개 class의 multinomial logistic regression이다.
- 만약 delta가 0이라면 Classifier blue가 실선에서 점선으로 이동할까? 이동하지 않는다. 이유는 실선일 때 Loss가 0이기 때문이다. Classifier blue가 "별 데이터"를 긍정으로 잘못 예측하고 있지만, Classifier red가 더 큰 값으로 잘 예측하고 있기 때문에 SVM loss는 0이다. loss가 0이면 더 이상 최적화가 이루어지지 않는다.
- 그래서 Delta 값을 줘서 loss 값을 키운다. 수식에서 `(S_j + delta) - S_y` 처럼 묶어서 표현해보면 다른 클래스로 판단하는 score 값을 키우는 것과 같은 의미다. 위 이미지에서 빨간 별 데이터의 blue score를 더 키워서 red score보다 커지게 한 후 loss로 계산하는 것이다.
- 그러면 loss가 생기고, 줄이기 위해 학습이 되면서 파란 실선이 파란 점선으로 좀 더 최적화될 가능성이 생긴다.

![delta3](http://cs231n.github.io/assets/margin.jpg)

> delta가 다른 클래스의 score를 높여준다는 것을 시각적으로 보여준다.(source: CS231n Lecture note)

### 2.3 Delta를 주면 좋은 예제 2

![delta1](https://i.imgur.com/kJmYw0q.png)

- SVM 모델의 목표는 데이터를 분류하는 hyperplane이 margin을 가장 크게 갖는 것이다. margin은 hyperplane과 가장 가까운 데이터인 support vector와의 거리를 말한다.
- 위 이미지에서 만약 delta 값이 없다면 weight 벡터가 어떻게 초기화되고, 최적화되느냐에 따라 Classifier blue의 위치가 위 둘 중 어떤것도 될 수 있다. 왜냐면 둘 모두 SVM loss가 0이기 때문.
- 그런데 둘 다 딱히 좋은 hyperplane은 아니다. 하나는 yellow에 치우쳐있고, 하나는 blue에 너무 치우쳐져있다. 이걸 적절하게 고르도록 도와주는 것이 delta다.

![delta2](https://i.imgur.com/IEJQPOW.png)

- 예를 들어서 지금 blue classifier만 조정되어야한다고 해보자. yellow 데이터에 대해서 로스를 계산할 때, delta를 주면 blue, red score가 올라가게 된다.
- blue score를 올린다는 말은 위처럼 데이터가 마치 위로 이동된 것처럼 생각해서 score를 계산한다고 볼 수 있겠다. 그에 맞춰서 classifier의 위치도 최적화 될 것이고 이미지에서처럼 blue classifier가 저렇게 꼭 평행이동하는 것은 아니겠지만 그에 맞게 이동할 것이다.
- delta는 hyperparameter이고 Cross validation을 통해 얼마나 이동하는 것이 가장 적절한 hyperplane의 위치인지 알아내야한다.
- delta로 score를 잘 조정하면 classifier들을 전체 데이터에 생김새(분포)에 맞게 적절하게 잘 위치시킬 수 있다.
