# Optimizer

- Gradient descent
    + Batch GD
    + Stochastic GD
    + Mini-batch GD
- Momentum
- NAG(Nesterov)
- Adagrad
- RMSProp
- Adam

## 1. Gradient descent

![gd](https://wikimedia.org/api/rest_v1/media/math/render/svg/26a319f33db70a80f8c5373f4348a198a202056c)

- notation
    + `a` : 특정 시점의 weight vector
    + `gamma` : learning rate
    + `F` : loss function
- 이전 시점의 weight에 F의 기울기(미분값)에 learning rate를 곱한 값을 빼준다.
- 데이터 사용 방법에 따라 다음 세 가지로 나뉜다.
    + (Batch) Gradient descent: **전체 데이터**를 모두 활용해 propagation, backpropagation해서 각 weight의 미분값을 구한다.
    + Stochastic GD: 전체 데이터가 아니라 샘플링한 데이터를 여러번 활용해서 weight를 업데이트하는 방식. 기본적으로 **단 하나**의 sample을 활용하는 것.
    + Mini-batch stochastic GD : 위 stochastic GD에서 한 개 샘플을 쓰는게 아니라 **subset**을 사용해서 weight 업데이트하는 방식
- stochastic 방법은 꼭 gradient descent로만 쓰는 것이 아니라 아래에 나오는 다른 방법에도 적용해서 쓴다.

## 2. Momentum

![gd-problem](https://i.imgur.com/m7YYZge.png)

> source: CS231n 2016 winter, Lecture 6

- Gradient descent의 문제점
    + Steep gradient: gradient가 크기 때문에 optimum point를 넘어서 큰 진폭으로 왔다갔다한다. 필요없는 움직임을 갖기 때문에 낭비다.
    + Shallow gradient: optimum point까지 가야하는 길이 멀 뿐더러, gradient가 작기 때문에 굉장히 조금씩 움직인다. optimum까지 너무 오래 걸린다.
- 위 문제를 해결한 것이 Momentum.















