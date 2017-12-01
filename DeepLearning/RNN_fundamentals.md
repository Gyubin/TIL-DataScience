# RNN fundamentals

- MLP, CNN은 모든 데이터가 서로 I.I.D라고 가정
- Sequential data는 의존 관계가 있기 때문에 기존 아키텍처로는 분석이 잘 안된다.
- 문장에서 빈칸을 채우거나, 다음 단어를 예측하기 위해선 주변 단어를 통해 유추하는 것이 바람직하다.

## 9. 왜 ReLU가 아니라 tanh를 쓸까

출처: [TensorFlow KR 주재걸 교수님 댓글](https://www.facebook.com/groups/TensorFlowKR/permalink/478174102523653/?comment_id=478528069154923&comment_tracking=%7B%22tn%22%3A%22R3%22%7D)

- 단일 activation function을 쓴다면 ReLU보다 Sigmoid, tanh가 비선형성 추가에 더 좋다.
    + 당연히 ReLU에서 양의 부분은 들어온 값 그대로 리턴하므로 선형의 성격이 강하다.
    + sigmoid나 tanh가 훨씬 본래의 비선형성에 알맞고, 다만 여러 레이어를 거칠 때 vanishing gradient 문제가 크므로 CNN 쪽에선 ReLU를 많이 쓰는 것
- output value가 zero-centered면 좋다.
    + 텍스트 처리에선 동일한 weight 값이 담긴 RNN/LSTM cell을 굉장히 많이 반복하게된다.
    + output의 값이 평균적으로 몇인지가 중요해지는데, 음의 방향이든 양의 방향이든 shift가 된다면 많은 반복을 거치면서 input 값이 유의미한 범위가 아니게 된다.
    + 그래서 평균적으로 0의 ouput을 내도록 function의 값이 zero centered인 tanh 함수를 주로 쓰게 된다.
- 모델이 `y = 3x + 0` 으로 학습됐을 때 tanh를 쓴다면
    + input `[-1, 1]` -> output `[-3, 3]` -> after tanh `[-1.25, 1.25]`
    + input `[-1.25, 1.25]` -> 같은 작업 반복, zero centered 유지
- sigmoid를 쓴다면
    + input `[-1, 1]` -> output `[-3, 3]` -> after sigm `[0.1, 0.9]`
    + input `[0.1, 0.9]` -> `[0.3, 2.7]` -> after sigm `[0.57, 0.94]`
    + 자꾸 위쪽으로 shift되는 것을 볼 수 있다. 가장 sensitive한, 즉 gradient가 가장 큰 지점인 0 주위를 벗어나게된다.
    + 점점 양수쪽으로 치우치게되고 모두 1이란 답만 내게 된다. -1에서 1 사이의 값을 다양하게 내다가 1만 내는 안좋은 결과를 갖게된다.
- 위 같은 상황이 기존 Fully connected, conv layer에서는 bias를 조정함으로써 해결됐다.
    + 매 레이어마다 학습되는 bias들이 서로 다른 값(다른 변수)이기 때문에 자연스럽게 shifted 된 정도를 조정 가능
    + 하지만 RNN/LSTM에선 hidden state의 값 자체가 tanh의 결과물(bias는 tanh 안에 존재)이기도 하고, 매 sequence마다 동일한 bias를 써야하기 때문에 shift를 조절하기 힘들다.
