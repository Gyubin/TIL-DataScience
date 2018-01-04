# Keras

## 1. Simple regression example

모델 객체를 만들고, **add** 하는 형태로 사용

```py
from keras.models import Sequential
model = Sequential()
```

### 1.1 Dense, Activation

```py
from keras.layers import Activation, Dense

model.add(Dense(10, input_shape = (13,), activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(1))
```

- 첫 input layer는 데이터의 feature 개수를 정해줘야한다.
- 파라미터로 10을 주는 것은 output이 10차원(feature)이라는 의미.
- Activation도 쉽게 지정해서 쓸 수 있다. `relu`, `tanh`, `elu`, `sigmoid`
- 위 코드처럼 한 번에 쓰는게 더 간결하지만, 따로 쓸 수도 있다.
    + `model.add(Activation('sigmoid'))`

### 1.2 Optimizer

```py
from keras import optimizers

sgd = optimizers.SGD(lr = 0.01)

model.compile(optimizer = sgd, loss = 'mean_squared_error', metrics = ['mse'])
model.summary()
```

- SGD 사용하는 예시.
- `compile` 함수를 통해 optimizer, loss를 정해줄 수 있다.
- `summary`는 깔끔하게 모델 출력해준다.

### 1.3 Training

```py
model.fit(X_train, y_train, batch_size = 50, epochs = 100, verbose = 1)
```

- 데이터, batch, epochs 지정해서 학습
- `verbose`: 0은 출력 없고, 1은 progress bar, 2는 epoch마다 한 줄로.

### 1.4 Evaluation

```py
results = model.evaluate(X_test, y_test)
print(model.metrics_names)     # list of metric names the model is employing
print(results)                 # actual figure of metrics computed
print('loss: ', results[0])
print('mse: ', results[1])
```

- evaluate 함수로 테스트 데이터 평가
- results에 담긴 값으로 결과 확인 가능
