# Keras fundamentals

## 1. Simple example

## 1.1 Regression

모델 객체를 만들고, **add** 하는 형태로 사용

```py
from keras.models import Sequential
model = Sequential()
```

### 1.1.1 Dense, Activation

```py
from keras.layers import Activation, Dense

model.add(Dense(10, input_shape = (13,), activation = 'sigmoid'))
model.add(Dense(10, kernel_initializer='he_normal', activation = 'sigmoid'))
model.add(Dense(10, kernel_initializer='he_normal', activation = 'sigmoid'))
model.add(Dense(1))
```

- 첫 input layer는 데이터의 feature 개수를 정해줘야한다.
- 파라미터로 10을 주는 것은 output이 10차원(feature)이라는 의미.
- Activation도 쉽게 지정해서 쓸 수 있다. `relu`, `tanh`, `elu`, `sigmoid`
- 위 코드처럼 한 번에 쓰는게 더 간결하지만, 따로 쓸 수도 있다.
    + `model.add(Activation('sigmoid'))`
- weight initializer
    + 레이어의 파라미터 `kernel_initializer`, `bias_initializer`로 지정
    + `random_uniform`, `zeros`, `he_normal`, 

    ```py
    Dense(64, kernel_initializer='random_uniform',
              bias_initializer='zeros')
    ```

### 1.1.2 Optimizer

```py
from keras import optimizers

sgd = optimizers.SGD(lr = 0.01)

model.compile(optimizer = sgd, loss = 'mean_squared_error', metrics = ['mse'])
model.summary()
```

- Optimizers
    + `SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)`
    + `RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)`
    + `Adagrad(lr=0.01, epsilon=None, decay=0.0)`
    + `Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)`
    + `Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)`
    + `Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)`
    + `Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)`
- `compile` 함수를 통해 optimizer, loss를 정해줄 수 있다.
- `summary`는 깔끔하게 모델 출력해준다.

### 1.1.3 Training

```py
model.fit(X_train, y_train, batch_size = 50, epochs = 100, verbose = 1)
```

- 데이터, batch, epochs 지정해서 학습
- `verbose`: 0은 출력 없고, 1은 progress bar, 2는 epoch마다 한 줄로.

### 1.1.4 Evaluation

```py
results = model.evaluate(X_test, y_test)
print(model.metrics_names)
print(results)
print('loss: ', results[0])
print('mse: ', results[1])
```

- evaluate 함수로 테스트 데이터 평가
- results에 담긴 값으로 결과 확인 가능

## 1.2 Simple classification

### 1.2.1 Prepare data

```py
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

whole_data = load_breast_cancer()
X_data = whole_data.data
y_data = whole_data.target

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3, random_state = 7) 
```

### 1.2.2 Train

```py
from keras.models import Sequential
from keras.layers import Activation, Dense

model = Sequential()

model.add(Dense(10, input_shape = (13,), activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(1, activation = 'sigmoid'))

sgd = optimizers.SGD(lr = 0.01)
model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size = 50, epochs = 100, verbose = 2)

results = model.evaluate(X_test, y_test)
print(model.metrics_names)
print(results)
```

- regression 문제와 다른 점은 `loss`와 `metrics` 밖에 없다.
- 각각 `binary_crossentropy`, `accuracy`로 정해서 학습

## 2. Something beyond

### 2.1 Batch normalization

```py
from keras.layers import BatchNormalization

# Codes ommited
model.add(BatchNormalization())
```

- ReLU같은 nonlinear activation 하기 전에, 모든 mini-batch에 대해서 수행
- 다음 순서가 반복: **Dense-BatchNorm-Activation-Dropout**

### 2.2 Dropout

```py
from keras.layers import Dropout

# Codes ommited
model.add(Dropout(0.2))
```

- Dense, Activation 다음에 넣어주면 된다.
- 다음 순서가 반복: **Dense-BatchNorm-Activation-Dropout**

### 2.3 Model ensemble

```py
import numpy as np

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

def mlp_model():
    model = Sequential()
    
    model.add(Dense(50, input_shape=(784, ), activation='sigmoid'))
    model.add(Dense(50), activation='sigmoid')
    model.add(Dense(50, activation='sigmoid'))
    model.add(Dense(50, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))
    
    sgd = optimizers.SGD(lr = 0.001)
    model.compile(optimizer = sgd, loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    
    return model

model1 = KerasClassifier(build_fn = mlp_model, epochs = 100, verbose = 0)
model2 = KerasClassifier(build_fn = mlp_model, epochs = 100, verbose = 0)
model3 = KerasClassifier(build_fn = mlp_model, epochs = 100, verbose = 0)

ensemble_clf = VotingClassifier(estimators=[('model1', model1),
                                            ('model2', model2),
                                            ('model3', model3)],
                                voting = 'soft')

ensemble_clf.fit(X_train, y_train)

y_pred = ensemble_clf.predict(X_test)
print('Test accuracy:', accuracy_score(y_pred, y_test))
```

- 주로 8-10개를 종합한다.
- scikit-learn의 [VotingClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html) 사용해야한다. 그러기 위해선 keras의 모델을 scikit-learn에서 사용할 수 있도록 wrapping이 필요하다.
- `KerasClassifier`가 그 역할을 한다. 위처럼 모델 함수를 넣고 파라미터 지정해준다.
- 래핑한 객체를 `VotingClassifier`로 묶어서 학습시킨다.
- voting
    + `hard` : majority rule voting
    + `soft` : 앙상블하는 모든 classifier의 확률 결과값들을 합해서 그 중 제일 높은 것을 취한다. well-calibrated classifiers일 때 사용을 추천한다고 한다.

## 3. CNN with MNIST

### 3.1 Prepare data

```py
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

(X_tr, y_tr), (X_te, y_te) = mnist.load_data()

# X_tr = X_tr.reshape((X_tr.shape[0], X_tr.shape[1] * X_tr.shape[2]))
# X_te = X_te.reshape((X_te.shape[0], X_te.shape[1] * X_te.shape[2]))
y_tr = to_categorical(y_tr)
y_te = to_categorical(y_te)
```

- 데이터셋은 `keras.datasets`에서 쉽게 가져올 수 있다.
- 위 주석 친 부분은 이미지를 flatten해서 벡터로 만드는 코드
- `to_categorical(data)` 형태로 one-hot vector로 만들 수 있다.

### 3.2 Library

```py
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, ZeroPadding2D, Input
from keras.models import Model
from keras.preprocessing import image

img = image.load_img('dog.jpg', target_size = (100, 100))
img = image.img_to_array(img)
```

`keras.preprocessing.image` : 이미지 쉽게 로드하고 조정 가능. numpy array 형태로 아래처럼 변환할 수 있다.

### 3.3 Padding

```py
# when padding = 'valid'
model = Sequential()
model.add(Conv2D(input_shape = (10, 10, 3), filters = 10, kernel_size = (3,3), strides = (1,1), padding = 'valid'))
print(model.output_shape)

# when padding = 'same'
model = Sequential()
model.add(Conv2D(input_shape = (10, 10, 3), filters = 10, kernel_size = (3,3), strides = (1,1), padding = 'same'))
print(model.output_shape)

# user-customized padding
input_layer = Input(shape = (10, 10, 3))
padding_layer = ZeroPadding2D(padding = (1,1))(input_layer)
model = Model(inputs = input_layer, outputs = padding_layer)
print(model.output_shape)
```

- valid는 패딩 없는 것, same은 resolution 유지되도록 padding 자동으로 준다.
- `ZeroPadding2D`를 이용해서 패딩하는 레이어를 만들어서 사용할 수도 있다. 위처럼 내가 지정해서 (1,1) 패딩을 줄 수 있음

### 3.4 Convolution layer

```py
model = Sequential()
model.add(Conv2D(input_shape = (10, 10, 3), filters = 10, kernel_size = (3,3), strides = (1,1), padding = 'same'))
print(model.output_shape)
```

`filters` 파라미터 값을 통해 필터 몇 개 쓸건지 정해준다. 나머지는 일반적인 딥러닝의 개념들

### 3.5 Pooling

```py
# (10, 10, 10) -> (5, 5, 10)
model.add(MaxPooling2D(pool_size = (2,2), padding = 'valid'))

# (10, 10, 10) -> (9, 9, 10)
model.add(MaxPooling2D(pool_size = (2,2), strides = (1,1), padding = 'valid'))

# (10, 10, 10) -> (5, 5, 10)
model.add(AveragePooling2D(pool_size = (2,2), padding = 'valid'))

# (10, 10, 10) -> (None, 10)
model.add(GlobalMaxPooling2D())
```

- 일반 `MaxPooling2D`에서 stride를 지정하지 않으면 `pool_size`와 같은 값이 된다.
- Average, Global Max pooling 다 가능

### 3.6 Flatten

```py
model = Sequential()
model.add(Conv2D(input_shape = (10, 10, 3), filters = 10, kernel_size = (3,3), strides = (1,1), padding = 'same'))
model.add(Flatten())
```

`Flatten()` 레이어를 추가해주면 벡터로 쫙 펴준다. Fully connected 하기 직전.

### 3.7 Example

![inception](https://raw.githubusercontent.com/iamaaditya/iamaaditya.github.io/master/images/inception_1x1.png)

```py
def NetworkInNetwork():
    model = Sequential()
    
    model.add(Conv2D(input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same', kernel_initializer='he_normal'))
    model.add(Conv2D(filters = 25, kernel_size = (1,1), strides = (1,1), padding = 'valid', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same', kernel_initializer='he_normal'))
    model.add(Conv2D(filters = 25, kernel_size = (1,1), strides = (1,1), padding = 'valid', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same', kernel_initializer='he_normal'))
    model.add(Conv2D(filters = 25, kernel_size = (1,1), strides = (1,1), padding = 'valid', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same', kernel_initializer='he_normal'))
    model.add(Conv2D(filters = 25, kernel_size = (1,1), strides = (1,1), padding = 'valid', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same', kernel_initializer='he_normal'))
    model.add(Conv2D(filters = 25, kernel_size = (1,1), strides = (1,1), padding = 'valid', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size = (2,2)))
    
    model.add(Flatten())
    model.add(Dense(50, activation = 'relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax', kernel_initializer='he_normal'))
    
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model 
```
