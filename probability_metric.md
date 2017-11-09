# Probability metric

Logistic regression에서 binary case 문제를 풀 때 단순히 accuracy만 측정해선 안된다. 만약 데이터가 class imbalance하다면 단순히 accuracy로만 학습시키면 잘못될 가능성이 있다.

어떤 metric을 사용하느냐에 따라 정답을 잘 맞출 것인가, 실수를 줄일 것인가가 정해진다.

## 1. Metrics

||정상 판정|암 판정|-|
|-|-|-|-|
|**정상**|988(TN)|2(FP)|TNR|
|**암**|1(FN)|9(TP)|recall|
|-|NPV|precision|

> FP를 False alarm이라고도 한다.

- **class imbalance** : 정상 데이터가 990개, 암 환자 데이터가 10개로 많이 편중돼있다.
- T(True), F(False) : 예측한게 맞았는지, 틀렸는지
- P(Positive), N(Negative) : 정하기 나름인데 "암"과 관련되서는 종양이 발견되면 Positive, 종양이 발견되지 않으면 Negative다.
- metric 간단 정리
    + precision: P 판정 중에서 맞춘 비율
    + recall(=TPR=sensitivity): 실제 P 중에서 맞춘 비율, True positive rate
    + TNR(=specificity): 실제 N 중에서 맞춘 비율, True negative rate
    + NPV(Negative predictive value) : N 판정 중에 맞춘 비율
    + FPR(False positive rate, `1-TNR` : N인데 P라고 한 비율

### 1.1 Accuracy

```
accuracy = (TN + TP) / (TN + FN + FP + TP)
```

- binary case가 아닌 multi class classification 문제에서는 다른 메트릭을 쓰기가 까다롭기 때문에 단순히 accuracy만 계산해서 쓴다. 전체 케이스 중 잘 추정한 것의 비율이다.
- 맨 위 데이터셋처럼 class imbalance하고 binary case라면 단순히 모두 정상이라고 판단만 해도 accuracy가 높게 나온다. 그래서 이거로만 하면 안된다.

### 1.2 Precision

```sh
precision = TP / (TP + FP)
# 9 / (9 + 2) = 81.8%
```

- P 판정한 것중에서 맞춘 비율
- precision이 높다는 것은: 진짜 암일 수 밖에 없는 애들만 골라냈다.

### 1.3 Recall

```sh
recall = TP / (TP + FN)
# 9 / (9 + 1) = 90%
```

- 실제 P 중에서 맞춘 비율
- recall이 높다면: 암일 것 같은 애들은 최대한 다 골라냈다. 마치 1차 검진처럼 먼저 최대한 골라낸 다음 나중에 제대로 검사할 것처럼.
- recall = TPR(True positive rate) = sensitivity

### 1.4 TNR

```sh
TNR = TN / (TN + FP)
# 988 / (988 + 2) = 99.8
```

- 실제 N 중에서 맞춘 비율
- TNR = True negative rate = specificity

### 1.5 FNR, FPR

- `FNR = 1 - TPR` : P 중에서 틀린 비율
- `FPR = 1 - TNR` : N 중에서 틀린 비율

### 1.6 F1-score

```sh
F1_score = 2 / (1/recall + 1/precision)
```

- recall과 precision의 조화평균
- 두 메트릭을 모두 고려하는 방식

## 2. ROC curve, AUC
