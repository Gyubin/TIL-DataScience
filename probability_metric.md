# Probability metric

Logistic regression에서 binary case 문제를 풀 때 단순히 accuracy만 측정해선 안된다. 만약 데이터가 class imbalance하다면 단순히 accuracy로만 학습시키면 잘못될 가능성이 있다.

어떤 metric을 사용하느냐에 따라 정답을 잘 맞출 것인가, 실수를 줄일 것인가가 정해진다.

## 1. Metrics

||정상 판정|암 판정|-|
|-|-|-|-|
|**정상**|988(TN)|2(FP)|TNR|
|**암**|1(FN)|9(TP)|recall|
|-|NPV|precision|

**class imbalance** : 정상 데이터가 990개, 암 환자 데이터가 10개로 많이 편중돼있다.

=False alarm

### 1.1 Accuracy

```
accuracy = (TN + TP) / (TN + FN + FP + TP)
```

- binary case가 아닌 multi class classification 문제에서는 다른 메트릭을 쓰기가 까다롭기 때문에 단순히 accuracy만 계산해서 쓴다. 전체 케이스 중 잘 추정한 것의 비율이다.
- 맨 위 데이터셋처럼 class imbalance하고 binary case라면 단순히 모두 정상이라고 판단만 해도 accuracy가 높게 나온다. 그래서 이거로만 하면 안된다.

### 1.2 Precision

```
precision = TP / (TP + FP)
```

### 1.3 Recall






