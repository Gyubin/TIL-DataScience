# Probability metric

Logistic regression에서 binary case 문제를 풀 때 단순히 accuracy만 측정해선 안된다. 만약 class imbalance 문제가 있다면, 즉 데이터 1000개 중 정상이 990개, 암이 10개 있다면 단순히 accuracy로만 학습시키면 잘못될 가능성이 있다.

어떤 metric을 사용하느냐에 따라 정답을 잘 맞출 것인가, 실수를 줄일 것인가가 정해진다.

## 1. Metrics

||정상 판정|암 판정|
|-|-|-|
|**정상**|988(TN)|2(FP=False alarm)|
|**암**|1(FN)|9(TP)|



### 1.1 Precision

