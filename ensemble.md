# Ensemble

- Udacity video [bagging](https://www.youtube.com/watch?v=2Mg8QD0F1dQ)
- Udacity video [boosting](https://www.youtube.com/watch?v=GM3CDQfQ4sw)
- scikit-learn [article](http://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html)
- Maryland univ [slide](http://www.cs.umd.edu/class/spring2006/cmsc726/Lectures/EnsembleMethods.pdf)

머신러닝에서 ensemble 개념은 어떤 task를 수행할 떄 2개 이상의 모델을 사용해서 각 모델의 결과를 majority voting(classification) 혹은 averge(regression) 내서 사용하는 것을 말한다.

## 1. Bagging(Bootstrap Aggregating)

![bagging](https://i.imgur.com/4tGnVNR.png)

> source: Udacity youtube video

- Training data에서 Random with replacement(복원추출)로 샘플링
    + 30개의 subset을 만들었다면 각각을 D1, D2, ..., D30이라고 지칭한다.
    + 데이터를 1개씩 뽑기 때문에 같은 subset 내에 동일한 데이터가 여러개 들어갈 수 있다.
    + subset의 크기는 training data의 60%정도로 한다.
- 만들어진 데이터셋으로 각각 모델을 학습한다. 같은 모델을 여러개 쓸 수도 있고, 다른 모델을 쓸 수도 있다.(SVM, Logistic regression, decision tree, ...)
- test data를 모든 모델에 적용해서 앙상블로 하나의 결과를 낸다.

## 2. AdaBoost(Boosting)

![adaboost](https://i.imgur.com/v8VoSLk.png)

> source: Udacity youtube video


## 3. GBM, XGBoost

## 4. Stacking
