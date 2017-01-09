---
layout: post
title: 机器学习中的损失函数
comments: false
---

<!--more-->

在机器学习中，损失函数是用来衡量预测结果与实际值之间差别大小的指标。一般的损失函数有5五种：

### 一. Gold Standard（标准式，0-1式）###

主要用于理想sample，这种一般很少有实践场景，这个方法的作用更多的是用来衡量其他损失函数的效果。表达式如下：

$$L(m)=\begin{cases}
0 & \textrm{if} \ \ m\geq0 \\
1 & \textrm{if} \ \ m<0
\end{cases}$$

### 二. Hinge ###

主要用于maximum-margin的分类算法，如svm算法。Hinge损失函数的描述如下式：

$$L(y)=\max(0,1-t\dot{}y)$$

这里$t=1~or~-1$，$y$是预测值，
而$t$是实际真实值，可以看出，当分类正确时，$y$和$t$会有相同的符号且$|y|\geqslant 1$
（$|y|>1$表示：相比于支持向量，该点距离分类边界更远），
此时损失函数$L(y)$的值为0；
分类错误时，$y$和$t$符号相反，
$L(y)$将随y变大。

### 三. Logarithmic Loss（对数损失）###

主要用于逻辑回归算法（Logistric Regression），在kaggle比赛里面衡量算法性能的指标往往是[logloss](https://www.kaggle.com/wiki/LogarithmicLoss)。表达式如下：

$$logloss=-\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{M}y_{ij}\log(p_{ij})$$

这里$$N$$是样本的数量，$M$是类别数量，$y_{ij}$和$p_{ij}$都是二值型标志位，表示第i个样本是否属于第j类，$y$表示真实值，$p$表示预测值。logloss越小说明算法越好。在实际编程应用中注意添加一个冗余项$(1e-15)$之类的，避免出现$\log 0$这样的情况。

### 四. squared loss（平方损失） ###

主要用于线性回归（Liner Regression），平方损失也可以理解为最小二乘法，基本原则很好理解，即最优拟合曲线应该是是点到回归曲线的距离和最小的直线，也就是平方和最小，表达式如下：

$$L(Y,f(X))=\sum^{N}_{i=1}(y_i-f(x_i))^2$$

这里$N$是样本的数量，$y$是真实值，$f(x)$是预测值。

### 五. exp-loss（指数损失） ###

主要用于Boosting算法，对于拥有$N$个样本的情况下，指数损失的函数表达式如下：

$$L(Y,f(X))=\frac{1}{N}\sum_{i=1}^{N}\exp[-y_i~f(x_i)]$$

$y$是真实值，$f(x)$是预测值。
