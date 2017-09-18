---
layout: post
title: PRML Chapter14 Combining Models
category: PRML
tags: machine_learning
---

* content
{:toc}




### Model combination的两种方式

一般主要分为两种形式：model averaging 和 model selection。

1. Model averaging 的例子是： Boost， mixture of linear regressioin， mixture of logtistic regression以及之前的 GMM。
2. Model selection 的例子是： Decision tree。



### Bayesian model averaging V.S. Combined model

二者的区别是：对于观察到的 data set D， Bayesian model averaging 认为它来自 a single model，尽管当前还无法确定到底是哪一个；而 Combined model 中， D 的数据可能来自不同的 model，并由它们混合而成。 后者的例子是 GMM，对于一个 D，它里面的数据可能是来自 K 个不同的 Gaussian model。



### Decision tree

Decision tree 包括： CART (Classification And Regression Tree)及其变化体 ID3 和 C4.5。

**作为 combined model 的 decision tree**
PRML把 decision tree视为一种 combined model：每片树叶就是一个 model，负责 input space中某一个 region 的 data point；对于给定一个 data point 的 regression/classification，从 decision tree 的 root 到 leaf 的决策过程，就是一个 model selection 的过程，即决定这个 data point 将由哪一个 model（ leaf）来处理。



**Decision tree 的目标函数**
假设有 D 维的数据集$X = \{ x_n:n=1,\dots,N \}$，以及他们相对应的labels $t = \{t_n:n =1,\dots,N \}$（对
regression 是连续值，对 classification 是离散值）。假设最后训练所得的树中 leaf 的集合是 T。第$\tau$个leaf代表input space 中的 region $R_{\tau}$，并设其中含有个 $N_{\tau}$data point。

对于 regression，区域$R_{\tau}$的optimal prediction 是：$y_{\tau} = \frac{1}{N_{\tau}} \sum\limits_{x_n \in R_{\tau}} t_n$

因此该区域中的 sum-of-squares error 是：


$$
Q_{\tau}(T) = \sum\limits_{x_n \in R_{\tau}} \{ t_n - y_{\tau} \}^2
$$


对于 classification，设$p_{\tau k}$是 region $R_{\tau}$中属于类别 k 的样本数。有两个常用的函数来度量训练的目标：




$$
\begin{align} Q_{\tau} &= \sum\limits_k p_{\tau k} ln [p_{\tau k}] --- cross entropy \\ Q_{\tau}(T) &= \sum\limits_k  p_{\tau k} (1-p_{\tau k})  --- Gini index\end{align}
$$


为了避免 over-fitting（比如每个 leaf 只有一个 data point，那么 training error 就是 0 ，但这样显然不合理），需要对 decision tree 进行控制 model complexity。一种方法是控制 Leaf的数目。这时候得到 regression/classification 的目标函数：


$$
C(T) = \sum\limits_{\tau = 1}^{\parallel T \parallel} Q_{\tau}(T) + \lambda \mid T \mid
$$


Decision tree 的训练过程就是最小化 C(T)的过程。

**Decision tree 的训练过程**
初始时整个 tree 就一个 root。然后反复进行下述步骤，直到满足一定的停止准则：

1. 选择当前 tree 的一个 leaf；
2. 选择 D 个 feature（ D-dimension data point 就是每个 data point 有 D 个 feature）中的一个feature；
3. 为选出来的 feature 选择一个 threshold；
4. 根据该 threshold，从当前 leaf 分裂出两个子节点， leaf 中所有其 feature 值大于 threshold的 data point 分入一个子节点，剩下的 data point 分入另一个子节点；从而得到一颗新的树，新树的 leaf 比原树多一个。

**Deision tree 的优缺点**
优点： human interpretability 好
缺点： tree struture 对 data set 很敏感，对 training data 稍许的改变，可能就导致训练出一个 tree struture 很不一样的树。



### 不同类型的Mixture Model

**Unconditional Mixture Model**：例如 ， 经典的混合模型 Gaussian Mixture Model (GMM)，有：

$p(x) = \sum\limits_{k=1}^K \pi_k N(x\mid \mu_k, \Sigma_k)$。这个分布由参数$\pi, \mu, \Sigma$控制。



**Conditional Mixture Model**： 例如， Mixtures of linear regression models，即将多个 linear regression model 混合成一个模型：$p(t \mid \theta) = \sum\limits_{k=1}^K \pi_k N(t\mid w_k^T \phi, \beta^{-1})$，其中$\theta = \{ W,\pi, \beta\}$。这个分布不仅依赖于参数$\theta = \{ W,\pi,\beta \}$，还依赖于 input $\phi$ ，所以是 conditional mixturemodel。



**Mixture of Expert**：在 Mixtures of linear regression models 中， mixing coefficients $\pi$ 是不依赖于 input 的；如果把对此条件放松，使$\pi$也依赖于 input，那么就得到了 mixture of expert模型：$p(t\mid x) = \sum\limits_{k=1}^K \pi_k(x) p_k(t\mid x)$。其中一个分支$p_k(t\mid x)$就是一个 expert。



**Hierarchical Mixture of Expert**：让 mixture of expert 的每一个分支$p_k(t\mid x)$本身为一个mixture of expert，这样就有了层次结构。

以上连最简单的 Unconditional Mixture Model 例如 GMM 都没有 closed-form solution，因此都需要 EM 算法来求解。



### AdaBoost

属于 model combination中的 model averaging。假设训练出 M个 model$\{y_m(x):m=1,\dots,M \}$，那么最后的 model 由这 M 个 Model 加权组合得到$Y(x) = sgn(\sum\limits_{m=1}^M \alpha_m y_m(x))$，其中$\alpha_m$是模型$y_m(x)$的权值，将在训练中给出。 这里考虑的是 binary classification，所以用符号 sgn 判断类别。

M 个 model $y_m(x)$将按顺序依次训练得到。而且每个模型$y_m(x)$的训练将根据前一模型的情况$y_{m-1}(x)$进行。具体做法是：

1. 为每个 data point 赋予一个权重 $w_n^{(m)}$，初始时(m=1)为$\frac{1}{N}$
2. 每次训练一个模型 $y_m(x)$时，将以最小化函数$J_m = \sum\limits_{n=1}^N w_n^{(m)} I(y_m(x_n) \neq t_n)$为目标，其中$t_n \in \{ -1,+1 \}$是类别标记，$I$是 indicator function；
3. 计算模型 $y_m(x)$的加权错误率$\varepsilon_m = \frac{\sum\limits_{n=1}^N w_n^{(m)}I(y_m(x_n)\neq t_n)}{\sum\limits_{n=1}^N w_n^{(m)}}$，因此得到模型 $y_m(x)$的权值$\alpha_m = ln\{ \frac{1-\varepsilon_m}{\varepsilon_m} \}$；
4. 根据$y_m(x)$的训练情况重新调整每个 data point 的权值，以为下一个模型的训练用：$w_n^{(m+1)} = w_n^{(m)} exp\{ \alpha_m I(y_m(x_n) \neq t_n) \}$。



问题： AdaBoost 整体(即$Y(x) =sgn(\sum\limits_{m=1}^M \alpha_m y_m(x))$)的目标优化函数是什么？

回答：是一个 exponential error function，$E = \sum\limits_{n=1}^N exp\{ -t_n f(x_n) \}$，其中$t_n \in \{ -1,+1 \}$是类别标记。$f(x_n) = \frac{1}{2}\sum\limits_{m=1}^M \alpha_m y_m(x)$是combined model。 AdaBoost 是一个对此目标函数的 sequential optimization。