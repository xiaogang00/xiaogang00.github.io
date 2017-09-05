---
layout: post
title: PRML Chapter7 Sparse Kernel Machine
category: PRML
tags: machine_learning
---

* content
{:toc}




### 关于SVM

* SVM 是一个 discriminant function，即直接把一个 input 判别到一个分类中。相比之下， RVM是一个 discriminant model，即训练后得到一个后验概率；
* SVM 是一个 sparse model，即训练后只需要保存一部分 training data（ support vectors）。相比之下， Gaussian process 训练后仍旧要保存全部 training data 以用于做 prediction；
* SVM 的应用： classification， regression 和 novelty detection。



### SVM的modeling过程： linearly separable的情况

我们考虑二分类的问题，有N个数据$\{ x_n,t_n \}$，其中$t_n \in \{ -1,1 \}$

*  modeling 的基本思路:

  假设有$y(x) = w^T \phi(x) +b$，那么$y(x) =0$就定义了一个高维空间的超平面。 SVM 希望找
  到一个使得 margin 最大化的超平面作为 decision boundary。 所谓 margin，指的是在全部 N
  个数据中，离 decision boundary 最近的数据点到该超平面的距离。因此只要找到作为 decision
  boundary 的超平面$y(x) = 0$的参数w和b的值，即可完成SVM的训练。

* modeling 的 mathematical formulation

  首先，任意一个点$x_n$到超平面$y(x)=0$的距离是$\frac{t_n y(x_n)}{\parallel w \parallel}$，而当 linearly separable 时，约束条件$t_n y(x_n) > 0$对每个点都可以满足。因此可以写出 margin 最大化的目标函数：


$$
\begin{align} &arg \max \limits_{w,b} \{ \frac{1}{\parallel w \parallel}  \min\limits_n [t_n y(x_n)]\}  \\ &s.t. \quad t_ny(x_n)>0 , n = 1,\dots,N \end{align}
$$
​       

其中，约束条件表明找出来的超平面必须是将两个类正确分开的；而目标函数表明在所有正确分开两个类别的    超平面中，找 margin 最大的一个。我们可以据此人为规定：距超平面最近的点，到该平面的距离是 1，即$t_ny(x_n)=1$，那么，所有点到超平面的距离显然就有：$t_n y(x_n) \geq 1, n=1,\dots,N$。于是原目标函数就变成为在该约束条件下：


$$
\begin{align} &arg \max \limits_{w,b} \{ \frac{1}{2}\parallel w \parallel^2  \\ &s.t. \quad t_ny(x_n)\geq 1 , n = 1,\dots,N \end{align}
$$


之后的步骤是先构造 Lagrange function，通过kernel函数，KKT条件来找到上述问题的解。

SVM的解是有特点的，即只有 support vector才起作用，这些点满足$t_n y(x_n)=1$。SVM 模型的解的这些特点使其为一个 sparse model。



### multiclass情况的SVM

这仍旧是一个 open problem。一般采用 one-versus-the rest 的方法，这样就要训练 K 个 SVM。 这种有两方面的不足：第一，存在 ambiguous region，这个区域中的点会被判定为属于多个类别，出现了 inconsistency；第二，面临正负样本不均衡的可能，因为”the rest“的一方很可能远比”one“的一方数据要多。



### single-class情况的SVM

属于 unsupervised learning， 类似于 probability density estimation， 但其目的并不是估计density of data，而是 find a smooth boundary enclosing a region of high density。 该 boundary的选取方法是：给定一个 0 到 1 之间的概率值，使得从数据的 distribution 中采样出的数据点落入 region 的概率为给定的概率值。
有两种常见的思路解决该问题。

1. 第一种，在 feature space 中，找一个最小的超球，使其中包含的数据点不低于总数据的某一个比例值。
2. 第二种， 在 feature space 中，找一个超平面，要求该超平面与原点的距离最大化，同时满足在超平面的原点一侧的数据点数不超过总数据的某一个比例值。

与 Single-Class Classification 相类似的术语有： Outlier Detection， Novelty Detection 以及Concept Learning。



### SVM for regression

$\varepsilon-SVM$用于 regression 是这样：寻找一个尽可能 flat 的函数 y，使得对于所有的 training data， target value 与函数 y 计算的 value 之间的差值不超过 $\varepsilon$， 所谓尽可能 flat 的函数，就是要求这个函数尽可能的“简单”（ model
complexity），以免 overfitting。 这一点正可用 regularizer 来表达。因此可以把这个思想形式化：


$$
\begin{align}  &\min\{ \frac{1}{2}\parallel w \parallel^2 \} \\ & s.t. \quad \mid y(x_n)-t_n \mid < \varepsilon, n=1,\dots,N \end{align}
$$


以上 modeling 存在的问题跟 hard margin 分类的 SVM 一样，包括： (1) 可能无解（对应线性不可分）； (2) 即使有解，导致的模型也可能 complexity 太高，处于 overfitting 状态，对泛化不利。为此， 我们允许每个点可以超出在$\varepsilon − tube$ 之外——当然前提是需要为此付出一些“代价”。 $\varepsilon − SVM$ 用来衡量超出$\varepsilon − tube$ 的代价的函数是：


$$
 E_{\varepsilon}(y(x)-t) = \begin{cases} 0 & \mid y(x) -t \mid < \varepsilon\\ \mid y(x)-t \mid - \varepsilon & otherwise \end{cases}
$$


### Relevance Vector Machine： RVM

这是一个 sparse Bayesian model。相比 discriminant function 的 SVM， RVM 是一个 discriminant model，训练得到后验概率。而且 RVM 得到的结果一般比 SVM 更稀疏，因此更加有利 prediction 的效率。而 SVM与之相比，缺点是：complexity参数 C需要事先指定或者用 cross-validation来寻找；by definition， SVM 只能用来二类分类等。对于 RVM，方便而自然的思路是先从 regression 开始，然后应用于 classification。正和 SVM的过程正好相反。



#### RVM for regression

RVM 其实就是一个 linear basis function model，只不过参数w的先验概率稍有不同。 因此仍旧有以下：


$$
\begin{align} y(x,w) &= w^T \phi(x) \\ p(t\mid x,w,\beta) &=N(t \mid y(x,w), \beta^{-1}) \end{align}
$$


但是对于先验概率，则采用：$p(w \mid \alpha) = \prod\limits_{i=1}^M N(w_i \mid 0, \alpha_i^{-1})$

作为一个 Bayesian model， RVM 以通过 marginalize 来求 predictive distribution 为目标；但在此之前，需要用 maximize marginal likelihood 来估计出 hyper-parameter 的值， 也就是$\alpha$和$\beta$。

RVM 的稀疏性：在对 marginal likelihood 最大化（ with respect to $\alpha$和$\beta$ ）的过程中， $\alpha$向量的大量分量都将趋于无穷大，从而使分布$N(w_i \mid 0, \alpha_i ^{-1})$退化为一个固定的值 0；也就是说，这导致权重向量 $w$ 中大量的 $w_i$ 都将是 0。



#### RVM for classification

考虑 binary classification。 将$y(x,w)$做一个 logistic 变换，使其具有概率解释（认为是后验概率），也就是$y(x,w) = \sigma(w^T \phi(x)) = p(C_1 \mid x,w)$。然后假定参数的先验概率分布：


$$
p(w \mid \alpha) = \prod \limits_{i=1}^M N(w_i \mid 0, \alpha_i^{-1})
$$


先用 marginal likelihood 最大化的方法求出$\alpha^*$（对于 classification，已经没有$\beta$ ）。 剩下的工作就和 Bayesian Logistic Regression 类似了。 即：先用 Laplace approximation 估计出后验概率，然后再用 marginalization 的方法求出 predictive distribution。

RVM 和 Bayesian Logistic Regression 的不同只有一个，即 RVM 的 prior 和后者不同（ Bayesian Logistic Regression 的 prior 是$p(w)=N(w \mid m_0, S_0)$，没有用超参数，而是取了固定的 parameter）。 所以 RVM 有一个先行的工作是估计$\alpha^*$。 之后就跟 Bayesian Logistic Regression一样。