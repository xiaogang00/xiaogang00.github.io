---
layout: post
title: PRML Chapter6 Kernel methods
category: PRML
tags: machine_learning
---

* content
{:toc}






### 两种model类型

在训练完成后， training data 有可能舍弃也有可能保留用于做 prediction。

1. 第一种类型是把 training data 仅仅用于参数估计，估计出来后训练数据即可舍弃。例如：linear basis function model， generalized linear model， neural network 等；
2. 第二种类型是在做 predication 时仍旧需要用到 training data。这又可细分为两种。一种是全部 training data需要保存，例如：kNN，Gaussian process等；另一种只需要保存一部分 training data，例如： SVM（只需要保存 Support Vector）。



### Kernel的定义和构造

假设$phi$ 是一个non-linear feature space mapping，将 input x 映射到特征空间中。那么 kernel
function 就是： $k(x,x') = \phi(x)^T \phi(x')^T$

两 种 典 型 kernel ： stationary kernel 和 homogeneous kernel 。前者的性质是 $k(x,x') = k(x-x')$，具有平移不变性；后者的性质是$k(x,x') = k(\parallel x-x' \parallel)$，也就是说仅依赖于参数之间的距离（ radial basis kernel）。



Kernel 函数的构造有两条途径：

1. 第一种，按照 kernel 的定义，即先确定出一个特征映射，然后求得 kernel； 
2. 第二种，直接确定一个函数 k，使其存在一个特征映射，满足$k(x,x') = \phi(x)^T \phi(x')$。为了更方便的利用第二种方法构造 kernel，需要研究按前面方式定义的 kernel 函数，具有怎样的性质。 例如，可以证明两个 kernel 函数相加、相乘后都还是 kernel；一个 kernel 乘上一个正数后也还是 kernel；一个 kernel 的指数函数也还是 kernel。 基于这些性质，可以通过简单的 kernel 构造出复杂的 kernel。



### 基于generative model构造kernel

一般来说，融合 generative model 与 discriminant model 的 方法是：先用 generative model 定义出一个kernel 函数，然后将 kernel 使用于 discriminant model。

利用 generative model，有两种方式方法定义 kernel：

1. 简单定义$k(x,x') = p(x)p(x')$，这显然是一个 kernel，其中概率$p(x) \geq 0$。因此 k 相当于是 1 维空间里的内积。 还可以构建出更加复杂的函数形式：$k(x,x') = \sum\limits_i p(x\mid i)p(x' \mid i) p(i)$

2. 第二种， 构建 Fish kernel。 考虑一个 parametric generative model， $p(x \mid \theta)$ ，其中$\theta$是一个 parameter vector。首先定义Fisher score：$g(\theta, x)=\triangledown_{\theta}ln(p(x\mid \theta))$，这是一个 scalar 函数对一个 vector 求导，所以得到的是一个 vector（梯度）。

   由此我们可以得到Fisher kernel:


$$
k(x,x') = g(\theta, x)^T F^{-1} g(\theta, x')
$$


其中F是 Fisher information matrix，公式是：$F = E_x[g(\theta, x) g(\theta, x)^T]$。Fisher 信息量在实际中可以用 sample average来近似。

Fisher score 是对数概率密度函数关于参数的梯度， Fisher information matrix 是 Fisher score（一个 random vector）的 covariance matrix；而 Fisher kernel 则是两个Fisher score 之间的 Mahalanobis 距离。 



### Gaussian process for regression

一般来说对于任意的$x_1,\dots,x_N$，得到一组随机变量$y(x_1),\dots,y(x_N)$，这组随机变量是一个
Gaussian distribution。我们用 kernel函数来确定$(y(x_n),y(x_m))$的 covariance：


$$
E[y(x_n)y(x_m)] = k(x_n, x_m)
$$


考虑到每个 observation 都还有一定的 noise，也就是对于随机变量$y(x_n)$，真正观察到的值的分布$t_n$应该是：$p(t_n \mid y_n) = N(t_n \mid y_n,\beta^{-1})$

根据 Gaussian process 的定义， $y=(y_1, \dots,y_N) $是一个Gaussian 分布，也就是说：$p(y) = N(y\mid 0,K)$，其中K是kernel。根据上面两个结论可以获得联合分布$p(t)$。

**为Gaussian process选kernel**

例如可以这样选择 kernel：


$$
k(x_n, x_m) = \theta_0 exp\{ -\frac{\theta1}{2} \parallel x_n - x_m \parallel \} + \theta_2 + \theta_3 x_n^T x_m
$$


这里即含有 Gaussian kernel，也含有 linear kernel 与 constant。 对于多个 kernel 线性组合出来的 kernel 而言，这个过程似乎是在进行“ kernel selection”。$\theta$可视为一个 hyper-parameter，因此对于它的估计可视为是 empirical Bayes。此时 marginal likelihood是基于联合概率分布 $p( t )$ 得到。



